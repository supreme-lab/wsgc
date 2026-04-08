# ============================================================
# run_pipeline.py  —  Main orchestration script
# ============================================================
"""
Full pipeline for safety geometry drift analysis.

Steps per model:
  1.  Load original aligned model (frozen reference)
  2.  Build probe + benign datasets
  3.  Extract original representations + evaluate original refusal rate
  4.  Compute safety subspace (SVD per layer)
  5.  UMAP reduction of original activations
  6.  Baseline fine-tuning → extract representations → evaluate → delete model
  7.  FW-SSR fine-tuning  → extract representations → evaluate → delete model
  8.  Compute all geometry metrics using saved activations
  9.  Generate 6 publication figures
  10. Combine all evaluation results and print final table

Usage:
    python run_pipeline.py                        # Uses defaults in config.py
    python run_pipeline.py --model mistralai/...  # Override model
    python run_pipeline.py --no-eval              # Skip slow generation eval
"""
import os
import json
import argparse
import random
import numpy as np
import torch

from config       import PipelineConfig
from data_utils   import SafetyProbeDataset, BenignFinetuneDataset, get_dataloader
from model_utils  import (
    load_model_and_tokenizer, wrap_with_lora,
    get_layer_indices, extract_representations,
    reference_on_gpu,
)
from geometry_utils import (
    compute_safety_subspace, compute_safety_drift,
    compute_cka, compute_class_separation, reduce_dimensions,
)
from trainers      import BaselineTrainer, FWSSRTrainer
from evaluation    import compute_refusal_rate, print_results_table
from visualization import (
    plot_latent_space_comparison,
    plot_drift_over_training,
    plot_drift_metrics_bar,
    plot_layer_wise_drift,
    plot_fisher_weights,
    plot_cka_matrix,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_behavior(model, tokenizer, geo_metrics, device, n_prompts, condition_name):
    """Evaluate refusal behavior of a model and merge with geometry metrics."""
    print(f"  [Eval] Behavioral evaluation: '{condition_name}'")
    behavior = compute_refusal_rate(
        model, tokenizer, device=device, n_prompts=n_prompts
    )
    print(f"    Refusal={behavior['refusal_rate']:.1%}  "
          f"Compliance={behavior['compliance_rate']:.1%}  "
          f"Ambiguous={behavior['ambiguous_rate']:.1%}")
    return {
        "refusal_rate":         behavior["refusal_rate"],
        "compliance_rate":      behavior["compliance_rate"],
        "ambiguous_rate":       behavior["ambiguous_rate"],
        **{k: v for k, v in geo_metrics.items()
           if isinstance(v, (int, float, np.floating))},
    }


def run_single_model(model_name: str, config: PipelineConfig, run_eval: bool = True):
    """Full pipeline for a single model."""
    set_seed(config.seed)
    device     = config.device if torch.cuda.is_available() else "cpu"
    short_name = model_name.split("/")[-1]
    n_eval     = 20

    os.makedirs(config.figures_dir,    exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir,     exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*65}")

    # ── Step 1: Load Original Model (frozen reference) ───────────────────────
    print("\n[Step 1] Loading original aligned model (reference)...")
    offload = getattr(config, "offload_reference_model", False)
    original_model, tokenizer = load_model_and_tokenizer(
        model_name, config, for_reference=offload
    )
    if not offload:
        original_model.to(device)
    for p in original_model.parameters():
        p.requires_grad_(False)

    # ── Step 2: Build Datasets ───────────────────────────────────────────────
    print("\n[Step 2] Building datasets...")
    probe_dataset = SafetyProbeDataset(
        tokenizer,
        n_samples=config.safety_probe_samples,
        max_length=128,
        seed=config.seed,
    )
    benign_dataset = BenignFinetuneDataset(
        tokenizer,
        max_samples=config.benign_max_samples,
        max_length=config.max_seq_length,
        seed=config.seed,
    )
    probe_dl  = get_dataloader(probe_dataset,  batch_size=32, shuffle=False)
    benign_dl = get_dataloader(benign_dataset, batch_size=config.finetune_batch_size)
    print(f"  Probe samples:  {len(probe_dataset)}")
    print(f"  Benign samples: {len(benign_dataset)}")

    # ── Step 3: Extract Original Representations + Evaluate ──────────────────
    print("\n[Step 3] Extracting original model representations...")
    layer_indices = get_layer_indices(original_model, config)

    orig_acts, labels = extract_representations(
        original_model, probe_dl, layer_indices, device=device,
    )

    # Evaluate original model immediately while it's loaded
    orig_geo = {
        "safety_drift": 0.0, "drift_ratio": 0.0,
        "orthogonal_drift": 0.0, "cosine_similarity": 1.0,
        "cka_vs_original": 1.0,
    }
    if run_eval:
        # Move reference to GPU if offloaded for evaluation
        with reference_on_gpu(original_model, device):
            orig_results = _eval_behavior(
                original_model, tokenizer, orig_geo, device, n_eval, "original"
            )
    else:
        orig_results = orig_geo.copy()

    # ── Step 4: Compute Safety Subspace ──────────────────────────────────────
    print("\n[Step 4] Computing safety subspace (SVD per layer)...")
    labels_arr = np.array(labels)
    safety_subspaces = {}

    for layer_idx in layer_indices:
        acts        = orig_acts[layer_idx]
        harm_acts   = acts[labels_arr == 1]
        benign_acts = acts[labels_arr == 0]
        U, sv = compute_safety_subspace(
            harm_acts, benign_acts, k=config.safety_subspace_dim
        )
        safety_subspaces[layer_idx] = U.to(device)
        print(f"  Layer {layer_idx}: subspace dim={U.shape[1]}, "
              f"top-SV={sv[0]:.4f}, explained={sv.sum():.2f}")

    # ── Step 5: UMAP of Original Representations ─────────────────────────────
    print("\n[Step 5] Reducing original representations for visualization...")
    vis_layer = layer_indices[-1]
    orig_2d   = reduce_dimensions(
        orig_acts[vis_layer].cpu(),
        method=config.reduction_method,
        pca_components=config.pca_components,
        seed=config.seed,
    )
    orig_class_sep = compute_class_separation(orig_acts[vis_layer].cpu(), labels)
    print(f"  Original Fisher Score (layer {vis_layer}): {orig_class_sep['fisher_score']:.4f}")
    orig_results.update({k: v for k, v in orig_class_sep.items()
                         if isinstance(v, (int, float, np.floating))})

    # ── Step 6: Baseline Fine-tuning → Evaluate → Delete ─────────────────────
    print("\n[Step 6] Baseline fine-tuning (no safety constraints)...")
    baseline_model, _ = load_model_and_tokenizer(model_name, config, for_reference=False)
    if config.use_lora:
        baseline_model = wrap_with_lora(baseline_model, config)
    for p in original_model.parameters():
        p.requires_grad_(False)

    baseline_trainer = BaselineTrainer(
        model=baseline_model,
        original_model=original_model,
        safety_subspaces=safety_subspaces,
        probe_dataloader=probe_dl,
        config=config,
        device=device,
    )
    baseline_trainer.train(benign_dl, epochs=config.finetune_epochs)

    # Extract baseline representations
    baseline_acts, _ = extract_representations(
        baseline_model, probe_dl, layer_indices, device=device,
    )
    baseline_2d = reduce_dimensions(
        baseline_acts[vis_layer].cpu(),
        method=config.reduction_method,
        pca_components=config.pca_components,
        seed=config.seed,
    )

    # Evaluate baseline immediately
    baseline_geo = {}   # filled in Step 8 after fwssr; placeholder for eval
    if run_eval:
        baseline_results = _eval_behavior(
            baseline_model, tokenizer, baseline_geo, device, n_eval, "finetuned"
        )
    else:
        baseline_results = {}

    # Save training histories before deleting
    baseline_drift_history = list(baseline_trainer.drift_history)
    baseline_loss_history  = list(baseline_trainer.loss_history)

    # Delete baseline model to free VRAM
    del baseline_model, baseline_trainer
    torch.cuda.empty_cache()
    print("  [Memory] Baseline model deleted.")

    # ── Step 7: FW-SSR Fine-tuning → Evaluate → Delete ───────────────────────
    print("\n[Step 7] FW-SSR mitigated fine-tuning...")
    fwssr_model, _ = load_model_and_tokenizer(model_name, config, for_reference=False)
    if config.use_lora:
        fwssr_model = wrap_with_lora(fwssr_model, config)

    fwssr_trainer = FWSSRTrainer(
        model=fwssr_model,
        original_model=original_model,
        safety_subspaces=safety_subspaces,
        probe_dataloader=probe_dl,
        config=config,
        device=device,
    )
    fwssr_trainer.train(benign_dl, epochs=config.finetune_epochs)

    # Extract FW-SSR representations
    fwssr_acts, _ = extract_representations(
        fwssr_model, probe_dl, layer_indices, device=device,
    )
    fwssr_2d = reduce_dimensions(
        fwssr_acts[vis_layer].cpu(),
        method=config.reduction_method,
        pca_components=config.pca_components,
        seed=config.seed,
    )

    # Evaluate FW-SSR immediately
    fwssr_geo = {}   # filled in Step 8
    if run_eval:
        fwssr_results = _eval_behavior(
            fwssr_model, tokenizer, fwssr_geo, device, n_eval, "mitigated"
        )
    else:
        fwssr_results = {}

    # Save training histories before deleting
    fwssr_drift_history   = list(fwssr_trainer.drift_history)
    fwssr_loss_history    = list(fwssr_trainer.loss_history)
    fisher_weights_saved  = {
        idx: fwssr_trainer.fisher_weights[idx].cpu().numpy()
        for idx in layer_indices
    }

    # Delete FW-SSR model to free VRAM
    del fwssr_model, fwssr_trainer
    torch.cuda.empty_cache()
    print("  [Memory] FW-SSR model deleted.")

    # ── Step 8: Geometry Metrics (using saved activations) ───────────────────
    print("\n[Step 8] Computing geometry metrics...")

    layer_drift_baseline = {}
    layer_drift_fwssr    = {}
    per_condition_metrics = {
        "original":  dict(orig_results),
        "finetuned": dict(baseline_results),
        "mitigated": dict(fwssr_results),
    }

    for idx in layer_indices:
        U = safety_subspaces[idx].cpu()

        b_drift = compute_safety_drift(orig_acts[idx].cpu(), baseline_acts[idx].cpu(), U)
        f_drift = compute_safety_drift(orig_acts[idx].cpu(), fwssr_acts[idx].cpu(),    U)

        layer_drift_baseline[idx] = b_drift["safety_drift"]
        layer_drift_fwssr[idx]    = f_drift["safety_drift"]

        b_sep = compute_class_separation(baseline_acts[idx].cpu(), labels)
        f_sep = compute_class_separation(fwssr_acts[idx].cpu(),    labels)
        o_sep = compute_class_separation(orig_acts[idx].cpu(),     labels)

        cka_bo = compute_cka(orig_acts[idx][:64].cpu(), baseline_acts[idx][:64].cpu())
        cka_fo = compute_cka(orig_acts[idx][:64].cpu(), fwssr_acts[idx][:64].cpu())

        print(f"\n  Layer {idx}:")
        print(f"    Baseline  | safety_drift={b_drift['safety_drift']:.4f} | "
              f"drift_ratio={b_drift['drift_ratio']:.4f} | fisher={b_sep['fisher_score']:.4f} | "
              f"CKA(vs orig)={cka_bo:.4f}")
        print(f"    FW-SSR    | safety_drift={f_drift['safety_drift']:.4f} | "
              f"drift_ratio={f_drift['drift_ratio']:.4f} | fisher={f_sep['fisher_score']:.4f} | "
              f"CKA(vs orig)={cka_fo:.4f}")

        # Merge geometry into results dicts (all conditions, vis_layer used for table)
        if idx == vis_layer:
            per_condition_metrics["original"].update({
                **{k: v for k, v in o_sep.items() if isinstance(v, (int, float, np.floating))},
                "safety_drift": 0.0, "drift_ratio": 0.0,
                "orthogonal_drift": 0.0, "cosine_similarity": 1.0,
                "cka_vs_original": 1.0,
            })
            per_condition_metrics["finetuned"].update({
                **{k: v for k, v in b_sep.items() if isinstance(v, (int, float, np.floating))},
                **b_drift, "cka_vs_original": cka_bo,
            })
            per_condition_metrics["mitigated"].update({
                **{k: v for k, v in f_sep.items() if isinstance(v, (int, float, np.floating))},
                **f_drift, "cka_vs_original": cka_fo,
            })

    # CKA matrix values
    cka_values = {
        ("original",  "finetuned"): compute_cka(orig_acts[vis_layer][:64].cpu(), baseline_acts[vis_layer][:64].cpu()),
        ("original",  "mitigated"): compute_cka(orig_acts[vis_layer][:64].cpu(), fwssr_acts[vis_layer][:64].cpu()),
        ("finetuned", "mitigated"): compute_cka(baseline_acts[vis_layer][:64].cpu(), fwssr_acts[vis_layer][:64].cpu()),
    }

    # ── Step 9: Generate Figures ──────────────────────────────────────────────
    print("\n[Step 9] Generating publication figures...")
    fig_dir = config.figures_dir
    fmt     = config.plot_format

    plot_latent_space_comparison(
        orig_2d, baseline_2d, fwssr_2d, labels, vis_layer,
        save_path=f"{fig_dir}/{short_name}_fig1_latent_space", fmt=fmt,
    )
    plot_drift_over_training(
        baseline_drift_history, fwssr_drift_history,
        baseline_loss_history,  fwssr_loss_history,
        save_path=f"{fig_dir}/{short_name}_fig2_drift_over_training", fmt=fmt,
    )
    plot_drift_metrics_bar(
        per_condition_metrics,
        save_path=f"{fig_dir}/{short_name}_fig3_drift_metrics", fmt=fmt,
    )
    plot_layer_wise_drift(
        {"finetuned": layer_drift_baseline, "mitigated": layer_drift_fwssr},
        save_path=f"{fig_dir}/{short_name}_fig4_layer_drift", fmt=fmt,
    )
    plot_fisher_weights(
        fisher_weights_saved,
        save_path=f"{fig_dir}/{short_name}_fig5_fisher_weights", fmt=fmt,
    )
    plot_cka_matrix(
        cka_values,
        save_path=f"{fig_dir}/{short_name}_fig6_cka_matrix", fmt=fmt,
    )

    # ── Step 10: Print Combined Results Table + Save ──────────────────────────
    print("\n[Step 10] Final combined evaluation results...")
    print_results_table(per_condition_metrics)

    out_path = f"{config.output_dir}/{short_name}_eval_results.json"
    with open(out_path, "w") as f:
        clean = {}
        for cond, metrics in per_condition_metrics.items():
            clean[cond] = {k: float(v) for k, v in metrics.items()
                           if isinstance(v, (int, float, np.floating))}
        json.dump(clean, f, indent=2)
    print(f"  [Saved] {out_path}")

    return per_condition_metrics


def main():
    parser = argparse.ArgumentParser(description="Safety Geometry Drift Pipeline")
    parser.add_argument("--model",   type=str,   default=None,
                        help="HuggingFace model name (overrides config)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip slow generation-based evaluation")
    parser.add_argument("--lambda",  type=float, default=None, dest="lam",
                        help="FW-SSR lambda override")
    parser.add_argument("--epochs",  type=int,   default=None)
    args = parser.parse_args()

    config = PipelineConfig()
    if args.model:
        config.model_names = [args.model]
    if args.lam:
        config.fwssr_lambda = args.lam
    if args.epochs:
        config.finetune_epochs = args.epochs

    print(f"\nFW-SSR Safety Geometry Pipeline")
    print(f"  Models: {config.model_names}")
    print(f"  FW-SSR λ: {config.fwssr_lambda}")
    print(f"  Epochs:   {config.finetune_epochs}")
    print(f"  Device:   {config.device if torch.cuda.is_available() else 'cpu (no CUDA)'}")

    all_results = {}
    for model_name in config.model_names:
        results = run_single_model(
            model_name, config, run_eval=not args.no_eval
        )
        all_results[model_name] = results

    # Cross-model comparison summary
    if len(all_results) > 1:
        print("\n" + "="*65)
        print("  CROSS-MODEL COMPARISON SUMMARY")
        print("="*65)
        for model_name, results in all_results.items():
            print(f"\n  {model_name}")
            for cond, metrics in results.items():
                r = metrics.get("refusal_rate", float("nan"))
                d = metrics.get("safety_drift",  float("nan"))
                print(f"    {cond:<12} | refusal={r:.1%}  safety_drift={d:.4f}")

    print("\n[Done] All figures saved to:", config.figures_dir)
    print("[Done] Results saved to:",      config.output_dir)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=3 nohup python run_pipeline.py > logs/run_pipeline_1.log 2>&1 &