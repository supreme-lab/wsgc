# ============================================================
# model_utils.py  —  Model loading and activation extraction
# ============================================================
"""
Handles:
  - Loading guard/aligned LLMs from HuggingFace
  - LoRA wrapping for parameter-efficient fine-tuning
  - Activation hook registration for representation extraction
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from config import PipelineConfig


def load_model_and_tokenizer(
    model_name: str,
    config: PipelineConfig,
    load_in_8bit: bool = False,      # legacy param; config.use_8bit preferred
    for_reference: bool = False,     # True = frozen reference copy → CPU offload
) -> Tuple[nn.Module, object]:
    """
    Loads a causal LM with memory-efficient settings.

    Memory strategy for large models (≥7B) on a single GPU:
      - QLoRA (4-bit NF4): base weights use ~4GB instead of ~24GB for a 12B model
      - Gradient checkpointing: recomputes activations instead of storing them
      - Reference model CPU offload: keeps frozen copy in RAM, moves to GPU per batch

    for_reference=True: loads in fp16 to CPU (no quantization needed since we
    never backprop through it). Saves the full quantized VRAM slot for the
    trainable model.
    """
    import os
    hf_token = config.hf_token or os.environ.get("HF_TOKEN", None)

    use_4bit = getattr(config, "use_4bit", False) and not for_reference
    use_8bit = (load_in_8bit or getattr(config, "use_8bit", False)) and not for_reference

    mode_str = "4-bit QLoRA" if use_4bit else ("8-bit" if use_8bit else "fp16")
    dest_str  = "CPU (reference)" if for_reference else "GPU"
    print(f"\n[Model] Loading '{model_name}' [{mode_str} → {dest_str}] ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Quantization config ───────────────────────────────────────────────────
    bnb_config = None
    if use_4bit:
        compute_dtype = getattr(torch, getattr(config, "bnb_4bit_compute_dtype", "float16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=getattr(config, "bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 bits/param
        )
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Load model ───────────────────────────────────────────────────────────
    load_kwargs = dict(
        token=hf_token,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    if for_reference:
        # Reference model: fp16 on CPU — no grad, no quant needed
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"]  = "cpu"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if config.fp16 else torch.float32
        load_kwargs["device_map"]  = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # ── Gradient checkpointing (trainable model only) ─────────────────────────
    if not for_reference and getattr(config, "use_gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("  [Memory] Gradient checkpointing enabled")

    # ── Llama 4 / LlamaGuard-4 patch ─────────────────────────────────────────
    chunk_size = 8192
    patched = []

    def _patch_cfg(cfg, label):
        if cfg is None:
            return
        if not getattr(cfg, "attention_chunk_size", None):
            try:
                cfg.attention_chunk_size = chunk_size
                patched.append(label)
            except Exception:
                pass

    _patch_cfg(model.config, "model.config")
    _patch_cfg(getattr(model.config, "text_config", None), "model.config.text_config")
    for key, val in vars(model.config).items():
        if hasattr(val, "__dict__") and not isinstance(val, (str, int, float, bool, list)):
            _patch_cfg(val, f"model.config.{key}")
    if patched:
        print(f"  [Llama4 patch] Set attention_chunk_size={chunk_size} on: {patched}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Parameters: {n_params:,}")
    return model, tokenizer


@contextmanager
def reference_on_gpu(model: nn.Module, device: str = "cuda"):
    """
    Context manager that temporarily moves a CPU-offloaded reference model
    to GPU for a forward pass, then moves it back to CPU.

    Usage:
        with reference_on_gpu(original_model, device):
            original_model(input_ids=..., ...)

    When config.offload_reference_model=False the model is already on GPU
    and this is a no-op (checks device before moving).
    """
    current_device = next(model.parameters()).device
    already_on_gpu = current_device.type == "cuda"

    if not already_on_gpu:
        model.to(device)
    try:
        yield model
    finally:
        if not already_on_gpu:
            model.to("cpu")
            torch.cuda.empty_cache()


def _detect_lora_target_modules(model: nn.Module) -> List[str]:
    """
    Auto-detects which linear layer names exist in the model so LoRA
    targets the correct modules regardless of architecture.

    Known patterns:
      - LLaMA / Mistral / Granite (MLP):  q/k/v/o_proj + gate/up/down_proj
      - Granite (some variants):          q/k/v/o_proj only (no gate/up/down)
      - GPT-2 / Falcon:                   c_attn, c_proj
      - GPT-NeoX:                         query_key_value, dense
    """
    all_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    candidates = [
        # Attention projections (present in almost all modern LLMs)
        ("q_proj",           "q_proj"           in all_names),
        ("k_proj",           "k_proj"           in all_names),
        ("v_proj",           "v_proj"            in all_names),
        ("o_proj",           "o_proj"            in all_names),
        # MLP projections (LLaMA-style SwiGLU)
        ("gate_proj",        "gate_proj"         in all_names),
        ("up_proj",          "up_proj"           in all_names),
        ("down_proj",        "down_proj"         in all_names),
        # GPT-2 style
        ("c_attn",           "c_attn"            in all_names),
        ("c_proj",           "c_proj"            in all_names),
        # Falcon / GPT-NeoX
        ("query_key_value",  "query_key_value"   in all_names),
        ("dense",            "dense"             in all_names),
    ]

    targets = [name for name, present in candidates if present]

    if not targets:
        raise ValueError(
            "Could not auto-detect LoRA target modules. "
            "Please set config.lora_target_modules explicitly."
        )

    print(f"  [LoRA] Auto-detected target modules: {targets}")
    return targets


def wrap_with_lora(model: nn.Module, config: PipelineConfig) -> nn.Module:
    """
    Wraps model with LoRA for parameter-efficient fine-tuning.
    Auto-detects target modules so it works across LLaMA, Mistral,
    Granite Guardian, GPT-2, Falcon, etc.
    """
    # Use explicit override from config if provided, else auto-detect
    target_modules = getattr(config, "lora_target_modules", None) \
                     or _detect_lora_target_modules(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_layer_indices(model: nn.Module, config) -> List[int]:
    """
    Resolves which transformer layers to probe, using one of two strategies:

    1. PROPORTIONAL (default): Probes layers at fixed fractions of total depth.
       Produces semantically comparable probe points across models of any size.
       E.g. ratios=[0.50, 0.65, 0.80, 1.0] on a 32-layer model → [16, 20, 25, 31]
                                             on a 16-layer model → [8, 10, 12, 15]
                                             on a 48-layer model → [24, 31, 38, 47]

    2. FIXED FALLBACK: If config.probe_layers_fixed is non-empty, those explicit
       absolute indices are used directly (supports negative indexing too).
       Useful for single-model ablations or reproducing exact prior results.

    Args:
        model:  Any causal LM (LLaMA, Mistral, Falcon, GPT-2 style)
        config: PipelineConfig — reads probe_layer_ratios and probe_layers_fixed

    Returns:
        Sorted list of absolute layer indices to probe.
    """
    layers = _find_layers(model)
    if layers is None:
        raise ValueError(
            "Cannot find transformer layers. "
            "Expected model.model.layers or model.transformer.h"
        )
    n = len(layers)

    # ── Strategy 2: explicit fixed indices (fallback / override) ────────────
    if getattr(config, "probe_layers_fixed", []):
        result = []
        for idx in config.probe_layers_fixed:
            abs_idx = n + idx if idx < 0 else idx
            abs_idx = max(0, min(abs_idx, n - 1))
            result.append(abs_idx)
        indices = sorted(set(result))
        print(f"  [Probing] Fixed indices → layers {indices}  (model depth={n})")
        return indices

    # ── Strategy 1: proportional probing (default) ───────────────────────────
    ratios  = getattr(config, "probe_layer_ratios", [0.50, 0.65, 0.80, 1.0])
    indices = sorted(set(
        min(int(r * n), n - 1) for r in ratios
    ))

    # Pretty depth summary for logging
    stages = {
        ratios[0]: "mid-network",
        ratios[1]: "upper-mid",
        ratios[2]: "late",
        ratios[3]: "final",
    } if len(ratios) == 4 else {}
    desc = "  ".join(
        f"L{idx}({stages.get(r,'')}{int(r*100)}%)"
        for idx, r in zip(indices, ratios)
    ) if stages else str(indices)

    print(f"  [Probing] Proportional layers ({n}-layer model) → {desc}")
    return indices


def _find_layers(model: nn.Module):
    """Traverses model to find the list of transformer layers."""
    # Direct LLaMA-style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 / Falcon style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # PEFT-wrapped
    if hasattr(model, "base_model"):
        inner = model.base_model.model
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return inner.model.layers
        if hasattr(inner, "transformer"):
            return inner.transformer.h
    return None


class ActivationExtractor:
    """
    Registers forward hooks on specified transformer layers to capture
    mean-pooled hidden states during a forward pass.

    Usage:
        extractor = ActivationExtractor(model, layer_indices)
        with extractor.capture():
            model(**inputs)
        acts = extractor.get_activations()   # {layer_idx: (batch, hidden_dim)}
    """

    def __init__(self, model: nn.Module, layer_indices: List[int]):
        self.model         = model
        self.layer_indices = layer_indices
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Mean-pool over sequence → (batch, hidden_dim)
            self._activations[layer_idx] = hidden.detach().float().mean(dim=1)
        return hook_fn

    @contextmanager
    def capture(self):
        """Context manager that registers hooks, then removes them."""
        self._activations = {}
        layers = _find_layers(self.model)
        try:
            for idx in self.layer_indices:
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self._hooks.append(hook)
            yield self
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks = []

    def get_activations(self) -> Dict[int, torch.Tensor]:
        return dict(self._activations)


@torch.no_grad()
def extract_representations(
    model: nn.Module,
    dataloader,
    layer_indices: List[int],
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[Dict[int, torch.Tensor], List[int]]:
    """
    Runs the model over a dataloader and collects hidden-state
    representations for each specified layer.

    Key optimisation: if the model is CPU-offloaded (reference model),
    we move it to GPU ONCE for the entire loop instead of per-batch,
    then move it back when done. This eliminates O(N_batches) transfers.

    Returns:
        acts   : {layer_idx: tensor (N, hidden_dim)}
        labels : List[int]  (if present in batch, else [])
    """
    model.eval()
    extractor  = ActivationExtractor(model, layer_indices)
    all_acts   = {idx: [] for idx in layer_indices}
    all_labels = []

    # Detect whether model lives on CPU (offloaded reference)
    try:
        current_device = next(model.parameters()).device
        is_offloaded = current_device.type == "cpu"
    except StopIteration:
        is_offloaded = False

    # Move to GPU once for the full extraction loop
    if is_offloaded:
        print(f"  [Extract] Moving reference model CPU→GPU for extraction...")
        model.to(device)

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch.get("label", None)

            with extractor.capture():
                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

            for idx in layer_indices:
                all_acts[idx].append(extractor.get_activations()[idx].cpu())

            if labels is not None:
                all_labels.extend(labels.cpu().tolist())
    finally:
        # Move back to CPU after extraction if it was offloaded
        if is_offloaded:
            model.to("cpu")
            torch.cuda.empty_cache()
            print(f"  [Extract] Reference model moved back to CPU.")

    result = {idx: torch.cat(all_acts[idx], dim=0) for idx in layer_indices}
    return result, all_labels
