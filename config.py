# ============================================================
# config.py  —  Pipeline Configuration
# ============================================================
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # ── Models ──────────────────────────────────────────────
    model_names: List[str] = field(default_factory=lambda: [
        # "mistralai/Mistral-7B-Instruct-v0.2",
        # "meta-llama/LlamaGuard-7b"
        # "meta-llama/Meta-Llama-Guard-2-8B"
        "meta-llama/Llama-Guard-3-8B"
        # "thu-coai/ShieldLM-7B-internlm2",
        # "allenai/wildguard",
        # "ibm-granite/granite-guardian-3.0-2b"
        # "meta-llama/Llama-Guard-4-12B"
        # "OpenSafetyLab/MD-Judge-v0.1"
    ])
    hf_token: Optional[str] = None

    # ── Data ────────────────────────────────────────────────
    benign_dataset: str       = "tatsu-lab/alpaca"
    benign_max_samples: int   = 2000
    safety_probe_samples: int = 80   # 80 is enough for stable SVD; was 200

    # ── Fine-tuning ──────────────────────────────────────────
    finetune_epochs: int             = 3
    finetune_lr: float               = 2e-5
    finetune_batch_size: int         = 4      # Reduced for large models; use grad accum
    gradient_accumulation_steps: int = 4     # Effective batch = 1 * 16 = 16
    max_seq_length: int              = 128    # Reduced from 256; cuts activation memory ~2x
    use_lora: bool                   = True
    lora_rank: int                   = 8      # Reduced from 16 for large models
    lora_alpha: int                  = 16
    lora_dropout: float              = 0.05
    # Set to non-empty list to override auto-detection (e.g. for unusual architectures)
    lora_target_modules: List[str]   = field(default_factory=lambda: [])

    # ── Memory Optimization ───────────────────────────────────
    # QLoRA: 4-bit quantization for the frozen base weights.
    # Cuts base model VRAM from ~24GB (fp16) to ~6GB (4-bit) for a 12B model,
    # making room for the LoRA adapters + activations + optimizer states.
    use_4bit: bool                   = True   # Enable for models >= 7B on single GPU
    use_8bit: bool                   = False  # Alternative: 8-bit (less aggressive)
    bnb_4bit_compute_dtype: str      = "float16"   # Compute dtype for 4-bit matmuls
    bnb_4bit_quant_type: str         = "nf4"       # "nf4" (NormalFloat4) recommended
    use_gradient_checkpointing: bool = True   # Trade compute for memory (~30% more steps)
    # Keep frozen reference model on CPU; move to GPU only for forward passes.
    # Saves ~half the VRAM for large models at the cost of CPU<->GPU transfer time.
    offload_reference_model: bool    = True

    # ── Geometry ─────────────────────────────────────────────
    # Proportional probing: fractions of total depth to probe.
    # Maps to semantically comparable stages across architectures:
    #   0.50 = mid-network (contextual semantics)
    #   0.65 = upper-mid  (task-specific representations)
    #   0.80 = late       (safety-decision boundary)
    #   1.00 = final      (output representations)
    # These proportions replace fixed relative indices like [-1,-4,-8,-12],
    # which land at very different semantic positions across model depths.
    probe_layer_ratios: List[float] = field(default_factory=lambda: [0.50, 0.65, 0.80, 1.0])

    # Fallback: set probe_layer_ratios=[] and populate probe_layers_fixed
    # to use explicit fixed indices (e.g. for single-model ablations).
    probe_layers_fixed: List[int]  = field(default_factory=lambda: [])

    safety_subspace_dim: int = 32
    pca_components: int      = 50
    umap_neighbors: int      = 15
    umap_min_dist: float     = 0.1

    # ── Novel Mitigation: FW-SSR ──────────────────────────────
    # L_total = L_task + lambda * sum_l sum_x || F_l (x) U_l^T(h_l(x;theta) - h_l(x;theta_0)) ||^2
    fwssr_lambda: float           = 0.1
    fwssr_fisher_samples: int     = 64
    fwssr_fisher_update_freq: int = 50
    fwssr_momentum: float         = 0.9
    fwssr_adaptive_lambda: bool   = True

    # ── Visualization ─────────────────────────────────────────
    reduction_method: str = "umap"   # "umap" | "tsne" | "pca"
    plot_format: str      = "pdf"

    # ── Paths ────────────────────────────────────────────────
    output_dir: str     = "./results"
    checkpoint_dir: str = "./checkpoints"
    figures_dir: str    = "./figures"

    # ── Misc ─────────────────────────────────────────────────
    seed: int       = 42
    device: str     = "cuda"
    fp16: bool      = True
    log_wandb: bool = False