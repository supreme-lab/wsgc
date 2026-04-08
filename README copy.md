# FW-SSR: Fisher-Weighted Safety Subspace Regularization

**A novel fine-tuning framework that prevents safety geometry collapse in aligned LLMs.**

---

## Overview

Fine-tuning aligned language models on benign data can inadvertently degrade their
safety alignment — even without any explicitly harmful training signal (Qi et al., 2023).
This project diagnoses this degradation through the lens of **latent space geometry** and
proposes **FW-SSR**, a curvature-aware mitigation technique.

### Core Problem

When safety-aligned LLMs (e.g., LlamaGuard, Mistral-Instruct) are fine-tuned on benign
instruction datasets, the internal geometry separating harmful from benign prompts collapses.
The model "forgets" to distinguish dangerous requests at the representation level.

### FW-SSR Solution

FW-SSR extends the Safety-Proximity Fine-Tuning (SPFT) framework by making regularization
adaptive to curvature in the safety subspace:

```
L_total = L_task + λ * Σ_l Σ_x || F_l ⊙ U_l^T(h_l(x;θ) - h_l(x;θ_0)) ||²
```

Where:

- `U_l` — safety subspace basis for layer l (from SVD on harmful–benign activation differences)
- `F_l` — diagonal Fisher information weights (curvature sensitivity per safety direction)
- `λ`   — adaptive regularization strength (scales with gradient conflict)

**Key innovations vs. uniform SPFT (Mode A):**

1. **Curvature-aware**: High-Fisher directions receive stronger protection
2. **Adaptive λ**: Reduces over-regularization when task & safety objectives naturally agree
3. **EMA stability**: Fisher weights updated with exponential moving average

---

## Project Structure

```
spft_project/
├── config.py          # All hyperparameters in one dataclass
├── data_utils.py      # Safety probe dataset + Alpaca fine-tune data
├── model_utils.py     # Model loading, LoRA wrapping, activation hooks
├── geometry_utils.py  # SVD subspace, drift metrics, CKA, UMAP
├── trainers.py        # BaselineTrainer & FWSSRTrainer
├── visualization.py   # 6 publication-quality figures
├── evaluation.py      # Refusal rates + geometry metrics table
├── run_pipeline.py    # Main orchestration script
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

For models requiring HuggingFace access (LlamaGuard):

```bash
export HF_TOKEN=hf_...
```

---

## Usage

```bash
# Run full pipeline with default model (Mistral-7B-Instruct)
python run_pipeline.py

# Use specific model
python run_pipeline.py --model meta-llama/LlamaGuard-2-8B

# Skip slow generation evaluation
python run_pipeline.py --no-eval

# Adjust FW-SSR strength
python run_pipeline.py --lambda 0.05 --epochs 2
```

---

## Outputs

**Figures** (saved to `./figures/`):


| Figure                   | Description                                |
| ------------------------ | ------------------------------------------ |
| fig1_latent_space        | 3-panel UMAP: Original / Baseline / FW-SSR |
| fig2_drift_over_training | Safety drift + task loss during training   |
| fig3_drift_metrics       | Grouped bar chart of geometry metrics      |
| fig4_layer_drift         | Per-layer drift heatmap                    |
| fig5_fisher_weights      | Curvature sensitivity per safety direction |
| fig6_cka_matrix          | 3×3 CKA similarity matrix                 |

**Results** (saved to `./results/`):

- `<model>_eval_results.json` — All metrics in JSON format

---

## Expected Results


| Condition  | Refusal Rate | Safety Drift | Fisher Score | CKA vs Original |
| ---------- | ------------ | ------------ | ------------ | --------------- |
| Original   | ~90%         | 0.000        | High         | 1.000           |
| Fine-tuned | ~20%         | High         | Low          | ~0.30           |
| FW-SSR     | ~85%         | Low          | High         | ~0.90           |

---

## Key References

1. Qi et al. (2023) *"Fine-tuning Aligned Language Models Compromises Safety"*
2. Richter et al. (ICLR 2025) *"An Auditing Test to Detect Behavioral Shift in Language Models"*
3. Lyu et al. (2024) *"Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates"* (SPFT basis)
4. Kornblith et al. (2019) *"Similarity of Neural Network Representations Revisited"* (CKA)

---

## Configuration

All hyperparameters are in `config.py`. Key FW-SSR settings:

```python
fwssr_lambda: float           = 0.1    # Base regularization strength
fwssr_fisher_samples: int     = 64     # Samples for Fisher approximation
fwssr_fisher_update_freq: int = 50     # Fisher update interval (steps)
fwssr_momentum: float         = 0.9    # EMA momentum
fwssr_adaptive_lambda: bool   = True   # Gradient-alignment-based λ scheduling
```
