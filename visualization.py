# ============================================================
# visualization.py  —  Publication-quality plots
# ============================================================
"""
Generates 6 figures comparing safety geometry across three conditions:
  1. Original aligned model
  2. Baseline fine-tuned (geometry collapsed)
  3. FW-SSR mitigated (geometry preserved)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {
    "original":   "#2196F3",   # Blue
    "finetuned":  "#F44336",   # Red
    "mitigated":  "#4CAF50",   # Green
    "harmful":    "#E91E63",   # Pink
    "benign":     "#00BCD4",   # Cyan
}


def _save(fig: plt.Figure, path: str, fmt: str = "pdf") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + "." + fmt, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [Saved] {path}.{fmt}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Latent Space Comparison (3-panel UMAP)
# ─────────────────────────────────────────────────────────────────────────────
def plot_latent_space_comparison(
    acts_2d_orig:     np.ndarray,     # (N, 2)
    acts_2d_finetuned:np.ndarray,     # (N, 2)
    acts_2d_mitigated:np.ndarray,     # (N, 2)
    labels:           List[int],       # 1=harmful, 0=benign
    layer_idx:        int,
    save_path:        str,
    fmt:              str = "pdf",
) -> None:
    """3-panel UMAP showing how geometry changes across the three conditions."""
    labels_arr = np.array(labels)
    conditions = [
        (acts_2d_orig,      "Original (Aligned)",       PALETTE["original"]),
        (acts_2d_finetuned, "Baseline Fine-tuned",       PALETTE["finetuned"]),
        (acts_2d_mitigated, "FW-SSR Mitigated",          PALETTE["mitigated"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Latent Space Geometry — Layer {layer_idx}\n"
        f"Harmful vs. Benign Prompt Separation",
        fontsize=15, fontweight="bold", y=1.02,
    )

    for ax, (acts, title, color) in zip(axes, conditions):
        harm_mask  = labels_arr == 1
        benign_mask = labels_arr == 0

        ax.scatter(
            acts[benign_mask, 0], acts[benign_mask, 1],
            c=PALETTE["benign"], alpha=0.6, s=40, marker="o",
            label="Benign", edgecolors="white", linewidths=0.3,
        )
        ax.scatter(
            acts[harm_mask, 0], acts[harm_mask, 1],
            c=PALETTE["harmful"], alpha=0.6, s=40, marker="^",
            label="Harmful", edgecolors="white", linewidths=0.3,
        )

        # Centroids
        c_h = acts[harm_mask].mean(0)
        c_b = acts[benign_mask].mean(0)
        ax.scatter(*c_h, c="black", s=200, marker="*", zorder=5, label="Centroid")
        ax.scatter(*c_b, c="black", s=200, marker="*", zorder=5)

        # Arrow between centroids
        ax.annotate(
            "", xy=c_h, xytext=c_b,
            arrowprops=dict(arrowstyle="<->", color="black", lw=2.0),
        )

        # Centroid distance annotation
        dist = np.linalg.norm(c_h - c_b)
        ax.set_title(f"{title}\n(Centroid Δ = {dist:.2f})", fontsize=13, color=color)
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    _save(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Safety Drift + Task Loss Over Training
# ─────────────────────────────────────────────────────────────────────────────
def plot_drift_over_training(
    baseline_drift:   List[Tuple[int, float]],
    mitigated_drift:  List[Tuple[int, float]],
    baseline_loss:    List[Tuple[int, float]],
    mitigated_loss:   List[Tuple[int, float]],
    save_path:        str,
    fmt:              str = "pdf",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Safety Geometry Drift & Task Loss During Fine-Tuning",
                 fontsize=15, fontweight="bold")

    # Panel A: Safety Drift
    if baseline_drift:
        steps_b, drift_b = zip(*baseline_drift)
        ax1.plot(steps_b, drift_b, color=PALETTE["finetuned"], lw=2.5,
                 label="Baseline (no regularization)", linestyle="--")
    if mitigated_drift:
        steps_m, drift_m = zip(*mitigated_drift)
        ax1.plot(steps_m, drift_m, color=PALETTE["mitigated"], lw=2.5,
                 label="FW-SSR Mitigated")

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Safety Drift (L2 in Subspace)", fontsize=12)
    ax1.set_title("(A) Safety Subspace Drift vs. Steps", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.4)

    # Panel B: Task Loss
    if baseline_loss:
        s_b, l_b = zip(*baseline_loss)
        s_b_sm   = _smooth(list(l_b), window=10)
        ax2.plot(s_b, s_b_sm, color=PALETTE["finetuned"], lw=2.5,
                 label="Baseline", linestyle="--")
    if mitigated_loss:
        s_m, l_m = zip(*mitigated_loss)
        s_m_sm   = _smooth(list(l_m), window=10)
        ax2.plot(s_m, s_m_sm, color=PALETTE["mitigated"], lw=2.5,
                 label="FW-SSR Mitigated")

    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Language Modeling Loss (CE)", fontsize=12)
    ax2.set_title("(B) Task Loss vs. Steps", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, save_path, fmt)


def _smooth(data: List[float], window: int = 10) -> List[float]:
    """Simple moving average."""
    result = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2 + 1)
        result.append(np.mean(data[lo:hi]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Drift Metrics Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_drift_metrics_bar(
    metrics: Dict,      # {condition: {metric: value}}
    save_path: str,
    fmt: str = "pdf",
) -> None:
    """
    Grouped bar chart comparing geometry metrics across conditions.
    Higher safety_drift, drift_ratio = worse. Higher Fisher score = better.
    """
    metric_labels = {
        "safety_drift":     "Safety Drift ↓",
        "orthogonal_drift": "Orthogonal Drift",
        "drift_ratio":      "Drift Ratio ↓",
        "fisher_score":     "Fisher Score ↑",
    }
    conditions  = list(metrics.keys())
    n_metrics   = len(metric_labels)
    n_conditions= len(conditions)
    bar_w       = 0.25
    x           = np.arange(n_metrics)

    condition_colors = {
        "original":  PALETTE["original"],
        "finetuned": PALETTE["finetuned"],
        "mitigated": PALETTE["mitigated"],
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cond in enumerate(conditions):
        vals = [metrics[cond].get(m, 0.0) for m in metric_labels]
        bars = ax.bar(
            x + i * bar_w, vals, bar_w,
            label=cond.capitalize(),
            color=condition_colors.get(cond, "#999999"),
            alpha=0.85, edgecolor="white",
        )

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(list(metric_labels.values()), fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Safety Geometry Metrics: Original vs. Fine-tuned vs. FW-SSR",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    _save(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-Layer Drift Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_layer_wise_drift(
    layer_drift: Dict,   # {condition: {layer_idx: drift_value}}
    save_path: str,
    fmt: str = "pdf",
) -> None:
    """Heatmap of safety drift per transformer layer for each condition."""
    conditions = list(layer_drift.keys())
    layers     = sorted(set(
        l for c in conditions for l in layer_drift[c].keys()
    ))

    matrix = np.array([
        [layer_drift[c].get(l, 0.0) for l in layers]
        for c in conditions
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.2), 4))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=[f"L{l}" for l in layers],
        yticklabels=[c.capitalize() for c in conditions],
        cmap="RdYlGn_r",
        annot=True, fmt=".3f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Safety Drift"},
    )
    ax.set_title("Per-Layer Safety Drift Heatmap\n(Lower = Geometry Better Preserved)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Transformer Layer", fontsize=12)
    plt.tight_layout()
    _save(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Fisher Weights (Curvature Sensitivity)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fisher_weights(
    fisher_weights: Dict,   # {layer_idx: np.ndarray (k,)}
    save_path: str,
    fmt: str = "pdf",
) -> None:
    """Bar charts showing curvature sensitivity per safety direction per layer."""
    n_layers = len(fisher_weights)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(
        1, n_layers, figsize=(6 * n_layers, 5), sharey=False
    )
    if n_layers == 1:
        axes = [axes]

    for ax, (layer_idx, weights) in zip(axes, fisher_weights.items()):
        w = np.array(weights)
        top_k = min(20, len(w))
        sorted_idx = np.argsort(w)[::-1][:top_k]
        ax.bar(
            range(top_k), w[sorted_idx],
            color=PALETTE["original"], alpha=0.85, edgecolor="white",
        )
        ax.set_xlabel("Safety Direction (sorted)", fontsize=11)
        ax.set_ylabel("Fisher Weight", fontsize=11)
        ax.set_title(f"Layer {layer_idx}\nCurvature Sensitivity", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Fisher Information Weights: Curvature of Safety Subspace Directions\n"
        "(Higher = Direction is More Safety-Critical & Fragile)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(fig, save_path, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: CKA Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_cka_matrix(
    cka_values: Dict,   # {(cond_a, cond_b): float}
    save_path: str,
    fmt: str = "pdf",
) -> None:
    """3×3 CKA heatmap across original, finetuned, mitigated conditions."""
    conditions = ["original", "finetuned", "mitigated"]
    n = len(conditions)
    matrix = np.zeros((n, n))

    for i, ca in enumerate(conditions):
        for j, cb in enumerate(conditions):
            key    = (ca, cb) if (ca, cb) in cka_values else (cb, ca)
            matrix[i, j] = cka_values.get(key, 1.0 if i == j else 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=[c.capitalize() for c in conditions],
        yticklabels=[c.capitalize() for c in conditions],
        cmap="RdYlGn", vmin=0, vmax=1,
        annot=True, fmt=".3f", annot_kws={"size": 14},
        linewidths=1.0, linecolor="white",
        cbar_kws={"label": "CKA Similarity"},
        square=True,
    )
    ax.set_title(
        "Centered Kernel Alignment (CKA)\nRepresentation Similarity Matrix\n"
        "(1.0 = Identical Geometry)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path, fmt)
