# ============================================================
# geometry_utils.py  —  Safety geometry analysis
# ============================================================
"""
Core geometric analysis:
  - Safety subspace extraction via SVD on class-conditional activation differences
  - Safety drift metrics (Δ_safety, drift ratio, cosine similarity)
  - CKA (Centered Kernel Alignment) between representation matrices
  - Class separation metrics (Fisher discriminant score)
  - Dimensionality reduction for visualization (UMAP / t-SNE / PCA)
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[Warning] umap-learn not installed. Install with: pip install umap-learn")


def compute_safety_subspace(
    harmful_acts: torch.Tensor,   # (N_harmful, hidden_dim)
    benign_acts:  torch.Tensor,   # (N_benign,  hidden_dim)
    k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the top-k safety-sensitive principal directions via SVD.

    Strategy:
      1. Mean-center each class
      2. Amplify the between-class direction (harmful centroid - benign centroid)
         to ensure safety directions are captured in top singular vectors
      3. SVD on augmented covariance matrix → top-k directions U

    Returns:
        U  : Safety subspace basis  (hidden_dim, k)  — orthonormal columns
        sv : Singular values        (k,)              — explained variance
    """
    h_mean = harmful_acts.mean(0, keepdim=True)
    b_mean = benign_acts.mean(0, keepdim=True)

    # Between-class difference (primary safety direction)
    diff = h_mean - b_mean                          # (1, d)

    # Center within class
    h_c = harmful_acts - h_mean                     # (N_h, d)
    b_c = benign_acts  - b_mean                     # (N_b, d)

    # Augmented matrix: between-class signal weighted 5x to dominate top SVs
    amplification = 5.0
    n_aug  = min(32, len(harmful_acts))
    aug    = diff.repeat(n_aug, 1) * amplification

    combined = torch.cat([h_c, b_c, aug], dim=0).float()  # (N_h+N_b+n_aug, d)

    # Truncated SVD via gram matrix for efficiency when d > N
    try:
        if combined.shape[0] < combined.shape[1]:
            # N < d: work in sample space
            G = combined @ combined.T               # (N, N)
            eigvals, eigvecs = torch.linalg.eigh(G)
            eigvals = eigvals.flip(0)
            eigvecs = eigvecs.flip(1)
            # Convert to data-space principal vectors
            sv  = eigvals[:k].clamp(min=0).sqrt()
            U   = combined.T @ eigvecs[:, :k]
            norms = U.norm(dim=0, keepdim=True).clamp(min=1e-8)
            U   = U / norms
        else:
            # d <= N: standard eigen-decomposition on covariance
            C = combined.T @ combined                # (d, d)
            eigvals, eigvecs = torch.linalg.eigh(C)
            eigvals = eigvals.flip(0)
            eigvecs = eigvecs.flip(1)
            sv = eigvals[:k].clamp(min=0).sqrt()
            U  = eigvecs[:, :k]
    except Exception as e:
        print(f"[Warning] SVD fallback triggered: {e}")
        U  = torch.randn(combined.shape[1], k)
        U, _ = torch.linalg.qr(U)
        sv = torch.ones(k)

    return U.float(), sv.float()


def compute_safety_drift(
    acts_original: torch.Tensor,   # (N, hidden_dim)
    acts_finetuned: torch.Tensor,  # (N, hidden_dim)
    U: torch.Tensor,               # (hidden_dim, k) safety subspace
) -> Dict[str, float]:
    """
    Computes safety geometry drift between original and fine-tuned model.

    Δ_safety = E_x [ || U^T (h(x;θ_ft) - h(x;θ_0)) || ]

    Key metrics:
      - total_drift:        mean L2 drift in full representation space
      - safety_drift:       mean L2 drift projected onto safety subspace U
      - orthogonal_drift:   drift orthogonal to safety subspace
      - drift_ratio:        safety_drift / total_drift (key: how much drift hits safety)
      - cosine_similarity:  mean cosine sim between original and FT representations
    """
    assert acts_original.shape == acts_finetuned.shape, \
        f"Shape mismatch: {acts_original.shape} vs {acts_finetuned.shape}"

    o = acts_original.float()
    f = acts_finetuned.float()
    U = U.float()

    delta = f - o                                    # (N, d)

    # Safety-subspace projection
    proj       = U @ U.T                             # (d, d) — projection matrix
    delta_safe = delta @ proj                        # (N, d)
    delta_orth = delta - delta_safe                  # (N, d)

    # Norms
    total_drift  = delta.norm(dim=-1).mean().item()
    safety_drift = delta_safe.norm(dim=-1).mean().item()
    orth_drift   = delta_orth.norm(dim=-1).mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(o, f, dim=-1).mean().item()

    # Per-direction projection magnitudes in safety subspace
    proj_mags = (delta @ U).abs().mean(dim=0)        # (k,)

    return {
        "total_drift":         total_drift,
        "safety_drift":        safety_drift,
        "orthogonal_drift":    orth_drift,
        "drift_ratio":         safety_drift / (total_drift + 1e-8),
        "cosine_similarity":   cos_sim,
        "mean_proj_magnitude": proj_mags.mean().item(),
        "max_proj_magnitude":  proj_mags.max().item(),
    }


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear Centered Kernel Alignment (CKA).
    Measures representational similarity between two matrices.
    CKA = 1.0 → identical geometry, CKA = 0.0 → orthogonal geometry.

    Reference: Kornblith et al. (2019), "Similarity of Neural Network Representations"
    """
    X = X.float()
    Y = Y.float()

    def centering(K: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    Kx   = X @ X.T
    Ky   = Y @ Y.T
    Kxc  = centering(Kx)
    Kyc  = centering(Ky)

    hsic_xy = (Kxc * Kyc).sum()
    hsic_xx = (Kxc * Kxc).sum().sqrt()
    hsic_yy = (Kyc * Kyc).sum().sqrt()

    return (hsic_xy / (hsic_xx * hsic_yy + 1e-10)).item()


def compute_class_separation(
    acts: torch.Tensor,
    labels: List[int],
) -> Dict:
    """
    Measures how well harmful (1) and benign (0) prompts are
    separated in the representation space.

    Metrics:
      - inter_class_distance: distance between class centroids
      - intra_class_variance: average within-class spread (L2)
      - fisher_score:         inter / intra — higher = cleaner safety boundary
    """
    acts   = acts.float()
    labels = torch.tensor(labels)

    h_acts = acts[labels == 1]
    b_acts = acts[labels == 0]

    if len(h_acts) == 0 or len(b_acts) == 0:
        return {"inter_class_distance": 0.0, "intra_class_variance": 0.0,
                "fisher_score": 0.0, "centroid_h": None, "centroid_b": None}

    c_h = h_acts.mean(0)
    c_b = b_acts.mean(0)

    inter = (c_h - c_b).norm().item()
    var_h = (h_acts - c_h).norm(dim=-1).mean().item()
    var_b = (b_acts - c_b).norm(dim=-1).mean().item()
    intra = (var_h + var_b) / 2.0

    return {
        "inter_class_distance": inter,
        "intra_class_variance": intra,
        "fisher_score":         inter / (intra + 1e-8),
        "centroid_h":           c_h.numpy(),
        "centroid_b":           c_b.numpy(),
    }


def reduce_dimensions(
    acts: torch.Tensor,
    method: str  = "umap",
    n_components: int = 2,
    pca_components: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Reduces high-dimensional activations to 2D for visualization.
    Pipeline: PCA(50) → UMAP(2) or t-SNE(2).

    PCA pre-reduction is required for UMAP/t-SNE to scale to large hidden dims.
    """
    X = acts.float().numpy()

    # Sanitize NaN/Inf values that can arise from fp16 overflow or
    # collapsed activations after heavy NaN-loss training
    if not np.isfinite(X).all():
        n_bad = (~np.isfinite(X)).sum()
        print(f"  [Warning] {n_bad} NaN/Inf values in activations — replacing with column means")
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        bad_mask  = ~np.isfinite(X)
        X[bad_mask] = np.take(col_means, np.where(bad_mask)[1])
        # Final safety net: zero out any remaining non-finite values
        X = np.where(np.isfinite(X), X, 0.0)

    n_pca = min(pca_components, X.shape[0] - 1, X.shape[1])

    # Step 1: PCA pre-reduction
    pca    = PCA(n_components=n_pca, random_state=seed)
    X_pca  = pca.fit_transform(X)
    cumvar = pca.explained_variance_ratio_.sum()
    print(f"  [Geometry] PCA ({n_pca} components) explains {cumvar:.1%} of variance")

    # Step 2: Non-linear embedding
    if method == "umap":
        if not UMAP_AVAILABLE:
            print("  [Warning] UMAP unavailable — falling back to t-SNE")
            method = "tsne"
        else:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=seed,
                verbose=False,
            )
            return reducer.fit_transform(X_pca)

    if method == "tsne":
        perp   = min(30, max(5, X_pca.shape[0] // 5))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perp,
            random_state=seed,
            metric="cosine",
            init="pca",
            n_iter=1000,
            verbose=0,
        )
        return reducer.fit_transform(X_pca)

    # Fallback: raw PCA 2D
    return X_pca[:, :2]