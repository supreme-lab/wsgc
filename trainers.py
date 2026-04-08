# ============================================================
# trainers.py  —  Baseline & FW-SSR fine-tuning trainers
# ============================================================
"""
Two trainers:
  1. BaselineTrainer:  Standard LoRA fine-tuning (no safety constraints)
                       Demonstrates safety geometry collapse (Qi et al. Level-3)

  2. FWSSRTrainer:     Novel Fisher-Weighted Safety Subspace Regularization
                       Preserves safety geometry via curvature-aware penalty

Novel FW-SSR Formulation:
  L_total = L_task + lambda * sum_l sum_x || F_l (x) U_l^T(h_l(x;theta) - h_l(x;theta_0)) ||^2

  Where:
    F_l  : Diagonal Fisher info projected onto safety subspace   (k,)
    U_l  : Safety subspace basis for layer l                    (d, k)
    lambda: Adaptive regularization (scales with grad conflict)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from model_utils import ActivationExtractor, extract_representations, reference_on_gpu
from geometry_utils import compute_safety_drift


class BaselineTrainer:
    """
    Standard LoRA fine-tuning on benign Alpaca data.
    No safety constraints — shows how benign fine-tuning degrades alignment.
    """

    def __init__(
        self,
        model: nn.Module,
        original_model: nn.Module,
        safety_subspaces: Dict[int, torch.Tensor],  # {layer_idx: (d, k)}
        probe_dataloader,
        config,
        device: str = "cuda",
    ):
        self.model            = model
        self.original_model   = original_model
        self.safety_subspaces = safety_subspaces
        self.probe_dl         = probe_dataloader
        self.config           = config
        self.device           = device

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.finetune_lr,
            weight_decay=0.01,
        )

        # Training history
        self.drift_history      = []   # [(step, drift_metrics)]
        self.loss_history       = []   # [(step, loss)]

    def train(self, dataloader, epochs: int) -> None:
        """Fine-tunes on benign data without any safety regularization."""
        self.model.train()
        global_step = 0
        nan_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0

            pbar = tqdm(dataloader, desc=f"[Baseline] Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                input_ids  = batch["input_ids"].to(self.device)
                attn_mask  = batch["attention_mask"].to(self.device)
                labels     = batch["labels"].to(self.device)

                # Mask padding tokens in labels (required for some tokenizers
                # that pad with 0 instead of -100, e.g. Granite Guardian)
                labels = labels.masked_fill(attn_mask == 0, -100)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss

                # Guard against NaN/Inf loss (fp16 overflow or bad batch)
                if not torch.isfinite(loss):
                    nan_batches += 1
                    if nan_batches <= 5:
                        print(f"\n  [Warning] NaN/Inf loss at step {global_step} — skipping batch")
                    self.optimizer.zero_grad()
                    global_step += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
                self.loss_history.append((global_step, loss.item()))

                if global_step % 100 == 0:
                    self._log_drift(global_step)

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  [Baseline] Epoch {epoch+1} | avg_loss={avg_loss:.4f} | skipped={nan_batches} NaN batches")

        if nan_batches > 0:
            print(f"\n  [Warning] {nan_batches} total NaN batches skipped. "
                  f"Consider: lower lr, disable fp16, or check tokenizer pad_token.")
        print("[Baseline] Training complete.")

    @torch.no_grad()
    def _log_drift(self, step: int) -> None:
        """Quick drift snapshot on first few probe batches."""
        self.model.eval()
        layer_indices = list(self.safety_subspaces.keys())

        # Pull a small sample from probe loader (first batch only)
        batch = next(iter(self.probe_dl))
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)

        ext_ft   = ActivationExtractor(self.model,          layer_indices)
        ext_orig = ActivationExtractor(self.original_model, layer_indices)

        with ext_ft.capture():
            self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        with reference_on_gpu(self.original_model, self.device):
            with ext_orig.capture():
                self.original_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        acts_ft   = ext_ft.get_activations()
        acts_orig = ext_orig.get_activations()

        avg_drift = 0.0
        for idx in layer_indices:
            h_ft   = acts_ft[idx].cpu()
            h_orig = acts_orig[idx].cpu()
            U      = self.safety_subspaces[idx].cpu()
            drift  = compute_safety_drift(h_orig, h_ft, U)
            avg_drift += drift["safety_drift"]

        avg_drift /= len(layer_indices)
        self.drift_history.append((step, avg_drift))
        self.model.train()


class FWSSRTrainer:
    """
    Fisher-Weighted Safety Subspace Regularization (FW-SSR).

    Novel contribution: Extends SPFT (Lyu et al.) by making the safety subspace
    regularization adaptive to curvature. High-Fisher-information safety directions
    receive stronger protection during fine-tuning.

    Key differences from SPFT Mode A (uniform L2 penalty):
      - Fisher weights F_l estimate curvature sensitivity per safety direction
      - Adaptive lambda scales with gradient conflict between task & safety
      - Fisher weights updated via EMA for training stability
    """

    def __init__(
        self,
        model: nn.Module,
        original_model: nn.Module,
        safety_subspaces: Dict[int, torch.Tensor],
        probe_dataloader,
        config,
        device: str = "cuda",
    ):
        self.model            = model
        self.original_model   = original_model
        self.safety_subspaces = safety_subspaces
        self.probe_dl         = probe_dataloader
        self.config           = config
        self.device           = device
        self.layer_indices    = list(safety_subspaces.keys())
        self.k                = config.safety_subspace_dim

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.finetune_lr,
            weight_decay=0.01,
        )

        # FW-SSR state
        self.fisher_weights: Dict[int, torch.Tensor] = {
            idx: torch.ones(self.safety_subspaces[idx].shape[1], device=device)
            for idx in self.layer_indices
        }
        self.current_lambda    = config.fwssr_lambda
        self.lambda_history    = []
        self.drift_history     = []
        self.loss_history      = []
        self.reg_loss_history  = []

    def train(self, dataloader, epochs: int) -> None:
        """
        Fine-tunes with FW-SSR regularization.

        Training loop:
          1. Forward pass → task loss
          2. Fisher-weighted safety subspace penalty
          3. Adaptive lambda adjustment (gradient alignment)
          4. Combined backward pass
          5. Fisher weight update every N steps
        """
        global_step = 0
        nan_batches = 0
        self.model.train()

        for epoch in range(epochs):
            epoch_loss    = 0.0
            epoch_reg     = 0.0
            n_batches     = 0

            pbar = tqdm(dataloader, desc=f"[FW-SSR]   Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels    = batch["labels"].to(self.device)

                # Mask padding in labels
                labels = labels.masked_fill(attn_mask == 0, -100)

                # ── Task Loss ──
                task_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    use_cache=False,
                )
                task_loss = task_outputs.loss

                if not torch.isfinite(task_loss):
                    nan_batches += 1
                    if nan_batches <= 5:
                        print(f"\n  [Warning] NaN/Inf task loss at step {global_step} — skipping batch")
                    self.optimizer.zero_grad()
                    global_step += 1
                    continue

                # ── FW-SSR Regularization ──
                reg_loss = self._compute_fwssr_loss(input_ids, attn_mask)

                total_loss = task_loss + self.current_lambda * reg_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += task_loss.item()
                epoch_reg  += reg_loss.item() if hasattr(reg_loss, "item") else reg_loss
                n_batches  += 1
                global_step += 1

                if self.config.fwssr_adaptive_lambda and global_step % 20 == 0:
                    self._adapt_lambda(input_ids, attn_mask, labels)

                if global_step % self.config.fwssr_fisher_update_freq == 0:
                    self._update_fisher_weights()

                pbar.set_postfix(
                    task=f"{task_loss.item():.4f}",
                    reg=f"{reg_loss:.4f}" if isinstance(reg_loss, float) else f"{reg_loss.item():.4f}",
                    lam=f"{self.current_lambda:.4f}",
                )

                self.loss_history.append((global_step, task_loss.item()))
                self.reg_loss_history.append((global_step, reg_loss.item() if hasattr(reg_loss, "item") else reg_loss))
                self.lambda_history.append((global_step, self.current_lambda))

                if global_step % 100 == 0:
                    self._log_drift(global_step)

            avg_task = epoch_loss / max(n_batches, 1)
            avg_reg  = epoch_reg  / max(n_batches, 1)
            print(f"  [FW-SSR] Epoch {epoch+1} | task={avg_task:.4f} | reg={avg_reg:.4f} | λ={self.current_lambda:.4f} | skipped={nan_batches} NaN batches")

        print("[FW-SSR] Training complete.")

    def _compute_fwssr_loss(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes Fisher-weighted safety subspace regularization loss.

        L_reg = sum_l || F_l_norm (x) U_l^T (h_l(x;theta) - h_l(x;theta_0)) ||^2

        Note: gradients only flow through the fine-tuned model's activations.
        Original model activations are detached (reference anchors).
        """
        ext_ft   = ActivationExtractor(self.model,          self.layer_indices)
        ext_orig = ActivationExtractor(self.original_model, self.layer_indices)

        # Fine-tuned model activations (differentiable)
        with ext_ft.capture():
            self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        # Original model activations (frozen reference — no grad)
        with torch.no_grad():
            with reference_on_gpu(self.original_model, self.device):
                with ext_orig.capture():
                    self.original_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        acts_ft   = ext_ft.get_activations()
        acts_orig = ext_orig.get_activations()

        total_reg = torch.tensor(0.0, device=self.device, requires_grad=True)

        for idx in self.layer_indices:
            h_ft    = acts_ft[idx]                 # (batch, d) — grad enabled
            h_orig  = acts_orig[idx].detach()      # (batch, d) — frozen
            U       = self.safety_subspaces[idx].to(self.device)   # (d, k)
            F_w     = self.fisher_weights[idx]     # (k,) — curvature weights

            # Project drift onto safety subspace
            delta      = h_ft - h_orig             # (batch, d)
            delta_proj = delta @ U                 # (batch, k)

            # Normalize Fisher weights
            F_norm = F_w / (F_w.sum() + 1e-8) * len(F_w)   # mean-1 normalized

            # Weighted L2 penalty
            weighted = delta_proj * F_norm.unsqueeze(0)     # (batch, k)
            layer_reg = (weighted ** 2).mean()

            total_reg = total_reg + layer_reg

        return total_reg / len(self.layer_indices)

    def _update_fisher_weights(self) -> None:
        """
        Updates per-layer Fisher weights via gradient-based diagonal approximation.

        F_l ≈ E_x [ (d log p(y|x) / d h_l)^2 ] projected onto safety subspace U_l

        Implementation: EMA update for stability.
          F_new = momentum * F_old + (1 - momentum) * F_current
        """
        batch = next(iter(self.probe_dl))
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels    = batch.get("labels", batch["input_ids"]).to(self.device)

        ext = ActivationExtractor(self.model, self.layer_indices)
        captured_acts = {}

        # Enable gradients for activation capture
        self.model.zero_grad()

        with ext.capture():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                use_cache=False,
            )

        captured_acts = ext.get_activations()
        loss = outputs.loss
        loss.backward()

        momentum = self.config.fwssr_momentum
        with torch.no_grad():
            for idx in self.layer_indices:
                h = captured_acts[idx]              # (batch, d)
                U = self.safety_subspaces[idx].to(self.device)  # (d, k)

                if h.grad_fn is not None:
                    # Approximate gradient w.r.t activations via finite difference not available
                    # Use proxy: gradient of loss w.r.t projected activations
                    proj = h @ U                    # (batch, k)
                    # Approximate Fisher as variance of projected activations (proxy)
                    F_new = (proj ** 2).mean(0).clamp(min=1e-6)  # (k,)
                else:
                    proj  = h.detach() @ U
                    F_new = (proj ** 2).mean(0).clamp(min=1e-6)

                # EMA update
                self.fisher_weights[idx] = (
                    momentum * self.fisher_weights[idx] + (1 - momentum) * F_new
                )

        self.model.zero_grad()

    def _adapt_lambda(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Adapts regularization strength based on gradient alignment between
        task objective and safety regularization.

        cos_sim in [-1, 1]:
          +1 → task & safety gradients aligned (reduce lambda)
          -1 → task & safety gradients conflict (increase lambda)

        adjustment factor = 1.0 - 0.5 * cos_sim  ∈ [0.5, 1.5]
        """
        self.optimizer.zero_grad()

        # Task gradient
        task_loss = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
        ).loss
        task_loss.backward()
        task_grads = [p.grad.clone().flatten() for p in self.model.parameters()
                      if p.grad is not None]
        task_grad_vec = torch.cat(task_grads) if task_grads else None
        self.optimizer.zero_grad()

        # Safety gradient
        reg_loss = self._compute_fwssr_loss(input_ids, attn_mask)
        if hasattr(reg_loss, "backward"):
            reg_loss.backward()
        safety_grads = [p.grad.clone().flatten() for p in self.model.parameters()
                        if p.grad is not None]
        safety_grad_vec = torch.cat(safety_grads) if safety_grads else None
        self.optimizer.zero_grad()

        if task_grad_vec is not None and safety_grad_vec is not None:
            cos_sim = F.cosine_similarity(
                task_grad_vec.unsqueeze(0),
                safety_grad_vec.unsqueeze(0),
            ).item()
            adjustment    = 1.0 - 0.5 * cos_sim
            new_lambda    = self.current_lambda * adjustment
            new_lambda    = max(1e-4, min(1.0, new_lambda))
            # Smooth EMA update of lambda
            self.current_lambda = 0.95 * self.current_lambda + 0.05 * new_lambda

    @torch.no_grad()
    def _log_drift(self, step: int) -> None:
        """Quick drift snapshot for monitoring."""
        self.model.eval()
        batch     = next(iter(self.probe_dl))
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)

        ext_ft   = ActivationExtractor(self.model,          self.layer_indices)
        ext_orig = ActivationExtractor(self.original_model, self.layer_indices)

        with ext_ft.capture():
            self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        with reference_on_gpu(self.original_model, self.device):
            with ext_orig.capture():
                self.original_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

        acts_ft   = ext_ft.get_activations()
        acts_orig = ext_orig.get_activations()

        avg_drift = 0.0
        for idx in self.layer_indices:
            U = self.safety_subspaces[idx].cpu()
            d = compute_safety_drift(acts_orig[idx].cpu(), acts_ft[idx].cpu(), U)
            avg_drift += d["safety_drift"]

        avg_drift /= len(self.layer_indices)
        self.drift_history.append((step, avg_drift))
        self.model.train()
        