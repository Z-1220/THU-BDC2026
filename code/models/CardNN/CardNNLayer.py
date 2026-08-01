"""CardNN allocation layer: differentiable Top-K selection + weight allocation.

Uses the Gumbel-Sinkhorn (optimal transport) relaxation of the cardinality-
constrained selection problem (Mena et al. 2018; Wang et al. 2023):

- Assets = N stocks + 1 virtual cash row.
- K positions; transport matrix P in R^{(N+1) x K} is column-stochastic
  (every position is filled) with row sums capped at 1 (each asset is used at
  most once; cash absorbs any unused positions).
- Selection mask m_i = sum_j P[i, j]  ->  sum(m) = K; weights are
  re-weighted by softmax(refined_score / tau_w) over the selected stocks and
  scaled by the invested fraction (1 - cash/K), so the weight sum is <= 1.

Training: Gumbel noise makes the assignment differentiable and stochastic.
Inference: `deterministic_weights` greedily takes the Top-K of
[stocks + cash] logits and applies the learned reweighting, keeping the
competition constraints (<= K stocks, weight sum <= 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSinkhornTopK(nn.Module):
    """Differentiable Top-K selection + allocation with a virtual cash asset."""

    def __init__(
        self,
        tau: float = 1.0,
        n_iter: int = 30,
        cash_logit: float = 0.0,
        reweight_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.tau = max(float(tau), 1e-3)
        self.n_iter = int(n_iter)
        # Learned logit of the virtual cash asset (a constant across stocks).
        self.cash_logit = nn.Parameter(torch.tensor(float(cash_logit)))
        # Learned temperature for the allocation reweighting (clamped > 0.05).
        self.reweight_tau = nn.Parameter(torch.tensor(float(reweight_tau)))

    @staticmethod
    def _log_sinkhorn(logits: torch.Tensor, n_iter: int) -> torch.Tensor:
        """Log-domain Sinkhorn: column-stochastic with row sums capped at 1.

        logits: (N+1, K). Returns log P with col sums = 1 and row sums <= 1.
        """
        L = logits
        for _ in range(n_iter):
            # Column normalization: softmax over rows for each position.
            L = L - torch.logsumexp(L, dim=0, keepdim=True)
            # Row cap: divide by max(row_sum, 1) <=> subtract max(log_row_sum, 0).
            row_lse = torch.logsumexp(L, dim=1, keepdim=True)
            L = L - torch.clamp(row_lse, min=0.0)
        L = L - torch.logsumexp(L, dim=0, keepdim=True)
        return L

    def forward(
        self,
        z: torch.Tensor,
        K: int,
        training: bool = True,
        noise_std: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Allocate portfolio weights from refined scores.

        Args:
            z: (N,) refined per-stock scores.
            K: maximum number of positions (<= 5 for the competition).
            training: add Gumbel noise during training.
            noise_std: Gumbel noise scale (defaults to tau).

        Returns:
            w: (N,) weights, sum <= 1, at most K nonzero entries.
            diagnostics dict (cash position, invested fraction, etc.).
        """
        N = z.shape[0]
        device = z.device
        K = int(K)

        logits = torch.cat([z, self.cash_logit.expand(1)])
        X = logits.unsqueeze(1).expand(N + 1, K).contiguous()
        if training:
            std = noise_std if noise_std is not None else self.tau
            u = torch.rand(X.shape, device=device).clamp_min(1e-8)
            gumbel = -torch.log(-torch.log(u)).clamp_min(1e-8)
            X = X + gumbel * std

        L = self._log_sinkhorn(X / self.tau, self.n_iter)
        P = torch.exp(L)
        mask = P.sum(dim=1)  # (N+1,), sum = K
        stock_mask = mask[:N]
        cash_mask = mask[N]
        invest = (1.0 - cash_mask / K).clamp(min=0.0, max=1.0)

        # Learned reweighting: softmax over refined scores (temperature learned).
        tau_w = F.softplus(self.reweight_tau) + 0.05
        sw = torch.softmax(z / tau_w, dim=0)
        w_raw = stock_mask * sw
        w = w_raw / (w_raw.sum() + 1e-8) * invest

        with torch.no_grad():
            diag = {
                "cash_positions": float(cash_mask.detach()),
                "invested_frac": float(invest.detach()),
                "n_positions": float((stock_mask.detach() > 1e-3).sum()),
                "max_weight": float(w.detach().max()),
            }
        return w, diag

    @torch.no_grad()
    def deterministic_weights(
        self, z: torch.Tensor, K: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Deterministic inference: greedy Top-K of [stocks + cash] logits."""
        N = z.shape[0]
        K = int(K)
        logits = torch.cat([z, self.cash_logit.expand(1)])
        top = torch.topk(logits, min(K, N + 1)).indices
        n_cash = int((top == N).sum())
        sel = top[top < N]
        invest = min(max(1.0 - n_cash / K, 0.0), 1.0)

        tau_w = F.softplus(self.reweight_tau) + 0.05
        w = torch.zeros(N, device=z.device)
        if sel.numel() > 0:
            sw = torch.softmax(z[sel] / tau_w, dim=0)
            w[sel] = sw * invest
        diag = {
            "cash_positions": float(n_cash),
            "invested_frac": float(invest),
            "n_positions": float(sel.numel()),
            "max_weight": float(w.max()),
        }
        return w, diag
