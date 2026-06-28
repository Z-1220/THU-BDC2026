"""Ranking loss functions for Plan A: Loss Function Structure Alignment.

Each loss function corresponds to a research plan experiment.
All losses operate on predicted scores and target labels (future returns).

Reference:
  A-E0: MSELoss (baseline CE regression)
  A-E1: PairwiseMarginLoss (standard pairwise ranking)
  A-E2: ListMLELoss (listwise ranking)
  A-E3: NDCGApproxLoss (NDCG surrogate)
  A-E4: CoarseToFineLoss (coarse categorization + fine ranking)
  A-E5: CandidateGroupRankingLoss (candidate recall + within-group ranking)
  A-E6: StructureConsistencyLoss (regularization + ranking loss)
  A-E7: combined CoarseToFineLoss + rank-weighted proxy
  A-E8-A-E11: Negative control variants
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# A-E0: MSE Baseline
# =============================================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE regression loss — A-E0 baseline.

    Args:
        pred: (N,) predicted scores
        target: (N,) true future returns (LABEL0)
    """
    return F.mse_loss(pred, target)


# =============================================================================
# A-E1: Pairwise Margin Ranking Loss
# =============================================================================

class PairwiseMarginLoss(nn.Module):
    """Pairwise margin ranking loss with configurable margin.

    For each pair (i, j) where target[i] > target[j], enforces:
        pred[i] - pred[j] >= margin
    """

    def __init__(self, margin: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pairwise margin loss.

        Args:
            pred: (N,) predicted scores
            target: (N,) true returns
        """
        n = pred.size(0)
        # Pairwise differences: (N, N)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # pred[i] - pred[j]
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)  # target[i] - target[j]

        # Mask: only consider pairs where target[i] > target[j]
        mask = (target_diff > 0).float()

        # Hinge loss: max(0, margin - (pred_i - pred_j))
        loss = F.relu(self.margin - pred_diff) * mask

        if self.reduction == "mean":
            n_pairs = mask.sum()
            if n_pairs > 0:
                return loss.sum() / n_pairs
            return torch.tensor(0.0, device=pred.device)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# A-E2: ListMLE (Listwise) Loss
# =============================================================================

class ListMLELoss(nn.Module):
    """ListMLE: Maximum Likelihood Estimation for listwise ranking.

    Defines a Plackett-Luce distribution over permutations and maximizes
    the likelihood of the true ordering (given by target labels).

    Implementation: For sorted target indices π, compute log-probability of
    each successive top-1 given the remaining items:

        P(item_k | remaining) = exp(score_k) / sum_{j in remaining} exp(score_j)
        loss = -sum_k log P(item_k | remaining)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            pred: (N,) predicted scores
            target: (N,) true returns (higher = better)
        """
        n = pred.size(0)
        # Sort by target descending to get true ranking
        _, idx = torch.sort(target, descending=True)
        sorted_pred = pred[idx] / self.temperature

        loss = torch.tensor(0.0, device=pred.device)
        for k in range(n - 1):
            # Remaining items from position k
            remaining = sorted_pred[k:]
            log_prob = remaining[0] - torch.logsumexp(remaining, dim=0)
            loss = loss - log_prob

        return loss / (n - 1)


# =============================================================================
# A-E3: NDCG Approximation (LambdaRank-style) Loss
# =============================================================================

class NDCGApproxLoss(nn.Module):
    """Approximate NDCG loss using a smooth sigmoid surrogate.

    λ-rank style approach: weights pairwise errors by NDCG delta.
    """

    def __init__(self, sigma: float = 1.0, k: int = 5):
        """
        Args:
            sigma: sigmoid temperature for smooth rank approximation.
            k: NDCG@k cutoff.
        """
        super().__init__()
        self.sigma = sigma
        self.k = k

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Approximate 1 - NDCG@k loss.

        Args:
            pred: (N,) predicted scores
            target: (N,) true returns
        """
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)

        # Pairwise sigmoid: P(i > j) = sigmoid(pred_i - pred_j)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        prob = torch.sigmoid(pred_diff * self.sigma)

        # DCG weights: gain = 2^rel - 1, use target directly as relevance
        # Normalize target to [0, 1] for gain computation
        t_min, t_max = target.min(), target.max()
        if t_max - t_min > 1e-8:
            rel = (target - t_min) / (t_max - t_min)
        else:
            return torch.tensor(0.0, device=pred.device)

        gain = 2 ** rel - 1  # (N,)

        # Approximate DCG@k: sum_i gain_i * P(rank_i <= k)
        # P(rank_i <= k) ≈ sigmoid — simpler approximation below

        # Simpler proxy: weighted pairwise loss with NDCG delta weights
        # ΔNDCG(i,j) = (gain_i - gain_j) * (1/log2(rank_i+1) - 1/log2(rank_j+1))
        gain_diff = gain.unsqueeze(1) - gain.unsqueeze(0)  # (N, N)

        # Discount factor differences (approximate with position bias)
        # Use predicted rank positions
        ranks = torch.argsort(torch.argsort(pred, descending=True)).float() + 1
        discount = 1.0 / torch.log2(ranks + 1)
        discount_diff = discount.unsqueeze(1) - discount.unsqueeze(0)

        delta_ndcg = torch.abs(gain_diff * discount_diff)  # (N, N)

        # Weighted logistic loss
        weight = delta_ndcg / (delta_ndcg.sum() + 1e-8)
        target_sign = (target.unsqueeze(1) > target.unsqueeze(0)).float()
        base_loss = F.binary_cross_entropy_with_logits(
            pred_diff * self.sigma, target_sign, reduction="none"
        )
        return (base_loss * weight).sum()


# =============================================================================
# A-E4: Coarse-to-Fine Ranking Loss
# =============================================================================

class CoarseToFineLoss(nn.Module):
    """Two-tier loss aligned with Kronos coarse/fine tokenizer design.

    Coarse layer: classify stock into "strong", "neutral", "weak" tiers.
    Fine layer: within each tier, rank stocks using pairwise loss.

    This mirrors Kronos's architecture where each K-line is encoded as
    coarse + fine subtokens.
    """

    def __init__(
        self,
        coarse_weight: float = 0.3,
        fine_weight: float = 0.7,
        n_tiers: int = 3,
        fine_margin: float = 0.05,
    ):
        """
        Args:
            coarse_weight: weight for coarse classification loss.
            fine_weight: weight for fine pairwise ranking loss.
            n_tiers: number of coarse tiers (default 3: strong/neutral/weak).
            fine_margin: margin for fine pairwise loss.
        """
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.n_tiers = n_tiers
        self.fine_margin = fine_margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coarse_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute coarse-to-fine loss.

        Args:
            pred: (N,) final refined scores (fine scores).
            target: (N,) true returns.
            coarse_logits: (N, n_tiers) logits for coarse classification.
                          If None, derived from pred via quantile binning.

        Returns:
            total_loss, loss_components dict
        """
        n = pred.size(0)
        if n < self.n_tiers:
            # Not enough stocks — fall back to MSE
            loss = F.mse_loss(pred, target)
            return loss, {"total": loss, "coarse": loss, "fine": torch.tensor(0.0)}

        # --- Coarse loss: cross-entropy on tier classification ---
        # Derive coarse labels from target quantiles
        with torch.no_grad():
            boundaries = torch.quantile(
                target, torch.linspace(0, 1, self.n_tiers + 1, device=target.device)
            )
            # Assign tiers (0 = weakest, n_tiers-1 = strongest)
            coarse_labels = torch.zeros(n, dtype=torch.long, device=target.device)
            for t in range(self.n_tiers):
                lower = boundaries[t]
                upper = boundaries[t + 1]
                if t == self.n_tiers - 1:
                    mask = (target >= lower) & (target <= upper)
                else:
                    mask = (target >= lower) & (target < upper)
                coarse_labels[mask] = t

        if coarse_logits is None:
            # Derive coarse logits from pred: use pred value as single-dim logit
            # and create tiers via distance to tier centers
            coarse_logits = torch.zeros(n, self.n_tiers, device=pred.device)
            for t in range(self.n_tiers):
                center = boundaries[t : t + 2].mean()
                coarse_logits[:, t] = -torch.abs(pred - center)

        coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)

        # --- Fine loss: within-tier pairwise ranking ---
        fine_loss = torch.tensor(0.0, device=pred.device)
        n_tier_pairs = 0
        for t in range(self.n_tiers):
            tier_mask = coarse_labels == t
            tier_indices = tier_mask.nonzero(as_tuple=True)[0]
            if len(tier_indices) < 2:
                continue
            tier_pred = pred[tier_indices]
            tier_target = target[tier_indices]
            tier_loss = PairwiseMarginLoss(margin=self.fine_margin)(
                tier_pred, tier_target
            )
            fine_loss = fine_loss + tier_loss
            n_tier_pairs += 1

        if n_tier_pairs > 0:
            fine_loss = fine_loss / n_tier_pairs

        total = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss
        return total, {"coarse": coarse_loss, "fine": fine_loss, "total": total}


# =============================================================================
# A-E5: Candidate + Within-Group Ranking Loss
# =============================================================================

class CandidateGroupRankingLoss(nn.Module):
    """Candidate recall + group ranking loss.

    Stage 1: Binary classification — does stock belong in Top-M?
    Stage 2: Ranking within the candidate group.
    """

    def __init__(
        self,
        top_m: int = 30,
        recall_weight: float = 0.3,
        rank_weight: float = 0.7,
        rank_margin: float = 0.05,
    ):
        super().__init__()
        self.top_m = top_m
        self.recall_weight = recall_weight
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        recall_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute candidate + group ranking loss.

        Args:
            pred: (N,) ranking scores.
            target: (N,) true returns.
            recall_logits: (N,) logits for binary candidate classification.
                          If None, derived from pred.

        Returns:
            total_loss, loss_components dict
        """
        n = pred.size(0)
        m = min(self.top_m, n)

        # --- Recall loss: is stock in Top-M by target? ---
        with torch.no_grad():
            _, top_indices = torch.topk(target, m)
            candidate_labels = torch.zeros(n, device=target.device)
            candidate_labels[top_indices] = 1.0

        if recall_logits is None:
            recall_logits = pred

        recall_loss = F.binary_cross_entropy_with_logits(
            recall_logits, candidate_labels
        )

        # --- Ranking loss: pairwise within candidate group ---
        candidate_mask = candidate_labels > 0.5
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        if len(candidate_indices) >= 2:
            cand_pred = pred[candidate_indices]
            cand_target = target[candidate_indices]
            rank_loss = PairwiseMarginLoss(margin=self.rank_margin)(cand_pred, cand_target)
        else:
            rank_loss = torch.tensor(0.0, device=pred.device)

        total = self.recall_weight * recall_loss + self.rank_weight * rank_loss
        return total, {
            "recall": recall_loss,
            "rank": rank_loss,
            "total": total,
        }


# =============================================================================
# A-E6: Structure Consistency Regularization
# =============================================================================

class StructureConsistencyLoss(nn.Module):
    """Ranking loss with structure consistency regularization.

    Penalizes the ranking head if its scores deviate too far from
    Kronos's original scores, preserving the hierarchical representation.

    L = L_rank + λ * L_consistency
    where L_consistency = ||pred_score - kronos_score||_2 (direction preserved)
    """

    def __init__(
        self,
        base_loss_type: str = "pairwise",
        consistency_weight: float = 0.1,
        consistency_type: str = "mse",  # "mse" or "cosine" or "spearman"
        margin: float = 0.1,
    ):
        """
        Args:
            base_loss_type: "mse", "pairwise", or "listmle"
            consistency_weight: λ weight for consistency regularization.
            consistency_type: type of consistency penalty.
            margin: margin for pairwise base loss.
        """
        super().__init__()
        self.base_loss_type = base_loss_type
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.margin = margin

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute regularized loss.

        Args:
            pred: (N,) predicted scores from ranking head.
            target: (N,) true returns.
            base_scores: (N,) original Kronos scores (before head).

        Returns:
            total_loss, loss_components dict
        """
        # --- Base ranking loss ---
        if self.base_loss_type == "pairwise":
            base_loss = PairwiseMarginLoss(margin=self.margin)(pred, target)
        elif self.base_loss_type == "listmle":
            base_loss = ListMLELoss()(pred, target)
        else:  # mse
            base_loss = F.mse_loss(pred, target)

        # --- Consistency loss ---
        if self.consistency_type == "cosine":
            # Cosine similarity loss: encourage same direction
            cos_sim = F.cosine_similarity(
                pred.unsqueeze(0), base_scores.unsqueeze(0)
            )
            consistency_loss = 1.0 - cos_sim.mean()
        elif self.consistency_type == "spearman":
            # Soft Spearman rank correlation proxy
            # Use differentiable rank approximation
            pred_rank = torch.argsort(torch.argsort(pred)).float()
            base_rank = torch.argsort(torch.argsort(base_scores)).float()
            consistency_loss = F.mse_loss(pred_rank, base_rank) / (pred.size(0) ** 2)
        else:  # mse
            # Encourage scores not to deviate too far
            consistency_loss = F.mse_loss(pred, base_scores)

        total = base_loss + self.consistency_weight * consistency_loss
        return total, {
            "base": base_loss,
            "consistency": consistency_loss,
            "total": total,
        }


# =============================================================================
# A-E7: Structure-Aligned + Rank-Weighted Allocation Proxy
# =============================================================================

class StructureAlignedRankWeightedLoss(nn.Module):
    """Combined coarse-to-fine structure + rank-weighted allocation proxy.

    Uses CoarseToFineLoss architecture but also adds a head that
    predicts optimal allocation weights for Top-K stocks.
    """

    def __init__(
        self,
        coarse_weight: float = 0.2,
        fine_weight: float = 0.5,
        alloc_weight: float = 0.3,
        top_k: int = 3,
        n_tiers: int = 3,
    ):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.alloc_weight = alloc_weight
        self.top_k = top_k
        self.n_tiers = n_tiers
        self.coarse_to_fine = CoarseToFineLoss(n_tiers=n_tiers)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alloc_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            pred: (N,) ranking scores.
            target: (N,) true returns.
            alloc_weights: (N,) predicted allocation weights (softmax).
                          If None, allocation loss is skipped.

        Returns:
            total_loss, loss_components dict
        """
        # Coarse-to-fine ranking component
        _, ctf_losses = self.coarse_to_fine(pred, target)

        # Allocation proxy: predict weights proportional to true Top-K returns
        alloc_loss = torch.tensor(0.0, device=pred.device)
        if alloc_weights is not None:
            with torch.no_grad():
                _, top_indices = torch.topk(target, self.top_k)
                target_alloc = torch.zeros_like(target)
                target_alloc[top_indices] = target[top_indices] - target.min() + 1e-8
                target_alloc = target_alloc / (target_alloc.sum() + 1e-8)

            alloc_loss = F.kl_div(
                (alloc_weights + 1e-8).log(), target_alloc, reduction="batchmean"
            )

        total = (
            self.coarse_weight * ctf_losses["coarse"]
            + self.fine_weight * ctf_losses["fine"]
            + self.alloc_weight * alloc_loss
        )
        return total, {
            "coarse": ctf_losses["coarse"],
            "fine": ctf_losses["fine"],
            "alloc": alloc_loss,
            "total": total,
        }


# =============================================================================
# Negative Control Losses (A-E8 through A-E11)
# =============================================================================

def shuffled_label_loss(
    loss_fn: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """A-E8: Apply loss with randomly shuffled target labels.

    If the structured loss relies on true ordering, shuffling should
    significantly degrade performance.
    """
    shuffled = target[torch.randperm(target.size(0))]
    return loss_fn(pred, shuffled)


def shuffled_stock_loss(
    loss_fn: nn.Module, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """A-E9: Apply loss with randomly shuffled stock order.

    If the cross-sectional structure matters, shuffling should hurt.
    """
    # Shuffle both pred and target together (breaks stock identity)
    perm = torch.randperm(pred.size(0))
    return loss_fn(pred[perm], target)


def coarse_only_loss(
    pred: torch.Tensor, target: torch.Tensor, n_tiers: int = 3
) -> torch.Tensor:
    """A-E10: Only coarse classification, no fine ranking.

    Tests if coarse tier alone is sufficient.
    """
    boundaries = torch.quantile(
        target, torch.linspace(0, 1, n_tiers + 1, device=target.device)
    )
    coarse_labels = torch.zeros(pred.size(0), dtype=torch.long, device=target.device)
    for t in range(n_tiers):
        lower = boundaries[t]
        upper = boundaries[t + 1]
        if t == n_tiers - 1:
            coarse_labels[(target >= lower) & (target <= upper)] = t
        else:
            coarse_labels[(target >= lower) & (target < upper)] = t

    # Use pred to classify into tiers
    coarse_logits = torch.zeros(pred.size(0), n_tiers, device=pred.device)
    for t in range(n_tiers):
        center = boundaries[t : t + 2].mean()
        coarse_logits[:, t] = -torch.abs(pred.unsqueeze(0) - center)

    return F.cross_entropy(coarse_logits, coarse_labels)


def fine_only_loss(
    pred: torch.Tensor, target: torch.Tensor, margin: float = 0.05
) -> torch.Tensor:
    """A-E11: Only fine pairwise ranking, no coarse structure.

    Tests if fine ranking alone is sufficient (standard approach).
    """
    return PairwiseMarginLoss(margin=margin)(pred, target)
