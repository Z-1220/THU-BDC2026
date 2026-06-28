"""KronosRankHead: Trainable ranking head on top of Kronos base scores.

Research Plan A: Loss Function Structure Alignment.

Architecture:
  Kronos base scores →[MLP Head | Transformer Head]→ refined scores

The head is trained with configurable ranking losses while Kronos weights
remain frozen. This allows systematic comparison of different loss functions
without retraining the 24.7M-parameter Kronos model.

fit() trains the head. predict() runs Kronos inference then applies the head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from qlib.data.dataset import DatasetH
from qlib.log import get_module_logger
from qlib.model.base import Model

from ..Kronos.KronosModel import KronosModel
from .RankingLosses import (
    PairwiseMarginLoss,
    ListMLELoss,
    NDCGApproxLoss,
    CoarseToFineLoss,
    CandidateGroupRankingLoss,
    StructureConsistencyLoss,
    StructureAlignedRankWeightedLoss,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# =============================================================================
# Score Refinement Head (MLP)
# =============================================================================

class ScoreRefinementMLP(nn.Module):
    """Lightweight MLP that refines per-stock Kronos scores.

    Takes raw Kronos scores and optional additional features,
    outputs refined ranking scores.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_coarse: bool = False,
        n_coarse_tiers: int = 3,
        output_alloc: bool = False,
        top_k: int = 3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd

        self.backbone = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden_dims[-1], 1)  # Refined score

        # Optional heads for structured losses
        self.output_coarse = output_coarse
        self.coarse_head = (
            nn.Linear(hidden_dims[-1], n_coarse_tiers) if output_coarse else None
        )

        self.output_alloc = output_alloc
        self.alloc_head = (
            nn.Linear(hidden_dims[-1], 1) if output_alloc else None
        )  # Per-stock allocation logit

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (N, input_dim) input features.

        Returns:
            dict with keys: "score", and optionally "coarse_logits", "alloc_logits"
        """
        h = self.backbone(x)
        out = {"score": self.score_head(h).squeeze(-1)}

        if self.coarse_head is not None:
            out["coarse_logits"] = self.coarse_head(h)
        if self.alloc_head is not None:
            out["alloc_logits"] = self.alloc_head(h).squeeze(-1)

        return out


# =============================================================================
# Qlib Model Wrapper
# =============================================================================

class KronosRankHeadModel(Model):
    """Qlib Model: Kronos base + trainable ranking head.

    Trains a lightweight head on top of Kronos scores with configurable
    loss functions from Plan A (Loss Function Structure Alignment).
    """

    LOSS_REGISTRY = {
        "mse": "A-E0: MSE baseline",
        "pairwise": "A-E1: Pairwise margin ranking",
        "listmle": "A-E2: ListMLE listwise ranking",
        "ndcg": "A-E3: NDCG approximation",
        "coarse_to_fine": "A-E4: Coarse-to-fine ranking",
        "candidate_group": "A-E5: Candidate + group ranking",
        "structure_consistency": "A-E6: Structure consistency regularization",
        "structure_aligned": "A-E7: Structure aligned + rank-weighted proxy",
        "shuffled_label": "A-E8: Negative control — shuffled labels",
        "shuffled_stock": "A-E9: Negative control — shuffled stocks",
        "coarse_only": "A-E10: Negative control — coarse only",
        "fine_only": "A-E11: Negative control — fine only",
    }

    def __init__(
        self,
        # Kronos base model params
        model_name: str = "small",
        pretrained_dir: str = "./model/kronos_pretrained",
        finetuned_dir: str | None = None,
        max_context: int = 512,
        pred_len: int = 5,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int = 42,
        cs_zscore: bool = False,  # We handle CS z-score in the head
        # Head params
        head_hidden_dims: list[int] | None = None,
        head_dropout: float = 0.1,
        head_input_features: list[str] | None = None,
        # Loss params
        loss_type: str = "mse",
        loss_kwargs: dict[str, Any] | None = None,
        # Training params
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        early_stop_patience: int = 15,
        batch_size: int = 256,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = get_module_logger("KronosRankHead")

        # Kronos base model
        self.kronos = KronosModel(
            model_name=model_name,
            pretrained_dir=pretrained_dir,
            finetuned_dir=finetuned_dir,
            max_context=max_context,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            seed=seed,
            cs_zscore=False,  # Raw scores — we handle normalization in head
        )

        # Head params
        self.head_hidden_dims = head_hidden_dims or [64, 32]
        self.head_dropout = head_dropout
        self.head_input_features = head_input_features or ["kronos_score"]

        # Loss params
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        self._build_loss()

        # Training params
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Build head
        self._build_head()

        self._fitted = False

    def _build_head(self) -> None:
        """Build the score refinement head."""
        input_dim = len(self.head_input_features)

        # Determine if we need special output heads
        needs_coarse = self.loss_type in ("coarse_to_fine", "structure_aligned")
        needs_alloc = self.loss_type == "structure_aligned"

        self.head = ScoreRefinementMLP(
            input_dim=input_dim,
            hidden_dims=self.head_hidden_dims,
            dropout=self.head_dropout,
            output_coarse=needs_coarse,
            n_coarse_tiers=self.loss_kwargs.get("n_tiers", 3),
            output_alloc=needs_alloc,
            top_k=self.loss_kwargs.get("top_k", 3),
        )
        self.head.to(self.device)

    def _build_loss(self) -> None:
        """Build the loss function based on loss_type."""
        kw = self.loss_kwargs

        if self.loss_type == "pairwise":
            self.loss_fn = PairwiseMarginLoss(
                margin=kw.get("margin", 0.1),
            )
        elif self.loss_type == "listmle":
            self.loss_fn = ListMLELoss(
                temperature=kw.get("temperature", 1.0),
            )
        elif self.loss_type == "ndcg":
            self.loss_fn = NDCGApproxLoss(
                sigma=kw.get("sigma", 1.0),
                k=kw.get("k", 5),
            )
        elif self.loss_type == "coarse_to_fine":
            self.loss_fn = CoarseToFineLoss(
                coarse_weight=kw.get("coarse_weight", 0.3),
                fine_weight=kw.get("fine_weight", 0.7),
                n_tiers=kw.get("n_tiers", 3),
                fine_margin=kw.get("fine_margin", 0.05),
            )
        elif self.loss_type == "candidate_group":
            self.loss_fn = CandidateGroupRankingLoss(
                top_m=kw.get("top_m", 30),
                recall_weight=kw.get("recall_weight", 0.3),
                rank_weight=kw.get("rank_weight", 0.7),
            )
        elif self.loss_type == "structure_consistency":
            self.loss_fn = StructureConsistencyLoss(
                base_loss_type=kw.get("base_loss_type", "pairwise"),
                consistency_weight=kw.get("consistency_weight", 0.1),
                consistency_type=kw.get("consistency_type", "mse"),
                margin=kw.get("margin", 0.1),
            )
        elif self.loss_type == "structure_aligned":
            self.loss_fn = StructureAlignedRankWeightedLoss(
                coarse_weight=kw.get("coarse_weight", 0.2),
                fine_weight=kw.get("fine_weight", 0.5),
                alloc_weight=kw.get("alloc_weight", 0.3),
                top_k=kw.get("top_k", 3),
                n_tiers=kw.get("n_tiers", 3),
            )
        else:  # mse (A-E0) and negative controls
            self.loss_fn = None  # Will use F.mse_loss directly

    # ------------------------------------------------------------------
    # Qlib Model Interface
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: DatasetH,
        evals_result: dict | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the ranking head on top of Kronos base scores.

        1. Get base scores from Kronos (frozen)
        2. Train the MLP head with the configured loss function
        3. Early stop on validation set
        """
        self.logger.info(f"KronosRankHead.fit() — loss_type={self.loss_type}")

        # Mark Kronos as fitted (its predict() requires this)
        self.kronos._fitted = True

        # Step 1: Get Kronos base scores for training data
        # We need to extract (score, label) pairs from the dataset
        # The dataset has (datetime, instrument) index with features and labels
        train_data, valid_data = self._extract_training_data(dataset)

        if len(train_data) == 0:
            self.logger.warning("No training data extracted — head will be identity.")
            self._fitted = True
            return

        # Step 2: Train the head
        X_train, y_train = train_data
        X_valid, y_valid = valid_data if valid_data is not None else (None, None)

        self.head.train()
        optimizer = AdamW(
            self.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Mini-batch training
            self.head.train()
            total_loss = 0.0
            n_batches = 0

            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i : i + self.batch_size]
                x_batch = X_train[idx].to(self.device)
                y_batch = y_train[idx].to(self.device)

                optimizer.zero_grad()
                out = self.head(x_batch)
                loss = self._compute_loss(out, y_batch, x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.head.parameters(), max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            scheduler.step()

            # Validation
            if X_valid is not None:
                self.head.eval()
                with torch.no_grad():
                    out = self.head(X_valid.to(self.device))
                    val_loss = self._compute_loss(out, y_valid.to(self.device), X_valid.to(self.device))
                val_loss_val = val_loss.item()
                self.head.train()

                if val_loss_val < best_loss:
                    best_loss = val_loss_val
                    patience_counter = 0
                    # Save best head state
                    self._best_head_state = {
                        k: v.cpu().clone()
                        for k, v in self.head.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if epoch % 20 == 0:
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={val_loss_val:.6f}"
                    )

                if patience_counter >= self.early_stop_patience:
                    self.logger.info(f"Early stop at epoch {epoch}")
                    break
            elif epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={avg_loss:.6f}")

        # Restore best head state
        if hasattr(self, "_best_head_state"):
            self.head.load_state_dict(self._best_head_state)
            del self._best_head_state

        self._fitted = True
        self.logger.info("KronosRankHead training complete.")

    def _compute_loss(
        self,
        out: dict[str, torch.Tensor],
        y_batch: torch.Tensor,
        x_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on loss_type."""
        pred = out["score"]

        if self.loss_type == "shuffled_label":
            from .RankingLosses import shuffled_label_loss

            base_loss = PairwiseMarginLoss(margin=self.loss_kwargs.get("margin", 0.1))
            return shuffled_label_loss(base_loss, pred, y_batch)
        elif self.loss_type == "shuffled_stock":
            from .RankingLosses import shuffled_stock_loss

            base_loss = PairwiseMarginLoss(margin=self.loss_kwargs.get("margin", 0.1))
            return shuffled_stock_loss(base_loss, pred, y_batch)
        elif self.loss_type == "coarse_only":
            from .RankingLosses import coarse_only_loss

            return coarse_only_loss(pred, y_batch, self.loss_kwargs.get("n_tiers", 3))
        elif self.loss_type == "fine_only":
            from .RankingLosses import fine_only_loss

            return fine_only_loss(pred, y_batch, self.loss_kwargs.get("margin", 0.05))
        elif self.loss_type == "coarse_to_fine":
            total, _ = self.loss_fn(pred, y_batch, out.get("coarse_logits"))
            return total
        elif self.loss_type == "candidate_group":
            total, _ = self.loss_fn(pred, y_batch, pred)  # Use pred as recall logits
            return total
        elif self.loss_type == "structure_consistency":
            base_scores = x_batch[:, 0]  # First feature is Kronos score
            total, _ = self.loss_fn(pred, y_batch, base_scores)
            return total
        elif self.loss_type == "structure_aligned":
            total, _ = self.loss_fn(
                pred, y_batch, out.get("alloc_logits")
            )
            return total
        elif self.loss_fn is not None:
            # pairwise, listmle, ndcg
            return self.loss_fn(pred, y_batch)
        else:
            # mse / default
            from .RankingLosses import mse_loss

            return mse_loss(pred, y_batch)

    def _extract_training_data(
        self, dataset: DatasetH
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None]:
        """Extract (features, labels) from Qlib dataset using Kronos scores.

        Uses kronos.predict() for efficient batch inference across all signal dates.

        Returns:
            ((X_train, y_train), (X_valid, y_valid)) tensors or None for valid
        """
        all_X_parts = []
        all_y_parts = []
        all_seg_parts = []

        for segment in ["train", "valid"]:
            try:
                seg_data = dataset.prepare(
                    segment, col_set=["feature", "label"], data_key="infer"
                )
            except (KeyError, ValueError):
                self.logger.warning(f"Could not prepare segment '{segment}', skipping.")
                continue

            if hasattr(seg_data, "get_index"):
                index = seg_data.get_index()
            else:
                index = seg_data.index

            if len(index) == 0:
                continue

            # Get Kronos scores for this segment (efficient batch inference)
            try:
                kronos_scores = self.kronos.predict(dataset, segment=segment)
            except Exception as e:
                self.logger.warning(f"Kronos predict failed for '{segment}': {e}")
                continue

            signal_dates = sorted(index.get_level_values(0).unique())

            for dt in signal_dates:
                # Get instruments for this date from kronos scores
                try:
                    dt_scores = kronos_scores.loc[dt]
                except KeyError:
                    continue

                instruments = list(dt_scores.index) if hasattr(dt_scores, 'index') else []
                if len(instruments) < 5:
                    continue

                # Get labels from dataset
                try:
                    dt_data = seg_data.loc[dt]
                except KeyError:
                    continue

                scores = []
                labels = []
                for inst in instruments:
                    try:
                        label_val = dt_data.loc[inst]
                        if hasattr(label_val, "iloc"):
                            label_val = float(label_val.iloc[:, -1].iloc[0])
                        else:
                            label_val = float(label_val)
                    except (KeyError, IndexError):
                        continue

                    score_val = float(dt_scores.loc[inst])

                    if np.isfinite(score_val) and np.isfinite(label_val):
                        scores.append(score_val)
                        labels.append(label_val)

                if len(scores) < 5:
                    continue

                X = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
                y = torch.tensor(labels, dtype=torch.float32)

                all_X_parts.append(X)
                all_y_parts.append(y)
                all_seg_parts.append(segment)

            self.logger.info(
                f"Extracted {sum(1 for s in all_seg_parts if s == segment)} date-groups from segment '{segment}'"
            )

        if not all_X_parts:
            return (torch.zeros(0, 1), torch.zeros(0)), None

        # Separate train/valid
        train_idx = [i for i, s in enumerate(all_seg_parts) if s == "train"]
        valid_idx = [i for i, s in enumerate(all_seg_parts) if s == "valid"]

        X_train = torch.cat([all_X_parts[i] for i in train_idx]) if train_idx else torch.zeros(0, 1)
        y_train = torch.cat([all_y_parts[i] for i in train_idx]) if train_idx else torch.zeros(0)

        if valid_idx:
            X_valid = torch.cat([all_X_parts[i] for i in valid_idx])
            y_valid = torch.cat([all_y_parts[i] for i in valid_idx])
            return (X_train, y_train), (X_valid, y_valid)

        return (X_train, y_train), None

    def predict(
        self,
        dataset: DatasetH,
        segment: str = "test",
        **kwargs: Any,
    ) -> pd.Series:
        """Predict refined scores: Kronos base → MLP head.

        Returns pd.Series with (datetime, instrument) MultiIndex.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get Kronos predictions
        kronos_pred = self.kronos.predict(dataset, segment=segment)

        if len(kronos_pred) == 0:
            return kronos_pred

        # Apply trained head
        self.head.eval()
        refined_parts = []

        for dt in sorted(kronos_pred.index.get_level_values(0).unique()):
            dt_mask = kronos_pred.index.get_level_values(0) == dt
            instruments = list(
                kronos_pred.index.get_level_values(1)[dt_mask]
            )
            scores = kronos_pred.loc[dt].values

            if len(scores) == 0:
                continue

            x = torch.tensor(
                scores.reshape(-1, 1), dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                out = self.head(x)
                refined = out["score"].cpu().numpy()

            refined_parts.append(
                pd.Series(
                    refined,
                    index=pd.MultiIndex.from_tuples(
                        [(dt, inst) for inst in instruments],
                        names=["datetime", "instrument"],
                    ),
                    name="score",
                )
            )

        if not refined_parts:
            return kronos_pred

        result = pd.concat(refined_parts)
        result.name = "score"
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def eval(self) -> "KronosRankHeadModel":
        self.head.eval()
        return self

    def to(self, device: torch.device) -> "KronosRankHeadModel":
        self.device = device
        self.head.to(device)
        self.kronos.to(device)
        return self
