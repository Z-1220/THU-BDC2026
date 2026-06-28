"""KronosContext: Context-aware Transformer for score refinement.

Research Plan B, Module B3.1: Context Transformer experiments (B-C1 through B-C7).

Architecture:
  [CLS_env] context token (market features)
  Stock tokens (per-stock: Kronos score + sector + CS stats + liquidity)
  Optional sector tokens (industry-level aggregation)
  → TransformerEncoder (self-attention) → per-stock refined scores

Experiments:
  B-C1: No context baseline (Kronos score only)
  B-C2: Market context only
  B-C3: Sector context only
  B-C4: Cross-sectional stats only
  B-C5: Full context (market + sector + CS)
  B-C6: Context shuffle (negative control)
  B-C7: Context zero (negative control)
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
from ..KronosRankHead.RankingLosses import (
    PairwiseMarginLoss,
    ListMLELoss,
)

from .ContextFeatures import ContextFeatureExtractor, load_stock_data


# =============================================================================
# Context Transformer Architecture
# =============================================================================

class ContextTransformer(nn.Module):
    """Transformer that processes stock tokens + market context token.

    Input: [CLS_context] + [stock_1, stock_2, ..., stock_N]
    CLS_context: market-level features (mom, vol, breadth, dispersion)
    Stock tokens: per-stock features (Kronos score, sector, CS stats, liquidity)

    Output: Refined per-stock scores (no CLS output needed).
    """

    def __init__(
        self,
        stock_feat_dim: int = 4,
        context_feat_dim: int = 5,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        use_sector_tokens: bool = False,
        n_sectors: int = 10,
        max_stocks: int = 300,
    ):
        """
        Args:
            stock_feat_dim: Dimension of per-stock features.
            context_feat_dim: Dimension of market context features.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: FFN hidden dimension.
            dropout: Dropout rate.
            use_sector_tokens: Whether to add sector-level tokens.
            n_sectors: Number of unique sector L1 codes.
            max_stocks: Max number of stocks per date (for positional encoding).
        """
        super().__init__()
        self.d_model = d_model
        self.use_sector_tokens = use_sector_tokens

        # Input projections
        self.stock_proj = nn.Linear(stock_feat_dim, d_model)
        self.context_proj = nn.Linear(context_feat_dim, d_model)

        if use_sector_tokens:
            self.sector_embed = nn.Embedding(n_sectors, d_model)
            self.sector_proj = nn.Linear(d_model, d_model)

        # Positional encoding (learned by default, no ordering bias)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1 + max_stocks + (n_sectors if use_sector_tokens else 0), d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        stock_features: torch.Tensor,
        market_features: torch.Tensor,
        sector_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            stock_features: (B, N, stock_feat_dim) — N stocks per batch.
            market_features: (B, context_feat_dim) — global context.
            sector_ids: (B, N,) — sector indices for sector tokens.

        Returns:
            (B, N,) refined scores.
        """
        B, N, _ = stock_features.shape

        # Project to d_model
        stock_emb = self.stock_proj(stock_features)  # (B, N, d_model)
        context_emb = self.context_proj(market_features).unsqueeze(1)  # (B, 1, d_model)

        # Collect tokens
        tokens = [context_emb, stock_emb]

        if self.use_sector_tokens and sector_ids is not None:
            # Aggregate stocks by sector to form sector tokens
            unique_sectors = sector_ids.unique()
            sector_tokens = []
            for sid in unique_sectors:
                mask = (sector_ids == sid).float().unsqueeze(-1)  # (B, N, 1)
                sector_avg = (stock_emb * mask).sum(dim=1, keepdim=True) / (
                    mask.sum(dim=1, keepdim=True) + 1e-8
                )  # (B, 1, d_model)
                sector_tokens.append(sector_avg)
            if sector_tokens:
                sector_emb = torch.cat(sector_tokens, dim=1)  # (B, n_sectors, d_model)
                tokens.append(sector_emb)

        x = torch.cat(tokens, dim=1)  # (B, 1+N+[S], d_model)

        # Add positional encoding
        total_tokens = x.size(1)
        pos = self.pos_encoding[:, :total_tokens, :]
        x = x + pos

        # Transformer
        x = self.transformer(x)  # (B, total_tokens, d_model)

        # Extract stock token outputs (skip CLS and sector tokens)
        stock_start = 1
        stock_end = 1 + N
        stock_out = x[:, stock_start:stock_end, :]  # (B, N, d_model)

        # Score prediction
        scores = self.score_head(stock_out).squeeze(-1)  # (B, N)

        return scores


# =============================================================================
# Qlib Model Wrapper
# =============================================================================

class KronosContextModel(Model):
    """Qlib Model: Kronos base + Context Transformer head.

    Implements Plan B Module B3.1 experiments (B-C1 through B-C7).

    Config via context_mode:
      "baseline" (B-C1): Only Kronos score as feature
      "market" (B-C2): + Market context (mom, vol, breadth, dispersion)
      "sector" (B-C3): + Sector context (sector momentum)
      "cs_stats" (B-C4): + Cross-sectional stats (CS rank, z-score)
      "full" (B-C5): All context features
      "shuffle" (B-C6): Full context but shuffled (negative control)
      "zero" (B-C7): Zero out stock features except Kronos score
    """

    def __init__(
        self,
        # Kronos params
        model_name: str = "small",
        pretrained_dir: str = "./model/kronos_pretrained",
        finetuned_dir: str | None = None,
        max_context: int = 512,
        pred_len: int = 5,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int = 42,
        cs_zscore: bool = False,
        # Context Transformer params
        context_mode: str = "full",
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        use_sector_tokens: bool = False,
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
    ):
        self.logger = get_module_logger("KronosContext")

        # Validate context_mode
        valid_modes = {"baseline", "market", "sector", "cs_stats", "full", "shuffle", "zero"}
        if context_mode not in valid_modes:
            raise ValueError(f"context_mode must be one of {valid_modes}, got '{context_mode}'")
        self.context_mode = context_mode

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
            cs_zscore=False,
        )

        # Context feature extractor
        self.extractor = ContextFeatureExtractor()
        self._sector_to_id = self._build_sector_id_map()

        # Model params
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_val = dropout
        self.use_sector_tokens = use_sector_tokens

        # Feature dimensions based on mode
        self._stock_feat_dim, self._context_feat_dim = self._get_feature_dims()

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

        # Build model
        self._build_model()

        self._fitted = False

    def _build_sector_id_map(self) -> dict[str, int]:
        """Map sector names to integer IDs."""
        from .ContextFeatures import load_sector_map
        s_map = load_sector_map()
        unique = sorted(set(s_map.values()))
        return {s: i for i, s in enumerate(unique)}

    def _get_feature_dims(self) -> tuple[int, int]:
        """Get stock feature dim and context feature dim based on mode."""
        # Available context features
        context_feat_dim = 5  # market_mom_5, market_mom_20, market_vol_20, breadth, dispersion

        # Stock features based on mode
        if self.context_mode == "baseline":
            stock_feat_dim = 1  # Only Kronos score
        elif self.context_mode == "market":
            stock_feat_dim = 2  # Kronos score + market context
        elif self.context_mode == "sector":
            stock_feat_dim = 2  # Kronos score + sector mom
        elif self.context_mode == "cs_stats":
            stock_feat_dim = 3  # Kronos score + CS rank + CS z-score
        else:  # full, shuffle, zero
            stock_feat_dim = 6  # Kronos score + sector_mom + cs_rank + cs_zscore + amount + turnover

        return stock_feat_dim, context_feat_dim

    def _build_model(self) -> None:
        """Build the Context Transformer."""
        self.model = ContextTransformer(
            stock_feat_dim=self._stock_feat_dim,
            context_feat_dim=self._context_feat_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_val,
            use_sector_tokens=self.use_sector_tokens,
            n_sectors=len(self._sector_to_id),
        )
        self.model.to(self.device)

    def _build_loss(self) -> None:
        """Build loss function."""
        kw = self.loss_kwargs
        if self.loss_type == "pairwise":
            self.loss_fn = PairwiseMarginLoss(margin=kw.get("margin", 0.1))
        elif self.loss_type == "listmle":
            self.loss_fn = ListMLELoss(temperature=kw.get("temperature", 1.0))
        else:
            self.loss_fn = None  # Use MSE

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
        """Train the Context Transformer on top of Kronos scores.

        1. Get Kronos scores for train/valid
        2. Extract context features
        3. Train the Context Transformer
        """
        self.logger.info(f"KronosContext.fit() — context_mode={self.context_mode}")
        self.kronos._fitted = True

        train_data, valid_data = self._extract_training_data(dataset)

        if train_data is None or len(train_data[0]) == 0:
            self.logger.warning("No training data. Head is identity.")
            self._fitted = True
            return

        X_train, y_train = train_data
        X_valid, y_valid = valid_data if valid_data is not None else (None, None)

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            # Train: process each date as a "batch" (cross-section)
            for gid in range(len(X_train)):
                stock_f, market_f, y = X_train[gid]
                if stock_f is None:
                    continue

                stock_f = stock_f.unsqueeze(0).to(self.device)  # (1, N, D)
                market_f = market_f.unsqueeze(0).to(self.device)  # (1, D_ctx)
                y_t = y.unsqueeze(0).to(self.device)  # (1, N)

                optimizer.zero_grad()
                pred = self.model(stock_f, market_f)  # (1, N)
                loss = self._compute_loss(pred.squeeze(0), y_t.squeeze(0))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            scheduler.step()

            # Validation
            if X_valid is not None:
                self.model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for gid in range(len(X_valid)):
                        stock_f, market_f, y = X_valid[gid]
                        if stock_f is None:
                            continue
                        stock_f = stock_f.unsqueeze(0).to(self.device)
                        market_f = market_f.unsqueeze(0).to(self.device)
                        y_t = y.unsqueeze(0).to(self.device)
                        pred = self.model(stock_f, market_f)
                        loss = self._compute_loss(pred.squeeze(0), y_t.squeeze(0))
                        val_loss += loss.item()
                        n_val += 1
                val_loss_avg = val_loss / max(n_val, 1)
                self.model.train()

                if val_loss_avg < best_loss:
                    best_loss = val_loss_avg
                    patience = 0
                    self._best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience += 1

                if epoch % 20 == 0:
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={val_loss_avg:.6f}"
                    )

                if patience >= self.early_stop_patience:
                    self.logger.info(f"Early stop at epoch {epoch}")
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
            del self._best_state

        self._fitted = True
        self.logger.info("KronosContext training complete.")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        if self.loss_fn is not None:
            return self.loss_fn(pred, target)
        return F.mse_loss(pred, target)

    def _extract_training_data(
        self, dataset: DatasetH
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    ]:
        """Extract (stock_features, market_features, labels) per signal date.

        Returns per-date lists to preserve cross-sectional structure.
        """
        train_groups = []
        valid_groups = []

        for segment in ["train", "valid"]:
            try:
                seg_data = dataset.prepare(segment, col_set=["feature", "label"], data_key="infer")
            except (KeyError, ValueError):
                continue

            if hasattr(seg_data, "get_index"):
                index = seg_data.get_index()
            else:
                index = seg_data.index
            if len(index) == 0:
                continue

            # Get Kronos scores
            try:
                kronos_scores = self.kronos.predict(dataset, segment=segment)
            except Exception:
                continue

            signal_dates = sorted(index.get_level_values(0).unique())

            for dt in signal_dates:
                try:
                    dt_scores = kronos_scores.loc[dt]
                except KeyError:
                    continue

                instruments = list(dt_scores.index) if hasattr(dt_scores, "index") else []
                if len(instruments) < 5:
                    continue

                try:
                    dt_data = seg_data.loc[dt]
                except KeyError:
                    continue

                # Build stock features and labels
                stock_feats = []
                labels = []
                valid_insts = []

                for inst in instruments:
                    try:
                        row = dt_data.loc[inst]
                        if ('label', 'LABEL0') in row.index:
                            label_val = float(row[('label', 'LABEL0')])
                        else:
                            label_val = float(row.iloc[-1])
                    except (KeyError, IndexError, TypeError):
                        continue

                    score_val = float(dt_scores.loc[inst])
                    stock_feats.append(score_val)
                    labels.append(label_val)
                    valid_insts.append(inst)

                if len(valid_insts) < 5:
                    continue

                # Extract context features
                try:
                    feat_df = self.extractor.extract(dt, valid_insts, dt_scores)
                except Exception:
                    continue

                # Build feature vectors based on context_mode
                stock_features = self._build_stock_features(
                    feat_df, valid_insts, dt_scores
                )

                market_features = self._build_market_features(feat_df)

                labels_t = torch.tensor(labels, dtype=torch.float32)

                group = (stock_features, market_features, labels_t)
                if segment == "train":
                    train_groups.append(group)
                else:
                    valid_groups.append(group)

            self.logger.info(
                f"Extracted {len(train_groups) if segment == 'train' else len(valid_groups)} "
                f"date-groups from '{segment}'"
            )

        return (train_groups if train_groups else None,
                valid_groups if valid_groups else None)

    def _build_stock_features(
        self, feat_df: pd.DataFrame, instruments: list[str], kronos_scores: pd.Series
    ) -> torch.Tensor:
        """Build per-stock feature tensor based on context_mode."""
        col_map = {
            "baseline": [],
            "market": ["market_mom_5", "market_mom_20", "market_vol_20", "market_breadth_1", "market_dispersion"],
            "sector": ["sector_mom_5"],
            "cs_stats": ["cs_rank", "cs_zscore"],
            "full": ["sector_mom_5", "cs_rank", "cs_zscore", "amount_log_60d", "turnover_log_60d"],
            "shuffle": ["sector_mom_5", "cs_rank", "cs_zscore", "amount_log_60d", "turnover_log_60d"],
            "zero": ["sector_mom_5", "cs_rank", "cs_zscore", "amount_log_60d", "turnover_log_60d"],
        }

        selected_cols = col_map.get(self.context_mode, [])
        n = len(instruments)

        # Kronos score as base feature
        scores = np.array([[kronos_scores.get(inst, 0.0)] for inst in instruments])

        if not selected_cols:
            feats = scores
        else:
            extra = feat_df[selected_cols].values.astype(np.float32)

            if self.context_mode == "shuffle":
                # Shuffle extra features (break context-stock alignment)
                perm = np.random.permutation(n)
                extra = extra[perm]
            elif self.context_mode == "zero":
                # Zero out non-Kronos features
                extra = np.zeros_like(extra)

            feats = np.concatenate([scores, extra], axis=1)

        # NaN handling
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(feats, dtype=torch.float32)

    def _build_market_features(self, feat_df: pd.DataFrame) -> torch.Tensor:
        """Build market context vector."""
        market_cols = [
            "market_mom_5", "market_mom_20", "market_vol_20",
            "market_breadth_1", "market_dispersion",
        ]
        vals = feat_df[market_cols].iloc[0].values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def predict(
        self, dataset: DatasetH, segment: str = "test", **kwargs: Any
    ) -> pd.Series:
        """Predict refined scores using Context Transformer.

        Returns pd.Series with (datetime, instrument) MultiIndex.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        kronos_pred = self.kronos.predict(dataset, segment=segment)
        if len(kronos_pred) == 0:
            return kronos_pred

        self.model.eval()
        refined_parts = []

        for dt in sorted(kronos_pred.index.get_level_values(0).unique()):
            try:
                dt_scores = kronos_pred.loc[dt]
            except KeyError:
                continue

            instruments = list(dt_scores.index) if hasattr(dt_scores, "index") else []
            if len(instruments) < 2:
                continue

            try:
                feat_df = self.extractor.extract(dt, instruments, dt_scores)
            except Exception:
                refined_parts.append(
                    pd.Series(
                        dt_scores.values,
                        index=pd.MultiIndex.from_tuples(
                            [(dt, inst) for inst in instruments],
                            names=["datetime", "instrument"],
                        ),
                        name="score",
                    )
                )
                continue

            stock_f = self._build_stock_features(feat_df, instruments, kronos_pred.loc[dt])
            market_f = self._build_market_features(feat_df)

            with torch.no_grad():
                stock_f = stock_f.unsqueeze(0).to(self.device)
                market_f = market_f.unsqueeze(0).to(self.device)
                refined = self.model(stock_f, market_f).squeeze(0).cpu().numpy()

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

    def eval(self) -> "KronosContextModel":
        self.model.eval()
        return self

    def to(self, device: torch.device) -> "KronosContextModel":
        self.device = device
        self.model.to(device)
        self.kronos.to(device)
        return self
