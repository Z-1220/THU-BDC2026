"""KronosCardNN: end-to-end stock selection + portfolio allocation model.

Pipeline (end-to-end, differentiable during training):
    fine-tuned Kronos scores
        + context features (sector momentum, CS rank/z-score, liquidity)
        -> Context Transformer (cross-stock self-attention + market CLS token)
        -> refined per-stock scores
        -> CardNN (Gumbel-Sinkhorn Top-K + learned reweighting, cash asset)
        -> portfolio weights (<= K stocks, weight sum <= 1)

The head is trained by `scripts/train_cardnn_e2e.py` (portfolio-return loss,
NDCG warm-start) and the weights are loaded by `fit()` for production.
`predict()` returns refined scores (Qlib interface); `allocate()` returns the
final portfolio for the signal date, used by `commit.py` (exposure_mode=cardnn).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from qlib.data.dataset import DatasetH
from qlib.log import get_module_logger
from qlib.model.base import Model

from ..Kronos.KronosModel import KronosModel
from ..KronosContext.ContextFeatures import ContextFeatureExtractor
from ..KronosContext.KronosContext import ContextTransformer
from .CardNNLayer import GumbelSinkhornTopK


_MARKET_COLS = [
    "market_mom_5",
    "market_mom_20",
    "market_vol_20",
    "market_breadth_1",
    "market_dispersion",
]
_STOCK_COLS = [
    "sector_mom_5",
    "cs_rank",
    "cs_zscore",
    "amount_log_60d",
    "turnover_log_60d",
]


class KronosCardNNModel(Model):
    """Qlib Model: Kronos base + Context Transformer + CardNN allocation."""

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
        # Context Transformer params
        context_mode: str = "full",
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        # CardNN params
        K: int = 3,
        cardnn_tau: float = 1.0,
        cardnn_n_iter: int = 30,
        cardnn_weights: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = get_module_logger("KronosCardNN")
        self.context_mode = context_mode
        self.K = int(K)
        self.cardnn_weights = cardnn_weights
        self.seed = seed

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
            cs_zscore=False,  # raw scores — CS handled by context features
        )
        self.extractor = ContextFeatureExtractor()

        self.transformer = ContextTransformer(
            stock_feat_dim=6,
            context_feat_dim=5,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_stocks=350,
        )
        self.cardnn = GumbelSinkhornTopK(
            tau=cardnn_tau, n_iter=cardnn_n_iter
        )
        self._net = nn.ModuleDict(
            {"transformer": self.transformer, "cardnn": self.cardnn}
        )

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._net.to(self.device)

        self._fitted = False
        self._last_raw: dict[pd.Timestamp, pd.Series] = {}

    # ------------------------------------------------------------------
    # Qlib Model interface
    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: DatasetH | None = None,
        evals_result: dict | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the trained head weights (training done by train_cardnn_e2e.py)."""
        if self.cardnn_weights and Path(self.cardnn_weights).exists():
            sd = torch.load(
                self.cardnn_weights, map_location="cpu", weights_only=False
            )
            self._net.load_state_dict(sd)
            self.logger.info(f"Loaded CardNN head weights from {self.cardnn_weights}")
        else:
            self.logger.warning(
                f"cardnn_weights not found at {self.cardnn_weights}; "
                "using untrained head — run train_cardnn_e2e.py first."
            )
        self._fitted = True

    def predict(
        self, dataset: DatasetH, segment: str = "test", **kwargs: Any
    ) -> pd.Series:
        """Refined scores via Context Transformer (raw Kronos scores stashed)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        kronos_pred = self.kronos.predict(dataset, segment=segment)
        if len(kronos_pred) == 0:
            return kronos_pred

        self._net.eval()
        refined_parts = []
        for dt in sorted(kronos_pred.index.get_level_values(0).unique()):
            dt_scores = kronos_pred.loc[dt]
            instruments = list(dt_scores.index) if hasattr(dt_scores, "index") else []
            if len(instruments) < 2:
                continue
            self._last_raw[dt] = dt_scores

            stock_f, market_f = self._build_features(dt, instruments, dt_scores)
            with torch.no_grad():
                sf_t = torch.tensor(stock_f, dtype=torch.float32).unsqueeze(0).to(self.device)
                mf_t = torch.tensor(market_f, dtype=torch.float32).unsqueeze(0).to(self.device)
                refined = self.transformer(sf_t, mf_t).squeeze(0).cpu().numpy()

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
    # Allocation (production entry point)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def allocate(
        self, scores: pd.Series, signal_date: str | pd.Timestamp
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Deterministic CardNN allocation for a single signal date.

        Args:
            scores: refined scores indexed by instrument (Qlib codes).
            signal_date: the Friday signal date.

        Returns:
            (weights {instrument: weight}, diagnostics).
        """
        dt = pd.Timestamp(signal_date)
        instruments = list(scores.index)
        raw = self._last_raw.get(dt)
        if raw is None:
            raise RuntimeError(
                f"Raw Kronos scores not stashed for {dt.date()}; call predict() first."
            )
        stock_f, market_f = self._build_features(dt, instruments, raw)
        sf_t = torch.tensor(stock_f, dtype=torch.float32).unsqueeze(0).to(self.device)
        mf_t = torch.tensor(market_f, dtype=torch.float32).unsqueeze(0).to(self.device)
        refined = self.transformer(sf_t, mf_t).squeeze(0)
        w, diag = self.cardnn.deterministic_weights(refined, self.K)
        weights = {
            inst: float(wi) for inst, wi in zip(instruments, w.cpu().numpy()) if wi > 0
        }
        return weights, diag

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------
    def _build_features(
        self,
        signal_date: pd.Timestamp,
        instruments: list[str],
        raw_scores: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        feat_df = self.extractor.extract(signal_date, instruments, raw_scores)
        scores = np.array(
            [[raw_scores.get(inst, 0.0)] for inst in instruments], dtype=np.float32
        )
        extra = feat_df[_STOCK_COLS].values.astype(np.float32)
        stock_f = np.concatenate([scores, extra], axis=1)
        market_f = feat_df[_MARKET_COLS].iloc[0].values.astype(np.float32)
        return (
            np.nan_to_num(stock_f, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(market_f, nan=0.0, posinf=0.0, neginf=0.0),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def eval(self) -> "KronosCardNNModel":
        self._net.eval()
        return self

    def to(self, device: torch.device) -> "KronosCardNNModel":
        self.device = device
        self._net.to(device)
        self.kronos.to(device)
        return self
