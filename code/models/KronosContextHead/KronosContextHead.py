"""Learned ranking head (multi-seed ensemble) for production — ML-driven
stock selection with learned soft risk screening.

Features per stock (order fixed, shared with scripts/train_context_head_d.py):
  [kronos_score, sector_mom_5, cs_rank, cs_zscore, amount_log_60d,
   turnover_log_60d, rev20, vol20, dd20]
Market context (5): [market_mom_5, market_mom_20, market_vol_20,
                     market_breadth_1, market_dispersion]

rev20/vol20/dd20 are risk features the attention head learns to use (soft
risk screening); hard indicator screens (ScreenProcessor) remain as
constraints. head_weights may be a single path or a list (ensemble: refined
scores are averaged across heads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

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
STOCK_FEAT_DIM = 7  # score + 5 context cols + rev20


class KronosContextHeadModel(Model):
    """Kronos base + learned Context Transformer ranking head (Top-3 equal)."""

    def __init__(
        self,
        model_name: str = "small",
        pretrained_dir: str = "./model/kronos_pretrained",
        finetuned_dir: str | None = None,
        max_context: int = 512,
        pred_len: int = 5,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int = 42,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        head_weights: Union[str, list[str], None] = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = get_module_logger("KronosContextHead")
        self.seed = seed
        paths = [head_weights] if isinstance(head_weights, str) else (head_weights or [])
        self.head_weights: list[str] = list(paths)

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
        self.extractor = ContextFeatureExtractor()

        def _new_head() -> ContextTransformer:
            return ContextTransformer(
                stock_feat_dim=STOCK_FEAT_DIM,
                context_feat_dim=5,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_stocks=350,
            )

        self._nets = nn.ModuleList([_new_head() for _ in range(max(1, len(self.head_weights)))])
        self._net = self._nets
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._net.to(self.device)
        self._fitted = False

    def _ensure_ready(self) -> None:
        if self._fitted:
            return
        loaded = 0
        for i, path in enumerate(self.head_weights):
            if Path(path).exists():
                sd = torch.load(path, map_location="cpu", weights_only=False)
                self._nets[i].load_state_dict(sd)
                loaded += 1
            else:
                self.logger.warning(f"head weight not found: {path}")
        if loaded == 0 and self.head_weights:
            self.logger.warning("no head weights loaded")
        self.kronos._fitted = True
        self._fitted = True

    def fit(
        self,
        dataset: DatasetH | None = None,
        evals_result: dict | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load the trained heads (training done by scripts/train_context_head_d.py)."""
        self._ensure_ready()

    def predict(
        self, dataset: DatasetH, segment: str = "test", **kwargs: Any
    ) -> pd.Series:
        """Ensemble refined ML scores (mean over heads)."""
        self._ensure_ready()
        kronos_pred = self.kronos.predict(dataset, segment=segment)
        if len(kronos_pred) == 0:
            return kronos_pred
        self._net.eval()
        parts = []
        for dt in sorted(kronos_pred.index.get_level_values(0).unique()):
            dt_scores = kronos_pred.loc[dt]
            instruments = list(dt_scores.index)
            if len(instruments) < 2:
                continue
            stock_f, market_f = self._build_features(dt, instruments, dt_scores)
            sf = torch.tensor(stock_f, dtype=torch.float32).unsqueeze(0).to(self.device)
            mf = torch.tensor(market_f, dtype=torch.float32).unsqueeze(0).to(self.device)
            refineds = []
            with torch.no_grad():
                for head in self._nets:
                    refineds.append(head(sf, mf).squeeze(0).cpu().numpy())
            refined = np.mean(refineds, axis=0)
            parts.append(
                pd.Series(
                    refined,
                    index=pd.MultiIndex.from_tuples(
                        [(dt, inst) for inst in instruments],
                        names=["datetime", "instrument"],
                    ),
                    name="score",
                )
            )
        if not parts:
            return kronos_pred
        result = pd.concat(parts)
        result.name = "score"
        return result

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
        rev20 = np.array(
            [[self._rev20(instrument, signal_date)] for instrument in instruments],
            dtype=np.float32,
        )
        stock_f = np.concatenate([scores, extra, rev20], axis=1)
        market_f = feat_df[_MARKET_COLS].iloc[0].values.astype(np.float32)
        return (
            np.nan_to_num(stock_f, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(market_f, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _rev20(self, instrument: str, signal_date: pd.Timestamp) -> float:
        code = instrument[2:] if instrument.startswith(("SH", "SZ")) else instrument
        d = self.extractor._df
        hist = d[(d["code"] == code) & (d["日期"] <= signal_date)].sort_values("日期")
        c = hist["close"].to_numpy()
        if len(c) < 21:
            return 0.0
        v = c[-1] / c[-21] - 1
        return float(v) if np.isfinite(v) else 0.0

    def eval(self) -> "KronosContextHeadModel":
        self._net.eval()
        return self

    def to(self, device: torch.device) -> "KronosContextHeadModel":
        self.device = device
        self._net.to(device)
        self.kronos.to(device)
        return self
