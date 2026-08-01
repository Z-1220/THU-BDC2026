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
        # Raw price data + sector map for the shared feature builder
        # (identical to scripts/train_context_head_d.py build_features).
        self._df_raw = pd.read_csv(
            Path(__file__).resolve().parent.parent.parent.parent / "model" / "data" / "stock_data.csv",
            encoding="utf-8-sig", parse_dates=["日期"],
        )
        _sector_csv = (
            Path(__file__).resolve().parent.parent.parent.parent / "resource" / "行业分类.csv"
        )
        _sdf = pd.read_csv(_sector_csv, encoding="utf-8-sig", dtype={"证券代码": str})
        self._sector_map = dict(zip(_sdf["证券代码"], _sdf["中证一级行业分类简称"]))
        self._rev_cache = self._build_rev_cache()

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

    def _build_rev_cache(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        df2 = self._df_raw.assign(code_s=self._df_raw["股票代码"].astype(str).str.zfill(6))
        cache = {}
        for code, grp in df2.groupby("code_s"):
            g = grp.sort_values("日期")
            cache[str(code)] = (g["日期"].to_numpy(), g["收盘"].to_numpy())
        return cache

    def _rev20(self, code: str, signal_date: pd.Timestamp) -> float:
        item = self._rev_cache.get(code)
        if item is None:
            return 0.0
        dates, closes = item
        i = int(np.searchsorted(dates, np.datetime64(signal_date), side="right")) - 1
        if i < 20:
            return 0.0
        v = closes[i] / closes[i - 20] - 1
        return float(v) if np.isfinite(v) else 0.0

    def _build_features(
        self,
        signal_date: pd.Timestamp,
        instruments: list[str],
        raw_scores: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from src.run_research_experiments import compute_context_features
        except ImportError:
            from code.src.run_research_experiments import compute_context_features

        codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in instruments]
        base = {c: float(v) for c, v in zip(codes, raw_scores.values)}
        ctx = compute_context_features(self._df_raw, signal_date, codes, base, self._sector_map)
        sf = ctx["stock_features"]
        rev = np.array(
            [[self._rev20(c, signal_date)] for c in codes], dtype=np.float32
        )
        stock_f = np.concatenate([sf, rev], axis=1)
        return (
            np.nan_to_num(stock_f, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ctx["market_features"], nan=0.0, posinf=0.0, neginf=0.0),
        )

    def eval(self) -> "KronosContextHeadModel":
        self._net.eval()
        return self

    def to(self, device: torch.device) -> "KronosContextHeadModel":
        self.device = device
        self._net.to(device)
        self.kronos.to(device)
        return self
