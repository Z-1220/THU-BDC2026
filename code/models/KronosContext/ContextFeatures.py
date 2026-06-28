"""Context feature extraction for Plan B: Context Transformer experiments.

Computes market-level, sector-level, and cross-sectional features that
Kronos cannot see from its per-stock OHLCV input.

Feature categories:
  Market context: HS300 20d return, 20d volatility, market breadth, dispersion
  Sector context: sector momentum, relative strength
  Cross-sectional: CS rank, CS z-score, liquidity features
  Liquidity: amount rank, turnover ratio
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Column name mapping (same as KronosModel)
_COL_MAP = {
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def load_stock_data() -> pd.DataFrame:
    """Load stock_data.csv with standardized column names."""
    csv_path = _PROJECT_ROOT / "data" / "stock_data.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["日期"])
    df = df.rename(columns={**{k: v for k, v in _COL_MAP.items()}, "股票代码": "code"})
    df["code"] = df["code"].astype(str)
    return df


def load_sector_map() -> dict[str, str]:
    """Load sector_l1 mapping from resource/行业分类.csv."""
    csv_path = _PROJECT_ROOT / "resource" / "行业分类.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"证券代码": str})
    return dict(zip(df["证券代码"], df["中证一级行业分类简称"]))


class ContextFeatureExtractor:
    """Extract context features for each signal date.

    Features computed per signal date:
      market_mom_5: 5-day HS300 average return
      market_mom_20: 20-day HS300 average return
      market_vol_20: 20-day HS300 return volatility
      market_breadth_1: % stocks with positive 5d return
      market_dispersion: cross-sectional std of 5d returns
      sector_mom_5: per-sector 5-day average return
      cs_rank: cross-sectional rank of Kronos score
      cs_zscore: cross-sectional z-score of Kronos score
      amount_rank: percentile rank of average turnover amount
      turnover: recent average turnover rate
    """

    def __init__(self, stock_data: pd.DataFrame | None = None):
        self._df = stock_data if stock_data is not None else load_stock_data()
        self._sector_map = load_sector_map()
        self._precompute_market_features()

    def _precompute_market_features(self) -> None:
        """Precompute market-level features for all dates."""
        df = self._df.copy()
        df = df.sort_values(["code", "日期"])

        # Daily HS300 average return (equal-weight proxy)
        daily_ret = df.groupby("日期")["close"].apply(
            lambda x: x.pct_change().mean()
        ).dropna()

        self._daily_ret = daily_ret

    def extract(
        self,
        signal_date: pd.Timestamp,
        instruments: list[str],
        kronos_scores: pd.Series,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """Extract context features for a signal date.

        Args:
            signal_date: The signal date (Friday).
            instruments: List of Qlib instrument codes (e.g. 'SH600000').
            kronos_scores: Per-stock Kronos scores indexed by instrument.
            lookback: Lookback window in trading days.

        Returns:
            DataFrame indexed by instrument with context feature columns.
        """
        df = self._df
        hist = df[df["日期"] <= signal_date]
        n = len(instruments)
        features = {}

        # ---- Market context features ----
        # 5-day and 20-day HS300 average return
        if len(self._daily_ret) > 0:
            ret_series = self._daily_ret[self._daily_ret.index <= signal_date]
            market_mom_5 = ret_series.tail(5).mean() if len(ret_series) >= 5 else 0.0
            market_mom_20 = ret_series.tail(20).mean() if len(ret_series) >= 20 else 0.0
            market_vol_20 = ret_series.tail(20).std() if len(ret_series) >= 20 else 0.0
        else:
            market_mom_5 = market_mom_20 = market_vol_20 = 0.0

        features["market_mom_5"] = [market_mom_5] * n
        features["market_mom_20"] = [market_mom_20] * n
        features["market_vol_20"] = [market_vol_20] * n

        # Market breadth: % stocks with positive 5d return
        breadth_vals = []
        dispersion_vals = []
        for inst in instruments:
            csv_code = inst[2:] if inst.startswith(("SH", "SZ")) else inst
            inst_hist = hist[hist["code"] == csv_code].sort_values("日期")
            if len(inst_hist) >= 6:
                ret_5d = inst_hist["close"].iloc[-1] / inst_hist["close"].iloc[-6] - 1
            else:
                ret_5d = 0.0
            breadth_vals.append(1.0 if ret_5d > 0 else 0.0)
            dispersion_vals.append(ret_5d)

        market_breadth = np.mean(breadth_vals) if breadth_vals else 0.0
        market_dispersion = np.std(dispersion_vals) if dispersion_vals else 0.0
        features["market_breadth_1"] = [market_breadth] * n
        features["market_dispersion"] = [market_dispersion] * n

        # ---- Sector context ----
        csv_codes = [inst[2:] if inst.startswith(("SH", "SZ")) else inst for inst in instruments]
        sectors = [self._sector_map.get(c, "unknown") for c in csv_codes]

        # Per-sector momentum (average 5d return by sector)
        sector_rets = {}
        for inst, sector in zip(instruments, sectors):
            csv_code = inst[2:] if inst.startswith(("SH", "SZ")) else inst
            inst_hist = hist[hist["code"] == csv_code].sort_values("日期")
            if len(inst_hist) >= 6:
                ret_5d = inst_hist["close"].iloc[-1] / inst_hist["close"].iloc[-6] - 1
            else:
                ret_5d = 0.0
            if sector not in sector_rets:
                sector_rets[sector] = []
            sector_rets[sector].append(ret_5d)

        sector_mom_map = {s: np.mean(rets) for s, rets in sector_rets.items()}
        features["sector_mom_5"] = [sector_mom_map.get(s, 0.0) for s in sectors]

        # ---- Cross-sectional features ----
        scores_arr = np.array([kronos_scores.get(inst, 0.0) for inst in instruments])

        # CS rank (percentile)
        from scipy.stats import rankdata
        ranks = rankdata(scores_arr, method="average")
        features["cs_rank"] = (ranks / (n + 1)).tolist()

        # CS z-score
        m, std = scores_arr.mean(), scores_arr.std(ddof=0)
        if std > 1e-8:
            features["cs_zscore"] = ((scores_arr - m) / std).tolist()
        else:
            features["cs_zscore"] = [0.0] * n

        # ---- Liquidity features ----
        amount_ranks = []
        avg_turnovers = []
        for inst in instruments:
            csv_code = inst[2:] if inst.startswith(("SH", "SZ")) else inst
            inst_hist = hist[hist["code"] == csv_code].sort_values("日期")
            if len(inst_hist) >= lookback:
                avg_amount = inst_hist["amount"].tail(lookback).mean()
                avg_volume = inst_hist["volume"].tail(lookback).mean()
            else:
                avg_amount = inst_hist["amount"].mean() if len(inst_hist) > 0 else 0
                avg_volume = inst_hist["volume"].mean() if len(inst_hist) > 0 else 0
            amount_ranks.append(np.log1p(avg_amount) if avg_amount > 0 else 0.0)
            avg_turnovers.append(np.log1p(avg_volume) if avg_volume > 0 else 0.0)

        features["amount_log_60d"] = amount_ranks
        features["turnover_log_60d"] = avg_turnovers

        # Build DataFrame
        result = pd.DataFrame(features, index=instruments)
        return result

    def get_context_vector(
        self,
        signal_date: pd.Timestamp,
        instruments: list[str],
        kronos_scores: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays for model input: (N, D) feature matrix.

        Returns:
            stock_features: (N, D) per-stock feature matrix
            market_features: (D_market,) global context vector
        """
        feat_df = self.extract(signal_date, instruments, kronos_scores)
        all_cols = feat_df.columns.tolist()

        # Market-level features (shared across all stocks)
        market_cols = [
            "market_mom_5", "market_mom_20", "market_vol_20",
            "market_breadth_1", "market_dispersion",
        ]
        market_features = feat_df[market_cols].iloc[0].values.astype(np.float32)

        # Stock-level features
        stock_cols = [c for c in all_cols if c not in market_cols]
        stock_features = feat_df[stock_cols].values.astype(np.float32)

        return stock_features, market_features
