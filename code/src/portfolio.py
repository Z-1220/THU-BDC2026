"""组合优化模块。workflow.py 和 submit.py 共用。

优化器（config → portfolio.optimizer）：
  - "equal"              : 等权
  - "score_proportional" : 按分数比例
  - "mean_variance"      : 模型分数作预期收益 + 风险模型优化
  - "min_variance"       : 纯风险驱动，不用预期收益
  - "risk_parity"        : HRP

风险模型（config → portfolio.params.risk_model）：
  直接传给 pypfopt.risk_models.risk_matrix(method=...)
  详见 https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def to_competition_code(qlib_code: str) -> str:
    q = str(qlib_code).upper()
    return q[2:] if q.startswith(("SH", "SZ")) else q


def fetch_daily_returns(
    start_time: str, end_time: str, instruments: str = "all"
) -> pd.DataFrame | None:
    """从 Qlib 获取日收益率矩阵。"""
    try:
        from qlib.data import D
        df = D.features(instruments=D.instruments(instruments), fields=["$close"],
                        start_time=start_time, end_time=end_time)
        if df is None or df.empty:
            return None
        close = df.iloc[:, 0] if not isinstance(df.columns, pd.MultiIndex) else df.iloc[:, 0]
        return close.unstack(level="instrument").pct_change(1).iloc[1:]
    except Exception as e:
        warnings.warn(f"无法获取收益率矩阵: {e}")
        return None


# ======================================================================
# 内部工具
# ======================================================================

def _ret_slice(returns_df, codes, lookback=60, min_days=20):
    if returns_df is None:
        return None
    avail = [c for c in codes if c in returns_df.columns]
    if not avail:
        return None
    ret = returns_df[avail].dropna(axis=1, how="all").tail(lookback)
    return ret if ret.shape[0] >= min_days and ret.shape[1] >= 2 else None


def _fallback_equal(scores, top_k):
    top = scores.nlargest(top_k)
    w = 1.0 / len(top)
    return {c: w for c in top.index}


def _clip(weights):
    total = sum(weights.values())
    if total > 1.0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ======================================================================
# 优化器（纯函数）
# ======================================================================

def optimize_equal(scores, returns_df=None, top_k=5, **_):
    top = scores.nlargest(top_k)
    w = 1.0 / len(top)
    return {c: w for c in top.index}


def optimize_score_proportional(scores, returns_df=None, top_k=5, **_):
    top = scores.nlargest(top_k)
    total = top.sum()
    if total <= 0:
        return _fallback_equal(scores, top_k)
    return {c: float(v / total) for c, v in top.items()}


def optimize_mean_variance(scores, returns_df=None, top_k=5,
                           risk_model="ledoit_wolf", risk_free_rate=0.02,
                           lookback=60, **risk_kwargs):
    try:
        from pypfopt import EfficientFrontier, risk_models
    except ImportError:
        warnings.warn("pip install pyportfolioopt")
        return _fallback_equal(scores, top_k)

    ret = _ret_slice(returns_df, scores.nlargest(top_k).index.tolist(), lookback)
    if ret is None:
        return _fallback_equal(scores, top_k)

    mu = pd.Series({c: scores.get(c, 0.0) for c in ret.columns}, dtype=float)
    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True, **risk_kwargs)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = {c: w for c, w in ef.clean_weights().items() if w > 1e-4}
    return _clip(weights) if weights else _fallback_equal(scores, top_k)


def optimize_min_variance(scores, returns_df=None, top_k=5,
                          risk_model="ledoit_wolf", lookback=60, **risk_kwargs):
    try:
        from pypfopt import EfficientFrontier, risk_models
    except ImportError:
        return _fallback_equal(scores, top_k)

    ret = _ret_slice(returns_df, scores.nlargest(top_k).index.tolist(), lookback)
    if ret is None:
        return _fallback_equal(scores, top_k)

    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True, **risk_kwargs)
    ef = EfficientFrontier(np.zeros(len(ret.columns)), S)
    ef.min_volatility()
    weights = {c: w for c, w in ef.clean_weights().items() if w > 1e-4}
    return _clip(weights) if weights else _fallback_equal(scores, top_k)


def optimize_risk_parity(scores, returns_df=None, top_k=5, lookback=60, **_):
    try:
        from pypfopt import HRPOpt
    except ImportError:
        return _fallback_equal(scores, top_k)

    ret = _ret_slice(returns_df, scores.nlargest(top_k).index.tolist(), lookback)
    if ret is None:
        return _fallback_equal(scores, top_k)

    weights = {c: w for c, w in HRPOpt(ret).optimize().items() if w > 1e-4}
    return _clip(weights) if weights else _fallback_equal(scores, top_k)


# ======================================================================
# 入口
# ======================================================================

_OPTIMIZERS = {
    "equal": optimize_equal,
    "score_proportional": optimize_score_proportional,
    "mean_variance": optimize_mean_variance,
    "min_variance": optimize_min_variance,
    "risk_parity": optimize_risk_parity,
}


def create_optimizer(config: dict):
    """返回一个 (scores, returns_df) → {code: weight} 的函数。"""
    port = config.get("portfolio", {})
    name = port.get("optimizer", "equal")
    top_k = port.get("top_k", 5)
    params = port.get("params", {})

    fn = _OPTIMIZERS.get(name)
    if fn is None:
        warnings.warn(f"未知 '{name}'，回退等权。可选: {list(_OPTIMIZERS)}")
        fn = optimize_equal

    return lambda scores, returns_df: fn(scores, returns_df, top_k=top_k, **params)
