"""PyPortfolioOpt 组合优化策略，继承 Qlib WeightStrategyBase。

内置策略（YAML 直接配置，无需自定义代码）：
  - TopkDropoutStrategy: Top-K 等权，Qlib 原生支持

本模块提供 Qlib 未内置的优化方法，通过 WeightStrategyBase 封装：
  - mean_variance    : 均值-方差优化（分数作预期收益）
  - min_variance     : 最小方差优化
  - risk_parity      : 层级风险平价 (HRP)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from qlib.contrib.strategy import WeightStrategyBase


# ----------------------------------------------------------------------
# PyPortfolioOpt 优化核心
# ----------------------------------------------------------------------
def _mean_variance(scores, returns_df, top_k, risk_model, lookback, risk_free_rate):
    from pypfopt import EfficientFrontier, risk_models

    if returns_df is None:
        return {}
    ret = returns_df.reindex(columns=scores.nlargest(top_k).index).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    mu = pd.Series({c: scores[c] for c in ret.columns}, dtype=float)
    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    return {c: w for c, w in ef.clean_weights().items() if w > 1e-4}


def _min_variance(scores, returns_df, top_k, risk_model, lookback):
    from pypfopt import EfficientFrontier, risk_models

    if returns_df is None:
        return {}
    ret = returns_df.reindex(columns=scores.nlargest(top_k).index).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True)
    ef = EfficientFrontier(np.zeros(len(ret.columns)), S)
    ef.min_volatility()
    return {c: w for c, w in ef.clean_weights().items() if w > 1e-4}


def _risk_parity(scores, returns_df, top_k, lookback):
    from pypfopt import HRPOpt

    if returns_df is None:
        return {}
    ret = returns_df.reindex(columns=scores.nlargest(top_k).index).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    hrp = HRPOpt(ret)
    return {c: w for c, w in hrp.optimize().items() if w > 1e-4}


_OPTIMIZERS = {
    "mean_variance": _mean_variance,
    "min_variance": _min_variance,
    "risk_parity": _risk_parity,
}


class PyPortfolioOptStrategy(WeightStrategyBase):
    """PyPortfolioOpt 组合优化策略，通过 YAML 配置切换优化方法。

    YAML 示例:
      strategy:
        class: PyPortfolioOptStrategy
        module_path: "code.strategies.portfolio_strategy"
        kwargs:
            optimizer: "mean_variance"
            top_k: 5
            risk_model: "ledoit_wolf"
            lookback: 60
            risk_free_rate: 0.02
    """

    def __init__(
        self,
        *,
        optimizer: str = "mean_variance",
        top_k: int = 5,
        risk_model: str = "ledoit_wolf",
        lookback: int = 60,
        risk_free_rate: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._optimizer_name = optimizer
        self._top_k = top_k
        self._risk_model = risk_model
        self._lookback = lookback
        self._risk_free_rate = risk_free_rate

        self._opt_func = _OPTIMIZERS.get(optimizer)
        if self._opt_func is None:
            raise ValueError(f"未知优化器 '{optimizer}'。可选: {list(_OPTIMIZERS)}")

    def generate_target_weight_position(self, score, current, trade_date):
        """生成目标仓位权重。"""
        returns_df = self._get_returns(trade_date)
        try:
            weights = self._opt_func(score, returns_df, self._top_k,
                                     self._risk_model, self._lookback, self._risk_free_rate)
        except Exception:
            warnings.warn("优化失败，返回空权重")
            weights = {}
        return weights

    def _get_returns(self, trade_date):
        """获取日收益率矩阵。"""
        from qlib.data import D
        start = (trade_date - pd.Timedelta(days=self._lookback * 2)).strftime("%Y-%m-%d")
        end = trade_date.strftime("%Y-%m-%d")
        df = D.features(instruments="all", fields=["$close"], start_time=start, end_time=end)
        if df is None or df.empty:
            return None
        close = df.iloc[:, 0].unstack(level="instrument")
        return close.pct_change().iloc[1:]