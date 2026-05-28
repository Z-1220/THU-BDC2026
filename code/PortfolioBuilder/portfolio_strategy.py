"""PyPortfolioOpt 组合优化策略。

内置策略（YAML 直接配置，无需自定义代码）：
  - TopkDropoutStrategy: Top-K 等权，Qlib 原生支持

本模块提供 Qlib 未内置的优化方法：
  - mean_variance    : 均值-方差优化（分数作预期收益）
  - min_variance     : 最小方差优化
  - risk_parity      : 层级风险平价 (HRP)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# PyPortfolioOpt 优化核心
# ----------------------------------------------------------------------
def _mean_variance(scores, returns_df, top_k, risk_model, lookback, risk_free_rate):
    from pypfopt import EfficientFrontier, risk_models

    if returns_df is None:
        return {}
    ret = returns_df.reindex(
        columns=scores.nlargest(top_k).index
    ).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    mu = pd.Series({c: scores[c] for c in ret.columns}, dtype=float)
    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True)

    # 年化无风险利率 → 周利率
    rf_weekly = risk_free_rate / 52.0

    ef = EfficientFrontier(mu, S)

    # 尝试 max_sharpe，失败则降级为 min_volatility
    try:
        ef.max_sharpe(risk_free_rate=rf_weekly)
    except ValueError:
        try:
            ef.min_volatility()
        except Exception:
            return {}

    return {c: w for c, w in ef.clean_weights().items() if w > 1e-4}



def _min_variance(scores, returns_df, top_k, risk_model, lookback, risk_free_rate=None):
    from pypfopt import EfficientFrontier, risk_models

    if returns_df is None:
        return {}
    ret = returns_df.reindex(
        columns=scores.nlargest(top_k).index
    ).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    S = risk_models.risk_matrix(ret, method=risk_model, returns_data=True)
    ef = EfficientFrontier(np.zeros(len(ret.columns)), S)
    ef.min_volatility()
    return {c: w for c, w in ef.clean_weights().items() if w > 1e-4}


def _risk_parity(scores, returns_df, top_k, lookback, risk_model=None, risk_free_rate=None):
    from pypfopt import HRPOpt

    if returns_df is None:
        return {}
    ret = returns_df.reindex(
        columns=scores.nlargest(top_k).index
    ).dropna(axis=1, how="all").tail(lookback)
    if ret.shape[0] < 20 or ret.shape[1] < 2:
        return {}

    hrp = HRPOpt(ret)
    return {c: w for c, w in hrp.optimize().items() if w > 1e-4}


_OPTIMIZERS = {
    "mean_variance": _mean_variance,
    "min_variance": _min_variance,
    "risk_parity": _risk_parity,
}


class PyPortfolioOptStrategy:
    """PyPortfolioOpt 组合优化策略（独立类，不依赖 Qlib backtest 框架）。

    YAML 示例:
      strategy:
        class: PyPortfolioOptStrategy
        module_path: "code.PortfolioBuilder.portfolio_strategy"
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
        # 不调用 super().__init__()，彻底脱离 Qlib backtest 框架
        self._optimizer_name = optimizer
        self._top_k = top_k
        self._risk_model = risk_model
        self._lookback = lookback
        self._risk_free_rate = risk_free_rate

        self._opt_func = _OPTIMIZERS.get(optimizer)
        if self._opt_func is None:
            raise ValueError(f"未知优化器 '{optimizer}'，可选: {list(_OPTIMIZERS)}")

        # 价格数据由外部 set_price_data() 注入，避免调用 D.features()
        self._close_df: pd.DataFrame | None = None

    def set_price_data(self, close_df: pd.DataFrame) -> None:
        """注入 close 价格数据。

        Parameters
        ----------
        close_df : pd.DataFrame
            行索引 = 日期 (datetime), 列 = 股票代码 (str)。
            由 run_all_model.py 从 handler._data 中提取并传入。
        """
        self._close_df = close_df

    def generate_target_weight_position(self, score, current, trade_date):
        """生成目标仓位权重。"""
        returns_df = self._get_returns(trade_date)
        try:
            weights = self._opt_func(score, returns_df, self._top_k,
                                     self._risk_model, self._lookback, self._risk_free_rate)
        except Exception as e:
            warnings.warn(f"优化失败 ({e.__class__.__name__}: {e})，返回空权重")
            weights = {}
        return weights


    def _get_returns(self, trade_date) -> pd.DataFrame | None:
        """从注入的 close_df 计算日收益率，完全不调用 D.features()。"""
        if self._close_df is None:
            raise RuntimeError(
                "未注入价格数据！请在调用 generate_target_weight_position() 之前 "
                "先调用 set_price_data(close_df)。"
            )

        mask = self._close_df.index <= trade_date
        available = self._close_df.loc[mask].tail(self._lookback * 2)
        if len(available) < self._lookback:
            return None
        return available.pct_change().iloc[1:]
