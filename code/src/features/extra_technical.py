"""Alpha158 未覆盖、无法用 Qlib 表达式描述的复杂技术指标。

对外暴露 `compute_extra_technical(close, high, low, volume)` 一个纯函数，
返回 {名称: np.ndarray}。本模块仅包含计算逻辑，I/O 和 DataFrame 整形由调用方
（processor.ExtraTechnicalProcessor）负责。
"""
from __future__ import annotations

import numpy as np


def compute_extra_technical(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> dict[str, np.ndarray]:
    """MACD signal / MACD hist / RSI14 / KDJ(K, D, J) / ATR14 / OBV。

    所有输入都应是 float ndarray，顺序为时间升序。缺失指标填 NaN。
    若环境没有 TA-Lib，会 raise ImportError —— 由调用方决定如何降级。
    """
    import talib

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    _, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    rsi = talib.RSI(close, timeperiod=14)
    k, d = talib.STOCH(
        high, low, close,
        fastk_period=9, slowk_period=3, slowd_period=3,
    )
    j = 3 * k - 2 * d
    atr = talib.ATR(high, low, close, timeperiod=14)
    obv = talib.OBV(close, volume)

    return {
        "MACD_SIGNAL": macd_signal,
        "MACD_HIST": macd_hist,
        "RSI14": rsi,
        "KDJ_K": k,
        "KDJ_D": d,
        "KDJ_J": j,
        "ATR14": atr,
        "OBV": obv,
    }
