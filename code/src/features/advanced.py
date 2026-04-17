"""高级个股特征：动量质量、风险度量、流动性、价格位置、波动率结构。"""
import numpy as np
import pandas as pd


def engineer_advanced_features(df):
    """
    在已有基础特征的 DataFrame 上追加高级个股特征。
    要求 df 已包含：return_1, return_5, return_10, 收盘, 最高, 最低, 成交量, 成交额, 换手率
    返回追加了新列的 df（不修改原有列）。
    """
    df = df.copy()
    daily_ret = df['return_1'] if 'return_1' in df.columns else df['收盘'].pct_change(1)
    close = df['收盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    amount = df['成交额'].astype(float)

    # --- 动量质量 (3) ---
    ret5 = df['return_5'] if 'return_5' in df.columns else close.pct_change(5)
    ret10 = df['return_10'] if 'return_10' in df.columns else close.pct_change(10)
    df['mom_accel_5_10'] = ret5 - ret10
    up_ret = daily_ret.clip(lower=0).rolling(5).sum()
    down_ret = daily_ret.clip(upper=0).rolling(5).sum()
    df['up_down_ratio_5d'] = up_ret / (down_ret.abs() + 1e-8)
    df['win_rate_5d'] = (daily_ret > 0).astype(float).rolling(5).mean()

    # --- 风险度量 (2) ---
    # 下行波动率：将正收益置零后计算滚动标准差（向量化，快速）
    neg_ret = daily_ret.clip(upper=0)
    df['downside_vol_10d'] = neg_ret.rolling(10).std()

    # 最大回撤：10日滚动最高价到当前价的回撤（标准短期回撤定义）
    rolling_high_10 = close.rolling(10).max()
    df['max_drawdown_10d'] = (rolling_high_10 - close) / (rolling_high_10 + 1e-12)

    # --- 流动性 (3) ---
    df['amihud_illiq_5d'] = (daily_ret.abs() / (amount + 1e-12)).rolling(5).mean()
    volume = df['成交量'].astype(float)
    df['volume_price_corr_10d'] = volume.rolling(10).corr(daily_ret.abs())
    turnover = df['换手率'].astype(float) if '换手率' in df.columns else volume
    df['turnover_accel_5d'] = turnover.rolling(5).mean() / (turnover.rolling(20).mean() + 1e-12)

    # --- 价格位置 (2) ---
    high_60 = close.rolling(60).max()
    low_60 = close.rolling(60).min()
    df['dist_high_60d'] = (close - high_60) / (high_60 + 1e-12)
    df['dist_low_60d'] = (close - low_60) / (low_60 + 1e-12)

    # --- 波动率结构 (2) ---
    vol_5 = daily_ret.rolling(5).std()
    vol_20 = daily_ret.rolling(20).std()
    df['vol_regime_ratio'] = vol_5 / (vol_20 + 1e-12)
    log_hl = np.log(high / (low + 1e-12))
    df['parkinson_vol_10d'] = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(10).mean())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


ADVANCED_COLUMNS = [
    'mom_accel_5_10', 'up_down_ratio_5d', 'win_rate_5d',
    'downside_vol_10d', 'max_drawdown_10d',
    'amihud_illiq_5d', 'volume_price_corr_10d', 'turnover_accel_5d',
    'dist_high_60d', 'dist_low_60d',
    'vol_regime_ratio', 'parkinson_vol_10d',
]
