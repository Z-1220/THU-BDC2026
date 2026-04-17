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
    # 有界差比（∈[-1,1]），避免原 up_ret/(down_ret.abs()+1e-8) 在纯涨/纯跌行情下炸到 1e6+
    df['up_down_ratio_5d'] = (up_ret - down_ret.abs()) / (up_ret + down_ret.abs() + 1e-12)
    df['win_rate_5d'] = (daily_ret > 0).astype(float).rolling(5).mean()

    # --- 风险度量 (2) ---
    # 下行波动率：将正收益置零后计算滚动标准差（向量化，快速）
    neg_ret = daily_ret.clip(upper=0)
    df['downside_vol_10d'] = neg_ret.rolling(10).std()

    # 最大回撤：10日滚动最高价到当前价的回撤（标准短期回撤定义）
    rolling_high_10 = close.rolling(10).max()
    df['max_drawdown_10d'] = (rolling_high_10 - close) / (rolling_high_10 + 1e-12)

    # --- 流动性 (3) ---
    # Amihud 非流动性原始值量纲极小（~1e-12）且极度右偏；停牌/涨跌停日成交额趋零会进一步放大。
    # 先放缩到 O(1) 再用 log1p，保持单调性但压缩尾部，避免 StandardScaler 被异常点主导。
    raw_illiq = (daily_ret.abs() / (amount + 1e-12)).rolling(5).mean()
    df['amihud_illiq_5d'] = np.log1p(raw_illiq * 1e9).clip(-20, 20)
    volume = df['成交量'].astype(float)
    df['volume_price_corr_10d'] = volume.rolling(10).corr(daily_ret.abs())
    turnover = df['换手率'].astype(float) if '换手率' in df.columns else volume
    # 换手率加速度改成对数比率：在长期停牌后复牌的窗口里，原版 ratio 可能炸到 1e10 级。
    turnover_5 = turnover.rolling(5).mean()
    turnover_20 = turnover.rolling(20).mean()
    df['turnover_accel_5d'] = (np.log(turnover_5 + 1e-6) - np.log(turnover_20 + 1e-6)).clip(-5, 5)

    # --- 价格位置 (2) ---
    high_60 = close.rolling(60).max()
    low_60 = close.rolling(60).min()
    df['dist_high_60d'] = (close - high_60) / (high_60 + 1e-12)
    df['dist_low_60d'] = (close - low_60) / (low_60 + 1e-12)

    # --- 波动率结构 (2) ---
    # 波动率机制比同样走对数比率：横盘后突放量时 vol_20 ≈ 0 会把原版 ratio 打爆。
    vol_5 = daily_ret.rolling(5).std()
    vol_20 = daily_ret.rolling(20).std()
    df['vol_regime_ratio'] = (np.log(vol_5 + 1e-6) - np.log(vol_20 + 1e-6)).clip(-5, 5)
    log_hl = np.log(high / (low + 1e-12))
    df['parkinson_vol_10d'] = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(10).mean())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 最后一道安全网：把所有 12 个高级特征 clip 到 [-50, 50]。
    # Alpha 和技术指标大多自然分布在 [-10, 10]，这个带宽足够宽到不压正常信号、又能挡住意外尾部。
    for col in ADVANCED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(-50, 50)

    return df


ADVANCED_COLUMNS = [
    'mom_accel_5_10', 'up_down_ratio_5d', 'win_rate_5d',
    'downside_vol_10d', 'max_drawdown_10d',
    'amihud_illiq_5d', 'volume_price_corr_10d', 'turnover_accel_5d',
    'dist_high_60d', 'dist_low_60d',
    'vol_regime_ratio', 'parkinson_vol_10d',
]
