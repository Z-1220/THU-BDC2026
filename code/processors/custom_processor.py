"""自定义 Qlib Processor：按金融概念解耦的特征工程模块。

每个 Processor 只生成一类特征，通过 YAML 的 infer_processors 列表独立控制。
依赖的基础字段（CLOSE0 / HIGH0 / LOW0 / VOLUME0 / AMOUNT0 及 RET*/sector*）均由
StockDataHandler 在表达式阶段提供，Processor 仅做衍生计算。

实现选择：
- 标准技术指标（RSI、KDJ、MACD signal/hist、ATR、OBV）使用 TA‑Lib 计算，
  因其难以用 Qlib 表达式简洁表达。
- 其他复合因子（动量质量、风险、流动性等）基于 pandas 滚动窗口实现，
  确保与 TA‑Lib 结果的一致性，同时避免额外依赖。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qlib.data.dataset.processor import Processor


def get_group_columns(df: pd.DataFrame, group: str) -> pd.Index:
    """选择 qlib DataFrame 中第一级 MultiIndex 匹配 group 的列。
    
    qlib 数据格式的列通常具有两级 MultiIndex（fields_group, field_name），
    例如 ('feature', 'CLOSE0')。
    """
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        # 处理 NaN 值：fillna 后再比较，避免 NaN 导致布尔掩码错误
        mask = level0.fillna("") == group if hasattr(level0, 'fillna') else level0 == group
        return df.columns[mask]
    # 对于单级列索引，尝试匹配以 fields_group 同名的前缀
    str_cols = df.columns.astype(str)
    return df.columns[str_cols.str.startswith(f"{group}_")]


def _get_feat_col(df: pd.DataFrame, col_name: str, fields_group: str = "feature") -> pd.Series:
    """从 qlib MultiIndex DataFrame 中安全获取指定列名的 Series。
    
    支持两种格式：
    1. MultiIndex: ('feature', 'CLOSE0') -> 使用 xs
    2. 单级索引: 'CLOSE0' -> 直接访问
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 使用 get_loc 在第二级查找，返回精确的列位置
        level1 = df.columns.get_level_values(1)
        mask = level1 == col_name
        if mask.any():
            result = df.iloc[:, mask]
            if result.shape[1] == 1:
                return result.iloc[:, 0]
            # 如果有多个匹配（不同 group），取指定 group 的
            level0 = df.columns.get_level_values(0)
            group_mask = mask & (level0 == fields_group)
            if group_mask.any():
                return df.iloc[:, group_mask].iloc[:, 0]
            return result.iloc[:, 0]
        raise KeyError(f"Column '{col_name}' not found in DataFrame columns: {df.columns.tolist()}")
    return df[col_name]


def _get_feat_col_single(df: pd.DataFrame, col_name: str) -> pd.Series:
    """从单 instrument 子 DataFrame 中获取列值，返回 1D Series。
    
    处理 MultiIndex 列的情况：当 groupby 后，子 DataFrame 仍保留 MultiIndex 列。
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 尝试在第二级查找
        level1_vals = df.columns.get_level_values(1)
        mask = level1_vals == col_name
        if mask.any():
            return df.iloc[:, mask].iloc[:, 0]
        # 尝试在第一级查找
        level0_vals = df.columns.get_level_values(0)
        mask0 = level0_vals == col_name
        if mask0.any():
            return df.iloc[:, mask0].iloc[:, 0]
        raise KeyError(f"Column '{col_name}' not found in MultiIndex columns: {list(df.columns)}")
    return df[col_name]


def _to_multiindex(df: pd.DataFrame, fields_group: str) -> pd.DataFrame:
    """将单级列 DataFrame 转换为 MultiIndex 列结构。"""
    df = df.copy()
    df.columns = pd.MultiIndex.from_product([[fields_group], df.columns])
    return df


# ---------------------------------------------------------------------------
# 纯 TA‑Lib 指标
# ---------------------------------------------------------------------------
class MomentumProcessor(Processor):
    """MACD signal/hist、RSI14、KDJ(K,D,J) — 动量与趋势类。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import talib
        except ImportError:
            return df

        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        required = {"CLOSE0", "HIGH0", "LOW0"}
        if not required.issubset(cols.get_level_values(-1)):
            return df

        names = ["MACD_SIGNAL", "MACD_HIST", "RSI14", "KDJ_K", "KDJ_D", "KDJ_J"]
        all_series = {n: [] for n in names}

        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").values.astype(float)
            high = _get_feat_col_single(sub, "HIGH0").values.astype(float)
            low = _get_feat_col_single(sub, "LOW0").values.astype(float)

            _, macd_signal, macd_hist = talib.MACD(close)
            rsi = talib.RSI(close, timeperiod=14)
            k, d = talib.STOCH(high, low, close)
            j = 3 * k - 2 * d

            all_series["MACD_SIGNAL"].append(pd.Series(macd_signal, index=sub.index))
            all_series["MACD_HIST"].append(pd.Series(macd_hist, index=sub.index))
            all_series["RSI14"].append(pd.Series(rsi, index=sub.index))
            all_series["KDJ_K"].append(pd.Series(k, index=sub.index))
            all_series["KDJ_D"].append(pd.Series(d, index=sub.index))
            all_series["KDJ_J"].append(pd.Series(j, index=sub.index))

        extra = pd.DataFrame({n: pd.concat(all_series[n]).sort_index() for n in names})
        extra = extra.reindex(feat.index).replace([np.inf, -np.inf], np.nan)
        extra = _to_multiindex(extra, self.fields_group)
        return pd.concat([df, extra], axis=1)

    def readonly(self) -> bool:
        return False


class VolatilityProcessor(Processor):
    """ATR14 — 波动性。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import talib
        except ImportError:
            return df

        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if not {"HIGH0", "LOW0", "CLOSE0"}.issubset(cols.get_level_values(-1)):
            return df

        series_list = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            atr = talib.ATR(
                _get_feat_col_single(sub, "HIGH0").values.astype(float),
                _get_feat_col_single(sub, "LOW0").values.astype(float),
                _get_feat_col_single(sub, "CLOSE0").values.astype(float),
                timeperiod=14,
            )
            series_list.append(pd.Series(atr, index=sub.index, name="ATR14"))

        atr_series = pd.concat(series_list).sort_index().reindex(feat.index)
        atr_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        extra = _to_multiindex(atr_series.to_frame(), self.fields_group)
        return pd.concat([df, extra], axis=1)

    def readonly(self) -> bool:
        return False


class VolumeProcessor(Processor):
    """OBV — 成交量能量。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import talib
        except ImportError:
            return df

        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if not {"CLOSE0", "VOLUME0"}.issubset(cols.get_level_values(-1)):
            return df

        obv_list = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            obv = talib.OBV(
                _get_feat_col_single(sub, "CLOSE0").values.astype(float),
                _get_feat_col_single(sub, "VOLUME0").values.astype(float),
            )
            obv_list.append(pd.Series(obv, index=sub.index, name="OBV"))

        obv_all = pd.concat(obv_list).sort_index().reindex(feat.index)
        obv_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        extra = _to_multiindex(obv_all.to_frame(), self.fields_group)
        return pd.concat([df, extra], axis=1)

    def readonly(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 复合因子 (pandas 实现)
# ---------------------------------------------------------------------------
def _replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


class MomentumQualityProcessor(Processor):
    """动量质量：收益/波动 (ADV_MOM_QUALITY_20)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if "CLOSE0" not in cols.get_level_values(-1):
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            ret1 = close.pct_change(1)
            ret20 = close.pct_change(20)
            vol20 = ret1.rolling(20).std()
            quality = (ret20 / vol20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            results.append(pd.DataFrame({"ADV_MOM_QUALITY_20": quality}, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class TailRiskProcessor(Processor):
    """下行波动率、最大回撤 (ADV_DOWNSIDE_VOL_20, ADV_MAX_DD_20)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if "CLOSE0" not in cols.get_level_values(-1):
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            ret1 = close.pct_change(1)
            downside = ret1.clip(upper=0).rolling(20).std()
            maxdd = (close / close.rolling(20).max() - 1.0)
            results.append(pd.DataFrame({
                "ADV_DOWNSIDE_VOL_20": downside,
                "ADV_MAX_DD_20": maxdd,
            }, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class DistributionProcessor(Processor):
    """收益偏度、峰度 (ADV_SKEW_20, ADV_KURT_20)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if "CLOSE0" not in cols.get_level_values(-1):
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            ret1 = close.pct_change(1)
            skew = ret1.rolling(20).skew()
            kurt = ret1.rolling(20).kurt()
            results.append(pd.DataFrame({
                "ADV_SKEW_20": skew,
                "ADV_KURT_20": kurt,
            }, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class LiquidityProcessor(Processor):
    """流动性 (ADV_AMIHUD_20, ADV_TURNOVER_RATIO_20_60)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        col_set = set(cols.get_level_values(-1))
        if "CLOSE0" not in col_set or "VOLUME0" not in col_set:
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            volume = _get_feat_col_single(sub, "VOLUME0").astype(float)
            try:
                amount = _get_feat_col_single(sub, "AMOUNT0").astype(float)
            except KeyError:
                amount = pd.Series(np.nan, index=sub.index)

            ret1 = close.pct_change(1)
            amihud = (ret1.abs() / amount.replace(0, np.nan)).rolling(20).mean()
            turnover = volume.rolling(20).mean() / volume.rolling(60).mean()

            results.append(pd.DataFrame({
                "ADV_AMIHUD_20": amihud,
                "ADV_TURNOVER_RATIO_20_60": turnover,
            }, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class PricePositionProcessor(Processor):
    """价格位置 (ADV_PRICE_POS_60, ADV_DIST_TO_HIGH_60, ADV_DIST_TO_LOW_60)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if "CLOSE0" not in cols.get_level_values(-1):
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            roll_max = close.rolling(60).max()
            roll_min = close.rolling(60).min()
            pos = (close - roll_min) / (roll_max - roll_min + 1e-12)
            dist_high = close / (roll_max + 1e-12) - 1.0
            dist_low = close / (roll_min + 1e-12) - 1.0

            results.append(pd.DataFrame({
                "ADV_PRICE_POS_60": pos,
                "ADV_DIST_TO_HIGH_60": dist_high,
                "ADV_DIST_TO_LOW_60": dist_low,
            }, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class VolatilityStructureProcessor(Processor):
    """波动率结构 (ADV_VOL_RATIO_5_60, ADV_HL_RANGE_MEAN_20)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        col_set = set(cols.get_level_values(-1))
        if not {"CLOSE0", "HIGH0", "LOW0"}.issubset(col_set):
            return df

        results = []
        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = _get_feat_col_single(sub, "CLOSE0").astype(float)
            high = _get_feat_col_single(sub, "HIGH0").astype(float)
            low = _get_feat_col_single(sub, "LOW0").astype(float)

            ret1 = close.pct_change(1)
            vol5 = ret1.rolling(5).std()
            vol60 = ret1.rolling(60).std()
            vol_ratio = (vol5 / vol60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

            hl_range = (high - low) / close.replace(0, np.nan)
            hl_mean = hl_range.rolling(20).mean()

            results.append(pd.DataFrame({
                "ADV_VOL_RATIO_5_60": vol_ratio,
                "ADV_HL_RANGE_MEAN_20": hl_mean,
            }, index=sub.index))
        out = pd.concat(results).sort_index().reindex(feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 横截面特征
# ---------------------------------------------------------------------------
class CrossSectionalRankProcessor(Processor):
    """横截面排名、Z‑score (CS_RANK_RET5, CS_ZSCORE_RET5)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        if "RET5" not in cols.get_level_values(-1):
            return df

        ret5 = _get_feat_col(feat, "RET5", self.fields_group)
        rank = ret5.groupby(level="datetime").rank(pct=True)
        mean = ret5.groupby(level="datetime").transform("mean")
        std = ret5.groupby(level="datetime").transform("std")
        zscore = (ret5 - mean) / (std + 1e-12)

        out = pd.DataFrame({
            "CS_RANK_RET5": rank,
            "CS_ZSCORE_RET5": zscore,
        }, index=feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class MarketLevelProcessor(Processor):
    """市场级特征 (MARKET_MOM_5, MARKET_MOM_10, MARKET_BREADTH_1, MARKET_DISPERSION)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        col_set = set(cols.get_level_values(-1))
        if not {"RET1", "RET5", "RET10"}.issubset(col_set):
            return df

        ret1 = _get_feat_col(feat, "RET1", self.fields_group)
        ret5 = _get_feat_col(feat, "RET5", self.fields_group)
        ret10 = _get_feat_col(feat, "RET10", self.fields_group)

        out = pd.DataFrame({
            "MARKET_MOM_5": ret5.groupby(level="datetime").transform("mean"),
            "MARKET_MOM_10": ret10.groupby(level="datetime").transform("mean"),
            "MARKET_BREADTH_1": ret1.groupby(level="datetime").transform(
                lambda x: (x > 0).mean()
            ),
            "MARKET_DISPERSION": ret1.groupby(level="datetime").transform("std"),
        }, index=feat.index)
        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class SectorLevelProcessor(Processor):
    """行业级特征 (基于 sector_l1)。"""

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)
        feat = df[cols].copy()
        col_set = set(cols.get_level_values(-1))
        if not {"RET1", "RET5", "RET10", "sector_l1"}.issubset(col_set):
            # 缺少行业列，返回全零特征
            for c in [
                "SECTOR_MOM_5", "SECTOR_MOM_10",
                "VS_SECTOR_MOM_5", "VS_SECTOR_MOM_10",
                "EXCESS_SECTOR_MOM_5", "EXCESS_SECTOR_MOM_10",
                "SECTOR_RANK_5", "SECTOR_BREADTH_1",
            ]:
                df[c] = 0.0
            return df

        ret1 = _get_feat_col(feat, "RET1", self.fields_group)
        ret5 = _get_feat_col(feat, "RET5", self.fields_group)
        ret10 = _get_feat_col(feat, "RET10", self.fields_group)
        sector = _get_feat_col(feat, "sector_l1", self.fields_group).astype(int).astype(str)

        out = pd.DataFrame(index=feat.index)
        for suffix, ret in [("5", ret5), ("10", ret10)]:
            group_keys = [feat.index.get_level_values("datetime"), sector]
            sector_avg = ret.groupby(group_keys).transform("mean")
            market_avg = ret.groupby(level="datetime").transform("mean")

            out[f"SECTOR_MOM_{suffix}"] = sector_avg
            out[f"VS_SECTOR_MOM_{suffix}"] = ret - sector_avg
            out[f"EXCESS_SECTOR_MOM_{suffix}"] = sector_avg - market_avg

        out["SECTOR_RANK_5"] = out["SECTOR_MOM_5"].groupby(level="datetime").rank(pct=True)
        out["SECTOR_BREADTH_1"] = ret1.groupby(
            [feat.index.get_level_values("datetime"), sector]
        ).transform(lambda x: (x > 0).mean())

        out = _to_multiindex(_replace_inf(out), self.fields_group)
        return pd.concat([df, out], axis=1)

    def readonly(self) -> bool:
        return False


class FridayFilterProcessor(Processor):
    """仅保留每周五的样本，确保周频标签对齐。"""
    def fit(self, df: pd.DataFrame = None) -> "FridayFilterProcessor":
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # df 的 index 是 (datetime, instrument) 的 MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values("datetime")
        else:
            dates = df.index
        # 只保留星期五 (weekday == 4)
        mask = dates.dayofweek == 4
        return df[mask]


class ScreenProcessor(Processor):
    """金融筛选：剔除流动性枯竭、长期下行、连续暴跌的股票。

    YAML 示例:
      - class: ScreenProcessor
        module_path: "code.processors.custom_processor"
        kwargs:
          min_amount_rank: 0.3    # 保留成交额前 70% 的股票
          trend_ma: 60             # 价格需 > MA60
          max_drawdown: 0.15       # 20日最大回撤 < 15%
    """

    def __init__(self, min_amount_rank: float = 0.3, trend_ma: int = 60,
                 max_drawdown: float = 0.15):
        self.min_amount_rank = min_amount_rank
        self.trend_ma = trend_ma
        self.max_drawdown = max_drawdown

    def fit(self, df: pd.DataFrame = None) -> "ScreenProcessor":
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            return df

        mask = pd.Series(True, index=df.index)

        # Access base OHLCV fields via the "feature" fields_group
        close = df.xs("CLOSE0", axis=1, level=1).squeeze()
        amount = df.xs("AMOUNT0", axis=1, level=1).squeeze()

        # 1) 趋势过滤: close > MA(trend_ma)
        if self.trend_ma and self.trend_ma > 0 and close is not None:
            ma = close.groupby("instrument").transform(
                lambda x: x.shift(1).rolling(self.trend_ma, min_periods=self.trend_ma).mean()
            )
            mask &= close > ma

        # 2) 回撤过滤: 20日最大回撤 < max_drawdown
        if self.max_drawdown and self.max_drawdown < 1.0 and close is not None:
            cummax = close.groupby("instrument").transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).max()
            )
            dd = (close / cummax - 1).abs()
            mask &= dd < self.max_drawdown

        # 3) 成交额过滤: 日均成交额排名 > min_amount_rank
        if self.min_amount_rank and self.min_amount_rank > 0 and amount is not None:
            avg_amount = amount.groupby("instrument").transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).mean()
            )
            rank = avg_amount.groupby("datetime").rank(pct=True)
            mask &= rank >= self.min_amount_rank

        # 降级保护: 至少保留 50 只
        if mask.sum() < 50:
            return df

        return df[mask]