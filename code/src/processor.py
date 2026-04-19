"""自定义 Qlib Processor：额外技术指标 + 高级个股特征 + 行业跨截面特征。

所有 Processor 都遵守 Qlib 的约定：
- 输入/输出 DataFrame 的 index = (datetime, instrument) MultiIndex
- 输入/输出 DataFrame 的 columns = (fields_group, feature_name) MultiIndex
  其中 feature 组通常为 "feature"，label 组为 "label"

严格避免使用未来信息（所有运算都限制在当前日期及之前）。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from qlib.data.dataset.processor import Processor


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _get_feature_df(df: pd.DataFrame, fields_group: str = "feature") -> pd.DataFrame:
    """提取 feature 组的子 DataFrame（返回单层 column index）。"""
    if isinstance(df.columns, pd.MultiIndex):
        return df[fields_group]
    return df


def _set_feature_df(
    df: pd.DataFrame, feat_df: pd.DataFrame, fields_group: str = "feature"
) -> pd.DataFrame:
    """把 feat_df 的列写回 MultiIndex 的 fields_group 层。"""
    if isinstance(df.columns, pd.MultiIndex):
        for col in feat_df.columns:
            df[(fields_group, col)] = feat_df[col]
    else:
        for col in feat_df.columns:
            df[col] = feat_df[col]
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan))


# ----------------------------------------------------------------------
# 1. 额外技术指标（无法用 Qlib 表达式描述的那些）
# ----------------------------------------------------------------------
class ExtraTechnicalProcessor(Processor):
    """MACD signal / KDJ / ATR / OBV / RSI。

    依赖基础字段：$close / $high / $low / $volume（由 Alpha158 feature config 或
    handler 的额外 expression 暴露为小写列名；此处我们用 Qlib 原始列 "$close" 等）。
    """

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:  # noqa: D401 - Qlib 签名
        return

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from features.extra_technical import compute_extra_technical
        except ImportError:  # pragma: no cover
            # TA-Lib 未安装时降级为跳过
            return df

        feat = _get_feature_df(df, self.fields_group).copy()

        required = {"$close", "$high", "$low", "$volume"}
        if not required.issubset(set(feat.columns)):
            return df

        names = ["MACD_SIGNAL", "MACD_HIST", "RSI14", "KDJ_K", "KDJ_D", "KDJ_J", "ATR14", "OBV"]
        chunks: dict[str, list[pd.Series]] = {n: [] for n in names}

        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            try:
                indicators = compute_extra_technical(
                    close=sub["$close"].to_numpy(),
                    high=sub["$high"].to_numpy(),
                    low=sub["$low"].to_numpy(),
                    volume=sub["$volume"].to_numpy(),
                )
            except ImportError:  # pragma: no cover
                return df
            for name in names:
                chunks[name].append(pd.Series(indicators[name], index=sub.index))

        extra = pd.DataFrame(
            {name: pd.concat(series_list).sort_index() for name, series_list in chunks.items()}
        )
        extra = extra.reindex(feat.index)
        extra.replace([np.inf, -np.inf], np.nan, inplace=True)

        return _set_feature_df(df, extra, self.fields_group)


# ----------------------------------------------------------------------
# 2. 高级个股特征
# ----------------------------------------------------------------------
class AdvancedFeatureProcessor(Processor):
    """高级个股特征：动量质量 / 风险度量 / 流动性 / 价格位置 / 波动率结构。

    这些特征来自原项目 ablation 中被临时关闭、此次恢复并加固过数值稳定性的部分。
    """

    def __init__(self, fields_group: str = "feature") -> None:
        self.fields_group = fields_group

    def fit(self, df: pd.DataFrame = None) -> None:
        return

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = _get_feature_df(df, self.fields_group).copy()

        required = {"$close", "$high", "$low", "$volume", "$amount"}
        if not required.issubset(set(feat.columns)):
            return df

        results = []

        for instrument, sub in feat.groupby(level="instrument", sort=False):
            sub = sub.sort_index()
            close = sub["$close"].astype(float)
            high = sub["$high"].astype(float)
            low = sub["$low"].astype(float)
            volume = sub["$volume"].astype(float)
            amount = sub["$amount"].astype(float)

            ret1 = close.pct_change(1)
            ret5 = close.pct_change(5)
            ret20 = close.pct_change(20)

            # --- 动量质量：收益 / 波动（Sharpe 近似） ---
            vol20 = ret1.rolling(20).std()
            mom_quality_20 = _safe_div(ret20, vol20)

            # --- 风险度量 ---
            downside_vol20 = ret1.clip(upper=0).rolling(20).std()
            max_drawdown_20 = (close / close.rolling(20).max() - 1.0)
            skewness_20 = ret1.rolling(20).skew()
            kurtosis_20 = ret1.rolling(20).kurt()

            # --- 流动性 ---
            # Amihud 非流动性 = |ret1| / amount（金额越大越流动 -> 比值越小）
            amihud_20 = (ret1.abs() / amount.replace(0, np.nan)).rolling(20).mean()
            turnover_ratio_20 = volume.rolling(20).mean() / volume.rolling(60).mean()

            # --- 价格位置 ---
            roll_max_60 = close.rolling(60).max()
            roll_min_60 = close.rolling(60).min()
            price_pos_60 = (close - roll_min_60) / (roll_max_60 - roll_min_60 + 1e-12)
            dist_to_high_60 = close / (roll_max_60 + 1e-12) - 1.0
            dist_to_low_60 = close / (roll_min_60 + 1e-12) - 1.0

            # --- 波动率结构 ---
            vol5 = ret1.rolling(5).std()
            vol60 = ret1.rolling(60).std()
            vol_ratio_5_60 = _safe_div(vol5, vol60)
            hl_range_norm = (high - low) / (close.replace(0, np.nan))
            hl_range_mean_20 = hl_range_norm.rolling(20).mean()

            sub_out = pd.DataFrame(
                {
                    "ADV_MOM_QUALITY_20": mom_quality_20,
                    "ADV_DOWNSIDE_VOL_20": downside_vol20,
                    "ADV_MAX_DD_20": max_drawdown_20,
                    "ADV_SKEW_20": skewness_20,
                    "ADV_KURT_20": kurtosis_20,
                    "ADV_AMIHUD_20": amihud_20,
                    "ADV_TURNOVER_RATIO_20_60": turnover_ratio_20,
                    "ADV_PRICE_POS_60": price_pos_60,
                    "ADV_DIST_TO_HIGH_60": dist_to_high_60,
                    "ADV_DIST_TO_LOW_60": dist_to_low_60,
                    "ADV_VOL_RATIO_5_60": vol_ratio_5_60,
                    "ADV_HL_RANGE_MEAN_20": hl_range_mean_20,
                },
                index=sub.index,
            )
            results.append(sub_out)

        advanced = pd.concat(results).sort_index()
        advanced = advanced.reindex(feat.index)
        advanced.replace([np.inf, -np.inf], np.nan, inplace=True)

        return _set_feature_df(df, advanced, self.fields_group)


# ----------------------------------------------------------------------
# 3. 跨截面特征（需要行业分类数据）
# ----------------------------------------------------------------------
class CrossSectionalProcessor(Processor):
    """跨截面特征：市场级 + 板块级 + 横截面排名/Z-score。

    板块级特征依赖 sector_map（{QlibCode: sector_name}）。若未提供或匹配不到，
    自动降级为全部填 0（保证列数不变）。
    """

    def __init__(
        self,
        fields_group: str = "feature",
        sector_map_path: str | None = None,
    ) -> None:
        self.fields_group = fields_group
        self.sector_map_path = sector_map_path
        self._sector_map: dict[str, str] | None = self._load_sector_map()

    def _load_sector_map(self) -> dict[str, str] | None:
        if not self.sector_map_path:
            return None
        if not os.path.exists(self.sector_map_path):
            return None
        with open(self.sector_map_path, encoding="utf-8") as f:
            return json.load(f)

    def fit(self, df: pd.DataFrame = None) -> None:
        return

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = _get_feature_df(df, self.fields_group).copy()

        # 优先用 handler 中预先算好的 RET1/RET5/RET10 表达式；若不存在，从 $close 重建
        if {"RET1", "RET5", "RET10"}.issubset(set(feat.columns)):
            ret1 = feat["RET1"]
            ret5 = feat["RET5"]
            ret10 = feat["RET10"]
        elif "$close" in feat.columns:
            close = feat["$close"].astype(float)
            ret1 = close.groupby(level="instrument").pct_change(1)
            ret5 = close.groupby(level="instrument").pct_change(5)
            ret10 = close.groupby(level="instrument").pct_change(10)
        else:
            return df

        # ---- 横截面排名 / Z-score（按日期 groupby）----
        cs_rank_ret5 = ret5.groupby(level="datetime").rank(pct=True)
        mean_r5 = ret5.groupby(level="datetime").transform("mean")
        std_r5 = ret5.groupby(level="datetime").transform("std")
        cs_zscore_ret5 = (ret5 - mean_r5) / (std_r5 + 1e-12)

        # ---- 市场级特征 ----
        market_mom_5 = ret5.groupby(level="datetime").transform("mean")
        market_mom_10 = ret10.groupby(level="datetime").transform("mean")
        market_breadth_1 = ret1.groupby(level="datetime").transform(
            lambda x: (x > 0).mean()
        )
        market_dispersion = ret1.groupby(level="datetime").transform("std")

        out = pd.DataFrame(
            {
                "CS_RANK_RET5": cs_rank_ret5,
                "CS_ZSCORE_RET5": cs_zscore_ret5,
                "MARKET_MOM_5": market_mom_5,
                "MARKET_MOM_10": market_mom_10,
                "MARKET_BREADTH_1": market_breadth_1,
                "MARKET_DISPERSION": market_dispersion,
            },
            index=feat.index,
        )

        # ---- 板块级特征（依赖 sector_map）----
        sector_cols = [
            "SECTOR_MOM_5",
            "SECTOR_MOM_10",
            "VS_SECTOR_MOM_5",
            "VS_SECTOR_MOM_10",
            "EXCESS_SECTOR_MOM_5",
            "EXCESS_SECTOR_MOM_10",
            "SECTOR_RANK_5",
            "SECTOR_BREADTH_1",
        ]
        if self._sector_map is not None:
            instruments = feat.index.get_level_values("instrument")
            sector_series = pd.Series(
                [self._sector_map.get(str(i), "未知") for i in instruments],
                index=feat.index,
                name="_sector",
            )

            for suffix, ret in [("5", ret5), ("10", ret10)]:
                group_keys = [feat.index.get_level_values("datetime"), sector_series]
                sector_avg = ret.groupby(group_keys).transform("mean")
                market_avg = ret.groupby(level="datetime").transform("mean")
                out[f"SECTOR_MOM_{suffix}"] = sector_avg
                out[f"VS_SECTOR_MOM_{suffix}"] = ret - sector_avg
                out[f"EXCESS_SECTOR_MOM_{suffix}"] = sector_avg - market_avg

            out["SECTOR_RANK_5"] = out["SECTOR_MOM_5"].groupby(level="datetime").rank(pct=True)
            out["SECTOR_BREADTH_1"] = ret1.groupby(
                [feat.index.get_level_values("datetime"), sector_series]
            ).transform(lambda x: (x > 0).mean())
        else:
            for c in sector_cols:
                out[c] = 0.0

        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return _set_feature_df(df, out, self.fields_group)
