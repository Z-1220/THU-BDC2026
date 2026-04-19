"""自定义 DataHandler：继承 Qlib Alpha158，追加额外技术指标 + 自定义标签。

约定：
- 特征：Alpha158 内置 158 因子  +  Alpha158 未覆盖的、可以用 Qlib 表达式表达的额外指标
        （MACD line / EMA{12,26,60} / 成交量变化 / 价差等）。复杂到无法用表达式表达的
        指标（MACD signal / KDJ / ATR / OBV / RSI 等）放到 `processor.ExtraTechnicalProcessor`
        里用 pandas+TA-Lib 计算。
- 标签：LABEL0 = (未来第5个交易日开盘 - 未来第1个交易日开盘) / 未来第1个交易日开盘
        对应 Qlib 表达式：(Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)
- Processors：infer_processors 追加 ExtraTechnicalProcessor / AdvancedFeatureProcessor /
        CrossSectionalProcessor（按 config.yaml 中的开关决定是否启用），最后再跑
        Alpha158 默认的 RobustZScoreNorm + Fillna。
"""
from __future__ import annotations

from typing import Any

from qlib.contrib.data.handler import Alpha158


# Alpha158 未覆盖、可直接用 Qlib 表达式描述的额外指标
EXTRA_EXPR_FEATURES: list[tuple[str, str]] = [
    # --- MACD 相关 ---
    ("EMA($close, 12)", "EMA12"),
    ("EMA($close, 26)", "EMA26"),
    ("EMA($close, 60)", "EMA60"),
    ("EMA($close, 12) - EMA($close, 26)", "MACD_LINE"),
    # --- Bollinger 中轨/标准差（上下轨在 Processor 里算）---
    ("Mean($close, 20)", "BOLL_MID"),
    ("Std($close, 20)", "BOLL_STD"),
    # --- 量能变化与比率 ---
    ("($volume - Ref($volume, 1)) / (Ref($volume, 1) + 1e-12)", "VOL_CHANGE"),
    ("Mean($volume, 5) / (Mean($volume, 20) + 1e-12)", "VOL_RATIO"),
    # --- K 线各价格差 ---
    ("$high - $low", "HL_SPREAD"),
    ("$open - $close", "OC_SPREAD"),
    ("$high - $close", "HC_SPREAD"),
    ("$low - $close", "LC_SPREAD"),
    # --- 基础日收益，供跨截面特征使用 ---
    ("$close / Ref($close, 1) - 1", "RET1"),
    ("$close / Ref($close, 5) - 1", "RET5"),
    ("$close / Ref($close, 10) - 1", "RET10"),
]


# 自定义 5 日开盘价差标签
LABEL_EXPR = ["(Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)"]
LABEL_NAME = ["LABEL0"]


class StockDataHandler(Alpha158):
    """Alpha158 + 额外 Qlib 表达式特征 + 自定义 5 日开盘收益率标签。

    真正 Qlib 表达式无法实现的指标（MACD signal / KDJ / ATR / OBV / RSI /
    行业跨截面特征 / 高级个股特征）放到 `processor.py` 里的三个 Processor 中。
    """

    def __init__(
        self,
        instruments: str = "all",
        start_time: str | None = None,
        end_time: str | None = None,
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        infer_processors: list[Any] | None = None,
        learn_processors: list[Any] | None = None,
        enable_extra_technical: bool = True,
        enable_advanced: bool = True,
        enable_cross_sectional: bool = True,
        sector_map_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._enable_extra_technical = enable_extra_technical
        self._enable_advanced = enable_advanced
        self._enable_cross_sectional = enable_cross_sectional
        self._sector_map_path = sector_map_path

        if infer_processors is None:
            infer_processors = self._default_infer_processors()
        if learn_processors is None:
            learn_processors = [{"class": "DropnaLabel"}]

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Alpha158 的 feature/label 配置覆盖
    # ------------------------------------------------------------------
    def get_feature_config(self) -> tuple[list[str], list[str]]:
        fields, names = super().get_feature_config()
        extra_fields = [e for e, _ in EXTRA_EXPR_FEATURES]
        extra_names = [n for _, n in EXTRA_EXPR_FEATURES]
        return fields + extra_fields, names + extra_names

    def get_label_config(self) -> tuple[list[str], list[str]]:
        return LABEL_EXPR, LABEL_NAME

    # ------------------------------------------------------------------
    # 默认 infer_processors 组合：自定义 Processor 先跑，Alpha158 标配后跑
    # ------------------------------------------------------------------
    def _default_infer_processors(self) -> list[dict[str, Any]]:
        processors: list[dict[str, Any]] = []

        if self._enable_extra_technical:
            processors.append(
                {
                    "class": "ExtraTechnicalProcessor",
                    "module_path": "processor",
                    "kwargs": {"fields_group": "feature"},
                }
            )
        if self._enable_advanced:
            processors.append(
                {
                    "class": "AdvancedFeatureProcessor",
                    "module_path": "processor",
                    "kwargs": {"fields_group": "feature"},
                }
            )
        if self._enable_cross_sectional:
            processors.append(
                {
                    "class": "CrossSectionalProcessor",
                    "module_path": "processor",
                    "kwargs": {
                        "fields_group": "feature",
                        "sector_map_path": self._sector_map_path,
                    },
                }
            )

        # Alpha158 默认的稳健归一化 + NaN 填充（必须放在最后，已经包含新特征）
        processors.extend(
            [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        )
        return processors
