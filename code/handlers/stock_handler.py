"""自定义 DataHandler：继承 Qlib Alpha158，追加额外技术指标 + 自定义标签。

约定：
- 特征：Alpha158 内置 158 因子 + 基础字段（CLOSE0 / HIGH0 / LOW0 / VOLUME0 / AMOUNT0）
        + 行业分类特征（sector_l1 ~ sector_l4）
        + 可用 Qlib 表达式描述的额外技术指标（EMA / MACD line / 量价变化等）。
- 复杂指标（MACD signal / KDJ / ATR / OBV / RSI / 高级个股特征 / 横截面特征）
        由 YAML 中注入的 Processor 处理。
- 标签：LABEL0 = (未来第5个交易日开盘 - 未来第1个交易日开盘) / 未来第1个交易日开盘
        对应 Qlib 表达式：(Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)
- Processors：全部通过 YAML 的 infer_processors 字段注入，不在代码中组装。
"""
from __future__ import annotations

from typing import Any

from qlib.contrib.data.handler import Alpha158


# ==========================================================================
# Alpha158 未覆盖、可直接用 Qlib 表达式描述的额外指标
# ==========================================================================
EXTRA_EXPR_FEATURES: list[tuple[str, str]] = [
    # --- 基础字段（供自定义 Processor 使用） ---
    # 注意：HIGH0/LOW0 已在 Alpha158 中定义，此处不重复添加
    ("$close", "CLOSE0"),
    ("$volume", "VOLUME0"),
    ("$amount", "AMOUNT0"),
    # --- 行业分类（已写入 Qlib 二进制数据的静态特征） ---
    ("$sector_l1", "sector_l1"),
    ("$sector_l2", "sector_l2"),
    ("$sector_l3", "sector_l3"),
    ("$sector_l4", "sector_l4"),
    # --- 均线与 MACD 线 ---
    ("EMA($close, 12)", "EMA12"),
    ("EMA($close, 26)", "EMA26"),
    ("EMA($close, 60)", "EMA60"),
    ("EMA($close, 12) - EMA($close, 26)", "MACD_LINE"),
    # --- Bollinger 中轨与标准差 ---
    ("Mean($close, 20)", "BOLL_MID"),
    ("Std($close, 20)", "BOLL_STD"),
    # --- 成交量变化与比率 ---
    ("($volume - Ref($volume, 1)) / (Ref($volume, 1) + 1e-12)", "VOL_CHANGE"),
    ("Mean($volume, 5) / (Mean($volume, 20) + 1e-12)", "VOL_RATIO"),
    # --- K 线价格差 ---
    ("$high - $low", "HL_SPREAD"),
    ("$open - $close", "OC_SPREAD"),
    ("$high - $close", "HC_SPREAD"),
    ("$low - $close", "LC_SPREAD"),
    # --- 基础日频收益率（供横截面 Processor 使用） ---
    ("$close / Ref($close, 1) - 1", "RET1"),
    ("$close / Ref($close, 5) - 1", "RET5"),
    ("$close / Ref($close, 10) - 1", "RET10"),
    ("$close / Ref($close, 20) - 1", "RET20"),
    # --- 短期波动率（日收益率的滚动标准差） ---
    ("Std($close / Ref($close, 1) - 1, 5)", "VOL5"),
    ("Std($close / Ref($close, 1) - 1, 10)", "VOL10"),
    ("Std($close / Ref($close, 1) - 1, 20)", "VOL20"),
]


# ==========================================================================
# 自定义标签：5 日开盘价差收益率
# ==========================================================================
LABEL_EXPR: list[str] = [
    "(Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)"
]
LABEL_NAME: list[str] = ["LABEL0"]


class StockDataHandler(Alpha158):
    """Alpha158 + 基础字段 + 行业分类 + 额外表达式特征 + 自定义标签。

    Processor 列表（如 ExtraTechnicalProcessor / CrossSectionalProcessor 等）
    通过 YAML 的 `infer_processors` 字段注入，不在代码中组装。
    """

    def __init__(
        self,
        instruments: str = "all",
        start_time: str | None = None,
        end_time: str | None = None,
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        infer_processors: list[dict[str, Any]] | None = None,
        learn_processors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # 默认 learn_processors 包含 DropnaLabel
        if learn_processors is None:
            learn_processors = [{"class": "DropnaLabel"}]
        # 如果 YAML 未提供 infer_processors，使用空列表（不执行额外处理）
        if infer_processors is None:
            infer_processors = []

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
        """在 Alpha158 的 158 个因子基础上追加额外表达式特征。"""
        fields, names = super().get_feature_config()
        extra_fields = [e for e, _ in EXTRA_EXPR_FEATURES]
        extra_names = [n for _, n in EXTRA_EXPR_FEATURES]
        return fields + extra_fields, names + extra_names

    def get_label_config(self) -> tuple[list[str], list[str]]:
        """覆盖 Alpha158 默认标签，使用比赛要求的 5 日开盘收益率。"""
        return LABEL_EXPR, LABEL_NAME
