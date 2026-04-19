"""比赛提交脚本：加载训练好的模型 -> 对最后一个交易日打分 -> 输出 Top-5 等权 CSV。

比赛评测脚本（test/score_self.py）要求：
- 列：stock_id（6 位纯数字）, weight
- 最多 5 只股票，权重和 ∈ (0, 1]

旧 predict.py 被本文件替代。
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.data.dataset import TSDatasetH

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import StockDataHandler  # noqa: E402  供 pickle 反序列化


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_competition_code(qlib_code: str) -> str:
    """SH600000 / SZ000001 -> 600000 / 000001"""
    q = str(qlib_code).upper()
    if q.startswith(("SH", "SZ")):
        return q[2:]
    return q


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    qi = cfg["qlib_init"]
    qlib.init(
        provider_uri=qi["provider_uri"],
        region=REG_CN if qi.get("region", "cn") == "cn" else qi.get("region"),
    )

    model_dir = Path(cfg["output"]["model_dir"])
    model_path = model_dir / "model.pkl"
    dataset_path = model_dir / "dataset.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到 dataset: {dataset_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dataset_path, "rb") as f:
        dataset: TSDatasetH = pickle.load(f)

    # 对 test 段打分（config 中 test_start..test_end 应覆盖到最新推理窗口）
    pred = model.predict(dataset, segment="test")
    if pred.empty:
        raise RuntimeError("test 段为空，无法生成提交。请检查 config 中的 test_start/test_end。")

    # 取最后一个交易日横截面的预测
    last_date = pred.index.get_level_values("datetime").max()
    day_scores = pred.xs(last_date, level="datetime").sort_values(ascending=False)

    top_k = int(cfg["backtest"].get("top_k", 5))
    top = day_scores.head(top_k)
    if len(top) < top_k:
        raise RuntimeError(
            f"可预测股票数 {len(top)} 小于 top_k={top_k}，请检查数据覆盖与 sequence_length。"
        )

    weight = 1.0 / top_k
    out_df = pd.DataFrame(
        {
            "stock_id": [to_competition_code(c) for c in top.index],
            "weight": [weight] * len(top),
        }
    )

    result_path = Path(cfg["output"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(result_path, index=False)

    print(f"预测日期: {pd.Timestamp(last_date).date()}")
    print(f"参与排序股票数: {len(day_scores)}")
    print(f"结果已写入: {result_path}")
    print(out_df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/src/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
