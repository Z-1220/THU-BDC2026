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
from qlib.data.dataset.handler import DataHandlerLP
from portfolio import create_optimizer, fetch_daily_returns, to_competition_code

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import TSDatasetHWithFill  # noqa: E402
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

    if not hasattr(dataset.handler, "_infer"):
        dataset.handler.setup_data(init_type=DataHandlerLP.IT_FIT_SEQ)

    # 1. 对最后一个交易日打分
    # ---------------------------------------------------------------
    pred = model.predict(dataset, segment="test")
    if pred.empty:
        pred = model.predict(dataset, segment="valid")
    if pred.empty:
        raise RuntimeError("没有任何可预测的数据。")

    last_date = pred.index.get_level_values("datetime").max()
    day_scores = pred.xs(last_date, level="datetime").sort_values(ascending=False)

    # ---------------------------------------------------------------
    # 2. 组合优化
    # ---------------------------------------------------------------
    optimizer = create_optimizer(cfg)
    data_cfg = cfg["data"]
    daily_returns = fetch_daily_returns(
        start_time=data_cfg["train_start"],
        end_time=data_cfg["oos_end"],
        instruments=data_cfg.get("instruments", "all"),
    )
    weights_dict = optimizer(day_scores, daily_returns)

    # ---------------------------------------------------------------
    # 3. 输出
    # ---------------------------------------------------------------
    result_df = pd.DataFrame({
        "stock_id": [to_competition_code(c) for c in weights_dict.keys()],
        "weight": list(weights_dict.values()),
    })
    result_path = Path(cfg["output"]["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)

    print(f"预测日期: {pd.Timestamp(last_date).date()}")
    print(f"优化器: {cfg.get('portfolio', {}).get('optimizer', 'equal')}")
    print(f"结果已写入: {result_path}")
    print(result_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/src/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
