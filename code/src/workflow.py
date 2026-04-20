"""Qlib workflow：初始化 -> 构建 Dataset -> 训练 -> 评估 -> 回测。

用法：
    uv run code/src/workflow.py --config code/src/config.yaml

严格的时间切分 + 无数据泄露由 Qlib 框架（TSDatasetH + `DataHandlerLP.setup_data`）保证。
"""
from __future__ import annotations

import argparse
import importlib
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP

# 确保自定义 Handler / Processor / Model 所在目录可 import
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_utils import TSDatasetHWithFill  # noqa: E402
from handler import StockDataHandler  # noqa: E402
from model import PointwiseTransformerModel  # noqa: E402


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_handler(cfg: dict[str, Any]) -> DataHandlerLP:
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    return StockDataHandler(
        instruments=data_cfg.get("instruments", "all"),
        start_time=data_cfg["train_start"],
        end_time=data_cfg["test_end"],
        fit_start_time=data_cfg["train_start"],
        fit_end_time=data_cfg["train_end"],
        enable_extra_technical=feat_cfg.get("enable_extra_technical", True),
        enable_advanced=feat_cfg.get("enable_advanced", True),
        enable_cross_sectional=feat_cfg.get("enable_cross_sectional", True),
        sector_map_path=data_cfg.get("sector_map_path"),
    )


def build_dataset(cfg: dict[str, Any], handler: DataHandlerLP) -> TSDatasetH:
    data_cfg = cfg["data"]
    return TSDatasetHWithFill(
        handler=handler,
        segments={
            "train": (data_cfg["train_start"], data_cfg["train_end"]),
            "valid": (data_cfg["valid_start"], data_cfg["valid_end"]),
            "test": (data_cfg["test_start"], data_cfg["test_end"]),
        },
        step_len=cfg["features"]["sequence_length"],
        fillna_type="ffill+bfill",
    )


def build_model(cfg: dict[str, Any]):
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    # 动态加载模型类，兼容 config.yaml 中 class 字段
    model_class = PointwiseTransformerModel
    class_path = mcfg.get("class")
    if class_path:
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except Exception:
            model_class = PointwiseTransformerModel

    # 模型初始化参数（原样传递）
    model_kwargs = {
        "seq_len": cfg["features"]["sequence_length"],
        "d_model": mcfg.get("d_model", 256),
        "nhead": mcfg.get("nhead", 4),
        "num_layers": mcfg.get("num_layers", 3),
        "dim_feedforward": mcfg.get("dim_feedforward", 512),
        "dropout": mcfg.get("dropout", 0.1),
        "batch_size": tcfg.get("batch_size", 256),
        "n_epochs": tcfg.get("num_epochs", 50),
        "lr": float(tcfg.get("learning_rate", 1e-4)),
        "weight_decay": float(tcfg.get("weight_decay", 1e-5)),
        "max_grad_norm": tcfg.get("max_grad_norm", 5.0),
        "enable_grad_clip": tcfg.get("enable_grad_clip", True),
        "early_stop": tcfg.get("early_stop_patience", 10),
        "loss": tcfg.get("loss", "mse"),
        "scheduler": tcfg.get("scheduler", "cosine"),
        "num_workers": tcfg.get("num_workers", 0),
        "seed": tcfg.get("seed", 42),
    }
    
    return model_class(**model_kwargs)


def compute_ic_metrics(pred: pd.Series, label: pd.Series) -> dict[str, float]:
    """按日横截面计算 IC / ICIR / Rank IC / Rank ICIR。"""
    aligned = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return {k: float("nan") for k in ["IC", "ICIR", "Rank IC", "Rank ICIR"]}

    def _daily_ic(group: pd.DataFrame, method: str) -> float:
        if len(group) < 2:
            return np.nan
        return group[["pred", "label"]].corr(method=method).iloc[0, 1]

    daily_ic = aligned.groupby(level="datetime").apply(lambda g: _daily_ic(g, "pearson"))
    daily_ric = aligned.groupby(level="datetime").apply(lambda g: _daily_ic(g, "spearman"))

    ic_mean = daily_ic.mean()
    ric_mean = daily_ric.mean()
    ic_std = daily_ic.std()
    ric_std = daily_ric.std()
    return {
        "IC": float(ic_mean),
        "ICIR": float(ic_mean / ic_std) if ic_std and ic_std > 0 else float("nan"),
        "Rank IC": float(ric_mean),
        "Rank ICIR": float(ric_mean / ric_std) if ric_std and ric_std > 0 else float("nan"),
    }


def simple_top_k_backtest(
    pred: pd.Series,
    label: pd.Series,
    top_k: int,
    rebalance_freq: int,
) -> dict[str, float]:
    """最简 Top-K 选股回测：每 rebalance_freq 个交易日调仓，等权买入前 k 名。"""
    aligned = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return {}

    dates = sorted({d for d, _ in aligned.index})
    rebalance_dates = dates[::rebalance_freq]
    returns = []
    for d in rebalance_dates:
        if d not in aligned.index.get_level_values("datetime"):
            continue
        day = aligned.xs(d, level="datetime").sort_values("pred", ascending=False).head(top_k)
        if day.empty:
            continue
        returns.append(day["label"].mean())

    if not returns:
        return {}
    arr = np.array(returns)
    cum = float(np.prod(1 + arr) - 1)
    return {
        "rebalance_count": len(arr),
        "avg_return": float(arr.mean()),
        "cum_return": cum,
        "win_rate": float((arr > 0).mean()),
    }


def run(config_path: str) -> None:
    cfg = load_config(config_path)

    qi = cfg["qlib_init"]
    provider_uri = qi["provider_uri"]
    region = qi.get("region", "cn")
    qlib.init(
        provider_uri=provider_uri,
        region=REG_CN if region == "cn" else region,
    )

    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    handler = build_handler(cfg)
    dataset = build_dataset(cfg, handler)

    # 持久化 handler + dataset（submit.py 推理时复用）
    with open(out_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    model = build_model(cfg)
    evals_result: dict[str, list[float]] = {}
    model.fit(dataset, evals_result=evals_result, save_path=str(out_dir / "best_model.pth"))

    with open(out_dir / "evals_result.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(evals_result, f)

    # 保存模型封装（含超参），推理时直接 load
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # --- 通过 handler 拉取原始 label（raw，不受 RobustZScoreNorm 等 Processor 影响） ---
    raw_label_all = dataset.handler.fetch(col_set=["label"], data_key=DataHandlerLP.DK_R)
    if isinstance(raw_label_all.columns, pd.MultiIndex):
        label_all = raw_label_all[("label", "LABEL0")]
    else:
        label_all = raw_label_all.iloc[:, 0]

    # --- 评估（valid） ---
    pred = model.predict(dataset, segment="valid")
    label = label_all.reindex(pred.index)
    metrics = compute_ic_metrics(pred, label)
    print("[valid] IC metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")

    # --- 回测（test） ---
    pred_test = model.predict(dataset, segment="test")
    label_test = label_all.reindex(pred_test.index)
    bt = simple_top_k_backtest(
        pred_test,
        label_test,
        top_k=cfg["backtest"]["top_k"],
        rebalance_freq=cfg["backtest"]["rebalance_freq"],
    )
    if bt:
        print("[test] Top-K backtest:")
        for k, v in bt.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # --- 预测分数落盘（供人工 check） ---
    pd.concat(
        [pred_test.rename("score"), label_test.rename("label")], axis=1
    ).to_csv(out_dir / "test_predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/src/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
