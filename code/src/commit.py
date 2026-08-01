"""比赛推理入口：加载最优模型，生成 result.csv。

完全由 model/result_model.yaml 驱动：
- qlib 初始化
- 模型定义与权重加载
- 数据集构建（测试日期从 train.csv 自动推断）
- 组合优化（等权 / PyPortfolioOpt 等方法，通过 portfolio 字段配置）
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# ----------------------------------------------------------------------
# 项目根目录：容器内为 /app，本地为当前文件所在目录的上两级
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))          # 确保 code 包可以被导入


def set_seed(seed: int = 42) -> None:
    """固定随机种子以保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_test_date(train_csv: Path) -> str:
    """从 train.csv 的最大日期推断测试日。"""
    df = pd.read_csv(train_csv, dtype={"股票代码": str})
    df["日期"] = pd.to_datetime(df["日期"])
    return df["日期"].max().strftime("%Y-%m-%d")


def init_qlib(cfg: dict[str, Any]) -> None:
    """使用配置中的 qlib_init 字段初始化 Qlib。"""
    import qlib
    qlib_init = cfg.get("qlib_init", {})
    if not qlib_init:
        qlib_init = {
            "provider_uri": str(PROJECT_ROOT / "temp" / "qlib_data"),
            "region": "cn",
        }
    qlib.init(**qlib_init)
    print(f"[test] Qlib 初始化完成，数据路径: {qlib_init.get('provider_uri')}")


def build_model(cfg: dict[str, Any]) -> Any:
    """构造模型并优先加载预训练权重。"""
    from qlib.utils import init_instance_by_config

    model = init_instance_by_config(cfg["task"]["model"])

    pth_path = PROJECT_ROOT / "model" / "result_model.pth"
    if pth_path.exists():
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
        # 尝试注入底层 PyTorch 网络
        if hasattr(model, "_net") and model._net is not None:
            model._net.load_state_dict(state_dict)
            print("[test] 已加载预训练模型 (via _net)")
        elif hasattr(model, "load_state_dict"):
            model.load_state_dict(state_dict)
            print("[test] 已加载预训练模型 (via load_state_dict)")
        else:
            warnings.warn("无法注入权重，将通过 fit() 激活模型")
            dataset = init_instance_by_config(cfg["task"]["dataset"])
            model.fit(dataset)
        if hasattr(model, "to"):
            device = model.device if hasattr(model, "device") else torch.device("cpu")
            model.to(device)
    else:
        # 降级方案：从配置新建模型并训练（通常不会在比赛流程中触发）
        warnings.warn("未找到预训练模型，将从头训练，可能超时。请先运行 train.sh")
        dataset = init_instance_by_config(cfg["task"]["dataset"])
        model.fit(dataset)

    model.eval()
    return model


def build_test_dataset(cfg: dict[str, Any], test_date: str) -> Any:
    """构造仅用于推理的测试数据集，时序范围覆盖 test_date。"""
    from qlib.utils import init_instance_by_config

    dataset_cfg = cfg["task"]["dataset"]
    # 复制配置并修改 segments 为单日测试区间
    kwargs = dict(dataset_cfg.get("kwargs", {}))
    kwargs["segments"] = {"test": (test_date, test_date)}
    new_cfg = {**dataset_cfg, "kwargs": kwargs}
    return init_instance_by_config(new_cfg)


def generate_scores(model: Any, dataset: Any) -> pd.Series:
    """调用模型预测，返回以 instrument 为 index 的分数序列。"""
    pred = model.predict(dataset, segment="test")
    if isinstance(pred.index, pd.MultiIndex):
        pred = pred.droplevel("datetime")
    return pred


def optimize_portfolio(scores: pd.Series, config: dict[str, Any]) -> dict[str, float]:
    """根据 YAML 中的 task.strategy / task.evaluation 字段执行组合优化。

    注意：推理阶段无历史价格数据，PyPortfolioOpt 优化器不可用，
    自动降级为分数最高的 top-k 等权。
    """
    eval_cfg = config.get("task", {}).get("evaluation", {})
    port_cfg = config.get("task", {}).get("strategy", {})

    # Check for rank_weighted mode (champion config)
    if eval_cfg.get("exposure_mode") == "rank_weighted":
        rank_weights = eval_cfg.get("rank_weights", [0.333, 0.333, 0.334])
        top = scores.nlargest(len(rank_weights))
        return {c: float(rank_weights[i]) for i, c in enumerate(top.index)}

    top_k = port_cfg.get("top_k", 5)
    optimizer_name = port_cfg.get("optimizer", "equal")

    if optimizer_name in ("mean_variance", "min_variance", "risk_parity"):
        warnings.warn(
            f"推理阶段不支持 '{optimizer_name}' 优化器（缺少历史价格数据），降级为等权"
        )

    top = scores.nlargest(top_k)
    w = 1.0 / len(top)
    return {c: float(w) for c in top.index}


def build_blend_rev05_weights(
    scores: pd.Series, signal_date: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Plan D regime strategy for the 'old-economy rotation' state:
    exclude IT, require close > MA60 (already screened), blend Kronos score
    z-score with 0.5 * (-20d return z-score), take Top-3 equal weight."""
    sector_map: dict[str, str] = {}
    sector_csv = PROJECT_ROOT / "resource" / "行业分类.csv"
    if sector_csv.exists():
        sdf = pd.read_csv(sector_csv, encoding="utf-8-sig", dtype={"证券代码": str})
        sector_map = dict(zip(sdf["证券代码"], sdf["中证一级行业分类简称"]))

    stock_df = pd.read_csv(
        PROJECT_ROOT / "data" / "stock_data.csv",
        encoding="utf-8-sig", parse_dates=["日期"], dtype={"股票代码": str},
    )
    stock_df["code"] = stock_df["股票代码"].str.zfill(6)
    closes = stock_df.pivot_table(index="日期", columns="code", values="收盘", aggfunc="last").sort_index()
    dt = pd.Timestamp(signal_date)
    if dt not in closes.index:
        prior = closes.index[closes.index <= dt]
        if len(prior) == 0:
            raise ValueError(f"signal date {dt.date()} before data start")
        dt = prior[-1]
    pos = closes.index.get_loc(dt)
    if pos < 20:
        raise ValueError(f"insufficient history for 20d return at {dt.date()}")
    ret20 = closes.iloc[pos] / closes.iloc[pos - 20] - 1

    scores_map = {
        c[2:] if c.startswith(("SH", "SZ")) else c: float(v)
        for c, v in scores.items()
    }
    ma60_daily = closes.rolling(60).mean()
    cands = [
        c for c in scores_map
        if sector_map.get(c, "") != "信息技术"
        and c in ret20.index and np.isfinite(ret20[c])
        and c in ma60_daily.columns and np.isfinite(ma60_daily.loc[dt, c])
        and closes.loc[dt, c] > ma60_daily.loc[dt, c]
    ]
    sc = pd.Series({c: scores_map[c] for c in cands})
    rv = pd.Series({c: ret20[c] for c in cands})

    def z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    blend = z(sc) + 0.5 * z(-rv)
    top = blend.sort_values(ascending=False).head(3)
    w = 1.0 / 3
    weights = {c: float(w) for c in top.index}
    diag = {
        "strategy": "blend_rev05",
        "n_candidates": len(cands),
        "picks": list(top.index),
    }
    return weights, diag


def save_result(weights: dict[str, float], output_path: Path) -> None:
    """保存 result.csv（stock_id 为 6 位数字字符串）。"""
    result = []
    for qcode, w in weights.items():
        stock_id = qcode[2:] if qcode.startswith(("SH", "SZ")) else qcode
        result.append({"stock_id": stock_id, "weight": round(w, 6)})
    df = pd.DataFrame(result, columns=["stock_id", "weight"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[test] 结果已保存至 {output_path}")


def main() -> None:
    # 1. 固定随机种子
    set_seed(42)

    # 2. 加载最优配置文件
    config_path = PROJECT_ROOT / "model" / "result_model.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件缺失: {config_path}")
    cfg = load_config(config_path)

    # 3. 初始化 Qlib（从配置读取）
    init_qlib(cfg)

    # 4. 推断测试日期
    train_csv = PROJECT_ROOT / "data" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"训练数据缺失: {train_csv}")
    test_date = infer_test_date(train_csv)
    print(f"[test] 推理日期: {test_date}")

    # 5. 加载模型
    model = build_model(cfg)

    # 6. 构建测试数据集
    dataset = build_test_dataset(cfg, test_date)

    # 7. 生成预测分数
    scores = generate_scores(model, dataset)
    print(f"[test] 预测了 {len(scores)} 只股票")

    # 8. 组合优化（从配置读取）
    eval_cfg = cfg.get("task", {}).get("evaluation", {})
    if eval_cfg.get("exposure_mode") == "cardnn" and hasattr(model, "allocate"):
        weights, diag = model.allocate(scores, test_date)
        print(
            f"[test] CardNN 端到端配权: {len(weights)} 只股票, "
            f"投入比例 {diag.get('invested_frac', 0.0):.3f}, 现金位 {diag.get('cash_positions', 0.0)}"
        )
    elif eval_cfg.get("exposure_mode") == "blend_rev05":
        weights, diag = build_blend_rev05_weights(scores, test_date)
        print(
            f"[test] Plan D blend_rev05 配权: {len(weights)} 只股票, "
            f"候选 {diag.get('n_candidates')}, 标的 {diag.get('picks')}"
        )
    else:
        weights = optimize_portfolio(scores, cfg)
        print(f"[test] 优化后持有 {len(weights)} 只股票，权重: {weights}")

    # 9. 写入 output/result.csv
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_result(weights, output_dir / "result.csv")


if __name__ == "__main__":
    main()