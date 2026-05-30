"""训练最终模型并保存检查点。

读取 model/result_model.yaml，执行一次完整训练，将最优模型保存为
model/result_model.pth，同时 dump 一份配置快照 model/config_snapshot.yaml。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    set_seed(42)

    config_path = PROJECT_ROOT / "model" / "result_model.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    cfg = load_config(config_path)

    # 初始化 Qlib
    import qlib
    provider_uri = cfg.get("qlib_init", {}).get(
        "provider_uri", str(PROJECT_ROOT / "temp" / "qlib_data")
    )
    qlib.init(provider_uri=provider_uri, region="cn")

    # 通过配置创建模型和数据集
    from qlib.utils import init_instance_by_config

    model = init_instance_by_config(cfg["task"]["model"])
    dataset = init_instance_by_config(cfg["task"]["dataset"])

    # 训练
    model.fit(dataset)

    # 保存模型权重（尝试提取底层 PyTorch 状态字典）
    model_path = PROJECT_ROOT / "model" / "result_model.pth"
    if hasattr(model, "_net") and model._net is not None:
        torch.save(model._net.state_dict(), model_path)
        print(f"[train] 模型权重已保存至 {model_path}")
    elif hasattr(model, "state_dict"):
        torch.save(model.state_dict(), model_path)
        print(f"[train] 模型权重已保存至 {model_path}")
    else:
        # 回退：保存整个模型对象（可能依赖 Qlib）
        torch.save(model, model_path)
        print(f"[train] 模型对象已保存至 {model_path}")

    # 保存配置快照，以便复现
    snapshot_path = PROJECT_ROOT / "model" / "config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"[train] 配置快照已保存至 {snapshot_path}")


if __name__ == "__main__":
    main()