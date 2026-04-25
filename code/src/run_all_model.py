from __future__ import annotations

import functools
import inspect
import json
import statistics
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import pandas as pd
import qlib
import torch
import yaml
from qlib.config import REG_CN
from qlib.data import D
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from qlib.workflow.task.gen import RollingGen, task_generator

import sys
import random
import numpy as np
import os

def set_seed(seed=42):
    """设置全局随机种子以确保结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



"""运行多个 Qlib 模型配置并进行滚动评估。

本模块提供命令行入口，用于执行一个或多个 YAML 模型配置，生成滚动任务，
在每个任务中运行独立的 Qlib 实验，并将汇总评价指标写入 markdown、CSV 和 JSON 清单。
"""

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROLLING_TRAIN_START = "2024-01-01"
ROLLING_TRAIN_END = "2026-02-10"
ROLLING_TEST_START = "2026-02-11"
ROLLING_TEST_END = "2026-04-17"
ROLLING_STEP = 5

SUMMARY_METRICS = {
    "IC": ["IC"],
    "ICIR": ["ICIR"],
    "Rank IC": ["Rank IC"],
    "Rank ICIR": ["Rank ICIR"],
    "Annualized Return": [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_without_cost.annualized_return",
    ],
    "Max Drawdown": [
        "1day.excess_return_with_cost.max_drawdown",
        "1day.excess_return_without_cost.max_drawdown",
    ],
}


def only_allow_defined_args(function_to_decorate):
    """装饰函数以拒绝未定义的关键字参数。

    这对于 Fire 命令非常有用，可以避免用户传入未知的命名参数。
    """
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError(
                    "Unknown argument seen '%s', expected: [%s]"
                    % (arg_name, ", ".join(sorted(valid_names)))
                )
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


def parse_yaml_paths(yaml_paths: str | list[str] | None) -> list[Path]:
    """规范化 YAML 路径输入并校验文件是否存在。

    支持逗号分隔字符串或路径列表，若为相对路径则相对于仓库根目录解析。
    """

    if yaml_paths is None:
        raise ValueError("Please provide YAML config paths via `yaml_paths`.")
    if isinstance(yaml_paths, str):
        text = yaml_paths.strip()
        if text.startswith("[") and text.endswith("]"):
            items = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        else:
            items = [item.strip() for item in text.split(",") if item.strip()]
    elif isinstance(yaml_paths, list):
        items = [str(item).strip() for item in yaml_paths if str(item).strip()]
    else:
        raise ValueError("`yaml_paths` must be a list or a comma-separated string.")

    paths = [(Path(item) if Path(item).is_absolute() else (REPO_ROOT / item)).resolve() for item in items]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"YAML config files not found: {missing}")
    return paths


def load_yaml(path: Path) -> dict[str, Any]:
    """从磁盘加载 YAML 配置。

    返回解析后的 Python 字典。
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_local_path(path_str: str, base_dir: Path) -> str:
    """将本地路径字符串相对于基目录解析为绝对路径。

    若已为绝对路径则直接返回。
    """

    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def normalize_qlib_init(raw_cfg: dict[str, Any], config_path: Path, mlruns_root: Path, experiment_name: str) -> dict[str, Any]:
    """规范化 Qlib 初始化配置以便本地执行。

    确保 provider_uri 和 MLflow 路径解析为绝对本地路径，缺失时补全默认实验管理器配置。
    """
    cfg = deepcopy(raw_cfg)
    if "provider_uri" in cfg:
        cfg["provider_uri"] = resolve_local_path(str(cfg["provider_uri"]), REPO_ROOT)

    region = cfg.get("region", "cn")
    cfg["region"] = REG_CN if region == "cn" else region

    if "exp_manager" not in cfg:
        cfg["exp_manager"] = {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{mlruns_root.resolve().as_posix()}",
                "default_exp_name": experiment_name,
            },
        }
        return cfg

    exp_manager = cfg["exp_manager"]
    kwargs = exp_manager.setdefault("kwargs", {})
    uri = kwargs.get("uri")
    if uri is None:
        kwargs["uri"] = f"file:{mlruns_root.resolve().as_posix()}"
    elif isinstance(uri, str) and uri.startswith("file:"):
        raw_uri_path = uri[5:]
        # Handle both "file:path" and "file://path" URI formats
        if raw_uri_path.startswith("//"):
            raw_uri_path = raw_uri_path[2:]
        if raw_uri_path and not Path(raw_uri_path).is_absolute():
            kwargs["uri"] = f"file:{(REPO_ROOT / raw_uri_path).resolve().as_posix()}"
    return cfg


def get_task_config(full_cfg: dict[str, Any]) -> dict[str, Any]:
    """从完整 YAML 配置中提取并复制 task 定义。"""
    if "task" not in full_cfg:
        raise KeyError("Each YAML must contain a top-level `task` config.")
    return deepcopy(full_cfg["task"])


def infer_experiment_name(full_cfg: dict[str, Any], config_path: Path) -> str:
    """从配置元数据中推断实验名称，若不存在则使用文件名。"""
    for key in ("experiment_name", "name"):
        if key in full_cfg and full_cfg[key]:
            return str(full_cfg[key])
    task = full_cfg.get("task", {})
    for key in ("name", "experiment_name"):
        if key in task and task[key]:
            return str(task[key])
    return config_path.stem


def infer_model_name(full_cfg: dict[str, Any], config_path: Path) -> str:
    """推断可读的模型名称用于汇报。"""
    for key in ("model_name", "name"):
        if key in full_cfg and full_cfg[key]:
            return str(full_cfg[key])
    model_cfg = full_cfg.get("task", {}).get("model", {})
    class_name = model_cfg.get("class")
    if class_name:
        return str(class_name).split(".")[-1]
    return config_path.stem


def get_calendar(start_time: str, end_time: str) -> pd.DatetimeIndex:
    """从 Qlib 获取指定日期范围内的交易日历并排序。"""
    return pd.DatetimeIndex(pd.to_datetime(D.calendar(start_time=start_time, end_time=end_time))).sort_values()

def get_weekly_periods(
    calendar: pd.DatetimeIndex,
    start: str,
    end: str,
) -> list[tuple[str, str]]:
    """从日历中提取所有连续5个交易日的完整周（跳过不足5天的残周）。"""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (calendar >= start_ts) & (calendar <= end_ts)
    cal_sub = calendar[mask]

    weeks: list[tuple[str, str]] = []
    i = 0
    while i + 4 < len(cal_sub):
        # 日期差为4天，且中间不存在非交易日（已经过日历过滤）
        if (cal_sub[i + 4] - cal_sub[i]).days == 4:
            weeks.append((
                cal_sub[i].strftime("%Y-%m-%d"),
                cal_sub[i + 4].strftime("%Y-%m-%d"),
            ))
            i += 5
        else:
            i += 1
    return weeks


def build_weekly_rolling_tasks(
    base_task: dict[str, Any],
    calendar: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """根据全局常量生成一组周度滚动任务.
    
    每个任务：
      - trainset: 从 ROLLING_TRAIN_START 到该周起始日的前一交易日
      - validset: 该周的5个交易日（无 test）
    """
    weeks = get_weekly_periods(calendar, ROLLING_TEST_START, ROLLING_TEST_END)
    tasks: list[dict[str, Any]] = []

    for w_start, w_end in weeks:
        task = deepcopy(base_task)

        # 训练截止日 = 本周开始之前最后一个交易日
        previous_dates = calendar[calendar < pd.Timestamp(w_start)]
        if len(previous_dates) == 0:
            continue  # 没有历史数据，不构成训练集
        train_end = previous_dates[-1].strftime("%Y-%m-%d")

        segs = task["dataset"]["kwargs"]["segments"]
        segs["train"] = (ROLLING_TRAIN_START, train_end)
        segs["valid"] = (w_start, w_end)
        segs.pop("test", None)  # 移除 test（如果存在）

        # 同步 handler 时间
        handler_cfg = task["dataset"]["kwargs"]["handler"]
        if isinstance(handler_cfg, dict):
            handler_kwargs = handler_cfg.setdefault("kwargs", {})
            handler_kwargs["start_time"] = ROLLING_TRAIN_START
            handler_kwargs["fit_start_time"] = ROLLING_TRAIN_START
            handler_kwargs["fit_end_time"] = train_end
            handler_kwargs["end_time"] = w_end

        tasks.append(task)

    return tasks

def record_class_name(record_cfg: dict[str, Any]) -> str:
    """返回记录配置中的短类名。"""
    return str(record_cfg.get("class", "")).split(".")[-1]


def get_record_config(task: dict[str, Any], class_name: str) -> dict[str, Any] | None:
    """从 task 中查找并返回匹配的记录配置。"""
    for record_cfg in task.get("record", []) or []:
        if record_class_name(record_cfg) == class_name:
            return deepcopy(record_cfg)
    return None


def get_port_analysis_config(task: dict[str, Any], full_cfg: dict[str, Any], port_record_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """解析 portfolio 分析配置，优先级为 record kwargs > task > 全局配置。"""
    if port_record_cfg:
        kwargs = deepcopy(port_record_cfg.get("kwargs", {}))
        if "config" in kwargs:
            return kwargs.pop("config")
    for key in ("port_analysis_config", "port_analysis"):
        if key in task:
            return deepcopy(task[key])
        if key in full_cfg:
            return deepcopy(full_cfg[key])
    return None


def generate_records(
    task: dict[str, Any],
    full_cfg: dict[str, Any],
    model: Any,
    dataset: Any,
    recorder: Any,
) -> None:
    """模型训练后生成标准 Qlib 记录。"""
    signal_record = SignalRecord(model, dataset, recorder)
    signal_record.generate()

    sig_record_cfg = get_record_config(task, "SigAnaRecord")
    sig_kwargs = deepcopy(sig_record_cfg.get("kwargs", {})) if sig_record_cfg else {}
    SigAnaRecord(recorder=recorder, **sig_kwargs).generate()

    port_record_cfg = get_record_config(task, "PortAnaRecord")
    port_kwargs = deepcopy(port_record_cfg.get("kwargs", {})) if port_record_cfg else {}
    # PortAnaRecord 需要从 record kwargs 或 task/full 配置中获取组合分析配置，
    # 这里统一解析该配置。
    port_config = port_kwargs.pop("config", None) or get_port_analysis_config(task, full_cfg, port_record_cfg)
    if port_config is None:
        raise KeyError(
            "PortAnaRecord requires a `config`. Put it in `task.port_analysis_config`, "
            "`port_analysis_config`, or `task.record[].kwargs.config`."
        )
    PortAnaRecord(recorder=recorder, config=port_config, **port_kwargs).generate()

    for record_cfg in task.get("record", []) or []:
        class_name = record_class_name(record_cfg)
        if class_name in {"SignalRecord", "SigAnaRecord", "PortAnaRecord"}:
            continue
        record = init_instance_by_config(
            record_cfg,
            try_kwargs={"recorder": recorder},
        )
        if hasattr(record, "generate"):
            record.generate()


def run_single_task(
    task: dict[str, Any],
    full_cfg: dict[str, Any],
    experiment_name: str,
    recorder_name: str,
    source_config_path: Path,
) -> str:
    """执行单个 Qlib 任务并将训练产物写入 recorder。"""
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        recorder = R.get_recorder()
        R.log_params(**flatten_dict(task))
        R.save_objects(**{"task_config": task})
        R.save_objects(**{"source_config_text": source_config_path.read_text(encoding="utf-8")})

        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])

        model.fit(dataset)
        R.save_objects(**{"trained_model": model})
        generate_records(task=task, full_cfg=full_cfg, model=model, dataset=dataset, recorder=recorder)
        return recorder.id


def get_metric_value(metrics: dict[str, Any], candidates: list[str]) -> float | None:
    """使用优先候选键列表提取数值型指标。"""
    for key in candidates:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, float)):
                return float(value)
    for metric_key, metric_value in metrics.items():
        if not isinstance(metric_value, (int, float)):
            continue
        if any(metric_key.endswith(candidate.split(".")[-1]) for candidate in candidates):
            return float(metric_value)
    return None


def aggregate_experiment_metrics(experiment_name: str, recorder_ids: list[str]) -> dict[str, float | None]:
    """聚合完成的滚动实验 recorder 的汇总指标。"""
    aggregated: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
    if not recorder_ids:
        return {metric_name: None for metric_name in SUMMARY_METRICS}

    for recorder_id in recorder_ids:
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
        if recorder is None or recorder.status != "FINISHED":
            continue
        metrics = recorder.list_metrics()
        for metric_name, candidates in SUMMARY_METRICS.items():
            value = get_metric_value(metrics, candidates)
            if value is not None:
                aggregated[metric_name].append(value)

    return {
        metric_name: (statistics.mean(values) if values else None)
        for metric_name, values in aggregated.items()
    }


def format_metric(value: float | None) -> str:
    """格式化数值型指标，便于 Markdown 展示。"""
    return f"{value:.4f}" if value is not None else "nan"


def render_summary_table(rows: list[dict[str, Any]]) -> str:
    """将聚合后的摘要行渲染为 Markdown 表格。"""
    header = (
        "| Model Name | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Max Drawdown |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {model_name} | {IC} | {ICIR} | {rank_ic} | {rank_icir} | {annualized_return} | {max_drawdown} |".format(
                model_name=row["model_name"],
                IC=format_metric(row["IC"]),
                ICIR=format_metric(row["ICIR"]),
                rank_ic=format_metric(row["Rank IC"]),
                rank_icir=format_metric(row["Rank ICIR"]),
                annualized_return=format_metric(row["Annualized Return"]),
                max_drawdown=format_metric(row["Max Drawdown"]),
            )
        )
    return header + "\n".join(body) + ("\n" if body else "")


class ModelRunner:
    @only_allow_defined_args
    def run(
        self,
        yaml_paths: str | list[str] | None = None,
        output_dir: str = "output/run_all_model",
    ) -> dict[str, Any]:
        """执行所有提供的 YAML 配置，并保存滚动实验结果摘要。

        参数:
            yaml_paths: 逗号分隔的 YAML 文件路径或 YAML 路径列表。
            output_dir: 在仓库根目录下用于存放运行产物的目录。

        返回:
            包含运行路径和错误信息的字典。
        """
        yaml_files = parse_yaml_paths(yaml_paths)
        run_dir = (REPO_ROOT / output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        mlruns_root = run_dir / "mlruns"
        mlruns_root.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        manifests: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        for yaml_path in yaml_files:
            full_cfg = load_yaml(yaml_path)
            experiment_name = infer_experiment_name(full_cfg, yaml_path)
            model_name = infer_model_name(full_cfg, yaml_path)

            qlib_init_kwargs = normalize_qlib_init(
                raw_cfg=full_cfg.get("qlib_init", {}),
                config_path=yaml_path,
                mlruns_root=mlruns_root,
                experiment_name=experiment_name,
            )
            qlib.init(**qlib_init_kwargs)

            calendar = get_calendar(ROLLING_TRAIN_START, ROLLING_TEST_END)
            base_task = get_task_config(full_cfg)
            rolling_tasks = build_weekly_rolling_tasks(
                base_task=base_task,
                calendar=calendar,
            )
            recorder_ids: list[str] = []

            print(f"\n=== Running {model_name} from {yaml_path} ===")
            print(f"experiment={experiment_name}, rolling_tasks={len(rolling_tasks)}")

            try:
                for idx, task in enumerate(rolling_tasks, start=1):
                    recorder_name = f"{yaml_path.stem}_roll_{idx:02d}"
                    recorder_id = run_single_task(
                        task=task,
                        full_cfg=full_cfg,
                        experiment_name=experiment_name,
                        recorder_name=recorder_name,
                        source_config_path=yaml_path,
                    )
                    recorder_ids.append(recorder_id)
            except Exception:
                errors[str(yaml_path)] = traceback.format_exc()
                print(errors[str(yaml_path)])

            manifests.append(
                {
                    "yaml_path": str(yaml_path),
                    "model_name": model_name,
                    "experiment_name": experiment_name,
                    "recorder_ids": recorder_ids,
                }
            )

            aggregated = aggregate_experiment_metrics(experiment_name, recorder_ids)
            summary_rows.append(
                {
                    "yaml_path": str(yaml_path),
                    "model_name": model_name,
                    **aggregated,
                }
            )

        summary_table = render_summary_table(summary_rows)
        summary_path = run_dir / "summary.md"
        summary_path.write_text(summary_table, encoding="utf-8")

        summary_csv = run_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "yaml_paths": [str(path) for path in yaml_files],
                    "manifests": manifests,
                    "errors": errors,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if summary_table:
            print("\n=== Summary Table ===")
            print(summary_table)
        if errors:
            print("\n=== Errors ===")
            for path, message in errors.items():
                print(f"[{path}]\n{message}")

        return {
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "summary_csv": str(summary_csv),
            "manifest_path": str(manifest_path),
            "errors": errors,
        }


if __name__ == "__main__":
    fire.Fire(ModelRunner)
