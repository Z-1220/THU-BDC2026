"""
比赛调参引擎 — 单次训练，多周评测。

用法:
    python code/src/run_all_model.py run --yaml_paths=models/LightGBM/LightGBM.yaml
    python code/src/run_all_model.py run --yaml_paths="models/LightGBM/LightGBM.yaml,models/XGB/XGB.yaml"
"""

from __future__ import annotations

import functools
import inspect
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.config import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config

import sys
import random

import torch

# ── 路径设置 ──────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent          # code/src
REPO_ROOT = CURRENT_DIR.parent                         # code/
_PROJECT_ROOT = REPO_ROOT.parent                       # THU-BDC2026/

for _p in [CURRENT_DIR, REPO_ROOT, _PROJECT_ROOT]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def only_allow_defined_args(func):
    """Fire 命令装饰器：拒绝未知参数。"""
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        spec = inspect.getfullargspec(func)
        valid = set(spec.args + spec.kwonlyargs)
        for k in kwargs:
            if k not in valid:
                raise ValueError(f"Unknown arg '{k}'")
        return func(*args, **kwargs)
    return _wrapped


def parse_yaml_paths(yaml_paths: str | list[str]) -> list[Path]:
    if isinstance(yaml_paths, str):
        items = [x.strip() for x in yaml_paths.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in yaml_paths]
    paths = [(Path(i) if Path(i).is_absolute() else REPO_ROOT / i).resolve() for i in items]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"YAML not found: {missing}")
    return paths


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_model_name(cfg: dict) -> str:
    cls = cfg.get("task", {}).get("model", {}).get("class", "")
    return cls.split(".")[-1] if cls else "unknown"


# ============================================================
# Qlib 版本兼容补丁（在 qlib.init() 之后调用）
# ============================================================

def apply_qlib_patch() -> None:
    """修复 D.features() 的 freq/inst_processors 参数位置冲突。"""
    try:
        from qlib.data.storage.file_storage import LocalDatasetProvider
        _orig = LocalDatasetProvider.dataset

        def _safe(self, instruments, fields, start_time=None, end_time=None,
                  freq="day", inst_processors=None, **kw):
            return _orig(self,
                         instruments=instruments, fields=fields,
                         start_time=start_time, end_time=end_time,
                         freq=freq, inst_processors=inst_processors, **kw)

        LocalDatasetProvider.dataset = _safe
        print("  [PATCH] ✅ LocalDatasetProvider.dataset")
    except ImportError:
        # 某些版本的类名/路径不同，尝试其他入口
        try:
            from qlib.data.data import DatasetD
            _orig_dataset = DatasetD.dataset

            def _safe_dataset(instruments, fields, start_time=None, end_time=None,
                              freq="day", inst_processors=None):
                return _orig_dataset(
                    instruments=instruments, fields=fields,
                    start_time=start_time, end_time=end_time,
                    freq=freq, inst_processors=inst_processors,
                )

            DatasetD.dataset = staticmethod(_safe_dataset)
            print("  [PATCH] ✅ DatasetD.dataset (fallback)")
        except Exception as e2:
            print(f"  [PATCH] ⚠️  无法应用补丁: {e2}")
    except Exception as e:
        print(f"  [PATCH] ⚠️  无法应用补丁: {e}")


# ============================================================
# 安全的 Qlib 数据接口
# ============================================================

def _safe_get_calendar(start: str, end: str) -> pd.DatetimeIndex:
    try:
        cal = D.calendar(start_time=start, end_time=end)
        return pd.DatetimeIndex(pd.to_datetime(cal)).sort_values()
    except Exception:
        return pd.DatetimeIndex([])


# ============================================================
# 从 handler._data 提取价格（彻底绕过 D.features）
# ============================================================

def _extract_prices_from_handler(handler) -> tuple[pd.Series, pd.Series]:
    """
    从 handler._data 中提取 $open 和 $close 价格列。

    返回:
        (open_series, close_series)
        每个 Series 的 Index 都是 MultiIndex (datetime, instrument)
    """
    raw = handler._data
    open_col = None
    close_col = None

    if isinstance(raw.columns, pd.MultiIndex):
        level1 = raw.columns.get_level_values(1)

        # Qlib 原始字段名通常是 "$open" / "$close"
        for candidate in ["$open", "$close"]:
            if candidate in level1:
                mask = level1 == candidate
                col = raw.loc[:, mask].iloc[:, 0]
                if "open" in candidate.lower():
                    open_col = col
                else:
                    close_col = col

        # 兜底：Alpha158 可能重命名为 "OPEN" / "CLOSE"
        if open_col is None and "OPEN" in level1:
            open_col = raw.loc[:, level1 == "OPEN"].iloc[:, 0]
        if close_col is None and "CLOSE" in level1:
            close_col = raw.loc[:, level1 == "CLOSE"].iloc[:, 0]

        # 模糊匹配
        if open_col is None or close_col is None:
            for val in level1.unique():
                val_str = str(val).lower()
                if open_col is None and "open" in val_str:
                    open_col = raw.loc[:, level1 == val].iloc[:, 0]
                if close_col is None and "close" in val_str:
                    close_col = raw.loc[:, level1 == val].iloc[:, 0]
    else:
        # 非 MultiIndex 的罕见情况
        for c in raw.columns:
            c_str = str(c).lower()
            if open_col is None and "open" in c_str:
                open_col = raw[c]
            if close_col is None and "close" in c_str:
                close_col = raw[c]

    if open_col is None or close_col is None:
        if isinstance(raw.columns, pd.MultiIndex):
            avail = raw.columns.get_level_values(1).unique().tolist()[:30]
        else:
            avail = raw.columns.tolist()[:30]
        missing = []
        if open_col is None:
            missing.append("open")
        if close_col is None:
            missing.append("close")
        raise KeyError(f"Missing {missing} in handler._data. Available: {avail}")

    open_col.name = "open"
    close_col.name = "close"

    return open_col, close_col


# ============================================================
# 评测周期提取
# ============================================================

def get_eval_periods(
    calendar: pd.DatetimeIndex,
    test_start: str,
    test_end: str,
    data_end: str,
    hold_days: int = 5,
) -> list[dict[str, pd.Timestamp]]:
    """提取非重叠评测周期: signal(Friday) → buy(Mon) → sell(next Fri + hold_days)

    signal day 对齐到周五，与 FridayFilterProcessor 一致。
    hold_days: signal 到 sell 的交易日偏移量，默认 5（完整交易周）。
               特殊节假日缩短时设为 4。
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    td = pd.Timestamp(data_end)
    mask = (calendar >= ts) & (calendar <= te)
    days = calendar[mask]

    # 找第一个周五作为起始 signal day，对齐 FridayFilterProcessor
    first_idx = 0
    for idx, d in enumerate(days):
        if d.dayofweek == 4:  # Friday
            first_idx = idx
            break

    periods = []
    i = first_idx
    while i + hold_days < len(days):
        sig, buy, sell = days[i], days[i + 1], days[i + hold_days]
        if sell > td:
            break
        periods.append({"signal_day": sig, "buy_day": buy, "sell_day": sell})
        i += hold_days
    return periods


# ============================================================
# 核心评测
# ============================================================

def evaluate_config(task_cfg: dict, eval_cfg: dict) -> dict:
    segs = task_cfg["dataset"]["kwargs"]["segments"]
    test_start, test_end = segs["test"]
    data_end = eval_cfg.get("data_end", test_end)

    # ── 1) 训练 ──
    print("  [1/5] Training model...")
    dataset = init_instance_by_config(task_cfg["dataset"])
    model = init_instance_by_config(task_cfg["model"])
    model.fit(dataset)

    # ── 2) 预测 ──
    print("  [2/5] Predicting...")
    pred = model.predict(dataset)

    # ── 3) 日历 ──
    print("  [3/5] Getting calendar...")
    cal = _safe_get_calendar(test_start, data_end)
    if len(cal) == 0:
        cal = pd.DatetimeIndex(
            pred.index.get_level_values("datetime").unique()
        ).sort_values()
        print("        (derived from predictions)")

    # ── 4) 价格数据 ──
    print("  [4/5] Getting price data...")
    try:
        open_series, close_series = _extract_prices_from_handler(dataset.handler)
        print(f"        ✅ open  shape={open_series.shape}  "
              f"index={open_series.index.names}")
        print(f"        ✅ close shape={close_series.shape}  "
              f"index={close_series.index.names}")
    except Exception as e:
        return {
            "metrics": {"n_weeks": 0}, "weekly_details": [],
            "error": f"price_extract_failed: {e}",
        }

    # open: MultiIndex(datetime, instrument) → unstack datetime → DataFrame(rows=stocks, cols=dates)
    price_df = open_series.unstack(level="datetime")
    print(f"        price_df: {price_df.shape[0]} stocks × {price_df.shape[1]} days")

    # close: MultiIndex(datetime, instrument) → unstack instrument → DataFrame(rows=dates, cols=stocks)
    close_df = close_series.unstack(level="instrument")
    print(f"        close_df: {close_df.shape[0]} days × {close_df.shape[1]} stocks")

    # ── 5) 策略初始化 + 注入价格 ──
    print("  [5/5] Weekly evaluation...")
    strategy = None
    scfg = task_cfg.get("strategy")
    if scfg:
        try:
            strategy = init_instance_by_config(scfg)
            print(f"        ✅ Strategy: {scfg.get('class')}")
        except Exception as e:
            print(f"        ⚠️  Strategy init failed ({e}), fallback to top-5 equal weight")
            strategy = None

    # 注入 close 价格到策略（PyPortfolioOptStrategy 需要）
    if strategy is not None and hasattr(strategy, "set_price_data"):
        strategy.set_price_data(close_df)
        print("        ✅ Price data injected into strategy")

    # ── 6) 提取评测周期 ──
    hold_days = eval_cfg.get("hold_days", 5)
    periods = get_eval_periods(cal, test_start, test_end, data_end, hold_days=hold_days)
    if not periods:
        return {
            "metrics": {"n_weeks": 0}, "weekly_details": [],
            "error": "no_complete_weeks",
        }
    print(f"        {len(periods)} evaluation periods (hold={hold_days}d): "
          f"{periods[0]['signal_day'].date()} → {periods[-1]['sell_day'].date()}")

    # ── 7) 逐周评测 ──
    details = []

    for p in periods:
        sd, bd, sed = p["signal_day"], p["buy_day"], p["sell_day"]

        # 获取当日预测
        try:
            dp = pred.xs(sd, level="datetime")
        except KeyError:
            details.append(_mk_detail(p, 0.0, "no_pred"))
            continue
        if dp.empty or dp.isna().all():
            details.append(_mk_detail(p, 0.0, "empty_pred"))
            continue

        # 选股 + 权重
        if strategy is not None:
            try:
                w = strategy.generate_target_weight_position(
                    score=dp, current=None, trade_date=sd,
                )
            except Exception as e:
                details.append(_mk_detail(p, 0.0, f"strat_err:{e}"))
                continue
        else:
            top5 = dp.nlargest(5)
            w = {s: 0.2 for s in top5.index}

        if not w:
            details.append(_mk_detail(p, 0.0, "no_weights"))
            continue

        sel = list(w.keys())

        # 获取买卖价格
        try:
            bp = price_df.loc[sel, bd]
            sp = price_df.loc[sel, sed]
        except KeyError as e:
            details.append(_mk_detail(p, 0.0, f"price_missing:{e}"))
            continue

        valid = bp.notna() & sp.notna() & (bp > 0)
        if valid.sum() == 0:
            details.append(_mk_detail(p, 0.0, "suspended"))
            continue

        # 计算周收益
        stock_rets = (sp[valid] / bp[valid]) - 1.0
        total_weight = sum(w.get(s, 0.0) for s in stock_rets.index)
        port_ret = sum(w.get(s, 0.0) * stock_rets[s] for s in stock_rets.index)

        details.append(_mk_detail(
            p, port_ret, "ok",
            top5=sel, weight=round(total_weight, 4), n_valid=int(valid.sum()),
        ))

    returns = pd.Series([d["return"] for d in details])
    return {"metrics": _calc_metrics(returns), "weekly_details": details, "error": None}


def _mk_detail(period, ret, status, **extra):
    d = {
        "signal_day": str(period["signal_day"].date()),
        "buy_day": str(period["buy_day"].date()),
        "sell_day": str(period["sell_day"].date()),
        "return": round(ret, 6),
        "status": status,
    }
    d.update(extra)
    return d


def _calc_metrics(returns: pd.Series) -> dict:
    n = len(returns)
    if n == 0:
        return {"n_weeks": 0}
    m, s = returns.mean(), returns.std(ddof=0)
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    downside = returns[returns < 0].std(ddof=0)
    cum = (1 + returns).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    sorted_ret = returns.sort_values()
    worst_n = sorted_ret.head(min(3, n)).mean()
    best_n = sorted_ret.tail(min(3, n)).mean()
    return {
        "n_weeks": n,
        "mean_weekly_return": round(float(m), 6),
        "std_weekly_return": round(float(s), 6),
        "weekly_sharpe": round(float(m / s), 4) if s > 0 else 0.0,
        "sortino": round(float(m / downside), 4) if downside and downside > 0 else 0.0,
        "win_rate": round(float((returns > 0).mean()), 4),
        "max_weekly_loss": round(float(returns.min()), 6),
        "max_weekly_gain": round(float(returns.max()), 6),
        "cumulative_return": round(float(cum.iloc[-1] - 1), 6) if len(cum) > 0 else 0.0,
        "max_drawdown": round(float(dd), 6),
        "median_return": round(float(returns.median()), 6),
        "worst_3_avg": round(float(worst_n), 6),
        "best_3_avg": round(float(best_n), 6),
        "avg_gain": round(float(pos.mean()), 6) if len(pos) > 0 else 0.0,
        "avg_loss": round(float(neg.mean()), 6) if len(neg) > 0 else 0.0,
        "gain_loss_ratio": round(float(pos.mean() / abs(neg.mean())), 4)
        if len(pos) > 0 and len(neg) > 0 and neg.mean() != 0 else 0.0,
        "pos_weeks": int(len(pos)),
        "neg_weeks": int(len(neg)),
    }


# ============================================================
# 输出
# ============================================================

def render_summary(rows: list[dict]) -> str:
    lines = []
    # ── 主表：收益与风险 ──
    lines.append("## 收益与风险")
    lines.append("")
    h1 = ("| Model | Weeks | Mean | Std | Sharpe | Sortino | Win% | "
          "Cum Ret | MaxDD |\n"
          "|-------|------:|-----:|----:|-------:|--------:|-----:|"
          "--------:|------:|\n")
    lines.append(h1)
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['model_name']} "
            f"| {m.get('n_weeks', 0)} "
            f"| {m.get('mean_weekly_return', 0):+.4f} "
            f"| {m.get('std_weekly_return', 0):.4f} "
            f"| {m.get('weekly_sharpe', 0):.4f} "
            f"| {m.get('sortino', 0):.4f} "
            f"| {m.get('win_rate', 0):.1%} "
            f"| {m.get('cumulative_return', 0):+.4f} "
            f"| {m.get('max_drawdown', 0):+.4f} |"
        )
    # ── 分布表：极端值与盈亏比 ──
    lines.append("")
    lines.append("## 收益分布")
    lines.append("")
    h2 = ("| Model | Median | Worst3Avg | Best3Avg | AvgGain | AvgLoss | "
          "G/L Ratio | Gain:Neg |\n"
          "|-------|-------:|----------:|---------:|--------:|--------:|"
          "----------:|---------:|\n")
    lines.append(h2)
    for r in rows:
        m = r["metrics"]
        gain_neg = f"{m.get('pos_weeks', 0)}:{m.get('neg_weeks', 0)}"
        lines.append(
            f"| {r['model_name']} "
            f"| {m.get('median_return', 0):+.4f} "
            f"| {m.get('worst_3_avg', 0):+.4f} "
            f"| {m.get('best_3_avg', 0):+.4f} "
            f"| {m.get('avg_gain', 0):+.4f} "
            f"| {m.get('avg_loss', 0):+.4f} "
            f"| {m.get('gain_loss_ratio', 0):.2f} "
            f"| {gain_neg} |"
        )
    # ── 错误 ──
    for r in rows:
        if r.get("error"):
            lines.append(f"\n⚠️ {r['model_name']}: {r['error']}")
    return "\n".join(lines) + "\n"


# ============================================================
# CLI 入口
# ============================================================

class ModelRunner:

    @only_allow_defined_args
    def run(
        self,
        yaml_paths: str | list[str],
        output_dir: str = "output/run_all_model",
        seed: int = 42,
    ) -> dict[str, Any]:
        """执行一个或多个 YAML 配置的比赛规则评测。"""
        set_seed(seed)
        yaml_files = parse_yaml_paths(yaml_paths)

        run_dir = REPO_ROOT / output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        qlib_inited = False
        summary_rows: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        for yaml_path in yaml_files:
            print(f"\n{'='*60}")
            print(f"  Evaluating: {yaml_path.name}")
            print(f"{'='*60}")

            full_cfg = load_yaml(yaml_path)
            model_name = infer_model_name(full_cfg)
            task_cfg = full_cfg.get("task", {})
            eval_cfg = full_cfg.get("task", {}).get("evaluation", {})

            # 初始化 Qlib + 打补丁（只做一次）
            if not qlib_inited:
                init_kw = full_cfg.get("qlib_init", {})
                provider_uri = init_kw.get("provider_uri", "./temp/qlib_data")
                region = init_kw.get("region", "cn")
                qlib.init(
                    provider_uri=provider_uri,
                    region=REG_CN if region == "cn" else region,
                )
                apply_qlib_patch()
                qlib_inited = True

            try:
                result = evaluate_config(task_cfg, eval_cfg)
            except Exception:
                errors[str(yaml_path)] = traceback.format_exc()
                print(f"  ❌ FAILED:\n{errors[str(yaml_path)][:500]}")
                summary_rows.append({
                    "model_name": model_name,
                    "metrics": {"n_weeks": 0},
                    "error": "exception",
                })
                continue

            # 打印结果
            m = result["metrics"]
            print(f"  ✅ Weeks: {m.get('n_weeks', 0)}")
            if m.get("n_weeks", 0) > 0:
                print(f"     Mean / Median: {m['mean_weekly_return']:+.4f} / {m['median_return']:+.4f}")
                print(f"     Std / MaxDD  : {m['std_weekly_return']:.4f} / {m['max_drawdown']:+.4f}")
                print(f"     Sharpe/Sortino: {m['weekly_sharpe']:.4f} / {m['sortino']:.4f}")
                print(f"     Win Rate     : {m['win_rate']:.1%}  ({m.get('pos_weeks', 0)}W+ / {m.get('neg_weeks', 0)}W-)")
                print(f"     G/L Ratio    : {m['gain_loss_ratio']:.2f}  "
                      f"(Gain {m['avg_gain']:+.4f} / Loss {m['avg_loss']:+.4f})")
                print(f"     Worst3 / Best3: {m['worst_3_avg']:+.4f} / {m['best_3_avg']:+.4f}")
                print(f"     Cum Return   : {m['cumulative_return']:+.4f}")

            # 每周明细
            for d in result.get("weekly_details", []):
                icon = "✅" if d["status"] == "ok" else "⚠️"
                t5 = f"  top5={d['top5']}" if "top5" in d else ""
                print(f"     {icon} {d['signal_day']} → {d['sell_day']}  "
                      f"ret={d['return']:+.4f}{t5}")

            summary_rows.append({
                "model_name": model_name,
                "metrics": m,
                "error": result.get("error"),
            })

            # 保存明细 CSV
            if result.get("weekly_details"):
                pd.DataFrame(result["weekly_details"]).to_csv(
                    run_dir / f"{model_name}_weekly.csv",
                    index=False, encoding="utf-8-sig",
                )

        # 汇总输出
        summary_md = render_summary(summary_rows)
        (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
        pd.DataFrame(summary_rows).to_csv(
            run_dir / "summary.csv", index=False, encoding="utf-8-sig",
        )

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(summary_md)
        print(f"  📄 Saved to: {run_dir}")

        return {"run_dir": str(run_dir), "errors": errors}


if __name__ == "__main__":
    fire.Fire(ModelRunner)
