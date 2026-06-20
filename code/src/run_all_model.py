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
import os
os.environ["MLFLOW_TRACKING_URI"] = "/tmp/mlruns"

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
    """提取评测周期: 以每个周五为 signal day。

    signal(Friday) → buy(Mon) → sell(T+hold_days)
    与 FridayFilterProcessor 对齐。
    hold_days: signal 到 sell 的交易日偏移量，默认 5（完整交易周）。
               特殊节假日缩短时设为 4。
    """
    ts = pd.Timestamp(test_start)
    te = pd.Timestamp(test_end)
    td = pd.Timestamp(data_end)
    mask = (calendar >= ts) & (calendar <= te)
    days = calendar[mask]

    periods = []
    for idx, d in enumerate(days):
        if d.dayofweek != 4:  # only Fridays
            continue
        if idx + hold_days >= len(days):
            break
        sig, buy, sell = d, days[idx + 1], days[idx + hold_days]
        if sell > td:
            break
        periods.append({"signal_day": sig, "buy_day": buy, "sell_day": sell})
    return periods


# ============================================================
# 市场状态 (Stage 3: Dynamic Position Sizing)
# ============================================================

def _load_hs300_index() -> pd.DataFrame | None:
    """Load HS300 index daily data. Try Baostock-saved file first."""
    csv_path = _PROJECT_ROOT / "data" / "hs300_index.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date")
        df["close"] = df["close"].astype(float)
        return df
    return None


def _compute_market_features(
    index_df: pd.DataFrame, signal_date: pd.Timestamp
) -> dict | None:
    """Compute market state features at signal_date (only data ≤ signal_date)."""
    hist = index_df[index_df["date"] <= signal_date]
    if len(hist) < 60:
        return None
    hist = hist.tail(60)
    closes = hist["close"].values
    daily_rets = np.diff(closes) / closes[:-1]

    # 1. 20-day cumulative return
    ret_20d = float(np.prod(1 + daily_rets[-20:]) - 1) if len(daily_rets) >= 20 else float(np.prod(1 + daily_rets) - 1)

    # 2. 20-day return volatility (annualized)
    vol_20d = float(np.std(daily_rets[-20:])) if len(daily_rets) >= 20 else float(np.std(daily_rets))

    # 3. Market breadth: proportion of closes above MA60
    ma60 = np.mean(closes)
    breadth = float(np.mean(closes > ma60))

    # 4. Cross-sectional dispersion proxy: recent return volatility
    recent = daily_rets[-5:] if len(daily_rets) >= 5 else daily_rets
    dispersion = float(np.std(recent))

    return {
        "hs300_20d_return": ret_20d,
        "hs300_20d_vol": vol_20d,
        "market_breadth": breadth,
        "dispersion": dispersion,
    }


def _classify_market_state(features: dict, vol_threshold: float) -> tuple[int, int]:
    """Classify market state: (trend_flag, vol_flag)."""
    trend = 1 if features["hs300_20d_return"] > 0 else 0
    vol = 1 if features["hs300_20d_vol"] > vol_threshold else 0
    return (trend, vol)


def _get_exposure(state: tuple[int, int], version: str = "v1") -> float:
    """Map market state → total exposure."""
    if version == "v1":
        mapping = {(1, 0): 1.0, (1, 1): 0.8, (0, 0): 0.6, (0, 1): 0.2}
    elif version == "v2":
        mapping = {(1, 0): 1.0, (1, 1): 0.8, (0, 0): 0.4, (0, 1): 0.0}
    else:
        mapping = {(1, 0): 1.0, (1, 1): 1.0, (0, 0): 1.0, (0, 1): 1.0}
    return mapping.get(state, 1.0)


_STATE_LABELS = {(1, 0): "A_强势低波", (1, 1): "B_强势高波", (0, 0): "C_弱势低波", (0, 1): "D_弱势高波"}

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

    # ── 6.5) 分数统计与动态仓位 (E4: Confidence/Gap Exposure) ──
    exposure_mode = eval_cfg.get("exposure_mode", None)
    market_features_map: dict = {}
    score_stats_map: dict = {}  # sd -> {top5_mean_z, gap_1_5, gap_1_10, ...}

    if exposure_mode in ("confidence", "gap", "conf_gap"):
        # Pre-compute score statistics for each signal date
        all_gaps = []
        for p in periods:
            sd = p["signal_day"]
            try:
                dp = pred.xs(sd, level="datetime")
            except KeyError:
                continue
            if dp.empty or dp.isna().all():
                continue
            top5 = dp.nlargest(5)
            top10 = dp.nlargest(10)
            stats = {
                "top1_score_z": float(top5.iloc[0]),
                "top5_score_z": float(top5.iloc[-1]),
                "top5_mean_z": float(top5.mean()),
                "top5_std_z": float(top5.std(ddof=0)),
                "score_gap_1_5": float(top5.iloc[0] - top5.iloc[-1]),
                "score_gap_1_10": float(top5.iloc[0] - top10.iloc[-1]),
            }
            score_stats_map[sd] = stats
            all_gaps.append(stats["score_gap_1_5"])

        # Compute reference values (median across test weeks)
        gap_ref = float(np.median(all_gaps)) if all_gaps else 0.01
        all_conf_means = [s["top5_mean_z"] for s in score_stats_map.values()]
        conf_baseline = float(np.mean(all_conf_means)) if all_conf_means else 0.0

        for sd, stats in score_stats_map.items():
            stats["gap_norm"] = min(stats["score_gap_1_5"] / (2.0 * gap_ref + 1e-8), 1.0)
            # Confidence-based exposure (centered: high conf = above avg, low = below)
            conf_centered = stats["top5_mean_z"] - conf_baseline
            g_conf = np.clip(1.0 + 0.15 * conf_centered, 0.7, 1.3)
            # Gap-based exposure
            g_gap = 0.8 + 0.4 * stats["gap_norm"]
            stats["conf_centered"] = float(conf_centered)
            stats["g_conf"] = float(g_conf)
            stats["g_gap"] = float(g_gap)
            # Combined exposure
            if exposure_mode == "confidence":
                stats["exposure"] = float(g_conf)
            elif exposure_mode == "gap":
                stats["exposure"] = float(g_gap)
            else:  # conf_gap
                stats["exposure"] = float(np.clip(g_conf * g_gap, 0.6, 1.5))

        print(f"        E4 ({exposure_mode}): conf_baseline={conf_baseline:.4f}, "
              f"gap_ref={gap_ref:.4f}, "
              f"avg_exposure={np.mean([s['exposure'] for s in score_stats_map.values()]):.2%}")

    # ── 7) 逐周评测 ──
    details = []
    ranking_metrics = []  # per-week ranking metrics

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

        # ---- 排名质量指标（全股票） ----
        try:
            all_insts = dp.index.intersection(price_df.index)
            # 所有股票的实际收益率 (T+1 open → T+5 open)
            all_bp = price_df.loc[all_insts, bd]
            all_sp = price_df.loc[all_insts, sed]
            valid_all = all_bp.notna() & all_sp.notna() & (all_bp > 0)
            all_insts_valid = all_insts[valid_all]
            if len(all_insts_valid) >= 5:
                actual_rets = (all_sp[valid_all] / all_bp[valid_all]) - 1.0
                pred_scores = dp.loc[all_insts_valid]

                # RankIC (Spearman correlation)
                from scipy.stats import spearmanr
                rank_ic, _ = spearmanr(pred_scores, actual_rets)
                rank_ic = 0.0 if np.isnan(rank_ic) else float(rank_ic)

                # Top5 Hit Rate
                actual_top5 = set(actual_rets.nlargest(5).index)
                pred_top5_rank = set(pred_scores.nlargest(5).index)
                hit5 = float(len(actual_top5 & pred_top5_rank)) / 5.0

                # Top10 Recall
                actual_top10 = set(actual_rets.nlargest(10).index)
                top10_recall = float(len(actual_top10 & pred_top5_rank)) / 10.0

                # NDCG@5
                pred_top5_rets = actual_rets.loc[list(pred_top5_rank)]
                dcg5 = _dcg_at_k(pred_top5_rets.values, 5)
                idcg5 = _dcg_at_k(actual_rets.nlargest(5).values, 5)
                ndcg5 = float(dcg5 / idcg5) if idcg5 > 0 else 0.0

                ranking_metrics.append({
                    "rank_ic": rank_ic,
                    "hit5": hit5,
                    "top10_recall": top10_recall,
                    "ndcg5": ndcg5,
                })
        except Exception:
            pass  # skip ranking metrics if data unavailable

        # ---- 选股 + 权重 ----
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

        # ---- Stage 5b: Rank-Weighted Portfolio ----
        if exposure_mode == "rank_weighted" and not strategy:
            top5_sorted = dp.nlargest(5)
            rank_weights = eval_cfg.get("rank_weights", [0.35, 0.25, 0.18, 0.12, 0.10])
            # If fewer than 5 weights (e.g., top3-only), select fewer stocks
            k = len(rank_weights)
            w = {s: rank_weights[i] for i, s in enumerate(top5_sorted.head(k).index)}

        # ---- Stage 3/4: Dynamic Position Sizing ----
        g = 1.0  # default: full exposure
        if exposure_mode and sd in market_features_map:
            g = market_features_map[sd]["exposure"]
        elif exposure_mode and sd in score_stats_map:
            g = score_stats_map[sd]["exposure"]
        if g != 1.0 and exposure_mode != "rank_weighted":
            w = {s: g / max(len(w), 1) for s in w}

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

        # 计算周收益 + rank贡献 (V4)
        stock_rets = (sp[valid] / bp[valid]) - 1.0
        total_weight = sum(w.get(s, 0.0) for s in stock_rets.index)
        port_ret = sum(w.get(s, 0.0) * stock_rets[s] for s in stock_rets.index)
        # Per-rank contribution
        rank_contrib = {}
        for i, s in enumerate(sel):
            if s in stock_rets.index and s in w:
                rank_contrib[f"rank{i+1}_contrib"] = round(w[s] * stock_rets[s], 6)

        extra = {
            "top5": sel, "weight": round(total_weight, 4),
            "n_valid": int(valid.sum()),
            **rank_contrib,
        }
        if exposure_mode and sd in market_features_map:
            mf = market_features_map[sd]
            extra["exposure"] = round(mf["exposure"], 2)
            extra["cash_ratio"] = round(1.0 - total_weight, 4)
            extra["state"] = _STATE_LABELS[mf["state"]]
        if exposure_mode and sd in score_stats_map:
            ss = score_stats_map[sd]
            extra["exposure"] = round(ss["exposure"], 2)
            extra["cash_ratio"] = round(1.0 - total_weight, 4)
            extra["top5_mean_z"] = round(ss["top5_mean_z"], 4)
            extra["conf_centered"] = round(ss.get("conf_centered", ss["top5_mean_z"]), 4)
            extra["score_gap_1_5"] = round(ss["score_gap_1_5"], 4)
            extra["g_conf"] = round(ss["g_conf"], 4)
            extra["g_gap"] = round(ss.get("g_gap", 1.0), 4)

        details.append(_mk_detail(p, port_ret, "ok", **extra))

    returns = pd.Series([d["return"] for d in details])
    result = {"metrics": _calc_metrics(returns), "weekly_details": details, "error": None, "_score_stats": score_stats_map}

    # ---- 市场状态仓位指标 (Stage 3) ----
    if exposure_mode and (market_features_map or score_stats_map):
        exposures = [d.get("exposure", 1.0) for d in details]
        cash_ratios = [d.get("cash_ratio", 0.0) for d in details]
        em_dict = {
            "average_exposure": round(float(np.mean(exposures)), 4) if exposures else 1.0,
            "average_cash_ratio": round(float(np.mean(cash_ratios)), 4) if cash_ratios else 0.0,
        }
        if market_features_map:
            bull_rets, bear_rets = [], []
            for d in details:
                sd = pd.Timestamp(d["signal_day"])
                if sd in market_features_map:
                    f = market_features_map[sd]["features"]
                    if f["hs300_20d_return"] > 0:
                        bull_rets.append(d["return"])
                    else:
                        bear_rets.append(d["return"])
            em_dict.update({
                "bull_market_return": round(float(np.mean(bull_rets)), 6) if bull_rets else 0.0,
                "bear_market_return": round(float(np.mean(bear_rets)), 6) if bear_rets else 0.0,
                "bull_weeks": len(bull_rets),
                "bear_weeks": len(bear_rets),
            })
            sc = {}
            for d in details:
                st = d.get("state", "unknown")
                sc[st] = sc.get(st, 0) + 1
            em_dict["state_counts"] = sc
        result["exposure_metrics"] = em_dict

    # ---- E4 信号诊断相关性 ----
    if score_stats_map:
        confs, gaps, rets = [], [], []
        for d in details:
            sd = pd.Timestamp(d["signal_day"])
            if sd in score_stats_map:
                confs.append(score_stats_map[sd]["top5_mean_z"])
                gaps.append(score_stats_map[sd]["score_gap_1_5"])
                rets.append(d["return"])
        result["e4_diagnostics"] = {
            "corr_conf_return": round(float(np.corrcoef(confs, rets)[0, 1]), 4) if len(confs) >= 3 else 0.0,
            "corr_gap_return": round(float(np.corrcoef(gaps, rets)[0, 1]), 4) if len(gaps) >= 3 else 0.0,
        }

    # ---- E5a: Rank Stability Analysis ----
    rank_returns: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: [], 20: []}
    for p in periods:
        sd, bd, sed = p["signal_day"], p["buy_day"], p["sell_day"]
        try:
            dp = pred.xs(sd, level="datetime")
        except KeyError:
            continue
        if dp.empty or dp.isna().all():
            continue
        sorted_stocks = dp.sort_values(ascending=False)
        for k in rank_returns:
            top_k = sorted_stocks.head(k).index
            top_k = top_k.intersection(price_df.index)
            bp = price_df.loc[top_k, bd]
            sp = price_df.loc[top_k, sed]
            valid = bp.notna() & sp.notna() & (bp > 0)
            if valid.sum() == 0:
                continue
            rets = (sp[valid] / bp[valid]) - 1.0
            rank_returns[k].append(float(rets.mean()))
    if rank_returns[5]:
        result["rank_stability"] = {
            f"top{k}_mean_ret": round(float(np.mean(rank_returns[k])), 6)
            for k in rank_returns
        }
        result["rank_stability"].update({
            f"top{k}_std_ret": round(float(np.std(rank_returns[k], ddof=0)), 6)
            for k in rank_returns
        })
        result["rank_stability"].update({
            f"top{k}_pos_rate": round(float(np.mean([r > 0 for r in rank_returns[k]])), 4)
            for k in rank_returns
        })
        result["rank_stability"]["n_weeks"] = len(rank_returns[5])

    # ---- V4: Rank Contribution Attribution ----
    if details and "rank1_contrib" in details[0]:
        contrib_sums = {f"rank{i+1}_contrib": 0.0 for i in range(5)}
        for d in details:
            for k in contrib_sums:
                contrib_sums[k] += d.get(k, 0.0)
        total_abs = sum(abs(v) for v in contrib_sums.values()) + 1e-12
        result["rank_attribution"] = {
            f"rank{i+1}_pct": round(contrib_sums[f"rank{i+1}_contrib"] / total_abs * 100, 1)
            for i in range(5)
        }

    # ---- 聚合排名指标 ----
    if ranking_metrics:
        rdf = pd.DataFrame(ranking_metrics)
        result["ranking_metrics"] = {
            "rank_ic_mean": round(float(rdf["rank_ic"].mean()), 4),
            "rank_ic_std": round(float(rdf["rank_ic"].std()), 4),
            "rank_ic_positive_rate": round(
                float((rdf["rank_ic"] > 0).mean()), 4
            ),
            "hit5_mean": round(float(rdf["hit5"].mean()), 4),
            "top10_recall_mean": round(float(rdf["top10_recall"].mean()), 4),
            "ndcg5_mean": round(float(rdf["ndcg5"].mean()), 4),
        }
    else:
        result["ranking_metrics"] = {}

    return result


def _dcg_at_k(scores: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k (gain = score, no relevance binarization)."""
    scores = np.asarray(scores, dtype=float)[:k]
    discounts = np.log2(np.arange(2, k + 2, dtype=float))
    return float(np.sum(scores / discounts))


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
    # ── 排名质量表 ──
    lines.append("")
    lines.append("## 排名质量 (Ranking Quality)")
    lines.append("")
    h3 = ("| Model | RankIC | IC>0% | Hit@5 | Recall@10 | NDCG@5 |\n"
          "|-------|-------:|------:|------:|----------:|-------:|\n")
    lines.append(h3)
    for r in rows:
        rm = r.get("ranking_metrics", {})
        if rm:
            lines.append(
                f"| {r['model_name']} "
                f"| {rm.get('rank_ic_mean', 0):.4f} "
                f"| {rm.get('rank_ic_positive_rate', 0):.1%} "
                f"| {rm.get('hit5_mean', 0):.1%} "
                f"| {rm.get('top10_recall_mean', 0):.1%} "
                f"| {rm.get('ndcg5_mean', 0):.4f} |"
            )
        else:
            lines.append(f"| {r['model_name']} | — | — | — | — | — |")
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
            # 排名质量指标
            rm = result.get("ranking_metrics", {})
            if rm:
                print(f"     RankIC: {rm.get('rank_ic_mean', 0):.4f} (IC>0: {rm.get('rank_ic_positive_rate', 0):.1%})")
                print(f"     Hit@5: {rm.get('hit5_mean', 0):.1%}  |  Recall@10: {rm.get('top10_recall_mean', 0):.1%}  |  NDCG@5: {rm.get('ndcg5_mean', 0):.4f}")

            # 动态仓位指标 (Stage 3/4)
            em = result.get("exposure_metrics", {})
            e4 = result.get("e4_diagnostics", {})
            if em and em.get("state_counts"):
                print(f"     Avg Exposure: {em.get('average_exposure', 0):.2%}  |  "
                      f"Avg Cash: {em.get('average_cash_ratio', 0):.2%}")
                print(f"     Bull ({em.get('bull_weeks', 0)}W): {em.get('bull_market_return', 0):+.4f}"
                      f"  |  Bear ({em.get('bear_weeks', 0)}W): {em.get('bear_market_return', 0):+.4f}")
                sc = em.get("state_counts", {})
                print(f"     States: {sc}")
            if e4:
                print(f"     Corr(top5_mean_z, ret): {e4.get('corr_conf_return', 0):+.4f}"
                      f"  |  Corr(gap, ret): {e4.get('corr_gap_return', 0):+.4f}")
            # E5a: Rank Stability
            rs = result.get("rank_stability", {})
            if rs:
                print(f"     Rank Stability ({rs.get('n_weeks', 0)}W):")
                print(f"       {'Rank':>6} {'MeanRet':>8} {'Std':>8} {'PosRate':>8}")
                for k in [1, 3, 5, 10, 20]:
                    print(f"       Top{k:<3}  {rs.get(f'top{k}_mean_ret', 0):+8.4f} "
                          f"{rs.get(f'top{k}_std_ret', 0):8.4f} {rs.get(f'top{k}_pos_rate', 0):8.1%}")
            # V4: Rank Contribution
            ra = result.get("rank_attribution", {})
            if ra:
                print(f"     Rank Contribution: {ra}")

            # 每周明细
            for d in result.get("weekly_details", []):
                icon = "✅" if d["status"] == "ok" else "⚠️"
                t5 = f"  top5={d['top5']}" if "top5" in d else ""
                extra = ""
                if "exposure" in d:
                    if "state" in d:
                        extra = f"  exp={d['exposure']:.1f} cash={d.get('cash_ratio', 0):.2f} [{d['state']}]"
                    elif "top5_mean_z" in d:
                        cc = d.get("conf_centered", d["top5_mean_z"])
                        extra = (f"  exp={d['exposure']:.2f}"
                                 f" conf_c={cc:+.3f}"
                                 f" gap={d['score_gap_1_5']:.3f}")
                print(f"     {icon} {d['signal_day']} → {d['sell_day']}  "
                      f"ret={d['return']:+.4f}{t5}{extra}")

            summary_rows.append({
                "model_name": model_name,
                "metrics": m,
                "ranking_metrics": result.get("ranking_metrics", {}),
                "exposure_metrics": result.get("exposure_metrics", {}),
                "error": result.get("error"),
            })

            # 保存明细 CSV
            if result.get("weekly_details"):
                pd.DataFrame(result["weekly_details"]).to_csv(
                    run_dir / f"{model_name}_weekly.csv",
                    index=False, encoding="utf-8-sig",
                )
            # 保存 E4 信号诊断 CSV
            score_stats = result.get("_score_stats", {})
            if score_stats:
                diag_rows = []
                for d in result.get("weekly_details", []):
                    sd = pd.Timestamp(d["signal_day"])
                    row = {"date": d["signal_day"], "weekly_return": d["return"]}
                    if sd in score_stats:
                        ss = score_stats[sd]
                        row.update({
                            "top1_score_z": ss["top1_score_z"],
                            "top5_score_z": ss["top5_score_z"],
                            "top5_mean_z": ss["top5_mean_z"],
                            "top5_std_z": ss["top5_std_z"],
                            "score_gap_1_5": ss["score_gap_1_5"],
                            "score_gap_1_10": ss["score_gap_1_10"],
                            "gap_norm": ss["gap_norm"],
                            "g_conf": ss["g_conf"],
                            "g_gap": ss.get("g_gap", 1.0),
                            "exposure": ss["exposure"],
                        })
                    if "top5" in d:
                        row["selected_stocks"] = ",".join(d["top5"])
                    diag_rows.append(row)
                pd.DataFrame(diag_rows).to_csv(
                    run_dir / f"{model_name}_signal_diagnostics.csv",
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
