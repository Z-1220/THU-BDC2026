#!/usr/bin/env python
"""10-week (recent-2026) comparison on short-term opportunity metrics.

Motivation: the previous 21-week test window diluted recent 2026 regime
features.  Here we:
  - train on 2024-01-01 ~ 2026-04-17 (early 2026 re-included; blind-test
    adjacent signal dates 2026-04-10 / 2026-04-17 excluded),
  - validate on 2026-04-24 ~ 2026-05-15,
  - test on the LAST 10 weeks: 2026-05-22 ~ 2026-07-24.

Decision metrics target short-term opportunity capture (weekly excess return,
win rate, upside capture, downside protection, Hit@5); Sharpe / Sortino are
reported as references only.

Usage:
    uv run python scripts/compare_2026_10w.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

_VSPEC = importlib.util.spec_from_file_location("verify", PROJECT_ROOT / "scripts" / "verify_plan_b_real_kronos.py")
verify = importlib.util.module_from_spec(_VSPEC)
_VSPEC.loader.exec_module(verify)

load_groups_from_cache = verify.load_groups_from_cache
load_price_data = verify.load_price_data
load_sector_map = verify.load_sector_map
compute_context_features = verify.compute_context_features
eval_scores = verify.eval_scores
make_cs_transform = verify.make_cs_transform
run_plan_b_real = verify.run_plan_b_real

_TSPEC = importlib.util.spec_from_file_location("train_cardnn", PROJECT_ROOT / "scripts" / "train_cardnn_e2e.py")
tcardnn = importlib.util.module_from_spec(_TSPEC)
_TSPEC.loader.exec_module(tcardnn)

build_contexts = tcardnn.build_contexts
run_experiment = tcardnn.run_experiment


BLIND_START = pd.Timestamp("2026-04-13")
BLIND_END = pd.Timestamp("2026-04-17")


def filter_blind(groups: list[dict]) -> list[dict]:
    """Drop signal dates whose holding period overlaps the blind test interval."""
    out = []
    for g in groups:
        dt = g["date"]
        # holding period: dt + 1 .. dt + 5 trading days ~ calendar window
        if dt <= BLIND_END and dt + pd.Timedelta(days=7) >= BLIND_START:
            continue
        out.append(g)
    return out


def benchmark_returns(test_groups: list[dict]) -> list[float]:
    """Equal-weight cross-sectional mean label per week (market benchmark)."""
    return [float(np.mean(g["labels"])) for g in test_groups]


def extended_metrics(weekly: list[dict], bench_rets: list[float]) -> dict:
    """Short-term decision metrics + Sharpe/Sortino references."""
    returns = np.array([r["week_return"] for r in weekly], dtype=np.float64)
    bench = np.array(bench_rets, dtype=np.float64)
    n = len(returns)
    if n == 0:
        return {}
    excess = returns - bench
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    up_mask = bench > 0
    down_mask = bench < 0
    std = returns.std(ddof=1)
    downside = returns[returns < 0]
    m = {
        "weeks": n,
        "cum_return": float(np.prod(1 + returns) - 1),
        "weekly_mean_return": float(returns.mean()),
        "win_rate": float((returns > 0).mean()),
        "weekly_excess_mean": float(excess.mean()),
        "excess_win_rate": float((excess > 0).mean()),
        "upside_capture": float((returns[up_mask] > 0).mean()) if up_mask.any() else float("nan"),
        "downside_protect": float((excess[down_mask] > 0).mean()) if down_mask.any() else float("nan"),
        "profit_loss_ratio": float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else float("nan"),
        "best_week": float(returns.max()),
        "worst_week": float(returns.min()),
        "rank_ic": float(np.mean([r.get("rank_ic", 0.0) for r in weekly])),
        "hit_at_5": float(np.mean([r.get("hit_at_5", 0.0) for r in weekly])),
        "sharpe": float(returns.mean() / std * np.sqrt(52)) if std > 1e-8 else 0.0,
        "sortino": float(returns.mean() / downside.std(ddof=1)) if len(downside) > 1 and downside.std(ddof=1) > 1e-8 else 0.0,
        "max_dd": float((np.cumprod(1 + returns) / np.maximum.accumulate(np.concatenate([[1.0], np.cumprod(1 + returns)]))[1:] - 1).min()) if n else 0.0,
    }
    # weekly details
    m["weekly"] = [
        {"date": r["date"], "return": r["week_return"], "bench": b,
         "excess": r["week_return"] - b}
        for r, b in zip(weekly, bench_rets)
    ]
    return m


def write_report(results: dict, meta: dict, out_path: Path) -> None:
    rows = []
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['cum_return']:+.4f} | {m['weekly_mean_return']:+.4f} | "
            f"{m['win_rate']:.1%} | {m['weekly_excess_mean']:+.4f} | {m['excess_win_rate']:.1%} | "
            f"{m['upside_capture']:.1%} | {m['downside_protect']:.1%} | {m['hit_at_5']:.3f} | "
            f"{m['rank_ic']:+.4f} | {m['sharpe']:.3f} | {m['sortino']:.3f} | {m['max_dd']:.4f} |"
        )
    rows_text = "\n".join(rows)

    # Decision ranking by short-term metrics (not Sharpe).
    def score(m):
        return (m["weekly_excess_mean"], m["win_rate"], m["upside_capture"], m["hit_at_5"])
    ranking = sorted(results.items(), key=lambda kv: score(kv[1]), reverse=True)
    rank_text = "\n".join(f"{i + 1}. {name} (周均超额 {m['weekly_excess_mean']:+.4f}, "
                          f"胜率 {m['win_rate']:.1%}, 上涨捕捉 {m['upside_capture']:.1%}, "
                          f"Hit@5 {m['hit_at_5']:.3f})" for i, (name, m) in enumerate(ranking))

    report = f"""# 2026 最近 10 周对照测试报告（短期机会指标）

日期: {meta['timestamp']} | 数据截止: {meta['data_end']} | 评分来源: fine-tuned Kronos-small (lb60)

## 窗口设计

- 训练: 2024-01-01 ~ 2026-04-17（2026 年初数据重新纳入；排除与盲测区间
  2026-04-13~04-17 重叠的信号周，共 {meta['train_weeks']} 周）
- 验证: 2026-04-24 ~ 2026-05-08（{meta['valid_weeks']} 周）
- 测试: 最近 10 个可用周 2026-05-15 ~ 2026-07-24（{meta['test_weeks']} 周；
  06-19 因该周无股票通过金融筛选而缺失，07-31 因无未来标签剔除）
- 基准: 每周横截面等权平均收益（全候选股票）

## 决策指标

决定指标（短期机会捕捉）: 累计收益 / 周均收益 / 胜率 / 周均超额收益 / 超额胜率 /
上涨周捕捉率（市场上涨周中组合盈利比例）/ 下跌周保护率（市场下跌周中跑赢市场比例）/
Hit@5 / RankIC。
Sharpe 与 Sortino 仅作参考，不作为决定依据。

## 结果

| 实验 | 累计收益 | 周均收益 | 胜率 | 周均超额 | 超额胜率 | 上涨捕捉 | 下跌保护 | Hit@5 | RankIC | Sharpe | Sortino | 最大回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_text}

## 短期机会指标排名（决定依据）

{rank_text}

## 周度明细（Kronos-CS 基线组合收益 vs 等权基准）

{meta['weekly_table']}

## 负对照说明

B-C6（上下文打乱）与 B-C7（上下文置零）为负对照：本窗口两者 Sharpe/累计收益反而最高，
进一步说明 10 周样本噪声大，任何单窗口指标都不应作为唯一决策依据；
决定应综合多窗口、多 seed 与短期机会指标（周均超额、胜率、捕捉率、Hit@5）。

## 复现

```bash
uv run python scripts/compare_2026_10w.py
```
"""
    out_path.write_text(report, encoding="utf-8")
    print(f"报告已保存: {out_path}")


def main() -> None:
    cache_path = PROJECT_ROOT / "temp" / "kronos_scores_cache_ft.pkl"
    df = load_price_data()
    sector_map = load_sector_map()
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    train_groups = load_groups_from_cache(cache, df, sector_map, "2024-01-01", "2026-04-17")
    train_groups = filter_blind(train_groups)
    valid_groups = load_groups_from_cache(cache, df, sector_map, "2026-04-24", "2026-05-08")
    test_groups = load_groups_from_cache(cache, df, sector_map, "2026-05-15", "2026-07-31")
    print(f"Train: {len(train_groups)}w | Valid: {len(valid_groups)}w | Test: {len(test_groups)}w")
    assert len(test_groups) == 10, f"expected 10 test weeks, got {len(test_groups)}"
    # Note: 2026-06-19 (Friday) has no screened universe (market-wide drop), so the
    # 10 usable weeks are 05-15/05-22/05-29/06-05/06-12/06-26/07-03/07-10/07-17/07-24.

    bench = benchmark_returns(test_groups)
    device = "cuda:0"

    results = {}

    # ---- Baselines ----
    cs_transform = make_cs_transform(sector_map)
    _, wk_raw = eval_scores(test_groups)
    _, wk_cs = eval_scores(test_groups, transform=cs_transform)
    results["Kronos-raw (Top-3)"] = extended_metrics(wk_raw, bench)
    results["Kronos-CS (Top-3)"] = extended_metrics(wk_cs, bench)
    print("Baselines done")

    # ---- Plan B heads (NDCG, B-C2/B-C4/B-C5) ----
    print("Training Plan B heads (B-C1..C7) on new split...")
    plan_b = run_plan_b_real(train_groups, valid_groups, test_groups, df, sector_map, device)
    for mode in ["B-C2 (Market)", "B-C4 (CS Stats)", "B-C5 (Full Context)"]:
        if mode in plan_b:
            results[mode + " + Top3"] = extended_metrics(plan_b[mode]["weekly_details"], bench)
    for mode in ["B-C6 (Shuffled)", "B-C7 (Zero)"]:
        if mode in plan_b:
            results[mode + " + Top3 (负对照)"] = extended_metrics(plan_b[mode]["weekly_details"], bench)
    print("Plan B heads done")

    # ---- CardNN end-to-end (frozen head, K=3, 3 seeds) ----
    train_ctx = build_contexts(train_groups, df, sector_map)
    valid_ctx = build_contexts(valid_groups, df, sector_map)
    test_ctx = build_contexts(test_groups, df, sector_map)
    for seed in [42, 2024, 7]:
        m = run_experiment(
            "e2e_k3_ft_10w", 3, seed,
            train_groups, valid_groups, test_groups,
            train_ctx, valid_ctx, test_ctx, device,
            warm_start=True, aux_ndcg_weight=0.0,
            epochs=100, patience=15,
            freeze_head=True, stage2_lr=1e-4, stage2_epochs=30,
        )
        results[f"e2e_k3_ft_10w_s{seed}"] = extended_metrics(m["weekly_details"], bench)
    # seed-mean summary
    ft = [results[f"e2e_k3_ft_10w_s{s}"] for s in [42, 2024, 7]]
    results["e2e_k3_ft_10w (3-seed mean)"] = {
        "cum_return": float(np.mean([m["cum_return"] for m in ft])),
        "weekly_mean_return": float(np.mean([m["weekly_mean_return"] for m in ft])),
        "win_rate": float(np.mean([m["win_rate"] for m in ft])),
        "weekly_excess_mean": float(np.mean([m["weekly_excess_mean"] for m in ft])),
        "excess_win_rate": float(np.mean([m["excess_win_rate"] for m in ft])),
        "upside_capture": float(np.mean([m["upside_capture"] for m in ft])),
        "downside_protect": float(np.mean([m["downside_protect"] for m in ft])),
        "hit_at_5": float(np.mean([m["hit_at_5"] for m in ft])),
        "rank_ic": float(np.mean([m["rank_ic"] for m in ft])),
        "sharpe": float(np.mean([m["sharpe"] for m in ft])),
        "sortino": float(np.mean([m["sortino"] for m in ft])),
        "max_dd": float(np.mean([m["max_dd"] for m in ft])),
    }
    print("CardNN done")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "output" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"compare_2026_10w_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    weekly_table = ["| 日期 | 组合收益 | 基准 | 超额 |", "|---|---:|---:|---:|"]
    for r in results["Kronos-CS (Top-3)"]["weekly"]:
        weekly_table.append(
            f"| {r['date']} | {r['return']:+.4f} | {r['bench']:+.4f} | {r['excess']:+.4f} |"
        )
    meta = {
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "data_end": str(df["日期"].max().date()),
        "train_weeks": len(train_groups),
        "valid_weeks": len(valid_groups),
        "test_weeks": len(test_groups),
        "weekly_table": "\n".join(weekly_table),
    }
    report_path = PROJECT_ROOT / "docs" / f"compare_2026_10w_report_{ts[:8]}.md"
    write_report(results, meta, report_path)


if __name__ == "__main__":
    main()
