#!/usr/bin/env python
"""Plan D — Stage D1: market-state definition (incl. sector rotation) and
state-conditioned weekly excess-return statistics for candidate strategies.

States (2x2):
  tech regime  : tech_rel = IT 20d return - mean(主要消费, 金融) 20d return
  leader regime: repeat = last week's top sector == this week's top sector
  => tech-leading/trend, tech-leading/rotation, old-economy/trend,
     old-economy/rotation

Strategies (all Top-3, no training needed):
  kronos_raw, kronos_cs, score_only(=raw), tech_avoid, sector_positive,
  sector_momentum, reversal_filtered

Usage:
    uv run python scripts/plan_d_state_conditioned.py
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import importlib.util

_A = importlib.util.spec_from_file_location(
    "analyze_regime", Path(__file__).resolve().parent / "analyze_regime_2026.py"
)
_amod = importlib.util.module_from_spec(_A)
_A.loader.exec_module(_amod)  # type: ignore[union-attr]
load_sector_map = _amod.load_sector_map

TECH_SECTOR = "信息技术"
OLD_SECTORS = ["主要消费", "金融"]
BLIND_START = pd.Timestamp("2026-04-13")
BLIND_END = pd.Timestamp("2026-04-17")


def weekly_returns(opens: pd.DataFrame, trading_days: list, signal_date: pd.Timestamp) -> pd.Series:
    """open[T+1] -> open[T+5] weekly return per code (vectorized)."""
    i = trading_days.index(signal_date)
    if i + 5 >= len(trading_days):
        return pd.Series(dtype=float)
    t1, t5 = trading_days[i + 1], trading_days[i + 5]
    r = opens.loc[t5] / opens.loc[t1] - 1
    return r.dropna()


def top3(score_map: dict, codes: list[str]) -> list[str]:
    sub = {c: score_map[c] for c in codes if c in score_map}
    return sorted(sub, key=sub.get, reverse=True)[:3]


def strat_return(codes: list[str], week_ret: pd.Series) -> float:
    if not codes or len(week_ret) == 0:
        return float("nan")
    return float(week_ret.reindex(codes).mean())


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "model" / "data" / "stock_data.csv",
                     encoding="utf-8-sig", parse_dates=["日期"], dtype={"股票代码": str})
    df["code"] = df["股票代码"].str.zfill(6)
    sector_map = load_sector_map()

    opens = df.pivot_table(index="日期", columns="code", values="开盘", aggfunc="last").sort_index()
    closes = df.pivot_table(index="日期", columns="code", values="收盘", aggfunc="last").sort_index()
    trading_days = list(opens.index)
    daily = closes.pct_change()
    daily_mean = daily.mean(axis=1)
    ret20_stock = closes / closes.shift(20) - 1
    ma60 = closes.rolling(60).mean()
    ret20_sector = pd.DataFrame(
        {s: ret20_stock[[c for c in ret20_stock.columns if sector_map.get(c, "") == s]].mean(axis=1)
         for s in set(sector_map.values())}
    )

    with open(PROJECT_ROOT / "temp" / "kronos_scores_cache_ft.pkl", "rb") as f:
        cache = pickle.load(f)
    score_by_date: dict[pd.Timestamp, pd.Series] = {}
    for (seg, dt), s in cache.items():
        score_by_date.setdefault(pd.Timestamp(dt), s)
    dates = sorted(score_by_date)
    blind_signals = {pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")}
    dates = [d for d in dates if d not in blind_signals]

    records = []
    for di, dt in enumerate(dates):
        if dt not in trading_days or trading_days.index(dt) + 5 >= len(trading_days):
            continue
        scores = score_by_date[dt]
        inst = list(scores.index)
        codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in inst]
        score_map = {c: float(s) for c, s in zip(codes, scores.values)}
        week_ret = weekly_returns(opens, trading_days, dt)
        bench = float(week_ret.mean()) if len(week_ret) else float("nan")

        # state features
        mom20 = float(daily_mean.loc[:dt].tail(20).mean()) if len(daily_mean.loc[:dt]) >= 20 else float("nan")
        vol20 = float(daily_mean.loc[:dt].tail(20).std()) if len(daily_mean.loc[:dt]) >= 20 else float("nan")
        breadth = float((closes.loc[dt] > ma60.loc[dt]).mean())
        tech_rel = float(ret20_sector.loc[dt, TECH_SECTOR] - ret20_sector.loc[dt, OLD_SECTORS].mean())

        # sector weekly returns this week
        sec_ret = week_ret.groupby(week_ret.index.map(lambda c: sector_map.get(c, "未知"))).mean()
        if len(sec_ret) == 0:
            continue
        this_winner = sec_ret.idxmax()
        prev_winner = None
        if di >= 1:
            prev_dt = dates[di - 1]
            prev_week_ret = weekly_returns(opens, trading_days, prev_dt)
            if len(prev_week_ret):
                prev_sec = prev_week_ret.groupby(prev_week_ret.index.map(lambda c: sector_map.get(c, "未知"))).mean()
                if len(prev_sec):
                    prev_winner = prev_sec.idxmax()
        leader_repeat = 1 if (prev_winner is not None and prev_winner == this_winner) else 0
        state = ("科技" if tech_rel > 0 else "老经济") + ("-趋势" if leader_repeat else "-轮动")

        # strategies
        raw_top = top3(score_map, codes)
        cs_map = {}
        sec_map = {c: sector_map.get(c) for c in codes}
        s_arr = np.array([score_map[c] for c in codes])
        s2 = s_arr.copy()
        sec_med = pd.Series(s2).groupby(pd.Series([sec_map[c] for c in codes])).transform("median")
        s2 = s2 - sec_med.fillna(0).to_numpy()
        lo, hi = np.quantile(s2, 0.01), np.quantile(s2, 0.99)
        s2 = np.clip(s2, lo, hi)
        m, sd = s2.mean(), s2.std()
        s2 = (s2 - m) / sd if sd > 1e-8 else s2
        cs_map = {c: float(v) for c, v in zip(codes, s2)}
        cs_top = top3(cs_map, codes)

        tech_avoid = top3(score_map, [c for c in codes if sector_map.get(c, "") != TECH_SECTOR])
        pos_sec = [c for c in codes if sector_map.get(c, "") in ret20_sector.columns and ret20_sector.loc[dt, sector_map.get(c, "")] > 0]
        sector_pos = top3(score_map, pos_sec)

        sm_codes = []
        if prev_winner is not None:
            sm_codes = top3(score_map, [c for c in codes if sector_map.get(c, "") == prev_winner])

        rev_candidates = [
            c for c in codes
            if c in ret20_stock.columns and c in ma60.columns
            and closes.loc[dt, c] > ma60.loc[dt, c] and np.isfinite(ret20_stock.loc[dt, c])
        ]
        rev_sorted = sorted(rev_candidates, key=lambda c: ret20_stock.loc[dt, c])[:3]

        records.append({
            "date": dt.date(), "state": state, "tech_rel": tech_rel,
            "leader_repeat": leader_repeat, "mom20": mom20, "vol20": vol20,
            "breadth": breadth, "bench": bench,
            "kronos_raw": strat_return(raw_top, week_ret),
            "kronos_cs": strat_return(cs_top, week_ret),
            "score_only": strat_return(raw_top, week_ret),
            "tech_avoid": strat_return(tech_avoid, week_ret),
            "sector_positive": strat_return(sector_pos, week_ret),
            "sector_momentum": strat_return(sm_codes, week_ret),
            "reversal_filtered": strat_return(rev_sorted, week_ret),
            "this_winner": this_winner, "prev_winner": prev_winner,
        })

    rec = pd.DataFrame(records)
    strategies = ["kronos_raw", "kronos_cs", "tech_avoid", "sector_positive", "sector_momentum", "reversal_filtered"]

    # ---- conditional stats per state ----
    lines = ["# Plan D — D1 状态条件化分析报告\n"]
    lines.append(f"数据范围: {rec['date'].min()} ~ {rec['date'].max()} | 周数: {len(rec)}\n")
    lines.append("## 状态分布\n")
    lines.append(rec["state"].value_counts().to_frame("周数").to_markdown())
    lines.append("\n## 各状态下的策略条件周超额（vs 等权基准）\n")
    lines.append("| 状态 | 策略 | 周数 | 周均超额 | 超额胜率 | 累计 | P(超额>1%) | 均值收益 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    cond_rows = []
    for state in sorted(rec["state"].unique()):
        sub = rec[rec["state"] == state]
        for st in strategies:
            exc = sub[st] - sub["bench"]
            exc = exc.dropna()
            if len(exc) < 3:
                continue
            lines.append(
                f"| {state} | {st} | {len(exc)} | {exc.mean():+.4f} | {(exc > 0).mean():.0%} | "
                f"{(np.prod(1 + sub[st].dropna()) - 1):+.4f} | {(exc > 0.01).mean():.0%} | "
                f"{sub[st].dropna().mean():+.4f} |"
            )
            cond_rows.append({"state": state, "strategy": st, "n": len(exc),
                              "excess_mean": exc.mean(), "excess_win": (exc > 0).mean(),
                              "p_exc_1pct": (exc > 0.01).mean()})

    # ---- current state (2026-07-31) ----
    dt = pd.Timestamp("2026-07-31")
    mom20 = float(daily_mean.loc[:dt].tail(20).mean())
    vol20 = float(daily_mean.loc[:dt].tail(20).std())
    breadth = float((closes.loc[dt] > ma60.loc[dt]).mean())
    tech_rel = float(ret20_sector.loc[dt, TECH_SECTOR] - ret20_sector.loc[dt, OLD_SECTORS].mean())
    lines.append(f"\n## 当前状态（2026-07-31）\n")
    lines.append(f"- mom20={mom20:+.4f}, vol20={vol20:.4f}, breadth(>MA60)={breadth:.1%}, tech_rel={tech_rel:+.4f}")
    lines.append(f"- 上周（07-24）领涨板块: {rec.iloc[-1]['this_winner']}, 前周: {rec.iloc[-1]['prev_winner']}")
    cur_state = ("科技" if tech_rel > 0 else "老经济") + ("-趋势" if rec.iloc[-1]["leader_repeat"] else "-轮动")
    lines.append(f"**当前状态判定: {cur_state}**\n")

    # conditional stats for the current state
    sub = rec[rec["state"] == cur_state]
    lines.append(f"## 当前状态（{cur_state}，n={len(sub)} 周）的策略排序\n")
    lines.append("| 策略 | 周均超额 | 超额胜率 | P(超额>1%) | 均值收益 |")
    lines.append("|---|---:|---:|---:|---:|")
    ranking = []
    for st in strategies:
        exc = (sub[st] - sub["bench"]).dropna()
        if len(exc) < 3:
            continue
        lines.append(
            f"| {st} | {exc.mean():+.4f} | {(exc > 0).mean():.0%} | {(exc > 0.01).mean():.0%} | "
            f"{sub[st].dropna().mean():+.4f} |"
        )
        ranking.append({"strategy": st, "excess_mean": exc.mean(), "excess_win": (exc > 0).mean(),
                        "p_exc_1pct": (exc > 0.01).mean(), "n": len(exc)})
    ranking.sort(key=lambda x: (x["excess_mean"], x["excess_win"]), reverse=True)
    lines.append(f"\n**当前状态最优策略: {ranking[0]['strategy']}**（周均超额 {ranking[0]['excess_mean']:+.4f}, "
                 f"胜率 {ranking[0]['excess_win']:.0%}, n={ranking[0]['n']}）\n")

    report = "\n".join(lines)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "docs" / f"d1_state_conditioned_{ts[:8]}.md"
    out.write_text(report, encoding="utf-8")
    with open(PROJECT_ROOT / "output" / "research" / f"d1_state_conditioned_{ts}.json", "w") as f:
        json.dump({"records": rec.to_dict("records"), "current_state": cur_state, "ranking": ranking}, f, indent=2, default=str)
    print(report)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
