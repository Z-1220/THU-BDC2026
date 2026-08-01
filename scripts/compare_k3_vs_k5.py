#!/usr/bin/env python
"""Equal-weight Top-3 vs Top-5 comparison for candidate strategies.

Windows: full test (2026-02-06..07-24), recent 10 weeks (05-15..07-24),
and the current state "老经济-轮动" (41 weeks).

Usage:
    uv run python scripts/compare_k3_vs_k5.py
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

_A = importlib.util.spec_from_file_location(
    "analyze_regime", PROJECT_ROOT / "scripts" / "analyze_regime_2026.py"
)
_amod = importlib.util.module_from_spec(_A)
_A.loader.exec_module(_amod)  # type: ignore[union-attr]
load_sector_map = _amod.load_sector_map

TECH_SECTOR = "信息技术"
OLD_SECTORS = ["主要消费", "金融", "房地产", "可选消费", "公用事业", "能源"]
BLIND_SIGNALS = {pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")}


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "stock_data.csv",
                     encoding="utf-8-sig", parse_dates=["日期"], dtype={"股票代码": str})
    df["code"] = df["股票代码"].str.zfill(6)
    sector_map = load_sector_map()
    opens = df.pivot_table(index="日期", columns="code", values="开盘", aggfunc="last").sort_index()
    closes = df.pivot_table(index="日期", columns="code", values="收盘", aggfunc="last").sort_index()
    trading_days = list(opens.index)
    daily_mean = closes.pct_change().mean(axis=1)
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
    dates = [d for d in dates if d not in BLIND_SIGNALS]

    def week_ret(dt):
        i = trading_days.index(dt)
        if i + 5 >= len(trading_days):
            return pd.Series(dtype=float)
        return (opens.loc[trading_days[i + 5]] / opens.loc[trading_days[i + 1]] - 1).dropna()

    records = []
    for di, dt in enumerate(dates):
        if dt not in trading_days or trading_days.index(dt) + 5 >= len(trading_days):
            continue
        scores = score_by_date[dt]
        codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in scores.index]
        score_map = {c: float(s) for c, s in zip(codes, scores.values)}
        wr = week_ret(dt)
        if len(wr) == 0:
            continue
        sec_ret = wr.groupby(wr.index.map(lambda c: sector_map.get(c, "未知"))).mean()
        this_winner = sec_ret.idxmax()
        prev_winner = None
        if di >= 1:
            pw = week_ret(dates[di - 1])
            if len(pw):
                ps = pw.groupby(pw.index.map(lambda c: sector_map.get(c, "未知"))).mean()
                if len(ps):
                    prev_winner = ps.idxmax()
        leader_repeat = int(prev_winner is not None and prev_winner == this_winner)
        tech_rel = float(ret20_sector.loc[dt, TECH_SECTOR] - ret20_sector.loc[dt, ["主要消费", "金融"]].mean())
        state = ("科技" if tech_rel > 0 else "老经济") + ("-趋势" if leader_repeat else "-轮动")
        records.append({
            "date": dt.date(), "state": state,
            "bench": float(wr.mean()), "wr": wr, "score": score_map, "codes": codes,
        })

    def pick(rec, name, K):
        wr, score, codes = rec["wr"], rec["score"], rec["codes"]
        dt = pd.Timestamp(rec["date"])
        if name == "kronos_raw":
            ranked = sorted(codes, key=lambda c: score[c], reverse=True)
        elif name == "kronos_cs":
            sec = {c: sector_map.get(c) for c in codes}
            s2 = pd.Series({c: score[c] for c in codes})
            med = s2.groupby(pd.Series([sec[c] for c in codes])).transform("median")
            s2 = s2 - med.fillna(0)
            lo, hi = s2.quantile(0.01), s2.quantile(0.99)
            s2 = s2.clip(lo, hi)
            s2 = zscore(s2)
            ranked = list(s2.sort_values(ascending=False).index)
        elif name == "tech_avoid":
            ranked = sorted([c for c in codes if sector_map.get(c, "") != TECH_SECTOR],
                            key=lambda c: score[c], reverse=True)
        elif name == "old_economy":
            ranked = sorted([c for c in codes if sector_map.get(c, "") in OLD_SECTORS],
                            key=lambda c: score[c], reverse=True)
        elif name == "rev_filter":
            cand = [c for c in codes if sector_map.get(c, "") != TECH_SECTOR
                    and c in ma60.columns and closes.loc[dt, c] > ma60.loc[dt, c]]
            ranked = sorted(cand, key=lambda c: ret20_stock.loc[dt, c])
        elif name == "blend_rev05":
            cand = [c for c in codes if sector_map.get(c, "") != TECH_SECTOR
                    and c in ma60.columns and closes.loc[dt, c] > ma60.loc[dt, c]]
            if not cand:
                ranked = []
            else:
                sc = pd.Series({c: score[c] for c in cand})
                rv = ret20_stock.loc[dt, cand].reindex(sc.index)
                bl = zscore(sc) + 0.5 * zscore(-rv)
                ranked = list(bl.sort_values(ascending=False).index)
        else:
            ranked = []
        picks = ranked[:K]
        if not picks:
            return float("nan")
        return float(wr.reindex(picks).mean())

    strategies = ["kronos_raw", "kronos_cs", "tech_avoid", "old_economy", "rev_filter", "blend_rev05"]
    windows = {
        "full_test(21w)": [r for r in records if pd.Timestamp("2026-02-06").date() <= r["date"] <= pd.Timestamp("2026-07-24").date()],
        "recent_10w": [r for r in records if pd.Timestamp("2026-05-15").date() <= r["date"] <= pd.Timestamp("2026-07-24").date()],
        "current_state(41w)": [r for r in records if r["state"] == "老经济-轮动"],
    }

    lines = ["# Top-3 vs Top-5 等权对比\n"]
    lines.append("基准: 全候选等权；指标为周均超额与超额胜率\n")
    lines.append("| 窗口 | 策略 | K=3 周均超额 | K=3 胜率 | K=5 周均超额 | K=5 胜率 | 差值(K3-K5) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    summary = []
    for wname, recs in windows.items():
        for st in strategies:
            e3, e5, n = [], [], 0
            for r in recs:
                b = r["bench"]
                v3 = pick(r, st, 3)
                v5 = pick(r, st, 5)
                if np.isfinite(v3) and np.isfinite(v5):
                    e3.append(v3 - b)
                    e5.append(v5 - b)
                    n += 1
            if n < 5:
                continue
            d = np.mean(e3) - np.mean(e5)
            lines.append(
                f"| {wname} | {st} | {np.mean(e3):+.4f} | {(np.array(e3) > 0).mean():.0%} | "
                f"{np.mean(e5):+.4f} | {(np.array(e5) > 0).mean():.0%} | {d:+.4f} |"
            )
            summary.append({"window": wname, "strategy": st, "n": n,
                            "k3_excess": np.mean(e3), "k3_win": (np.array(e3) > 0).mean(),
                            "k5_excess": np.mean(e5), "k5_win": (np.array(e5) > 0).mean(),
                            "diff": d})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "docs" / f"k3_vs_k5_{ts[:8]}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    with open(PROJECT_ROOT / "output" / "research" / f"k3_vs_k5_{ts}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n".join(lines))
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
