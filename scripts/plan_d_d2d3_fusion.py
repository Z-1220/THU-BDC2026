#!/usr/bin/env python
"""Plan D — Stages D2/D3: signal fusion within the current state
("老经济-轮动", n=41 weeks) with bootstrap confidence, plus 2026-07-31
candidate portfolios for the B-round decision.

Usage:
    uv run python scripts/plan_d_d2d3_fusion.py
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
    m, sd = s.mean(), s.std()
    return (s - m) / sd if sd > 1e-9 else s * 0


def top3_by(score: dict, key_fn) -> list[str]:
    return sorted(score, key=key_fn, reverse=True)[:3]


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "model" / "data" / "stock_data.csv",
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

    # ---- build state-labelled records (same as D1) ----
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
        bench = float(wr.mean())
        records.append({
            "date": dt.date(), "state": state, "tech_rel": tech_rel, "leader_repeat": leader_repeat,
            "mom20": float(daily_mean.loc[:dt].tail(20).mean()), "bench": bench,
            "wr": wr, "score": score_map, "codes": codes,
        })

    cur_recs = [r for r in records if r["state"] == "老经济-轮动"]
    print(f"当前状态周数: {len(cur_recs)}")

    # ---- candidate strategies (Top-3, cash scaling option) ----
    def strat_rets(r, name):
        wr, score, codes = r["wr"], r["score"], r["codes"]
        if name == "tech_avoid":
            picks = top3_by({c: score[c] for c in codes if sector_map.get(c, "") != TECH_SECTOR}, score.get)
        elif name == "old_economy":
            picks = top3_by({c: score[c] for c in codes if sector_map.get(c, "") in OLD_SECTORS}, score.get)
        elif name == "rev_filter":
            dt_r = pd.Timestamp(r["date"])
            cand = [c for c in codes if sector_map.get(c, "") != TECH_SECTOR
                    and c in ma60.columns and closes.loc[dt_r, c] > ma60.loc[dt_r, c]]
            picks = sorted(cand, key=lambda c: ret20_stock.loc[dt_r, c])[:3]
        elif name in ("blend_rev05", "blend_rev10"):
            dt = pd.Timestamp(r["date"])
            cand = [c for c in codes if sector_map.get(c, "") != TECH_SECTOR
                    and c in ma60.columns and closes.loc[dt, c] > ma60.loc[dt, c]]
            if not cand:
                picks = []
            else:
                w = 0.5 if name == "blend_rev05" else 1.0
                sc = pd.Series({c: score[c] for c in cand})
                rv = ret20_stock.loc[dt, cand].reindex(sc.index)
                bl = zscore(sc) + w * zscore(-rv)
                picks = list(bl.sort_values(ascending=False).index[:3])
        elif name == "sec_pos_nontech":
            dt = pd.Timestamp(r["date"])
            pos = [c for c in codes if sector_map.get(c, "") != TECH_SECTOR
                   and sector_map.get(c, "") in ret20_sector.columns
                   and ret20_sector.loc[dt, sector_map.get(c, "")] > 0]
            picks = top3_by({c: score[c] for c in pos}, score.get)
        else:  # kronos_raw
            picks = top3_by(score, score.get)
        if not picks:
            return float("nan"), picks
        return float(wr.reindex(picks).mean()), picks

    strategies = ["kronos_raw", "tech_avoid", "old_economy", "rev_filter", "blend_rev05", "blend_rev10", "sec_pos_nontech"]

    # ---- conditional stats + bootstrap (current state) ----
    lines = ["# Plan D — D2/D3 当前状态（老经济-轮动）信号融合与 bootstrap\n"]
    lines.append(f"状态周数: {len(cur_recs)} | 基准: 等权全候选\n")
    lines.append("| 策略 | 周均超额 | 超额胜率 | P(超额>1%) | 累计超额 | bootstrap 2.5% | 97.5% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    rows = []
    for st in strategies:
        rets, picks_list = [], []
        for r in cur_recs:
            v, pk = strat_rets(r, st)
            rets.append(v)
        exc = pd.Series(rets) - pd.Series([r["bench"] for r in cur_recs])
        exc = exc.dropna()
        if len(exc) < 5:
            continue
        rng = np.random.default_rng(42)
        boot = rng.choice(exc.to_numpy(), size=(2000, len(exc)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cum = float(np.prod(1 + pd.Series(rets).dropna()) - 1)
        cum_exc = float(np.prod(1 + exc) - 1)
        lines.append(
            f"| {st} | {exc.mean():+.4f} | {(exc > 0).mean():.0%} | {(exc > 0.01).mean():.0%} | "
            f"{cum_exc:+.4f} | {lo:+.4f} | {hi:+.4f} |"
        )
        rows.append({"strategy": st, "excess_mean": exc.mean(), "excess_win": (exc > 0).mean(),
                     "p_exc_1pct": (exc > 0.01).mean(), "cum_excess": cum_exc,
                     "boot_lo": lo, "boot_hi": hi, "n": len(exc), "cum": cum})

    # ---- 07-31 candidate portfolios ----
    dt = pd.Timestamp("2026-07-31")
    scores = score_by_date.get(dt)
    if scores is None:
        raise SystemExit("no 07-31 scores in cache")
    codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in scores.index]
    score_map = {c: float(s) for c, s in zip(codes, scores.values)}
    wr_dummy = pd.Series(dtype=float)
    fake = {"date": dt, "wr": wr_dummy, "score": score_map, "codes": codes, "bench": float("nan")}
    lines.append("\n## 2026-07-31 候选组合（策略 → 股票/行业）\n")
    lines.append("| 策略 | 股票 | 行业 |")
    lines.append("|---|---|---|")
    cand_picks = {}
    for st in strategies:
        _, picks = strat_rets(fake, st)
        cand_picks[st] = picks
        secs = "/".join(sector_map.get(c, "?") for c in picks)
        lines.append(f"| {st} | {','.join(picks) if picks else '-'} | {secs} |")

    lines.append("\n## 结论与建议\n")
    best = max(rows, key=lambda x: (x["excess_mean"], x["excess_win"]))
    lines.append(f"- 当前状态最佳策略: **{best['strategy']}** "
                 f"(周均超额 {best['excess_mean']:+.4f}, 胜率 {best['excess_win']:.0%}, "
                 f"bootstrap [{best['boot_lo']:+.4f}, {best['boot_hi']:+.4f}])")
    worst = min(rows, key=lambda x: x["excess_mean"])
    lines.append(f"- 当前状态最差策略: **{worst['strategy']}**（周均超额 {worst['excess_mean']:+.4f}）")
    lines.append(f"- 07-31 候选: {cand_picks}")

    report = "\n".join(lines)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / "docs" / f"d2d3_fusion_{ts[:8]}.md"
    out.write_text(report, encoding="utf-8")
    with open(PROJECT_ROOT / "output" / "research" / f"d2d3_fusion_{ts}.json", "w") as f:
        json.dump({"rows": rows, "candidates_0731": cand_picks}, f, indent=2, default=str)
    print(report)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
