#!/usr/bin/env python
"""Deep analysis of the 2026-05~07 market regime (sector rotation, style, picks).

Answers:
  1. Which sectors won/lost in the 10-week test window and in July 2026
     (tech selloff vs consumer / banks / baijiu strength)?
  2. Trend vs reversal: which style factor worked in June vs July?
  3. Sector-rotation persistence: does last week's winning sector repeat?
  4. What sectors did the baseline (Kronos Top-3) picks come from, and how
     would sector-momentum / reversal strategies have done instead?

Usage:
    uv run python scripts/analyze_regime_2026.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from src.run_research_experiments import compute_labels, get_friday_dates  # noqa: E402


def load_sector_map() -> dict[str, str]:
    p = PROJECT_ROOT / "resource" / "行业分类.csv"
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"证券代码": str})
    col = "中证一级行业分类简称"
    if col not in df.columns:
        col = "中证一级行业分类"
    return dict(zip(df["证券代码"], df[col]))


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "model" / "data" / "stock_data.csv",
                     encoding="utf-8-sig", parse_dates=["日期"], dtype={"股票代码": str})
    df["code"] = df["股票代码"].str.zfill(6)
    sector_map = load_sector_map()

    fridays = get_friday_dates(df, "2026-04-24", "2026-07-31")
    print("Fridays:", [d.strftime("%m-%d") for d in fridays])

    # ---- weekly stock returns (T+1 open -> T+5 open) ----
    week_rets: dict[pd.Timestamp, pd.Series] = {}
    for dt in fridays:
        codes = sorted(df[df["日期"] <= dt]["code"].unique())
        labels = compute_labels(df, dt, codes)
        week_rets[dt] = pd.Series(labels)

    # ---- sector weekly returns ----
    rows = []
    for dt, rets in week_rets.items():
        sec = rets.index.map(lambda c: sector_map.get(c, "未知"))
        sec_df = pd.DataFrame({"ret": rets.values, "sector": sec})
        agg = sec_df.groupby("sector")["ret"].agg(["mean", "count"])
        for s, r in agg.iterrows():
            rows.append({"date": dt.date(), "sector": s, "ret": r["mean"], "n": int(r["count"])})
    sec_weekly = pd.DataFrame(rows)

    # ---- market regime per week ----
    regime = []
    for dt, rets in week_rets.items():
        hist = df[df["日期"] <= dt]
        close_p = hist.pivot_table(values="收盘", index="日期", columns="code", aggfunc="last")
        daily = close_p.pct_change().mean(axis=1).dropna()
        mom20 = float(daily.tail(20).mean()) if len(daily) >= 20 else float("nan")
        vol20 = float(daily.tail(20).std()) if len(daily) >= 20 else float("nan")
        above_ma20 = float((close_p.iloc[-1] > close_p.iloc[-21:].mean()).mean()) if len(close_p) > 20 else float("nan")
        rets_s = rets.dropna()
        regime.append({
            "date": dt.date(),
            "bench_week": float(rets_s.mean()),
            "breadth": float((rets_s > 0).mean()),
            "dispersion": float(rets_s.std()),
            "mom20": mom20,
            "vol20": vol20,
            "above_ma20": above_ma20,
        })
    regime_df = pd.DataFrame(regime)

    # ---- style factors: 5d momentum vs 20d reversal IC ----
    style_rows = []
    for dt, rets in week_rets.items():
        hist = df[df["日期"] <= dt]
        ic = {"date": dt.date()}
        for lb, name in [(5, "mom5"), (20, "rev20")]:
            vals = []
            for c in rets.index:
                d = hist[hist["code"] == c].sort_values("日期")
                if len(d) >= lb + 1:
                    vals.append(d["收盘"].iloc[-1] / d["收盘"].iloc[-lb - 1] - 1)
                else:
                    vals.append(np.nan)
            f = pd.Series(vals, index=rets.index)
            common = rets.dropna().index.intersection(f.dropna().index)
            if len(common) > 5:
                ic[name] = float(rets[common].corr(f[common]))
            else:
                ic[name] = float("nan")
        style_rows.append(ic)
    style_df = pd.DataFrame(style_rows)

    # ---- simple strategies over the test window ----
    test_dates = [d for d in fridays if d.date() >= pd.Timestamp("2026-05-15").date()]
    cache_path = PROJECT_ROOT / "temp" / "kronos_scores_cache_ft.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    strat = []
    for dt in test_dates:
        # baseline picks: top-3 by raw Kronos score
        scores = None
        for (seg, d), s in cache.items():
            if pd.Timestamp(d) == dt and seg == "test":
                scores = s
                break
        if scores is None:
            continue
        codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in scores.index]
        score_vals = scores.values
        top3 = np.argsort(score_vals)[::-1][:3]
        top3_sectors = [sector_map.get(codes[i], "未知") for i in top3]
        rets = week_rets.get(dt, pd.Series(dtype=float))
        port_ret = float(rets.reindex([codes[i] for i in top3]).mean()) if len(rets) else float("nan")

        # sector-momentum strategy: buy top-3 stocks (by raw score) within the
        # best-performing sector of last week
        prev_date = (dt - pd.Timedelta(days=7)).date()
        sm = float("nan")
        if prev_date in set(sec_weekly["date"]):
            pw = sec_weekly[sec_weekly["date"] == prev_date].set_index("sector")["ret"]
            best_sec = pw.idxmax() if len(pw) else None
            if best_sec is not None:
                sec_codes = [c for c in codes if sector_map.get(c, "") == best_sec]
                if len(sec_codes) >= 3:
                    sub = {c: score_vals[i] for i, c in enumerate(codes) if c in sec_codes}
                    top = sorted(sub, key=sub.get, reverse=True)[:3]
                    sm = float(rets.reindex(top).mean()) if len(rets) else float("nan")

        # reversal strategy: buy top-3 20d laggards (with positive enough score?)
        hist = df[df["日期"] <= dt]
        rev = {}
        for c in codes:
            d = hist[hist["code"] == c].sort_values("日期")
            if len(d) >= 21:
                rev[c] = d["收盘"].iloc[-1] / d["收盘"].iloc[-21] - 1
        lag_top = sorted(rev, key=rev.get)[:3]
        rev_ret = float(rets.reindex(lag_top).mean()) if len(rets) else float("nan")

        bench = float(rets.mean())
        strat.append({
            "date": dt.date(),
            "bench": bench,
            "kronos_top3": port_ret,
            "kronos_top3_sectors": "/".join(top3_sectors),
            "sector_mom_top3": sm,
            "rev20_top3": rev_ret,
            "winning_sectors": "/".join(
                sec_weekly[sec_weekly["date"] == dt.date()]
                .sort_values("ret", ascending=False).head(3)["sector"].tolist()
            ),
            "losing_sectors": "/".join(
                sec_weekly[sec_weekly["date"] == dt.date()]
                .sort_values("ret").head(3)["sector"].tolist()
            ),
        })
    strat_df = pd.DataFrame(strat)

    # ---- report ----
    lines = []
    lines.append("# 2026 年 5-7 月市场状态与板块轮动分析\n")
    lines.append("## 行业周度收益（测试窗口）\n")
    pivot = sec_weekly.pivot_table(index="sector", columns="date", values="ret")
    pivot = pivot.loc[pivot.iloc[:, -4:].mean(axis=1).sort_values(ascending=False).index]
    fmt = pivot.copy()
    for c in fmt.columns:
        fmt[c] = fmt[c].map(lambda x: f"{x:+.3f}")
    lines.append(fmt.to_markdown())
    lines.append("\n## 市场状态逐周\n")
    lines.append(regime_df.round(4).to_markdown(index=False))
    lines.append("\n## 风格因子 IC（5 日动量 / 20 日反转 vs 下周收益）\n")
    lines.append(style_df.round(3).to_markdown(index=False))
    lines.append("\n## 策略对比（测试窗口）\n")
    lines.append(strat_df.round(4).to_markdown(index=False))

    # ---- July cumulative by sector (07-03 .. 07-24) ----
    jul = sec_weekly[sec_weekly["date"] >= pd.Timestamp("2026-07-01").date()]
    jul_cum = jul.groupby("sector")["ret"].agg(lambda x: float(np.prod(1 + x) - 1)).sort_values(ascending=False)
    jul_cum_txt = "\n".join(f"| {s} | {v:+.4f} |" for s, v in jul_cum.items())

    # ---- winner persistence: does last week's top sector repeat? ----
    persist = []
    dates = sorted(sec_weekly["date"].unique())
    for d in dates[1:]:
        prev_d = dates[dates.index(d) - 1]
        prev_win = sec_weekly[sec_weekly["date"] == prev_d].sort_values("ret", ascending=False).iloc[0]["sector"]
        cur_win = sec_weekly[sec_weekly["date"] == d].sort_values("ret", ascending=False).iloc[0]["sector"]
        persist.append({"date": d, "prev_winner": prev_win, "this_winner": cur_win, "repeat": prev_win == cur_win})
    persist_df = pd.DataFrame(persist)

    lines.append("\n## 7 月行业累计收益（07-03 ~ 07-24）\n")
    lines.append("| 行业 | 累计 |")
    lines.append("|---:|---:|")
    lines.append(jul_cum_txt)
    lines.append("\n## 周度领涨板块持续性\n")
    lines.append(persist_df.to_markdown(index=False))

    # summary stats
    if len(strat_df):
        def agg(col):
            s = strat_df[col].dropna()
            return f"{s.mean():+.4f} (胜率 {(s > 0).mean():.0%}, n={len(s)})"
        lines.append("\n## 汇总\n")
        lines.append(f"- 基准(等权): {agg('bench')}")
        lines.append(f"- Kronos Top-3: {agg('kronos_top3')}")
        lines.append(f"- 板块动量 Top-3: {agg('sector_mom_top3')}")
        lines.append(f"- 20日反转 Top-3: {agg('rev20_top3')}")
        lines.append(f"- 超额(基准=0): Kronos {strat_df['kronos_top3'].mean() - strat_df['bench'].mean():+.4f} | "
                     f"板块动量 {strat_df['sector_mom_top3'].mean() - strat_df['bench'].mean():+.4f} | "
                     f"反转 {strat_df['rev20_top3'].mean() - strat_df['bench'].mean():+.4f}")

    report = "\n".join(lines)
    out = PROJECT_ROOT / "docs" / "regime_analysis_2026.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
