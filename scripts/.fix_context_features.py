"""Temporary one-off: normalize stock codes inside compute_context_features
(string vs int mismatch fix) — behavior-equivalent for the proxy path."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tp = ROOT / "code" / "src" / "run_research_experiments.py"
t = tp.read_text(encoding="utf-8")

old = '''def compute_context_features(df, signal_date, instruments, base_scores, sector_map, lookback=60):
    hist = df[df["日期"] <= signal_date]
    n = len(instruments)

    # Market features
    daily_close = hist.pivot_table(values="收盘", index="日期", columns="股票代码", aggfunc="last")
    daily_ret = daily_close.pct_change().mean(axis=1).dropna()
    if len(daily_ret) >= 20:
        mm5, mm20, mv20 = float(daily_ret.tail(5).mean()), float(daily_ret.tail(20).mean()), float(daily_ret.tail(20).std())
    else:
        mm5 = mm20 = mv20 = 0.0

    rets_5d = []
    for code in instruments:
        d = hist[hist["股票代码"] == code].sort_values("日期")
        rets_5d.append(float(d["收盘"].iloc[-1] / d["收盘"].iloc[-6] - 1) if len(d) >= 6 else 0.0)
    rets_5d = np.array(rets_5d)'''
new = '''def compute_context_features(df, signal_date, instruments, base_scores, sector_map, lookback=60):
    # Normalize stock codes to zero-padded strings (fixes int/str mismatches).
    df2 = df.assign(code_s=df["股票代码"].astype(str).str.zfill(6))
    hist = df2[df2["日期"] <= signal_date]
    instruments_n = [str(c).zfill(6) for c in instruments]
    n = len(instruments_n)

    # Market features
    daily_close = hist.pivot_table(values="收盘", index="日期", columns="code_s", aggfunc="last")
    daily_ret = daily_close.pct_change().mean(axis=1).dropna()
    if len(daily_ret) >= 20:
        mm5, mm20, mv20 = float(daily_ret.tail(5).mean()), float(daily_ret.tail(20).mean()), float(daily_ret.tail(20).std())
    else:
        mm5 = mm20 = mv20 = 0.0

    rets_5d = []
    for code in instruments_n:
        d = hist[hist["code_s"] == code].sort_values("日期")
        rets_5d.append(float(d["收盘"].iloc[-1] / d["收盘"].iloc[-6] - 1) if len(d) >= 6 else 0.0)
    rets_5d = np.array(rets_5d)'''
assert t.count(old) == 1, f"sec1 count={t.count(old)}"
t = t.replace(old, new)

old2 = '''    # Sector
    sectors = [sector_map.get(c, "unknown") for c in instruments]'''
new2 = '''    # Sector
    sectors = [sector_map.get(c, "unknown") for c in instruments_n]'''
assert t.count(old2) == 1, f"sec2 count={t.count(old2)}"
t = t.replace(old2, new2)

old3 = '''    scores_arr = np.array([base_scores.get(inst, 0.0) for inst in instruments], dtype=np.float32)'''
new3 = '''    scores_arr = np.array([base_scores.get(inst, 0.0) for inst in instruments_n], dtype=np.float32)'''
assert t.count(old3) == 1, f"sec3 count={t.count(old3)}"
t = t.replace(old3, new3)

old4 = '''    al = np.zeros(n, dtype=np.float32)
    tl = np.zeros(n, dtype=np.float32)
    for i, code in enumerate(instruments):
        d = hist[hist["股票代码"] == code].sort_values("日期")'''
new4 = '''    al = np.zeros(n, dtype=np.float32)
    tl = np.zeros(n, dtype=np.float32)
    for i, code in enumerate(instruments_n):
        d = hist[hist["code_s"] == code].sort_values("日期")'''
assert t.count(old4) == 1, f"sec4 count={t.count(old4)}"
t = t.replace(old4, new4)

tp.write_text(t, encoding="utf-8")
print("compute_context_features normalized")
