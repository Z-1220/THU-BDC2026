#!/usr/bin/env python
"""Compute the state -> best-K table for automatic aggressiveness.

For every historical week (head_s2024 refined scores), classify the market
state (科技/老经济 x 趋势/轮动) and measure the conditional weekly excess of
Top-K equal-weight portfolios for K in {2, 3, 5}. The state with the highest
conditional excess (min 3 weeks) is chosen per state; the table is saved to
model/auto_k_table.json and consumed by commit.py exposure_mode "auto".

Usage:
    uv run python scripts/compute_auto_k.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

_VSPEC = importlib.util.spec_from_file_location("verify", PROJECT_ROOT / "scripts" / "verify_plan_b_real_kronos.py")
verify = importlib.util.module_from_spec(_VSPEC)
_VSPEC.loader.exec_module(verify)
load_groups_from_cache = verify.load_groups_from_cache
load_price_data = verify.load_price_data
load_sector_map = verify.load_sector_map

_TSPEC = importlib.util.spec_from_file_location("train_head", PROJECT_ROOT / "scripts" / "train_context_head_d.py")
tmod = importlib.util.module_from_spec(_TSPEC)
_TSPEC.loader.exec_module(tmod)
build_features = tmod.build_features
refined_for = tmod.refined_for

_RSPEC = importlib.util.spec_from_file_location("runner", PROJECT_ROOT / "code" / "src" / "run_research_experiments.py")
runner = importlib.util.module_from_spec(_RSPEC)
_RSPEC.loader.exec_module(runner)
ContextTransformer = runner.ContextTransformer

TECH_SECTOR = "信息技术"
BLIND_SIGNALS = {pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")}


def main() -> None:
    df = load_price_data()
    df["code"] = df["股票代码"].astype(str).str.zfill(6)
    sector_map = load_sector_map()
    with open(PROJECT_ROOT / "temp" / "kronos_scores_cache_ft.pkl", "rb") as f:
        cache = pickle.load(f)

    groups = load_groups_from_cache(cache, df, sector_map, "2024-01-01", "2026-07-31")
    groups = [g for g in groups if g["date"] not in BLIND_SIGNALS]
    feats = build_features(groups, df, sector_map)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    head = ContextTransformer(
        stock_feat_dim=7, context_feat_dim=5, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, dropout=0.1, max_stocks=350,
    ).to(device)
    sd = torch.load(PROJECT_ROOT / "model" / "context_head" / "head_s2024.pt",
                    map_location="cpu", weights_only=False)
    head.load_state_dict(sd)

    # state features (same as Plan D / commit.py auto mode)
    closes = df.pivot_table(index="日期", columns="code", values="收盘", aggfunc="last").sort_index()
    ret20_stock = closes / closes.shift(20) - 1
    ret20_sector = pd.DataFrame(
        {s: ret20_stock[[c for c in ret20_stock.columns if sector_map.get(c, "") == s]].mean(axis=1)
         for s in set(sector_map.values())}
    )

    def leader_repeat(dt: pd.Timestamp) -> int:
        """Whether the last completed week's top sector repeats the prior week."""
        idx = closes.index
        pos = idx.get_loc(dt) if dt in idx else np.searchsorted(idx, dt) - 1
        # find last two Fridays with complete future data before dt
        fridays = [d for d in idx[:pos] if d.dayofweek == 4 and (idx.get_loc(d) + 5) < len(idx)]
        if len(fridays) < 2:
            return 0
        winners = []
        for f in fridays[-2:]:
            i = idx.get_loc(f)
            wr = (closes.iloc[i + 5] / closes.iloc[i + 1] - 1).dropna()
            sec = wr.groupby(wr.index.map(lambda c: sector_map.get(c, "未知"))).mean()
            winners.append(sec.idxmax() if len(sec) else None)
        return int(winners[0] is not None and winners[1] is not None and winners[0] == winners[1])

    rows = []
    for g, f in zip(groups, feats):
        dt = g["date"]
        if dt not in closes.index or closes.index.get_loc(dt) < 20:
            continue
        tech_rel = float(ret20_sector.loc[dt, TECH_SECTOR]
                         - ret20_sector.loc[dt, ["主要消费", "金融"]].mean())
        state = ("科技" if tech_rel > 0 else "老经济") + ("-趋势" if leader_repeat(dt) else "-轮动")
        refined = refined_for(head, f, device)
        y = g["labels"]
        bench = float(np.mean(y))
        order = np.argsort(refined)[::-1]
        for K in (2, 3, 5):
            topk = order[:K]
            w = np.full(K, 1.0 / K)
            ret = float(np.dot(y[topk], w[: len(topk)]))
            rows.append({"state": state, "K": K, "excess": ret - bench})
    rec = pd.DataFrame(rows)

    table = {}
    lines = ["# 自动激进度：状态 → 最优 K 表\n"]
    lines.append("| 状态 | K | 周数 | 周均超额 | 超额胜率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for state in sorted(rec["state"].unique()):
        sub = rec[rec["state"] == state]
        best = None
        for K in (2, 3, 5):
            s = sub[sub["K"] == K]["excess"].dropna()
            if len(s) < 3:
                continue
            lines.append(
                f"| {state} | {K} | {len(s)} | {s.mean():+.4f} | {(s > 0).mean():.0%} |"
            )
            if best is None or s.mean() > best[1]:
                best = (K, s.mean(), len(s))
        if best is not None:
            table[state] = {"best_k": best[0], "excess": best[1], "n": best[2]}
            lines.append(f"| **{state} → 最优** | **K={best[0]}** | {best[2]} | {best[1]:+.4f} | |")

    out = PROJECT_ROOT / "model" / "auto_k_table.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines.append(f"\n表已保存: model/auto_k_table.json")
    report = PROJECT_ROOT / "docs" / "auto_aggressiveness_20260801.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已保存: {report}")


if __name__ == "__main__":
    main()
