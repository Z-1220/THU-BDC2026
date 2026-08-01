#!/usr/bin/env python
"""Aggressive variants on the best single head (seed42): concentrated
allocations to chase excess return. Evaluates on the 10-week OOS window and
shows the 2026-07-31 portfolio per variant.

Usage:
    uv run python scripts/eval_aggressive.py
"""
from __future__ import annotations

import importlib.util
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


def variants_summary(refined_list, test_groups, weights_by_k):
    out = {}
    for name, (K, w) in weights_by_k.items():
        weekly = []
        for g, r in zip(test_groups, refined_list):
            y = g["labels"]
            topk = np.argsort(r)[::-1][:K]
            ww = np.array(w[:K])
            ww = ww / ww.sum()
            week_return = float(np.dot(y[topk], ww[: len(topk)]))
            bench = float(np.mean(y))
            weekly.append({"week_return": week_return, "bench": bench, "excess": week_return - bench})
        rets = np.array([x["week_return"] for x in weekly])
        exc = np.array([x["excess"] for x in weekly])
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(np.concatenate([[1.0], cum]))
        mdd = float((cum / peak[1:] - 1).min())
        out[name] = {
            "excess_mean": float(exc.mean()),
            "excess_win": float((exc > 0).mean()),
            "win_rate": float((rets > 0).mean()),
            "cum_return": float(np.prod(1 + rets) - 1),
            "p_exc_1pct": float((exc > 0.01).mean()),
            "best_week": float(rets.max()),
            "worst_week": float(rets.min()),
            "max_dd": mdd,
            "weekly": weekly,
        }
    return out


def main() -> None:
    df = load_price_data()
    df["code"] = df["股票代码"].astype(str).str.zfill(6)
    sector_map = load_sector_map()
    with open(PROJECT_ROOT / "temp" / "kronos_scores_cache_ft.pkl", "rb") as f:
        cache = pickle.load(f)

    test_groups = load_groups_from_cache(cache, df, sector_map, "2026-05-15", "2026-07-31")
    test_feat = build_features(test_groups, df, sector_map)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    head = ContextTransformer(
        stock_feat_dim=7, context_feat_dim=5, d_model=32, nhead=4,
        num_layers=2, dim_feedforward=64, dropout=0.1, max_stocks=350,
    ).to(device)
    sd = torch.load(PROJECT_ROOT / "model" / "context_head" / "head_s2024.pt",
                    map_location="cpu", weights_only=False)
    head.load_state_dict(sd)
    refined_list = [refined_for(head, f, device) for f in test_feat]

    variants = {
        "top3_equal": (3, [1 / 3, 1 / 3, 1 / 3]),
        "top3_conc": (3, [0.5, 0.3, 0.2]),
        "top2_equal": (2, [0.5, 0.5]),
        "top1": (1, [1.0]),
    }
    res = variants_summary(refined_list, test_groups, variants)

    # ---- 07-31 portfolio per variant ----
    dt = pd.Timestamp("2026-07-31")
    scores = cache.get(("test", dt))
    if scores is None:
        for (seg, d), s in cache.items():
            if pd.Timestamp(d) == dt and seg == "test":
                scores = s
                break
    codes = [c[2:] if c.startswith(("SH", "SZ")) else c for c in scores.index]
    pseudo = {"date": dt, "instruments": codes, "scores": scores.values,
              "labels": np.zeros(len(codes))}
    fake_feat = build_features([pseudo], df, sector_map)[0]
    r_final = refined_for(head, fake_feat, device)
    order = np.argsort(r_final)[::-1]
    picks = {codes[i]: r_final[i] for i in order}

    lines = ["# 激进策略评估（单 seed2024 头，10 周样本外）\n"]
    lines.append("| 变体 | 周均超额 | 超额胜率 | 胜率 | 累计 | P(>1%) | 最佳周 | 最差周 | 最大回撤 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in res.items():
        lines.append(
            f"| {name} | {m['excess_mean']:+.4f} | {m['excess_win']:.0%} | {m['win_rate']:.0%} | "
            f"{m['cum_return']:+.4f} | {m['p_exc_1pct']:.0%} | {m['best_week']:+.4f} | "
            f"{m['worst_week']:+.4f} | {m['max_dd']:.4f} |"
        )
    lines.append("\n## 07-31 各变体组合（分数从高到低）\n")
    for i in range(5):
        if i < len(order):
            lines.append(f"{i + 1}. {codes[order[i]]} (score {r_final[order[i]]:+.4f})")

    out = PROJECT_ROOT / "docs" / "aggressive_variants_20260801.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
