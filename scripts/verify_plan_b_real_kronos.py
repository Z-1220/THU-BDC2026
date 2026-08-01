#!/usr/bin/env python
"""Verify Plan B (Context Transformer + Full Context) on real Kronos scores.

Reproduces the Plan A (MSE / NDCG MLP heads) and Plan B (B-C1..B-C7 Context
Transformer modes) experiments from code/src/run_research_experiments.py, but
with cached Kronos scores (temp/kronos_scores_cache.pkl) instead of the
momentum proxy.  Also evaluates the pure-Kronos baselines (raw and CS-Z-score
standardized, Top-3 rank-weighted), then writes a Markdown report to docs/.

Usage:
    uv run python scripts/verify_plan_b_real_kronos.py \
        --cache temp/kronos_scores_cache.pkl \
        --test-start 2026-02-02 --test-end 2026-07-31
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---- Load the research runner module (no side effects) ----
_RUNNER_PATH = PROJECT_ROOT / "code" / "src" / "run_research_experiments.py"
_SPEC = importlib.util.spec_from_file_location("runner", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)  # type: ignore[union-attr]

ContextTransformer = runner.ContextTransformer
NDCGApproxLoss = runner.NDCGApproxLoss
build_mlp_head = runner.build_mlp_head
compute_metrics = runner.compute_metrics
compute_context_features = runner.compute_context_features
compute_labels = runner.compute_labels
evaluate_plan_a = runner.evaluate_plan_a
evaluate_plan_b = runner.evaluate_plan_b
_make_loss_fn = runner._make_loss_fn
compute_plan_a_loss = runner.compute_plan_a_loss
load_sector_map = runner.load_sector_map
load_price_data = runner.load_price_data


def to_csv_code(inst: str) -> str:
    if isinstance(inst, str) and inst.startswith(("SH", "SZ")):
        return inst[2:]
    return str(inst)


def load_groups_from_cache(
    cache: dict,
    df: pd.DataFrame,
    sector_map: dict[str, str],
    start: str,
    end: str,
) -> list[dict]:
    """Build per-signal-date groups {date, instruments, scores, labels}."""
    groups = []
    for (seg, dt), scores_series in sorted(cache.items(), key=lambda kv: kv[0][1]):
        dt = pd.Timestamp(dt)
        if not (start <= str(dt.date()) <= end):
            continue
        instruments_str = [to_csv_code(i) for i in scores_series.index]
        instruments_int = [int(c) for c in instruments_str]
        score_map = {int(to_csv_code(i)): float(v) for i, v in scores_series.items()}
        labels = compute_labels(df, dt, instruments_int)
        common = sorted(set(instruments_int) & set(labels))
        if len(common) < 5:
            continue
        str_by_int = {ci: cs for ci, cs in zip(instruments_int, instruments_str)}
        groups.append(
            {
                "date": dt,
                "instruments": [str_by_int[c] for c in common],
                "scores": np.array([score_map[c] for c in common], dtype=np.float64),
                "labels": np.array([labels[c] for c in common], dtype=np.float64),
            }
        )
    return groups


def make_cs_transform(sector_map: dict[str, str]):
    """Replicate KronosModel cs_zscore=True post-processing (sector-neutral +
    winsorize + z-score)."""

    def transform(scores: np.ndarray, instruments: list[str]) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64).copy()
        sectors = [sector_map.get(c, None) for c in instruments]
        sec_df = pd.DataFrame({"score": s, "sector": sectors})
        med = sec_df.groupby("sector")["score"].transform("median")
        known = sec_df["sector"].notna().to_numpy()
        s[known] = s[known] - med.to_numpy()[known]
        lo, hi = np.quantile(s, 0.01), np.quantile(s, 0.99)
        s = np.clip(s, lo, hi)
        m, std = s.mean(), s.std(ddof=0)
        if std > 1e-8:
            s = (s - m) / std
        return s

    return transform


def eval_scores(groups: list[dict], transform=None) -> tuple[dict, list[dict]]:
    """Evaluate a pure score signal with Top-3 rank-weighted portfolio."""
    from scipy.stats import spearmanr

    results = []
    for g in groups:
        s = g["scores"].copy()
        if transform is not None:
            s = transform(s, g["instruments"])
        order = np.argsort(s)[::-1]
        y = g["labels"]
        n = len(y)
        top3 = order[:3]
        w = np.array([0.333, 0.333, 0.334])
        wr = float(np.dot(y[top3], w[: len(top3)])) if n >= 3 else 0.0
        ric, _ = spearmanr(s, y) if n >= 3 else (0.0, 1.0)
        h5 = (
            len(set(order[:5]) & set(np.argsort(y)[::-1][:5])) / 5
            if n >= 5
            else 0.0
        )
        results.append(
            {
                "date": str(g["date"].date()),
                "n_stocks": n,
                "week_return": wr,
                "rank_ic": ric,
                "hit_at_5": h5,
            }
        )
    return compute_metrics(results), results


def run_plan_a_real(
    train_groups: list[dict],
    valid_groups: list[dict],
    test_groups: list[dict],
    device: str,
) -> dict:
    """Plan A key experiments (MSE + NDCG MLP heads) on real Kronos scores."""
    experiments = [
        ("A-E0 (MSE Baseline)", "mse", {}),
        ("A-E3 (NDCG Approx)", "ndcg", {"sigma": 1.0, "k": 5}),
    ]
    results = {}
    for exp_name, loss_type, loss_kwargs in experiments:
        print(f"\n=== {exp_name} ===")
        head = build_mlp_head(in_dim=1).to(device)
        loss_fn = _make_loss_fn(loss_type, loss_kwargs)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        best_loss = float("inf")
        patience = 0
        best_state = None

        for epoch in range(100):
            head.train()
            perm = torch.randperm(len(train_groups))
            for gid in perm:
                g = train_groups[gid]
                x = torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32).to(device)
                y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
                optimizer.zero_grad()
                pred = head(x).squeeze(-1)
                loss = compute_plan_a_loss(loss_fn, loss_type, loss_kwargs, pred, y, x, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for g in valid_groups:
                    x = torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32).to(device)
                    y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
                    pred = head(x).squeeze(-1)
                    val_loss += compute_plan_a_loss(
                        loss_fn, loss_type, loss_kwargs, pred, y, x, device
                    ).item()
            val_loss /= len(valid_groups)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            else:
                patience += 1
            if patience >= 15:
                break

        if best_state:
            head.load_state_dict(best_state)
        weekly = evaluate_plan_a(head, test_groups, device)
        metrics = compute_metrics(weekly)
        metrics["experiment"] = exp_name
        metrics["loss_type"] = loss_type
        metrics["weekly_details"] = weekly
        results[exp_name] = metrics
        print(
            f"  Sharpe: {metrics['sharpe']:.4f} | CumRet: {metrics['cum_return']:+.4f} | "
            f"Win%: {metrics['win_rate']:.1%} | RankIC: {metrics['rank_ic']:.4f} | "
            f"Hit@5: {metrics['hit_at_5']:.3f}"
        )
    return results


def run_plan_b_real(
    train_groups: list[dict],
    valid_groups: list[dict],
    test_groups: list[dict],
    df: pd.DataFrame,
    sector_map: dict[str, str],
    device: str,
) -> dict:
    """Plan B Context Transformer experiments (B-C1..B-C7) on real scores."""
    print("\nPre-computing context features...")
    train_ctx = [
        compute_context_features(
            df, g["date"], g["instruments"],
            dict(zip(g["instruments"], g["scores"])), sector_map,
        )
        for g in train_groups
    ]
    valid_ctx = [
        compute_context_features(
            df, g["date"], g["instruments"],
            dict(zip(g["instruments"], g["scores"])), sector_map,
        )
        for g in valid_groups
    ]
    test_ctx = [
        compute_context_features(
            df, g["date"], g["instruments"],
            dict(zip(g["instruments"], g["scores"])), sector_map,
        )
        for g in test_groups
    ]

    ndcg = NDCGApproxLoss(sigma=1.0, k=5).to(device)
    modes = {
        "B-C1 (No Context)": {"stock_cols": [0], "use_market": False, "desc": "Baseline: score only"},
        "B-C2 (Market)": {"stock_cols": [0], "use_market": True, "desc": "+Market via CLS token"},
        "B-C3 (Sector)": {"stock_cols": [0, 1], "use_market": False, "desc": "+Sector momentum"},
        "B-C4 (CS Stats)": {"stock_cols": [0, 2, 3], "use_market": False, "desc": "+CS rank & z-score"},
        "B-C5 (Full Context)": {"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "desc": "All features"},
        "B-C6 (Shuffled)": {"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "shuffle": True, "desc": "Shuffled (neg ctrl)"},
        "B-C7 (Zero)": {"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "zero": True, "desc": "Zeroed (neg ctrl)"},
    }
    results = {}
    for mode_name, cfg in modes.items():
        print(f"\n=== {mode_name}: {cfg['desc']} ===")
        sc = cfg["stock_cols"]
        um = cfg["use_market"]
        ds = cfg.get("shuffle", False)
        dz = cfg.get("zero", False)

        model = ContextTransformer(
            stock_feat_dim=len(sc),
            context_feat_dim=5 if um else 1,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.1,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        best_loss = float("inf")
        patience = 0
        best_state = None

        for epoch in range(100):
            model.train()
            perm = torch.randperm(len(train_groups))
            for gid in perm:
                g, ctx = train_groups[gid], train_ctx[gid]
                sf = ctx["stock_features"][:, sc].copy()
                mf = ctx["market_features"].copy() if um else np.zeros(1, dtype=np.float32)
                if ds and len(sf) > 1:
                    perm_idx = np.random.permutation(len(sf))
                    sf[:, 1:] = sf[perm_idx, 1:]
                if dz:
                    sf[:, 1:] = 0.0
                sf_t = torch.tensor(sf, dtype=torch.float32).unsqueeze(0).to(device)
                mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0).to(device)
                y_t = torch.tensor(g["labels"], dtype=torch.float32).unsqueeze(0).to(device)
                opt.zero_grad()
                pred = model(sf_t, mf_t)
                loss = ndcg(pred.squeeze(0), y_t.squeeze(0))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for gid, g in enumerate(valid_groups):
                    ctx = valid_ctx[gid]
                    sf = ctx["stock_features"][:, sc].copy()
                    mf = ctx["market_features"].copy() if um else np.zeros(1, dtype=np.float32)
                    sf_t = torch.tensor(sf, dtype=torch.float32).unsqueeze(0).to(device)
                    mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0).to(device)
                    y_t = torch.tensor(g["labels"], dtype=torch.float32).unsqueeze(0).to(device)
                    pred = model(sf_t, mf_t)
                    val_loss += ndcg(pred.squeeze(0), y_t.squeeze(0)).item()
            val_loss /= len(valid_groups)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= 15:
                break

        if best_state:
            model.load_state_dict(best_state)
        weekly = evaluate_plan_b(model, test_groups, test_ctx, sc, um, ds, dz, device)
        metrics = compute_metrics(weekly)
        metrics["experiment"] = mode_name
        metrics["context_mode"] = cfg["desc"]
        metrics["weekly_details"] = weekly
        results[mode_name] = metrics
        print(
            f"  Sharpe: {metrics['sharpe']:.4f} | CumRet: {metrics['cum_return']:+.4f} | "
            f"Win%: {metrics['win_rate']:.1%} | RankIC: {metrics['rank_ic']:.4f} | "
            f"Hit@5: {metrics['hit_at_5']:.3f}"
        )
    return results


def write_report(
    baselines: dict,
    plan_a: dict,
    plan_b: dict,
    meta: dict,
    out_path: Path,
) -> None:
    rows = []
    for name, m in list(baselines.items()) + list(plan_a.items()) + list(plan_b.items()):
        rows.append(
            f"| {name} | {m['weeks']} | {m['sharpe']:.4f} | {m['cum_return']:+.4f} | "
            f"{m['win_rate']:.1%} | {m['max_dd']:.4f} | {m['rank_ic']:.4f} | {m['hit_at_5']:.3f} |"
        )
    rows_text = "\n".join(rows)

    kronos_cs = baselines.get("Kronos-CS (Top-3)")
    bc5 = plan_b.get("B-C5 (Full Context)")
    verdict_parts = []
    if kronos_cs and bc5:
        ratio = bc5["sharpe"] / kronos_cs["sharpe"] if abs(kronos_cs["sharpe"]) > 1e-8 else float("nan")
        verdict_parts.append(
            f"B-C5 Sharpe = {bc5['sharpe']:.4f} vs 生产基线 Kronos-CS = {kronos_cs['sharpe']:.4f} "
            f"(比值 {ratio:.2f}x; Plan C 门槛为 1.3x)"
        )
    best_b = max(plan_b.items(), key=lambda kv: kv[1]["sharpe"]) if plan_b else (None, None)
    best_a = max(plan_a.items(), key=lambda kv: kv[1]["sharpe"]) if plan_a else (None, None)
    if best_b[0] and best_a[0]:
        verdict_parts.append(
            f"Plan B 最佳 {best_b[0]} (Sharpe {best_b[1]['sharpe']:.4f}) vs "
            f"Plan A 最佳 {best_a[0]} (Sharpe {best_a[1]['sharpe']:.4f})"
        )
    verdict_text = "\n".join(verdict_parts)

    report = f"""# Plan B 真 Kronos 验证报告

日期: {meta['timestamp']} | 数据截止: {meta['data_end']} | 评分来源: {meta['score_source']}

## 目的

验证研究计划 B 的结论在真实 Kronos 评分（而非 20 日动量代理）上是否成立：
B-C5（Context Transformer + Full Context + NDCG 损失）是否显著优于纯 Kronos
基线与 Plan A 最佳（NDCG / MSE MLP 头部），为 B 榜提交提供依据。

## 方法

- 评分来源: Kronos-small {meta['score_source']}，通过 {meta['cache_file']} 预计算缓存
- 股票池: FridayFilter + ScreenProcessor（成交额前 70%、MA60、回撤 15%）筛选后的横截面
- 信号窗口: 训练 {meta['train_window']} / 验证 {meta['valid_window']} / 测试 {meta['test_window']}
- 标签: 实际 5 日开盘收益 (open[T+5]-open[T+1])/open[T+1]，与赛题一致
- 组合评估: Top-3 rank-weighted [33.3%, 33.3%, 33.4%]
- 对比项:
  - Kronos-raw (Top-3): 缓存原始分数直接选股
  - Kronos-CS (Top-3): 行业中性化 + Winsorize + Z-score 后选股（= 生产基线）
  - Plan A: MSE / NDCG MLP 排序头部
  - Plan B: B-C1..B-C7 Context Transformer（NDCG 损失，per-date-group 训练）

## 结果

| 实验 | 周数 | Sharpe | 累计收益 | 胜率 | 最大回撤 | RankIC | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows_text}

## 结论

{verdict_text}
"""
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="temp/kronos_scores_cache.pkl")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-05")
    parser.add_argument("--valid-end", default="2026-01-30")
    parser.add_argument("--test-start", default="2026-02-02")
    parser.add_argument("--test-end", default="2026-07-31")
    parser.add_argument("--score-source", default="pretrained zero-shot")
    parser.add_argument("--out-dir", default="output/research")
    args = parser.parse_args()

    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    print("Loading price data...")
    df = load_price_data()
    sector_map = load_sector_map()
    print(f"stock_data.csv: {df['日期'].min().date()} ~ {df['日期'].max().date()}, "
          f"{df['股票代码'].nunique()} stocks")

    print(f"Loading Kronos score cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Cache entries: {len(cache)}")

    train_groups = load_groups_from_cache(cache, df, sector_map, args.train_start, args.train_end)
    valid_groups = load_groups_from_cache(cache, df, sector_map, args.valid_start, args.valid_end)
    test_groups = load_groups_from_cache(cache, df, sector_map, args.test_start, args.test_end)
    print(f"Train: {len(train_groups)}w | Valid: {len(valid_groups)}w | Test: {len(test_groups)}w")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Baselines ----
    cs_transform = make_cs_transform(sector_map)
    m_raw, _ = eval_scores(test_groups)
    m_raw["experiment"] = "Kronos-raw (Top-3)"
    m_cs, _ = eval_scores(test_groups, transform=cs_transform)
    m_cs["experiment"] = "Kronos-CS (Top-3)"
    baselines = {"Kronos-raw (Top-3)": m_raw, "Kronos-CS (Top-3)": m_cs}
    print(f"\nKronos-raw: Sharpe {m_raw['sharpe']:.4f} | CumRet {m_raw['cum_return']:+.4f}")
    print(f"Kronos-CS:  Sharpe {m_cs['sharpe']:.4f} | CumRet {m_cs['cum_return']:+.4f}")

    # ---- Plan A key experiments ----
    plan_a = run_plan_a_real(train_groups, valid_groups, test_groups, device)

    # ---- Plan B experiments ----
    plan_b = run_plan_b_real(
        train_groups, valid_groups, test_groups, df, sector_map, device
    )

    # ---- Save JSON ----
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"baselines": baselines, "plan_a": plan_a, "plan_b": plan_b}
    json_path = out_dir / f"plan_b_real_kronos_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # ---- Report ----
    meta = {
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "data_end": str(df["日期"].max().date()),
        "score_source": args.score_source,
        "cache_file": str(cache_path),
        "train_window": f"{args.train_start} ~ {args.train_end}",
        "valid_window": f"{args.valid_start} ~ {args.valid_end}",
        "test_window": f"{args.test_start} ~ {args.test_end}",
    }
    report_path = PROJECT_ROOT / "docs" / f"plan_b_real_kronos_verification_{ts[:8]}.md"
    write_report(baselines, plan_a, plan_b, meta, report_path)


if __name__ == "__main__":
    main()
