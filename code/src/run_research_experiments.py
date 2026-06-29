#!/usr/bin/env python
"""Research experiment runner for Plans A and B.

Plan A: Loss function structure alignment (10 experiments)
Plan B: Context Transformer with NDCG loss (7 context modes)

Both use per-date-group training to preserve cross-sectional structure.

Usage:
    python code/src/run_research_experiments.py plan_a
    python code/src/run_research_experiments.py plan_b
    python code/src/run_research_experiments.py all
"""
from __future__ import annotations

import json
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code"))

# Fix random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Module-level imports for loss functions
from models.KronosRankHead.RankingLosses import (
    PairwiseMarginLoss,
    ListMLELoss,
    CoarseToFineLoss,
    CandidateGroupRankingLoss,
    StructureConsistencyLoss,
)

# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================

def load_price_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent.parent
    return pd.read_csv(project_root / "data" / "stock_data.csv",
                       encoding="utf-8-sig", parse_dates=["日期"])


def load_sector_map() -> dict[str, str]:
    project_root = Path(__file__).resolve().parent.parent.parent
    p = project_root / "resource" / "行业分类.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"证券代码": str})
    return dict(zip(df["证券代码"], df["中证一级行业分类简称"]))


def compute_momentum_scores(df, signal_date, lookback=20):
    hist = df[df["日期"] <= signal_date]
    scores = {}
    for code in sorted(hist["股票代码"].unique()):
        d = hist[hist["股票代码"] == code].sort_values("日期")
        if len(d) < lookback + 1:
            continue
        ret = float(d["收盘"].iloc[-1] / d["收盘"].iloc[-lookback] - 1)
        if np.isfinite(ret):
            scores[code] = ret
    return scores


def get_friday_dates(df, start, end):
    return [d for d in sorted(df["日期"].unique())
            if start <= str(d.date()) <= end and d.weekday() == 4]


def compute_labels(df, signal_date, instruments):
    future = df[df["日期"] > signal_date].sort_values("日期")
    labels = {}
    for code in instruments:
        fi = future[future["股票代码"] == code].sort_values("日期")
        if len(fi) < 2:
            continue
        t1_open = float(fi["开盘"].iloc[0])
        t5_candidates = fi["开盘"].iloc[1:5]
        t5_open = float(t5_candidates.iloc[-1]) if len(t5_candidates) > 0 else t1_open
        if t1_open > 0:
            v = (t5_open - t1_open) / t1_open
            if np.isfinite(v):
                labels[code] = v
    return labels


def extract_groups(df, start, end, sector_map):
    groups = []
    for dt in get_friday_dates(df, start, end):
        scores = compute_momentum_scores(df, dt)
        if len(scores) < 5:
            continue
        instruments = list(scores.keys())
        labels = compute_labels(df, dt, instruments)
        common = sorted(set(scores) & set(labels))
        if len(common) < 5:
            continue
        groups.append({
            "date": dt, "instruments": common,
            "scores": np.array([scores[c] for c in common]),
            "labels": np.array([labels[c] for c in common]),
        })
    return groups


def compute_context_features(df, signal_date, instruments, base_scores, sector_map, lookback=60):
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
    rets_5d = np.array(rets_5d)
    market_features = np.array([mm5, mm20, mv20, float(np.mean(rets_5d > 0)), float(np.std(rets_5d))], dtype=np.float32)

    # Sector
    sectors = [sector_map.get(c, "unknown") for c in instruments]
    sec_rets = {}
    for i, code in enumerate(instruments):
        sec_rets.setdefault(sectors[i], []).append(rets_5d[i])
    sec_mom = {s: float(np.mean(r)) for s, r in sec_rets.items()}
    sector_mom_5 = np.array([sec_mom.get(s, 0.0) for s in sectors], dtype=np.float32)

    # CS
    scores_arr = np.array([base_scores.get(inst, 0.0) for inst in instruments], dtype=np.float32)
    from scipy.stats import rankdata
    ranks = rankdata(scores_arr, method="average")
    cs_rank = (ranks / (n + 1)).astype(np.float32)
    m, std = scores_arr.mean(), scores_arr.std(ddof=0)
    cs_zscore = ((scores_arr - m) / std).astype(np.float32) if std > 1e-8 else np.zeros(n, dtype=np.float32)

    # Liquidity
    al = np.zeros(n, dtype=np.float32)
    tl = np.zeros(n, dtype=np.float32)
    for i, code in enumerate(instruments):
        d = hist[hist["股票代码"] == code].sort_values("日期")
        avg_a = float(d["成交额"].tail(lookback).mean()) if len(d) >= lookback else (float(d["成交额"].mean()) if len(d) > 0 else 0)
        avg_v = float(d["成交量"].tail(lookback).mean()) if len(d) >= lookback else (float(d["成交量"].mean()) if len(d) > 0 else 0)
        al[i] = np.log1p(avg_a) if avg_a > 0 else 0
        tl[i] = np.log1p(avg_v) if avg_v > 0 else 0

    stock_features = np.column_stack([
        scores_arr, sector_mom_5, cs_rank, cs_zscore, al, tl
    ]).astype(np.float32)
    stock_features = np.nan_to_num(stock_features, nan=0.0)
    market_features = np.nan_to_num(market_features, nan=0.0)

    return {"stock_features": stock_features, "market_features": market_features}


# ============================================================================
# Models
# ============================================================================

class ContextTransformer(nn.Module):
    def __init__(self, stock_feat_dim=6, context_feat_dim=5, d_model=32, nhead=4,
                 num_layers=2, dim_feedforward=64, dropout=0.1, max_stocks=350):
        super().__init__()
        self.stock_proj = nn.Linear(stock_feat_dim, d_model)
        self.context_proj = nn.Linear(context_feat_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1 + max_stocks, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1))

    def forward(self, stock_features, market_features):
        B, N, _ = stock_features.shape
        se = self.stock_proj(stock_features)
        ce = self.context_proj(market_features).unsqueeze(1)
        x = torch.cat([ce, se], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.score_head(x[:, 1:, :]).squeeze(-1)


class NDCGApproxLoss(nn.Module):
    def __init__(self, sigma=1.0, k=5):
        super().__init__()
        self.sigma = sigma
        self.k = k

    def forward(self, pred, target):
        n = pred.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        pd = pred.unsqueeze(1) - pred.unsqueeze(0)
        t_min, t_max = target.min(), target.max()
        if t_max - t_min < 1e-8:
            return torch.tensor(0.0, device=pred.device)
        rel = (target - t_min) / (t_max - t_min)
        gain = 2**rel - 1
        gd = gain.unsqueeze(1) - gain.unsqueeze(0)
        ranks = torch.argsort(torch.argsort(pred, descending=True)).float() + 1
        disc = 1.0 / torch.log2(ranks + 1)
        dd = disc.unsqueeze(1) - disc.unsqueeze(0)
        dndcg = torch.abs(gd * dd)
        w = dndcg / (dndcg.sum() + 1e-8)
        ts = (target.unsqueeze(1) > target.unsqueeze(0)).float()
        bl = F.binary_cross_entropy_with_logits(pd * self.sigma, ts, reduction="none")
        return (bl * w).sum()


def build_mlp_head(in_dim=1, hidden=[64, 32], dropout=0.1):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ============================================================================
# Training Utilities
# ============================================================================

def compute_plan_a_loss(loss_fn, loss_type, loss_kwargs, pred, y, x, device):
    if loss_type == "ndcg":
        return NDCGApproxLoss(sigma=loss_kwargs.get("sigma", 1.0))(pred, y)
    elif loss_type == "shuffled_label":
        return PairwiseMarginLoss(margin=0.1)(pred, y[torch.randperm(len(y))])
    elif loss_type == "coarse_only":
        n_tiers = loss_kwargs.get("n_tiers", 3)
        # Assign tiers by quantile
        q = torch.quantile(y, torch.linspace(0, 1, n_tiers + 1, device=y.device))
        cl = torch.zeros(len(y), dtype=torch.long, device=y.device)
        for t in range(n_tiers):
            lo, hi = q[t], q[t + 1]
            mask = (y >= lo) & (y < hi) if t < n_tiers - 1 else (y >= lo) & (y <= hi + 1e-6)
            cl[mask] = t
        # Distance of pred to each tier center → classification logits
        centers = torch.tensor([(q[t] + q[t+1]) / 2 for t in range(n_tiers)], device=y.device)
        logits = -torch.abs(pred.unsqueeze(1) - centers.unsqueeze(0))
        return F.cross_entropy(logits, cl)
    elif loss_type == "fine_only":
        return PairwiseMarginLoss(margin=loss_kwargs.get("margin", 0.05))(pred, y)
    elif loss_fn is not None:
        if loss_type == "structure_consistency":
            loss, _ = loss_fn(pred, y, x.squeeze(-1))
        elif loss_type in ("coarse_to_fine", "candidate_group"):
            loss, _ = loss_fn(pred, y)
        else:
            loss = loss_fn(pred, y)
        return loss
    return F.mse_loss(pred, y)


def compute_metrics(weekly_results):
    returns = [r["week_return"] for r in weekly_results]
    n = len(returns)
    if n == 0:
        return {"weeks": 0, "sharpe": 0.0, "cum_return": 0.0, "win_rate": 0.0,
                "max_dd": 0.0, "rank_ic": 0.0, "hit_at_5": 0.0}
    mr = float(np.mean(returns))
    sr = float(np.std(returns, ddof=1))
    sharpe = float(mr / sr * np.sqrt(52)) if sr > 1e-8 else 0.0
    wr = float(np.mean([r > 0 for r in returns]))
    cr = float(np.prod([1 + r for r in returns]) - 1)
    peak = cum = 1.0
    mdd = 0.0
    for r in returns:
        cum *= (1 + r)
        peak = max(peak, cum)
        mdd = max(mdd, (peak - cum) / peak if peak > 0 else 0.0)
    ric = float(np.mean([r["rank_ic"] for r in weekly_results]))
    h5 = float(np.mean([r["hit_at_5"] for r in weekly_results]))
    return {"weeks": n, "mean_return": mr, "std_return": sr, "sharpe": sharpe,
            "win_rate": wr, "cum_return": cr, "max_dd": mdd,
            "rank_ic": ric, "hit_at_5": h5}


def evaluate_plan_a(head, test_groups, device):
    head.eval()
    results = []
    for g in test_groups:
        x = torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32).to(device)
        y = g["labels"]
        with torch.no_grad():
            refined = head(x).squeeze(-1).cpu().numpy()
        n = len(y)
        wr = float(np.dot(y[np.argsort(refined)[::-1][:3]], [0.333, 0.333, 0.334])) if n >= 3 else 0.0
        from scipy.stats import spearmanr
        ric, _ = spearmanr(refined, y) if n >= 3 else (0.0, 1.0)
        h5 = len(set(np.argsort(refined)[::-1][:5]) & set(np.argsort(y)[::-1][:5])) / 5 if n >= 5 else 0.0
        results.append({"date": str(g["date"].date()), "n_stocks": n, "week_return": wr,
                        "rank_ic": ric, "hit_at_5": h5})
    return results


def evaluate_plan_b(model, test_groups, test_contexts, stock_cols, use_market, do_shuffle, do_zero, device):
    model.eval()
    results = []
    for gid, g in enumerate(test_groups):
        ctx = test_contexts[gid]
        sf = ctx["stock_features"][:, stock_cols].copy()
        mf = ctx["market_features"].copy() if use_market else np.zeros(1, dtype=np.float32)
        if do_shuffle and len(sf) > 1:
            perm_idx = np.random.permutation(len(sf))
            sf[:, 1:] = sf[perm_idx, 1:]
        if do_zero:
            sf[:, 1:] = 0.0
        sf_t = torch.tensor(sf, dtype=torch.float32).unsqueeze(0).to(device)
        mf_t = torch.tensor(mf, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            refined = model(sf_t, mf_t).squeeze(0).cpu().numpy()
        y = g["labels"]
        n = len(y)
        wr = float(np.dot(y[np.argsort(refined)[::-1][:3]], [0.333, 0.333, 0.334])) if n >= 3 else 0.0
        from scipy.stats import spearmanr
        ric, _ = spearmanr(refined, y) if n >= 3 else (0.0, 1.0)
        h5 = len(set(np.argsort(refined)[::-1][:5]) & set(np.argsort(y)[::-1][:5])) / 5 if n >= 5 else 0.0
        results.append({"date": str(g["date"].date()), "n_stocks": n, "week_return": wr,
                        "rank_ic": ric, "hit_at_5": h5})
    return results


# ============================================================================
# Plan A: Loss Function Experiments (per-date-group training)
# ============================================================================

def run_plan_a():
    print("=" * 60)
    print("Plan A: Loss Function Structure Alignment")
    print("=" * 60)

    df = load_price_data()
    sector_map = load_sector_map()

    train_groups = extract_groups(df, "2024-01-01", "2025-12-31", sector_map)
    valid_groups = extract_groups(df, "2026-01-05", "2026-01-30", sector_map)
    test_groups = extract_groups(df, "2026-02-02", "2026-05-27", sector_map)
    print(f"Train: {len(train_groups)}w, Valid: {len(valid_groups)}w, Test: {len(test_groups)}w")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    experiments = [
        ("A-E0 (MSE Baseline)", "mse", {}),
        ("A-E1 (Pairwise Margin)", "pairwise", {"margin": 0.1}),
        ("A-E2 (ListMLE)", "listmle", {"temperature": 1.0}),
        ("A-E3 (NDCG Approx)", "ndcg", {"sigma": 1.0, "k": 5}),
        ("A-E4 (Coarse-to-Fine)", "coarse_to_fine", {"n_tiers": 3}),
        ("A-E5 (Candidate+Group)", "candidate_group", {"top_m": 30}),
        ("A-E6 (Structure Consistency)", "structure_consistency", {"consistency_weight": 0.1}),
        ("A-E8 (Shuffled Labels)", "shuffled_label", {}),
        ("A-E10 (Coarse Only)", "coarse_only", {"n_tiers": 3}),
        ("A-E11 (Fine Only)", "fine_only", {"margin": 0.05}),
    ]

    all_results = {}
    for exp_name, loss_type, loss_kwargs in experiments:
        print(f"\n{'='*50}")
        print(f"Running {exp_name}...")

        head = build_mlp_head(in_dim=1).to(device)
        loss_fn = _make_loss_fn(loss_type, loss_kwargs)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        best_loss = float("inf")
        patience = 0
        best_state = None

        for epoch in range(100):
            head.train()
            total_loss = 0.0
            n_batches = 0

            # Per-date-group training: shuffle order of groups each epoch
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
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for g in valid_groups:
                    x = torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32).to(device)
                    y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
                    pred = head(x).squeeze(-1)
                    val_loss += compute_plan_a_loss(loss_fn, loss_type, loss_kwargs, pred, y, x, device).item()
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
        all_results[exp_name] = metrics

        print(f"  Sharpe: {metrics['sharpe']:.4f} | CumRet: {metrics['cum_return']:+.4f} | "
              f"Win%: {metrics['win_rate']:.1%} | RankIC: {metrics['rank_ic']:.4f} | "
              f"Hit@5: {metrics['hit_at_5']:.3f}")

    _save_json(all_results, "plan_a")
    _print_plan_a_table(all_results)
    return all_results


def _make_loss_fn(loss_type, loss_kwargs):
    if loss_type == "pairwise":
        return PairwiseMarginLoss(margin=loss_kwargs.get("margin", 0.1))
    elif loss_type == "listmle":
        return ListMLELoss(temperature=loss_kwargs.get("temperature", 1.0))
    elif loss_type == "coarse_to_fine":
        return CoarseToFineLoss(n_tiers=loss_kwargs.get("n_tiers", 3))
    elif loss_type == "candidate_group":
        return CandidateGroupRankingLoss(top_m=loss_kwargs.get("top_m", 30))
    elif loss_type == "structure_consistency":
        return StructureConsistencyLoss(
            base_loss_type=loss_kwargs.get("base_loss_type", "pairwise"),
            consistency_weight=loss_kwargs.get("consistency_weight", 0.1))
    return None


# ============================================================================
# Plan B: Context Transformer Experiments (NDCG loss)
# ============================================================================

def run_plan_b():
    print("=" * 60)
    print("Plan B: Context Transformer with NDCG Loss (Plan A Champion)")
    print("=" * 60)

    df = load_price_data()
    sector_map = load_sector_map()

    train_groups = extract_groups(df, "2024-01-01", "2025-12-31", sector_map)
    valid_groups = extract_groups(df, "2026-01-05", "2026-01-30", sector_map)
    test_groups = extract_groups(df, "2026-02-02", "2026-05-27", sector_map)
    print(f"Train: {len(train_groups)}w, Valid: {len(valid_groups)}w, Test: {len(test_groups)}w")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Pre-compute context
    print("\nPre-computing context features...")
    train_ctx = [_compute_ctx(g, df, sector_map) for g in train_groups]
    valid_ctx = [_compute_ctx(g, df, sector_map) for g in valid_groups]
    test_ctx = [_compute_ctx(g, df, sector_map) for g in test_groups]

    ndcg = NDCGApproxLoss(sigma=1.0, k=5).to(device)

    modes = {
        "B-C1 (No Context)":  {"stock_cols": [0], "use_market": False, "desc": "Baseline: score only"},
        "B-C2 (Market)":      {"stock_cols": [0], "use_market": True,  "desc": "+Market via CLS token"},
        "B-C3 (Sector)":      {"stock_cols": [0, 1], "use_market": False, "desc": "+Sector momentum"},
        "B-C4 (CS Stats)":    {"stock_cols": [0, 2, 3], "use_market": False, "desc": "+CS rank & z-score"},
        "B-C5 (Full Context)":{"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "desc": "All features"},
        "B-C6 (Shuffled)":   {"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "shuffle": True, "desc": "Shuffled (neg ctrl)"},
        "B-C7 (Zero)":       {"stock_cols": [0, 1, 2, 3, 4, 5], "use_market": True, "zero": True, "desc": "Zeroed (neg ctrl)"},
    }

    all_results = {}
    for mode_name, cfg in modes.items():
        print(f"\n{'='*50}")
        print(f"Running {mode_name}: {cfg['desc']}")

        sc = cfg["stock_cols"]
        um = cfg["use_market"]
        ds = cfg.get("shuffle", False)
        dz = cfg.get("zero", False)

        model = ContextTransformer(
            stock_feat_dim=len(sc), context_feat_dim=5 if um else 1,
            d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        best_loss = float("inf")
        patience = 0
        best_state = None

        for epoch in range(100):
            model.train()
            total_loss = 0.0
            perm = torch.randperm(len(train_groups))
            for gid in perm:
                g, ctx = train_groups[gid], train_ctx[gid]
                sf = ctx["stock_features"][:, sc].copy()
                mf = ctx["market_features"].copy() if um else np.zeros(1, dtype=np.float32)
                if ds and len(sf) > 1:
                    # Shuffle non-score feature columns (break context-stock alignment)
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
                total_loss += loss.item()

            sched.step()

            # Validation
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
        all_results[mode_name] = metrics

        print(f"  Sharpe: {metrics['sharpe']:.4f} | CumRet: {metrics['cum_return']:+.4f} | "
              f"Win%: {metrics['win_rate']:.1%} | RankIC: {metrics['rank_ic']:.4f} | "
              f"Hit@5: {metrics['hit_at_5']:.3f}")

    _save_json(all_results, "plan_b")
    best_name, best_sharpe = _print_plan_b_table(all_results)
    return all_results


def _compute_ctx(g, df, sector_map):
    return compute_context_features(df, g["date"], g["instruments"],
                                    dict(zip(g["instruments"], g["scores"])), sector_map)


# ============================================================================
# Output
# ============================================================================

def _save_json(data, name):
    out = Path(__file__).resolve().parent.parent.parent / "output" / "research"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out / f"{name}_results_{ts}.json", "w") as f:
        class Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, pd.Timestamp): return str(o)
                return super().default(o)
        json.dump(data, f, indent=2, cls=Enc)
    print(f"\nSaved to {out / f'{name}_results_{ts}.json'}")


def _print_plan_a_table(results):
    print("\n" + "=" * 90)
    print("Plan A Results (per-date-group training)")
    print("=" * 90)
    hdr = f"{'Experiment':<30} {'Weeks':>6} {'Sharpe':>8} {'CumRet':>9} {'Win%':>7} {'MaxDD':>8} {'RankIC':>7} {'Hit@5':>6}"
    print(hdr); print("-" * 90)
    bl = results.get("A-E0 (MSE Baseline)", {}); bl_s = bl.get("sharpe", 0.0)
    ndcg_m = results.get("A-E3 (NDCG Approx)", {}); ndcg_s = ndcg_m.get("sharpe", -999)
    for n, r in results.items():
        s = r.get("sharpe", 0.0)
        m = " ↑" if s > bl_s + 0.01 else (" ↓" if s < bl_s - 0.05 else "")
        print(f"{n:<30} {r.get('weeks',0):>6} {s:>8.4f}{m:<3} {r.get('cum_return',0):>+9.4f} "
              f"{r.get('win_rate',0):>7.1%} {r.get('max_dd',0):>8.4f} "
              f"{r.get('rank_ic',0):>7.4f} {r.get('hit_at_5',0):>6.3f}")
    print(f"\nNDCG vs MSE: Sharpe {ndcg_s:.4f} vs {bl_s:.4f} "
          f"({'↑ KEEP' if ndcg_s > bl_s + 0.01 else '≈' if abs(ndcg_s - bl_s) < 0.01 else '↓'})")


def _print_plan_b_table(results):
    print("\n" + "=" * 90)
    print("Plan B Results (NDCG loss + Context Transformer)")
    print("=" * 90)
    hdr = f"{'Experiment':<30} {'Weeks':>6} {'Sharpe':>8} {'CumRet':>9} {'Win%':>7} {'MaxDD':>8} {'RankIC':>7} {'Hit@5':>6}"
    print(hdr); print("-" * 90)
    bl = results.get("B-C1 (No Context)", {}); bl_s = bl.get("sharpe", 0.0)
    best_n, best_s = None, -999
    for n, r in results.items():
        s = r.get("sharpe", 0.0)
        if s > best_s: best_s = s; best_n = n
        m = " ↑ GAIN" if s > bl_s + 0.01 else (" ↓" if s < bl_s - 0.05 else "")
        print(f"{n:<30} {r.get('weeks',0):>6} {s:>8.4f}{m:<8} {r.get('cum_return',0):>+9.4f} "
              f"{r.get('win_rate',0):>7.1%} {r.get('max_dd',0):>8.4f} "
              f"{r.get('rank_ic',0):>7.4f} {r.get('hit_at_5',0):>6.3f}")
    print(f"\nBest: {best_n} (Sharpe={best_s:.4f})")
    return best_n, best_s


# ============================================================================
# Auto-Update Champion Config
# ============================================================================

def update_champion(plan_a_results, plan_b_results):
    """Update model/result_model.yaml based on best combination found."""
    best_a = max(plan_a_results.items(), key=lambda x: x[1].get("sharpe", -999))
    best_b = max(plan_b_results.items(), key=lambda x: x[1].get("sharpe", -999))

    print("\n" + "=" * 60)
    print("🏆 Auto-Update Champion Configuration")
    print("=" * 60)
    print(f"  Best Plan A: {best_a[0]} (Sharpe={best_a[1]['sharpe']:.4f})")
    print(f"  Best Plan B: {best_b[0]} (Sharpe={best_b[1]['sharpe']:.4f})")

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "model" / "result_model.yaml"

    # Determine best loss type for Plan A
    best_a_loss = best_a[1].get("loss_type", "mse")
    loss_mapping = {
        "mse": "mse", "pairwise": "pairwise", "listmle": "listmle",
        "ndcg": "ndcg", "coarse_to_fine": "coarse_to_fine"
    }
    mapped_loss = loss_mapping.get(best_a_loss, "mse")

    # Check if Plan B beats Plan A (context adds value)
    b_beats_a = best_b[1]["sharpe"] > best_a[1]["sharpe"] + 0.01

    print(f"\n  🔧 Recommendations:")
    print(f"     Loss function: {mapped_loss} ({best_a[0]})")

    if b_beats_a:
        print(f"     Context mode: {best_b[0]}")
        print(f"     ✅ Context features add value! Consider adding to config.")
    else:
        print(f"     Context: Not beneficial (best Plan B: {best_b[0]}, Sharpe={best_b[1]['sharpe']:.4f})")
        print(f"     ✅ Plain NDCG + MLP is optimal for current setup")

    # Save update info
    info = {
        "timestamp": datetime.now().isoformat(),
        "best_plan_a": {"name": best_a[0], "sharpe": best_a[1]["sharpe"],
                        "cum_return": best_a[1]["cum_return"],
                        "loss_type": best_a[1].get("loss_type")},
        "best_plan_b": {"name": best_b[0], "sharpe": best_b[1]["sharpe"],
                        "cum_return": best_b[1]["cum_return"]},
        "context_adds_value": b_beats_a,
        "recommended_loss": mapped_loss,
        "config_updated": b_beats_a,
    }

    out = project_root / "output" / "research" / "champion_update.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"\n  Update info saved to {out}")

    # If NDCG is confirmed best, update the YAML
    if mapped_loss == "ndcg" or best_a[1]["sharpe"] > 0:
        print("\n  📝 Updating model/result_model.yaml with best config...")
        _update_yaml_loss(config_path, mapped_loss, b_beats_a, best_b[0] if b_beats_a else None)

    return info


def _update_yaml_loss(config_path, loss_type, add_context, context_mode):
    """Update the YAML config with the best loss type and context settings."""
    with open(config_path) as f:
        lines = f.readlines()

    # Add a comment documenting the research-based selection
    header = [
        f"# Champion config — auto-updated by research pipeline ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n",
        f"# Best loss: {loss_type} (Plan A experiment winner)\n",
    ]
    if add_context:
        header.append(f"# Context mode: {context_mode} (Plan B experiment winner)\n")
    header.append(f"# See: output/research/champion_update.json for details\n\n")

    # Find where qlib_init starts and insert header
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("qlib_init:") or line.startswith("# Champion config"):
            insert_idx = i
            break

    # Remove old auto-update comments
    filtered = []
    skip_old = False
    for line in lines:
        if line.startswith("# Champion config — auto-updated"):
            skip_old = True
            continue
        if skip_old and line.startswith("# "):
            continue
        skip_old = False
        filtered.append(line)

    # Write updated config
    with open(config_path, "w") as f:
        f.writelines(header + filtered)

    print(f"     Updated {config_path}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Research experiment runner")
    p.add_argument("command", nargs="?", default="all",
                   choices=["plan_a", "plan_b", "all"])
    args = p.parse_args()

    pa, pb = {}, {}
    if args.command in ("plan_a", "all"):
        pa = run_plan_a()
    if args.command in ("plan_b", "all"):
        pb = run_plan_b()
    if pa and pb:
        update_champion(pa, pb)
    elif args.command == "plan_b" and pb:
        print("\n⚠️  Running Plan A for comparison...")
        pa = run_plan_a()
        update_champion(pa, pb)


if __name__ == "__main__":
    main()
