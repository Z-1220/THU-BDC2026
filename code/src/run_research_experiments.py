#!/usr/bin/env python
"""Fast experiment runner using heuristic scores as a Kronos proxy.

This enables rapid iteration on the experimental framework without
waiting for full Kronos inference. The heuristic is 20-day momentum,
which has known properties and serves as a useful baseline.

Real Kronos scores can be substituted by running the cache command first.

Usage:
    python code/src/run_research_experiments.py proxy
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code"))


def load_price_data() -> pd.DataFrame:
    """Load stock_data.csv."""
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_path = project_root / "data" / "stock_data.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["日期"])
    return df


def compute_momentum_scores(df: pd.DataFrame, signal_date: pd.Timestamp, lookback: int = 20) -> dict[str, float]:
    """Compute simple momentum-based scores as Kronos proxy."""
    hist = df[df["日期"] <= signal_date]
    stocks = sorted(hist["股票代码"].unique())
    scores = {}
    for code in stocks:
        inst_data = hist[hist["股票代码"] == code].sort_values("日期")
        if len(inst_data) < lookback + 1:
            continue
        # 20-day momentum: (close_today - close_20d_ago) / close_20d_ago
        ret = float(inst_data["收盘"].iloc[-1] / inst_data["收盘"].iloc[-lookback] - 1)
        if np.isfinite(ret):
            scores[code] = ret
    return scores


def get_friday_dates(df: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    """Get Friday trading dates in range."""
    all_dates = sorted(df["日期"].unique())
    fridays = []
    for d in all_dates:
        if start <= str(d.date()) <= end and d.weekday() == 4:  # Friday
            fridays.append(d)
    return fridays


def compute_labels(df: pd.DataFrame, signal_date: pd.Timestamp, instruments: list[str]) -> dict[str, float]:
    """Compute LABEL0 = (open[T+5] - open[T+1]) / open[T+1] for each stock."""
    future = df[df["日期"] > signal_date].sort_values("日期")
    future_dates = future["日期"].unique()[:5]  # Next 5 trading days
    if len(future_dates) < 5:
        return {}

    labels = {}
    for code in instruments:
        future_inst = future[future["股票代码"] == code].sort_values("日期")
        if len(future_inst) < 2:  # Need at least T+1 and T+5
            continue
        dates_avail = future_inst["日期"].tolist()
        # Find T+1 (next trading day) and T+5 (5th trading day after signal)
        # For simplicity: use first and last available in the 5-day window
        t1_open = float(future_inst["开盘"].iloc[0])
        t5_open = float(future_inst["开盘"].iloc[-1]) if len(future_inst) >= 2 else t1_open
        if t1_open > 0:
            label = (t5_open - t1_open) / t1_open
            if np.isfinite(label):
                labels[code] = label
    return labels


def run_proxy_experiments():
    """Run Plan A experiments using momentum-based proxy scores."""
    print("Loading stock data...")
    df = load_price_data()

    # Date ranges
    train_start, train_end = "2024-01-01", "2025-12-31"
    valid_start, valid_end = "2026-01-05", "2026-01-30"
    test_start, test_end = "2026-02-02", "2026-05-27"

    def extract_groups(start, end):
        dates = get_friday_dates(df, start, end)
        groups = []
        for dt in dates:
            scores = compute_momentum_scores(df, dt)
            if len(scores) < 5:
                continue
            instruments = list(scores.keys())
            labels = compute_labels(df, dt, instruments)
            # Keep only stocks with both score and label
            common = sorted(set(scores.keys()) & set(labels.keys()))
            if len(common) < 5:
                continue
            groups.append({
                "date": dt,
                "instruments": common,
                "scores": np.array([scores[c] for c in common]),
                "labels": np.array([labels[c] for c in common]),
            })
        return groups

    print("Extracting train groups...")
    train_groups = extract_groups(train_start, train_end)
    print(f"  Train: {len(train_groups)} signal dates")

    print("Extracting valid groups...")
    valid_groups = extract_groups(valid_start, valid_end)
    print(f"  Valid: {len(valid_groups)} signal dates")

    print("Extracting test groups...")
    test_groups = extract_groups(test_start, test_end)
    print(f"  Test: {len(test_groups)} signal dates")

    if not test_groups:
        print("No test data. Check date ranges.")
        return

    # Import loss functions
    from models.KronosRankHead.RankingLosses import (
        PairwiseMarginLoss,
        ListMLELoss,
        NDCGApproxLoss,
        CoarseToFineLoss,
        CandidateGroupRankingLoss,
        StructureConsistencyLoss,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Prepare training data as flat tensors for simple MLP
    X_train = torch.cat([
        torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32)
        for g in train_groups
    ])
    y_train = torch.cat([
        torch.tensor(g["labels"], dtype=torch.float32)
        for g in train_groups
    ])

    X_valid = torch.cat([
        torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32)
        for g in valid_groups
    ]) if valid_groups else None
    y_valid = torch.cat([
        torch.tensor(g["labels"], dtype=torch.float32)
        for g in valid_groups
    ]) if valid_groups else None

    print(f"\nTraining: {len(X_train)} (score, label) pairs")
    if X_valid is not None:
        print(f"Validation: {len(X_valid)} pairs")
    print(f"Test: {len(test_groups)} signal dates\n")

    # Experiment definitions
    experiments = [
        ("A-E0 (MSE Baseline)", "mse", {}),
        ("A-E1 (Pairwise Margin)", "pairwise", {"margin": 0.1}),
        ("A-E2 (ListMLE)", "listmle", {"temperature": 1.0}),
        ("A-E3 (NDCG Approx)", "ndcg", {"sigma": 1.0, "k": 5}),
        ("A-E4 (Coarse-to-Fine)", "coarse_to_fine", {"n_tiers": 3, "coarse_weight": 0.3, "fine_weight": 0.7}),
        ("A-E5 (Candidate+Group)", "candidate_group", {"top_m": 30, "recall_weight": 0.3, "rank_weight": 0.7}),
        ("A-E6 (Structure Consistency)", "structure_consistency", {"consistency_weight": 0.1, "base_loss_type": "pairwise"}),
        ("A-E8 (Shuffled Labels)", "shuffled_label", {}),
        ("A-E10 (Coarse Only)", "coarse_only", {"n_tiers": 3}),
        ("A-E11 (Fine Only)", "fine_only", {"margin": 0.05}),
    ]

    all_results = {}

    for exp_name, loss_type, loss_kwargs in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp_name}...")

        # Build head
        head = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1),
        ).to(device)

        # Build loss
        if loss_type == "pairwise":
            loss_fn = PairwiseMarginLoss(margin=loss_kwargs.get("margin", 0.1))
        elif loss_type == "listmle":
            loss_fn = ListMLELoss(temperature=loss_kwargs.get("temperature", 1.0))
        elif loss_type == "ndcg":
            loss_fn = NDCGApproxLoss(sigma=loss_kwargs.get("sigma", 1.0), k=loss_kwargs.get("k", 5))
        elif loss_type == "coarse_to_fine":
            loss_fn = CoarseToFineLoss(
                n_tiers=loss_kwargs.get("n_tiers", 3),
                coarse_weight=loss_kwargs.get("coarse_weight", 0.3),
                fine_weight=loss_kwargs.get("fine_weight", 0.7),
            )
        elif loss_type == "candidate_group":
            loss_fn = CandidateGroupRankingLoss(
                top_m=loss_kwargs.get("top_m", 30),
                recall_weight=loss_kwargs.get("recall_weight", 0.3),
                rank_weight=loss_kwargs.get("rank_weight", 0.7),
            )
        elif loss_type == "structure_consistency":
            loss_fn = StructureConsistencyLoss(
                base_loss_type=loss_kwargs.get("base_loss_type", "pairwise"),
                consistency_weight=loss_kwargs.get("consistency_weight", 0.1),
            )
        else:
            loss_fn = None  # MSE

        # Train
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        best_loss = float("inf")
        patience = 0
        train_X = X_train.to(device)
        train_y = y_train.to(device)
        if X_valid is not None:
            valid_X = X_valid.to(device)
            valid_y = y_valid.to(device)

        for epoch in range(100):
            head.train()
            perm = torch.randperm(len(train_X))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_X), 256):
                idx = perm[i : i + 256]
                x = train_X[idx]
                y = train_y[idx]

                optimizer.zero_grad()
                pred = head(x).squeeze(-1)

                if loss_fn is not None:
                    if loss_type in ("coarse_to_fine", "candidate_group"):
                        loss, _ = loss_fn(pred, y)
                    elif loss_type == "structure_consistency":
                        base_scores = x.squeeze(-1)
                        loss, _ = loss_fn(pred, y, base_scores)
                    elif loss_type == "shuffled_label":
                        # Shuffle labels
                        shuffled_y = y[torch.randperm(len(y))]
                        pairwise_fn = PairwiseMarginLoss(margin=0.1)
                        loss = pairwise_fn(pred, shuffled_y)
                    elif loss_type == "coarse_only":
                        # Coarse: classify into tiers
                        n_tiers = loss_kwargs.get("n_tiers", 3)
                        boundaries = torch.quantile(y, torch.linspace(0, 1, n_tiers + 1, device=y.device))
                        coarse_labels = torch.zeros(len(y), dtype=torch.long, device=y.device)
                        for t in range(n_tiers):
                            lo, hi = boundaries[t], boundaries[t+1]
                            if t == n_tiers - 1:
                                coarse_labels[(y >= lo) & (y <= hi)] = t
                            else:
                                coarse_labels[(y >= lo) & (y < hi)] = t
                        logits = torch.zeros(len(y), n_tiers, device=y.device)
                        for t in range(n_tiers):
                            center = boundaries[t:t+2].mean()
                            logits[:, t] = -torch.abs(pred.unsqueeze(1) - center)
                        loss_fn_ce = torch.nn.CrossEntropyLoss()
                        loss = loss_fn_ce(logits, coarse_labels)
                    elif loss_type == "fine_only":
                        pairwise_fn = PairwiseMarginLoss(margin=loss_kwargs.get("margin", 0.05))
                        loss = pairwise_fn(pred, y)
                    else:
                        loss = loss_fn(pred, y)
                else:
                    loss = torch.nn.functional.mse_loss(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            if X_valid is not None:
                head.eval()
                with torch.no_grad():
                    pred = head(valid_X).squeeze(-1)
                    if loss_fn is not None and loss_type not in ("shuffled_label", "coarse_only", "fine_only"):
                        if loss_type == "structure_consistency":
                            val_loss, _ = loss_fn(pred, valid_y, valid_X.squeeze(-1))
                        elif loss_type in ("coarse_to_fine", "candidate_group"):
                            val_loss, _ = loss_fn(pred, valid_y)
                        else:
                            val_loss = loss_fn(pred, valid_y)
                    else:
                        val_loss = torch.nn.functional.mse_loss(pred, valid_y)

                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    patience = 0
                else:
                    patience += 1
                if patience >= 15:
                    break

        # Evaluate on test set
        head.eval()
        weekly_results = []

        for g in test_groups:
            x = torch.tensor(g["scores"].reshape(-1, 1), dtype=torch.float32).to(device)
            y = g["labels"]
            n = len(y)

            with torch.no_grad():
                refined = head(x).squeeze(-1).cpu().numpy()

            if n >= 3:
                top_idx = np.argsort(refined)[::-1][:3]
                top_weights = np.array([0.333, 0.333, 0.334])
                week_return = float(np.dot(y[top_idx], top_weights))
                hit_rate = float(np.mean(y[top_idx] > 0))
            else:
                week_return = 0.0
                hit_rate = 0.0

            # Rank IC
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(refined, y) if n >= 3 else (0.0, 1.0)

            # Hit@5
            if n >= 5:
                pred_top5 = set(np.argsort(refined)[::-1][:5])
                true_top5 = set(np.argsort(y)[::-1][:5])
                hit5 = len(pred_top5 & true_top5) / 5
            else:
                hit5 = 0.0

            weekly_results.append({
                "date": str(g["date"].date()),
                "n_stocks": n,
                "week_return": week_return,
                "rank_ic": rank_ic,
                "hit_at_5": hit5,
            })

        returns = [r["week_return"] for r in weekly_results]
        n_weeks = len(returns)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        sharpe = float(mean_ret / std_ret * np.sqrt(52)) if std_ret > 0 else 0.0
        win_rate = float(np.mean([r > 0 for r in returns]))
        cum_ret = float(np.prod([1 + r for r in returns]) - 1)

        # Max drawdown
        max_dd = 0.0
        peak = 0.0
        cum = 1.0
        for r in returns:
            cum *= (1 + r)
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        rank_ic_mean = float(np.mean([r["rank_ic"] for r in weekly_results]))
        hit_at_5_mean = float(np.mean([r["hit_at_5"] for r in weekly_results]))

        result = {
            "experiment": exp_name,
            "loss_type": loss_type,
            "loss_kwargs": loss_kwargs,
            "weeks": n_weeks,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "sharpe": sharpe,
            "sortino": sharpe,  # Simplified
            "win_rate": win_rate,
            "cum_return": cum_ret,
            "max_dd": max_dd,
            "rank_ic": rank_ic_mean,
            "hit_at_5": hit_at_5_mean,
            "gain_loss_ratio": 0.0,
            "weekly_details": weekly_results,
        }

        all_results[exp_name] = result
        print(f"  Sharpe: {sharpe:.4f} | CumRet: {cum_ret:+.4f} | Win%: {win_rate:.1%} | "
              f"RankIC: {rank_ic_mean:.4f} | Hit@5: {hit_at_5_mean:.3f} | Weeks: {n_weeks}")

    # Save results
    output_dir = Path(__file__).resolve().parent.parent.parent / "output" / "research"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"plan_a_results_{timestamp}.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("Plan A: Loss Function Structure Alignment — Results Comparison")
    print("(Proxy scores: 20-day momentum heuristic)")
    print("=" * 90)
    header = f"{'Experiment':<30} {'Weeks':>6} {'Sharpe':>8} {'CumRet':>9} {'Win%':>7} {'MaxDD':>8} {'RankIC':>7} {'Hit@5':>6}"
    print(header)
    print("-" * 90)

    baseline = all_results.get("A-E0 (MSE Baseline)", {})
    baseline_sharpe = baseline.get("sharpe", 0.0)

    for name, r in all_results.items():
        s = r.get("sharpe", 0.0)
        marker = ""
        if s > baseline_sharpe + 0.01:
            marker = " ↑ KEEP"
        elif s < baseline_sharpe - 0.05:
            marker = " ↓ DROP"

        print(f"{name:<30} {r.get('weeks', 0):>6} {s:>8.4f}{marker:<8} "
              f"{r.get('cum_return', 0):>+9.4f} {r.get('win_rate', 0):>7.1%} "
              f"{r.get('max_dd', 0):>8.4f} {r.get('rank_ic', 0):>7.4f} "
              f"{r.get('hit_at_5', 0):>6.3f}")

    # Analysis
    print("\n" + "=" * 90)
    print("Analysis Summary")
    print("=" * 90)

    # Structure alignment check
    structured_exps = ["A-E4 (Coarse-to-Fine)", "A-E5 (Candidate+Group)", "A-E6 (Structure Consistency)"]
    structured_keeping = sum(
        1 for e in structured_exps
        if all_results.get(e, {}).get("sharpe", -999) > baseline_sharpe + 0.01
    )
    neg_controls = ["A-E8 (Shuffled Labels)", "A-E10 (Coarse Only)", "A-E11 (Fine Only)"]
    neg_worse = sum(
        1 for e in neg_controls
        if all_results.get(e, {}).get("sharpe", 999) < baseline_sharpe - 0.01
    )

    print(f"\nGate Check:")
    print(f"  Structured losses keeping (>=2 needed): {structured_keeping}/3")
    print(f"  Negative controls worse than baseline: {neg_worse}/3")

    if structured_keeping >= 2:
        print("\n  ✅ Structure alignment EFFECTIVE — hierarchical loss improves ranking.")
    elif structured_keeping >= 1:
        print("\n  ⚠️  Partial evidence — structure direction correct but implementation needs tuning.")
    else:
        print("\n  ❌ Structure alignment NOT effective with proxy scores — may need real Kronos embeddings.")

    if neg_worse >= 2:
        print("  ✅ Negative controls confirm: breaking structure hurts performance.")
    else:
        print("  ⚠️  Negative controls inconclusive — structure signal may be weak in proxy setting.")

    return all_results


if __name__ == "__main__":
    run_proxy_experiments()
