#!/usr/bin/env python
"""End-to-end training + evaluation of Kronos + Context Transformer + CardNN.

The model learns, jointly and end-to-end:
  - cross-stock interactions (Context Transformer self-attention),
  - market environment (CLS context token),
  - selection + allocation (CardNN Gumbel-Sinkhorn Top-K with cash asset),
directly optimizing the competition metric (portfolio return), with an
optional NDCG warm-start / auxiliary ranking loss.

Usage:
    uv run python scripts/train_cardnn_e2e.py \
        --cache temp/kronos_scores_cache_ft.pkl \
        --test-start 2026-02-02 --test-end 2026-07-31 \
        --seeds 42 2024 7
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

# ---- Reuse helpers from the real-Kronos verification pipeline ----
_VERIFY = PROJECT_ROOT / "scripts" / "verify_plan_b_real_kronos.py"
_VSPEC = importlib.util.spec_from_file_location("verify", _VERIFY)
verify = importlib.util.module_from_spec(_VSPEC)
_VSPEC.loader.exec_module(verify)  # type: ignore[union-attr]

load_groups_from_cache = verify.load_groups_from_cache
load_price_data = verify.load_price_data
load_sector_map = verify.load_sector_map
compute_metrics = verify.compute_metrics
compute_context_features = verify.compute_context_features

_RUNNER = PROJECT_ROOT / "code" / "src" / "run_research_experiments.py"
_RSPEC = importlib.util.spec_from_file_location("runner", _RUNNER)
runner = importlib.util.module_from_spec(_RSPEC)
_RSPEC.loader.exec_module(runner)  # type: ignore[union-attr]

ContextTransformer = runner.ContextTransformer
NDCGApproxLoss = runner.NDCGApproxLoss

from models.CardNN.CardNNLayer import GumbelSinkhornTopK  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EndToEndModel(nn.Module):
    """Context Transformer + CardNN allocation (trainable jointly)."""

    def __init__(
        self,
        stock_feat_dim: int = 6,
        context_feat_dim: int = 5,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        K: int = 3,
        cardnn_tau: float = 1.0,
        cardnn_n_iter: int = 30,
    ) -> None:
        super().__init__()
        self.K = K
        self.transformer = ContextTransformer(
            stock_feat_dim=stock_feat_dim,
            context_feat_dim=context_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_stocks=350,
        )
        self.cardnn = GumbelSinkhornTopK(tau=cardnn_tau, n_iter=cardnn_n_iter)

    def forward(
        self, stock_f: torch.Tensor, market_f: torch.Tensor, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        refined = self.transformer(stock_f, market_f).squeeze(0)  # (N,)
        w, diag = self.cardnn(refined, self.K, training=training)
        return refined, w, diag

    def deterministic(self, stock_f: torch.Tensor, market_f: torch.Tensor):
        refined = self.transformer(stock_f, market_f).squeeze(0)
        w, diag = self.cardnn.deterministic_weights(refined, self.K)
        return refined, w, diag


def portfolio_return(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (w * y).sum()


def train_stage1_ndcg(
    model: EndToEndModel,
    train_groups: list[dict],
    valid_groups: list[dict],
    train_ctx: list[dict],
    valid_ctx: list[dict],
    device: str,
    epochs: int,
    patience: int,
) -> None:
    """Warm-start the Context Transformer with NDCG loss (head only)."""
    ndcg = NDCGApproxLoss(sigma=1.0, k=5).to(device)
    optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float("inf")
    patience_cnt = 0
    best_state = None

    for epoch in range(epochs):
        model.transformer.train()
        perm = torch.randperm(len(train_groups))
        for gid in perm:
            g, ctx = train_groups[gid], train_ctx[gid]
            sf = torch.tensor(ctx["stock_features"], dtype=torch.float32).unsqueeze(0).to(device)
            mf = torch.tensor(ctx["market_features"], dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            refined = model.transformer(sf, mf).squeeze(0)
            loss = ndcg(refined, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.transformer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gid, g in enumerate(valid_groups):
                ctx = valid_ctx[gid]
                sf = torch.tensor(ctx["stock_features"], dtype=torch.float32).unsqueeze(0).to(device)
                mf = torch.tensor(ctx["market_features"], dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
                refined = model.transformer(sf, mf).squeeze(0)
                val_loss += ndcg(refined, y).item()
        val_loss /= len(valid_groups)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.transformer.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break
    if best_state:
        model.transformer.load_state_dict(best_state)


def train_stage2_return(
    model: EndToEndModel,
    train_groups: list[dict],
    valid_groups: list[dict],
    train_ctx: list[dict],
    valid_ctx: list[dict],
    device: str,
    epochs: int,
    patience: int,
    aux_ndcg_weight: float,
    freeze_head: bool = False,
    lr: float = 1e-3,
) -> dict:
    """Joint end-to-end training with the portfolio-return loss."""
    ndcg = NDCGApproxLoss(sigma=1.0, k=5).to(device)
    params = list(model.cardnn.parameters()) if freeze_head else list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_ret = -float("inf")
    patience_cnt = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        if freeze_head:
            model.transformer.eval()
        perm = torch.randperm(len(train_groups))
        for gid in perm:
            g, ctx = train_groups[gid], train_ctx[gid]
            sf = torch.tensor(ctx["stock_features"], dtype=torch.float32).unsqueeze(0).to(device)
            mf = torch.tensor(ctx["market_features"], dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            refined, w, _ = model(sf, mf, training=True)
            loss = -portfolio_return(w, y)
            if aux_ndcg_weight > 0:
                loss = loss + aux_ndcg_weight * ndcg(refined, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()

        val_ret = evaluate_valid_return(model, valid_groups, valid_ctx, device)
        if val_ret > best_val_ret:
            best_val_ret = val_ret
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return {"best_val_return": float(best_val_ret)}


@torch.no_grad()
def evaluate_valid_return(
    model: EndToEndModel,
    valid_groups: list[dict],
    valid_ctx: list[dict],
    device: str,
) -> float:
    model.eval()
    rets = []
    for gid, g in enumerate(valid_groups):
        ctx = valid_ctx[gid]
        sf = torch.tensor(ctx["stock_features"], dtype=torch.float32).unsqueeze(0).to(device)
        mf = torch.tensor(ctx["market_features"], dtype=torch.float32).unsqueeze(0).to(device)
        y = torch.tensor(g["labels"], dtype=torch.float32).to(device)
        _, w, _ = model.deterministic(sf, mf)
        rets.append(float(portfolio_return(w, y)))
    return float(np.mean(rets)) if rets else 0.0


@torch.no_grad()
def evaluate_test(
    model: EndToEndModel,
    test_groups: list[dict],
    test_ctx: list[dict],
    device: str,
) -> tuple[dict, list[dict], list[dict]]:
    """Weekly evaluation with Top-3-style metrics + allocation diagnostics."""
    from scipy.stats import spearmanr

    model.eval()
    weekly = []
    diags = []
    for gid, g in enumerate(test_groups):
        ctx = test_ctx[gid]
        sf = torch.tensor(ctx["stock_features"], dtype=torch.float32).unsqueeze(0).to(device)
        mf = torch.tensor(ctx["market_features"], dtype=torch.float32).unsqueeze(0).to(device)
        y = g["labels"]
        refined, w, diag = model.deterministic(sf, mf)
        refined_np = refined.cpu().numpy()
        w_np = w.cpu().numpy()
        n = len(y)
        wr = float(np.dot(w_np, y))
        ric, _ = spearmanr(refined_np, y) if n >= 3 else (0.0, 1.0)
        h5 = (
            len(set(np.argsort(refined_np)[::-1][:5]) & set(np.argsort(y)[::-1][:5])) / 5
            if n >= 5
            else 0.0
        )
        weekly.append(
            {
                "date": str(g["date"].date()),
                "n_stocks": n,
                "week_return": wr,
                "rank_ic": ric,
                "hit_at_5": h5,
            }
        )
        diags.append(
            {
                "n_positions": diag["n_positions"],
                "invested_frac": diag["invested_frac"],
                "cash_positions": diag["cash_positions"],
                "max_weight": diag["max_weight"],
                "weight_sum": float(w_np.sum()),
                "hhi": float((w_np ** 2).sum()),
            }
        )
    metrics = compute_metrics(weekly)
    metrics["n_positions"] = float(np.mean([d["n_positions"] for d in diags]))
    metrics["invested_frac"] = float(np.mean([d["invested_frac"] for d in diags]))
    metrics["cash_positions"] = float(np.mean([d["cash_positions"] for d in diags]))
    metrics["max_weight"] = float(np.max([d["max_weight"] for d in diags]))
    metrics["mean_hhi"] = float(np.mean([d["hhi"] for d in diags]))
    metrics["weight_sum_ok"] = all(abs(d["weight_sum"] - 1.0 + (1 - d["invested_frac"])) < 1e-6 for d in diags)
    return metrics, weekly, diags


def run_experiment(
    name: str,
    K: int,
    seed: int,
    train_groups: list[dict],
    valid_groups: list[dict],
    test_groups: list[dict],
    train_ctx: list[dict],
    valid_ctx: list[dict],
    test_ctx: list[dict],
    device: str,
    warm_start: bool,
    aux_ndcg_weight: float,
    epochs: int = 100,
    patience: int = 15,
    freeze_head: bool = False,
    stage2_lr: float = 1e-3,
    stage2_epochs: int | None = None,
) -> dict:
    set_seed(seed)
    model = EndToEndModel(K=K).to(device)
    if warm_start:
        train_stage1_ndcg(
            model, train_groups, valid_groups, train_ctx, valid_ctx,
            device, epochs, patience,
        )
    stage2 = train_stage2_return(
        model, train_groups, valid_groups, train_ctx, valid_ctx,
        device, stage2_epochs or epochs, patience, aux_ndcg_weight,
        freeze_head=freeze_head, lr=stage2_lr,
    )
    metrics, weekly, diags = evaluate_test(model, test_groups, test_ctx, device)
    ckpt_dir = PROJECT_ROOT / "model" / "cardnn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items()},
        ckpt_dir / f"{name}_s{seed}.pt",
    )
    metrics["experiment"] = name
    metrics["K"] = K
    metrics["seed"] = seed
    metrics["warm_start"] = warm_start
    metrics["aux_ndcg_weight"] = aux_ndcg_weight
    metrics["freeze_head"] = freeze_head
    metrics.update(stage2)
    metrics["weekly_details"] = weekly
    metrics["allocation_diag"] = diags
    print(
        f"[{name} seed={seed}] Sharpe {metrics['sharpe']:.4f} | "
        f"CumRet {metrics['cum_return']:+.4f} | Win% {metrics['win_rate']:.1%} | "
        f"pos {metrics['n_positions']:.2f} | invest {metrics['invested_frac']:.3f} | "
        f"val_ret {metrics['best_val_return']:+.4f}"
    )
    return metrics


def build_contexts(groups: list[dict], df: pd.DataFrame, sector_map: dict) -> list[dict]:
    return [
        compute_context_features(
            df, g["date"], g["instruments"],
            dict(zip(g["instruments"], g["scores"])), sector_map,
        )
        for g in groups
    ]


def write_report(
    results: dict,
    baselines: dict,
    champion: dict,
    meta: dict,
    out_path: Path,
) -> None:
    rows = []
    for exp_name, r in results.items():
        rows.append(
            f"| {exp_name} | {r['seed']} | {r['K']} | {r['weeks']} | {r['sharpe']:.4f} | "
            f"{r['cum_return']:+.4f} | {r['win_rate']:.1%} | {r['max_dd']:.4f} | "
            f"{r['rank_ic']:.4f} | {r['hit_at_5']:.3f} | {r['n_positions']:.2f} | "
            f"{r['invested_frac']:.3f} | {r['max_weight']:.3f} |"
        )
    rows_text = "\n".join(rows)
    bl_rows = "\n".join(
        f"| {n} | {m['sharpe']:.4f} | {m['cum_return']:+.4f} | {m['win_rate']:.1%} | "
        f"{m['rank_ic']:.4f} | {m['hit_at_5']:.3f} |"
        for n, m in baselines.items()
    )
    report = f"""# CardNN 端到端验证报告

日期: {meta['timestamp']} | 数据截止: {meta['data_end']} | 评分来源: {meta['score_source']}

## 目的

在真实 Kronos 评分上验证端到端 CardNN 配权模型（Context Transformer 捕捉股票交互与
市场环境，Gumbel-Sinkhorn Top-K 选股配权，直接以组合收益为损失），并评估其相对
生产基线（Kronos-CS Top-3）与 Context Transformer 精排（B-C2/B-C4/B-C5）的增益。

## 方法

- 评分缓存: {meta['cache_file']}（训练 {meta['train_window']} / 验证 {meta['valid_window']} / 测试 {meta['test_window']}）
- 股票池: FridayFilter + ScreenProcessor 筛选后横截面
- 训练: 阶段 1 NDCG warm-start 精排头；阶段 2 端到端组合收益损失（可选 NDCG 辅助项），
  per-date-group，早停 patience=15，固定 seed
- 约束: 持仓 <= K 只，权重和 <= 1（现金资产承接剩余），K 取 3 / 5
- 评估: Top-K 确定性推理（无 Gumbel 噪声），周收益、Sharpe、RankIC、Hit@5 及配权诊断

## 基线与历史对照（21 周测试窗口）

| 实验 | Sharpe | 累计收益 | 胜率 | RankIC | Hit@5 |
|---|---:|---:|---:|---:|---:|
{bl_rows}

## 结果（本报告）

| 实验 | seed | K | 周数 | Sharpe | 累计收益 | 胜率 | 最大回撤 | RankIC | Hit@5 | 平均持仓 | 投入比例 | 最大权重 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_text}

## 关键结论

- Champion: {champion['config']} (seed {champion['seed']}, K={champion['K']},
  Sharpe {champion['sharpe']:.4f}, CumRet {champion['cum_return']:+.4f},
  平均持仓 {champion['n_positions']:.2f}, 投入比例 {champion['invested_frac']:.3f})
- 对比生产基线 Kronos-CS (Sharpe {baselines.get('Kronos-CS (Top-3)', {}).get('sharpe', float('nan')):.4f}) 与
  Plan B 最佳 B-C4 (Sharpe {baselines.get('B-C4 (CS Stats)', {}).get('sharpe', float('nan')):.4f})

## 复现

```bash
uv run python scripts/train_cardnn_e2e.py \\
  --cache temp/kronos_scores_cache_ft.pkl --test-start 2026-02-02 --test-end 2026-07-31
```

## 局限

- 测试窗口 21 周，含官方盲测周（仅评估）；端到端收益损失对验证集早停与 seed 敏感。
- 生产推理为确定性 Top-K 贪心近似（训练时 Sinkhorn 松弛）。
"""
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="temp/kronos_scores_cache_ft.pkl")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-05")
    parser.add_argument("--valid-end", default="2026-01-30")
    parser.add_argument("--test-start", default="2026-02-02")
    parser.add_argument("--test-end", default="2026-07-31")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only-configs", nargs="*", default=None,
                        help="Run only these configs (e.g. --only-configs e2e_k3_ft)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    print("Loading data...")
    df = load_price_data()
    sector_map = load_sector_map()
    with open(cache_path, "rb") as f:
        cache = __import__("pickle").load(f)

    train_groups = load_groups_from_cache(cache, df, sector_map, args.train_start, args.train_end)
    valid_groups = load_groups_from_cache(cache, df, sector_map, args.valid_start, args.valid_end)
    test_groups = load_groups_from_cache(cache, df, sector_map, args.test_start, args.test_end)
    print(f"Train: {len(train_groups)}w | Valid: {len(valid_groups)}w | Test: {len(test_groups)}w | Device: {device}")

    print("Pre-computing context features...")
    train_ctx = build_contexts(train_groups, df, sector_map)
    valid_ctx = build_contexts(valid_groups, df, sector_map)
    test_ctx = build_contexts(test_groups, df, sector_map)

    # ---- Experiments ----
    configs = {
        "e2e_k3_ret": {"K": 3, "warm_start": False, "aux": 0.0, "seeds": args.seeds},
        "e2e_k3_ndcg": {"K": 3, "warm_start": True, "aux": 0.1, "seeds": args.seeds},
        "e2e_k5_ndcg": {"K": 5, "warm_start": True, "aux": 0.1, "seeds": args.seeds},
        "e2e_k3_ft": {"K": 3, "warm_start": True, "aux": 0.0, "seeds": args.seeds,
                      "freeze_head": True, "stage2_lr": 1e-4, "stage2_epochs": 30, "patience2": 8},
        "e2e_k3_shuf": {"K": 3, "warm_start": True, "aux": 0.1, "seeds": [42]},
        "e2e_k3_zero": {"K": 3, "warm_start": True, "aux": 0.1, "seeds": [42]},
    }
    if args.only_configs:
        configs = {k: v for k, v in configs.items() if k in args.only_configs}

    # Shuffle / zero negative-control contexts
    shuf_ctx = []
    for ctx in train_ctx + valid_ctx + test_ctx:
        sf = ctx["stock_features"].copy()
        perm = np.random.permutation(len(sf))
        sf[:, 1:] = sf[perm, 1:]
        shuf_ctx.append({"stock_features": sf, "market_features": ctx["market_features"]})
    zero_ctx = [
        {
            "stock_features": np.column_stack(
                [ctx["stock_features"][:, 0], np.zeros_like(ctx["stock_features"][:, 1:])]
            ).astype(np.float32),
            "market_features": ctx["market_features"],
        }
        for ctx in train_ctx + valid_ctx + test_ctx
    ]

    n_train = len(train_groups)
    results = {}
    for name, cfg in configs.items():
        for seed in cfg["seeds"]:
            if name == "e2e_k3_shuf":
                t_ctx = shuf_ctx[:n_train]
                v_ctx = shuf_ctx[n_train : n_train + len(valid_groups)]
                e_ctx = shuf_ctx[n_train + len(valid_groups) :]
            elif name == "e2e_k3_zero":
                t_ctx = zero_ctx[:n_train]
                v_ctx = zero_ctx[n_train : n_train + len(valid_groups)]
                e_ctx = zero_ctx[n_train + len(valid_groups) :]
            else:
                t_ctx, v_ctx, e_ctx = train_ctx, valid_ctx, test_ctx
            key = f"{name}_s{seed}"
            results[key] = run_experiment(
                name, cfg["K"], seed,
                train_groups, valid_groups, test_groups,
                t_ctx, v_ctx, e_ctx, device,
                cfg["warm_start"], cfg["aux"], args.epochs,
                cfg.get("patience2", args.patience),
                cfg.get("freeze_head", False),
                cfg.get("stage2_lr", 1e-3),
                cfg.get("stage2_epochs"),
            )

    # ---- Save results + champion weights ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "output" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cardnn_e2e_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # baselines from the real-Kronos verification JSON
    base_json = None
    for p in sorted((PROJECT_ROOT / "output" / "research").glob("plan_b_real_kronos_*.json")):
        base_json = p
    baselines = {}
    if base_json is not None:
        with open(base_json, encoding="utf-8") as f:
            bd = json.load(f)
        for k in ["Kronos-CS (Top-3)", "B-C2 (Market)", "B-C4 (CS Stats)", "B-C5 (Full Context)"]:
            if k in bd.get("baselines", {}):
                baselines[k] = bd["baselines"][k]
            elif k in bd.get("plan_b", {}):
                baselines[k] = bd["plan_b"][k]
    else:
        baselines = {"Kronos-CS (Top-3)": {"sharpe": float("nan"), "cum_return": float("nan"),
                     "win_rate": 0.0, "rank_ic": 0.0, "hit_at_5": 0.0}}

    # champion: best mean validation return among main configs
    main_keys = [k for k in results if any(k.startswith(c) for c in ("e2e_k3_ret", "e2e_k3_ndcg", "e2e_k5_ndcg", "e2e_k3_ft"))]
    best_key = max(main_keys, key=lambda k: results[k]["best_val_return"])
    champion = {"config": results[best_key]["experiment"], "seed": results[best_key]["seed"],
                "K": results[best_key]["K"], "sharpe": results[best_key]["sharpe"],
                "cum_return": results[best_key]["cum_return"],
                "n_positions": results[best_key]["n_positions"],
                "invested_frac": results[best_key]["invested_frac"]}

    # save champion weights (best config, seed 42 preferred for reproducibility)
    save_key = best_key if best_key.endswith("_s42") else f"{best_key.split('_s')[0]}_s42"
    if save_key not in results:
        save_key = best_key
    ckpt_dir = PROJECT_ROOT / "model" / "cardnn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Champion export: reuse the already-saved run checkpoint.
    src_ckpt = ckpt_dir / f"{save_key}.pt"
    if src_ckpt.exists():
        torch.save(
            torch.load(src_ckpt, map_location="cpu", weights_only=False),
            ckpt_dir / "kronos_cardnn_champion.pt",
        )
    print(f"Champion weights saved: model/cardnn/kronos_cardnn_champion.pt ({save_key})")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "data_end": str(df["日期"].max().date()),
        "score_source": "fine-tuned Kronos-small (lb60)",
        "cache_file": str(cache_path),
        "train_window": f"{args.train_start} ~ {args.train_end}",
        "valid_window": f"{args.valid_start} ~ {args.valid_end}",
        "test_window": f"{args.test_start} ~ {args.test_end}",
    }
    report_path = PROJECT_ROOT / "docs" / f"cardnn_e2e_report_{ts[:8]}.md"
    write_report(results, baselines, champion, meta, report_path)

    # summary table for console
    print("\n" + "=" * 100)
    print("CardNN E2E Summary")
    print("=" * 100)
    for k in sorted(results):
        r = results[k]
        print(
            f"{k:<24} Sharpe {r['sharpe']:>7.4f} | CumRet {r['cum_return']:>+8.4f} | "
            f"Win {r['win_rate']:>6.1%} | pos {r['n_positions']:>5.2f} | invest {r['invested_frac']:>6.3f} | "
            f"val_ret {r['best_val_return']:>+8.4f}"
        )


if __name__ == "__main__":
    main()
