"""Fine-tune Kronos predictor on A-share OHLCV data.

Loads pretrained KronosTokenizer (frozen) and Kronos (trainable), then
fine-tunes with autoregressive next-token prediction on CSI 300 stocks.

Usage:
    python scripts/finetune_kronos.py --model small --epochs 20
    python scripts/finetune_kronos.py --model base --epochs 20 --lr 2e-5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

from models.Kronos.kronos_src import Kronos, KronosTokenizer  # noqa: E402

# ---- column mapping: stock_data.csv -> Kronos ----
_COL_MAP = {
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
_REQUIRED = ["open", "high", "low", "close", "volume", "amount"]

# ---- time split (consistent with YAML segments) ----
_TRAIN_END = "2025-12-31"
_VAL_START = "2026-01-05"
_VAL_END = "2026-01-30"


# =====================================================================
# Dataset
# =====================================================================
class KlineFinetuneDataset(Dataset):
    """Sliding-window OHLCV dataset for Kronos autoregressive fine-tuning.

    Each sample is a (window, 6) array for one stock. The window covers
    lookback + pred_len + 1 steps so there is one extra target token.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        lookback: int = 60,
        pred_len: int = 5,
    ):
        df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["日期"])
        df = df.rename(columns=_COL_MAP)
        df["code"] = df["股票代码"].astype(str)

        self.window = lookback + pred_len + 1
        self.lookback = lookback
        self.clip = 5.0
        self.required_cols = _REQUIRED

        if split == "train":
            # Train: samples must end on or before _TRAIN_END
            df = df[df["日期"] <= _TRAIN_END].copy()
        else:
            # Val: include data before _VAL_START for lookback context,
            # but only create samples whose last date is in [_VAL_START, _VAL_END]
            df = df[df["日期"] <= _VAL_END].copy()

        # Pre-load all stock data into memory for fast __getitem__
        self._stock_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        codes = sorted(df["code"].unique())
        for c in codes:
            cdf = df[df["code"] == c].sort_values("日期")
            if len(cdf) < self.window:
                continue
            vals = cdf[self.required_cols].values.astype(np.float32)
            dates = cdf["日期"].values
            self._stock_data[c] = (vals, dates)

        # Build flat index of (code, start_idx) tuples
        self.samples: list[tuple[str, int]] = []
        for c, (vals, dates) in self._stock_data.items():
            n = len(vals)
            valid_mask = ~np.isnan(vals).any(axis=1)
            for i in range(n - self.window + 1):
                if split == "val":
                    last_date = pd.Timestamp(dates[i + self.window - 1])
                    if last_date < pd.Timestamp(_VAL_START):
                        continue
                if valid_mask[i : i + self.window].all():
                    self.samples.append((c, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        code, start = self.samples[idx]
        vals, dates = self._stock_data[code]

        win = vals[start : start + self.window].copy()

        # Normalize using only lookback portion (no data leakage)
        past = win[: self.lookback]
        mean = past.mean(axis=0)
        std = past.std(axis=0)
        win = (win - mean) / (std + 1e-5)
        win = np.clip(win, -self.clip, self.clip)

        # Build time features [minute, hour, weekday, day, month]
        win_dates = pd.DatetimeIndex(dates[start : start + self.window])
        stamp = np.stack(
            [
                np.zeros(len(win_dates), dtype=np.float32),  # minute
                np.zeros(len(win_dates), dtype=np.float32),  # hour
                win_dates.weekday.values.astype(np.float32),
                win_dates.day.values.astype(np.float32),
                win_dates.month.values.astype(np.float32),
            ],
            axis=1,
        )

        return torch.from_numpy(win), torch.from_numpy(stamp)


# =====================================================================
# Training
# =====================================================================
def train_epoch(
    model: Kronos,
    tokenizer: KronosTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_stamp in loader:
        batch_x = batch_x.to(device)
        batch_stamp = batch_stamp.to(device)

        # Tokenize OHLCV (frozen tokenizer)
        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(batch_x, half=True)

        # Offset: input[0..T-1] -> target[1..T]
        s1_in = s1_ids[:, :-1]
        s2_in = s2_ids[:, :-1]
        s1_tgt = s1_ids[:, 1:]
        s2_tgt = s2_ids[:, 1:]
        stamp_in = batch_stamp[:, :-1, :]

        s1_logits, s2_logits = model(s1_in, s2_in, stamp=stamp_in)
        loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, s1_tgt, s2_tgt)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: Kronos,
    tokenizer: KronosTokenizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_stamp in loader:
        batch_x = batch_x.to(device)
        batch_stamp = batch_stamp.to(device)

        s1_ids, s2_ids = tokenizer.encode(batch_x, half=True)
        s1_in, s2_in = s1_ids[:, :-1], s2_ids[:, :-1]
        s1_tgt, s2_tgt = s1_ids[:, 1:], s2_ids[:, 1:]
        stamp_in = batch_stamp[:, :-1, :]

        s1_logits, s2_logits = model(s1_in, s2_in, stamp=stamp_in)
        loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, s1_tgt, s2_tgt)

        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / max(total_samples, 1)


def finetune(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_size is None:
        args.batch_size = 512 if args.model == "small" else 256
    print(f"Device: {device}")

    pretrained_dir = Path(args.pretrained_dir)
    tokenizer_dir = pretrained_dir / "Kronos-Tokenizer-base"
    model_dir = pretrained_dir / f"Kronos-{args.model}"

    for d in [tokenizer_dir, model_dir]:
        if not d.exists():
            raise FileNotFoundError(f"{d} not found. Run scripts/download_kronos_models.py first.")

    # ---- Load models ----
    print(f"Loading tokenizer from {tokenizer_dir} ...")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    print(f"Loading Kronos-{args.model} from {model_dir} ...")
    model = Kronos.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,}")

    # ---- Data ----
    csv_path = _PROJECT_ROOT / "data" / "stock_data.csv"
    train_ds = KlineFinetuneDataset(str(csv_path), split="train", lookback=args.lookback, pred_len=args.pred_len)
    val_ds = KlineFinetuneDataset(str(csv_path), split="val", lookback=args.lookback, pred_len=args.pred_len)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Save dir ----
    save_dir = Path(args.save_dir) / f"Kronos-{args.model}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, device, args.grad_clip)
        val_loss = validate(model, tokenizer, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.0f}s")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            model.save_pretrained(str(save_dir))
            print(f"  -> saved best to {save_dir}")

        # Early stop
        if epoch - best_epoch >= args.early_stop:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nDone. Best val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {save_dir}")


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Kronos on A-share OHLCV")
    parser.add_argument("--model", default="small", choices=["small", "base"])
    parser.add_argument("--pretrained-dir", default=str(_PROJECT_ROOT / "model" / "kronos_pretrained"))
    parser.add_argument("--save-dir", default=str(_PROJECT_ROOT / "model" / "kronos_finetuned"))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: 64 (small) / 32 (base)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--early-stop", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Kronos-{args.model} Fine-tuning")
    print(f"  lookback={args.lookback}  pred_len={args.pred_len}")
    print(f"  batch_size={args.batch_size}  epochs={args.epochs}  lr={args.lr}")
    print("=" * 60)
    finetune(args)


if __name__ == "__main__":
    main()
