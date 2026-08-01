"""Stage 1: Fine-tune Kronos Tokenizer on A-share OHLCV data.

Adapts the BSQ codebook to the target market's price/volume distribution.
The tokenizer is an autoencoder: encoder → BSQ quantizer → decoder.
Training objective: L_coarse + L_fine + λ * L_quant

Usage:
    python scripts/finetune_kronos_tokenizer.py --epochs 10
    python scripts/finetune_kronos_tokenizer.py --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

from models.Kronos.kronos_src import KronosTokenizer  # noqa: E402

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

# ---- time splits ----
_TRAIN_START = os.environ.get("FINETUNE_TRAIN_START", None)
_TRAIN_END = os.environ.get("FINETUNE_TRAIN_END", "2025-12-31")
_VAL_START = os.environ.get("FINETUNE_VAL_START", "2026-01-05")
_VAL_END = os.environ.get("FINETUNE_VAL_END", "2026-01-30")


# =====================================================================
# Dataset (reuse same sliding-window logic as predictor fine-tuning)
# =====================================================================
class KlineFinetuneDataset(torch.utils.data.Dataset):
    """Sliding-window OHLCV dataset for tokenizer fine-tuning.

    Each sample is a (window, 6) array. Unlike predictor training, the
    tokenizer only needs lookback-windows (no pred_len target offset),
    so window = lookback.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        lookback: int = 60,
    ):
        df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["日期"])
        df = df.rename(columns=_COL_MAP)
        df["code"] = df["股票代码"].astype(str).str.zfill(6)

        self.window = lookback
        self.clip = 5.0
        self.required_cols = _REQUIRED

        if split == "train":
            df = df[df["日期"] <= _TRAIN_END].copy()
            if _TRAIN_START is not None:
                df = df[df["日期"] >= _TRAIN_START].copy()
        else:
            df = df[df["日期"] <= _VAL_END].copy()

        # Pre-load all stock data into memory
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        code, start = self.samples[idx]
        vals, _dates = self._stock_data[code]

        win = vals[start : start + self.window].copy()

        # Per-sample z-score normalization (no data leakage)
        mean = win.mean(axis=0)
        std = win.std(axis=0)
        win = (win - mean) / (std + 1e-5)
        win = np.clip(win, -self.clip, self.clip)

        return torch.from_numpy(win)


# =====================================================================
# Training
# =====================================================================
def train_epoch(
    tokenizer: KronosTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_quant: float,
    grad_clip: float,
) -> float:
    tokenizer.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x in loader:
        batch_x = batch_x.to(device)

        # Forward: encoder → BSQ quantizer → decoder
        (z_pre, z_full), bsq_loss, _quantized, _z_indices = tokenizer(batch_x)

        # Reconstruction losses
        loss_coarse = nn.functional.mse_loss(z_pre, batch_x)
        loss_fine = nn.functional.mse_loss(z_full, batch_x)
        loss_recon = loss_coarse + loss_fine

        # Total loss with quantization commitment
        loss = loss_recon + lambda_quant * bsq_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tokenizer.parameters(), grad_clip)
        optimizer.step()

        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    tokenizer: KronosTokenizer,
    loader: DataLoader,
    device: torch.device,
    lambda_quant: float,
) -> float:
    tokenizer.eval()
    total_loss = 0.0
    total_samples = 0

    for batch_x in loader:
        batch_x = batch_x.to(device)

        (z_pre, z_full), bsq_loss, _quantized, _z_indices = tokenizer(batch_x)

        loss_coarse = nn.functional.mse_loss(z_pre, batch_x)
        loss_fine = nn.functional.mse_loss(z_full, batch_x)
        loss = loss_coarse + loss_fine + lambda_quant * bsq_loss

        n = batch_x.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / max(total_samples, 1)


def finetune(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_dir = Path(args.pretrained_dir) / "Kronos-Tokenizer-base"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(
            f"{tokenizer_dir} not found. Run scripts/download_kronos_models.py first."
        )

    # ---- Load pretrained tokenizer ----
    print(f"Loading tokenizer from {tokenizer_dir} ...")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tokenizer.to(device)

    n_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"Tokenizer params: {n_params:,}")

    # ---- Data ----
    csv_path = _PROJECT_ROOT / "model" / "data" / "stock_data.csv"
    train_ds = KlineFinetuneDataset(str(csv_path), split="train", lookback=args.lookback)
    val_ds = KlineFinetuneDataset(str(csv_path), split="val", lookback=args.lookback)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=4,
    )

    # ---- Optimizer (lower LR for tokenizer codebook) ----
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=args.lr, weight_decay=0.01,
    )

    # ---- Save dir ----
    save_dir = Path(args.save_dir) / "Kronos-Tokenizer-base-ft"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            tokenizer, train_loader, optimizer, device,
            args.lambda_quant, args.grad_clip,
        )
        val_loss = validate(
            tokenizer, val_loader, device, args.lambda_quant,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"{elapsed:.0f}s"
        )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            tokenizer.save_pretrained(str(save_dir))
            print(f"  -> saved best to {save_dir}")

        # Early stop
        if epoch - best_epoch >= args.early_stop:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nDone. Best val loss: {best_val:.6f} at epoch {best_epoch}")
    print(f"Tokenizer saved to: {save_dir}")


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Kronos Tokenizer on A-share OHLCV",
    )
    parser.add_argument(
        "--pretrained-dir",
        default=str(_PROJECT_ROOT / "model" / "kronos_pretrained"),
    )
    parser.add_argument(
        "--save-dir",
        default=str(_PROJECT_ROOT / "model" / "kronos_phase2"),
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-quant", type=float, default=0.25,
                        help="Weight of BSQ commitment loss")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  Kronos Tokenizer Fine-tuning (Stage 1)")
    print(f"  lookback={args.lookback}  epochs={args.epochs}  lr={args.lr}")
    print(f"  lambda_quant={args.lambda_quant}")
    print("=" * 60)
    finetune(args)


if __name__ == "__main__":
    main()
