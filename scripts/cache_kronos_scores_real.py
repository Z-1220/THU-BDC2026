#!/usr/bin/env python
"""Pre-compute real Kronos scores for all signal dates and cache to disk.

Standalone variant of cache_kronos_scores.py that supports:
  - configurable start/end dates (train/valid/test segments)
  - optional fine-tuned Kronos weights (production champion pipeline)

Each experiment loads the cached scores instead of re-running Kronos.

Usage:
    uv run python scripts/cache_kronos_scores_real.py \
        --start-date 2024-01-01 --end-date 2026-07-31 \
        --finetuned-dir ./model/kronos_finetuned/Kronos-small-lb60
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from qlib import init as qlib_init
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import DropnaLabel, Fillna

from handlers.stock_handler import StockDataHandler
from processors.custom_processor import FridayFilterProcessor, ScreenProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit signal dates (0=all)")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument(
        "--handler-start",
        default="2022-01-01",
        help="Qlib handler data start (pre-2024 history needed for ScreenProcessor MA60 warm-up)",
    )
    parser.add_argument("--end-date", default="2026-07-31")
    parser.add_argument(
        "--finetuned-dir",
        default=None,
        help="Optional fine-tuned Kronos weights dir (default: pretrained zero-shot)",
    )
    parser.add_argument("--output", default="temp/kronos_scores_cache.pkl")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing Qlib...")
    qlib_init(provider_uri="./temp/qlib_data", region="cn")

    print("Building Kronos model...")
    from models.Kronos.KronosModel import KronosModel

    model = KronosModel(
        model_name="small",
        pretrained_dir="./model/kronos_pretrained",
        finetuned_dir=args.finetuned_dir,
        max_context=512,
        pred_len=5,
        seed=42,
        cs_zscore=False,  # Save raw scores — CS processing done later
    )
    model._fitted = True

    print("Building dataset...")
    handler = StockDataHandler(
        instruments="all",
        start_time=args.handler_start,
        end_time=args.end_date,
        fit_start_time=args.handler_start,
        fit_end_time="2025-12-31",
        infer_processors=[
            FridayFilterProcessor(),
            Fillna(fields_group="feature"),
            ScreenProcessor(min_amount_rank=0.3, trend_ma=60, max_drawdown=0.15),
        ],
        learn_processors=[DropnaLabel(fields_group="label")],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": [args.start_date, "2025-12-31"],
            "valid": ["2026-01-05", "2026-01-30"],
            "test": ["2026-02-02", args.end_date],
        },
    )

    all_scores = {}
    for segment in ["train", "valid", "test"]:
        print(f"\nRunning Kronos on segment '{segment}'...")
        try:
            scores = model.predict(dataset, segment=segment)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        signal_dates = sorted(scores.index.get_level_values(0).unique())
        print(f"  Got {len(scores)} scores across {len(signal_dates)} signal dates")

        if args.limit > 0 and len(signal_dates) > args.limit:
            signal_dates = signal_dates[-args.limit :]
            print(
                f"  Limiting to last {args.limit} dates: "
                f"{signal_dates[0].date()} to {signal_dates[-1].date()}"
            )

        for dt in signal_dates:
            try:
                all_scores[(segment, dt)] = scores.loc[dt]
            except KeyError:
                continue

    with open(output_path, "wb") as f:
        pickle.dump(all_scores, f)

    print(f"\nSaved {len(all_scores)} date-segment score entries to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    train_dates = sum(1 for k in all_scores if k[0] == "train")
    valid_dates = sum(1 for k in all_scores if k[0] == "valid")
    test_dates = sum(1 for k in all_scores if k[0] == "test")
    print(f"Dates: train={train_dates}, valid={valid_dates}, test={test_dates}")


if __name__ == "__main__":
    main()
