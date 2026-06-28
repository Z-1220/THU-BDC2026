#!/usr/bin/env python
"""Pre-compute Kronos base scores for all signal dates and cache to disk.

This eliminates redundant Kronos inference across experiments.
Each experiment loads cached scores instead of re-running Kronos.

Usage:
    python scripts/cache_kronos_scores.py            # Full cache
    python scripts/cache_kronos_scores.py --limit 20 # Only last 20 dates
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
from qlib.data.dataset.processor import DropnaLabel

from handlers.stock_handler import StockDataHandler
from processors.custom_processor import FridayFilterProcessor, ScreenProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit signal dates (0=all)")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-06-26")
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
        max_context=512,
        pred_len=5,
        seed=42,
        cs_zscore=False,  # Save raw scores — CS processing done later
    )
    model._fitted = True

    print("Building dataset...")
    handler = StockDataHandler(
        instruments="all",
        start_time=args.start_date,
        end_time=args.end_date,
        fit_start_time=args.start_date,
        fit_end_time="2025-12-31",
        infer_processors=[
            FridayFilterProcessor(),
            type("Fillna", (), {"__call__": lambda s, df: df.fillna(0)})() if False else None,
            ScreenProcessor(min_amount_rank=0.3, trend_ma=60, max_drawdown=0.15),
        ],
        learn_processors=[DropnaLabel(fields_group="label")],
    )
    # Note: infer_processors with Fillna need special handling
    # Use simpler approach — let StockDataHandler handle Fillna internally

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ["2022-01-01", "2025-12-31"],
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
            print(f"  Limiting to last {args.limit} dates: {signal_dates[0].date()} to {signal_dates[-1].date()}")

        for dt in signal_dates:
            try:
                dt_scores = scores.loc[dt]
                key = (segment, dt)
                all_scores[key] = dt_scores
            except KeyError:
                continue

    # Save cache
    with open(output_path, "wb") as f:
        pickle.dump(all_scores, f)

    print(f"\nSaved {len(all_scores)} date-segment score entries to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    train_dates = sum(1 for k in all_scores if k[0] == "train")
    valid_dates = sum(1 for k in all_scores if k[0] == "valid")
    test_dates = sum(1 for k in all_scores if k[0] == "test")
    print(f"Dates: train={train_dates}, valid={valid_dates}, test={test_dates}")


if __name__ == "__main__":
    main()
