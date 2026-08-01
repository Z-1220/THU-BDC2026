"""从 stock_data.csv 生成 train.csv，排除主办方 test.csv 的日期区间。

test.csv 是主办方下发的最终盲测集（2026-04-13 ~ 2026-04-17），不可修改。
本脚本确保 train.csv 包含除盲测区间外所有可用数据。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 stock_data.csv 生成 train.csv（排除盲测日期区间）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "model" / "data" / "stock_data.csv"),
        help="原始 stock_data.csv 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "model" / "data" / "train.csv"),
        help="输出 train.csv 路径",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "test.csv"),
        help="主办方下发的 test.csv 路径（用于推断盲测日期区间）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    test_csv_path = Path(args.test_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"stock_data.csv 不存在: {input_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"test.csv 不存在: {test_csv_path}")

    # 读取原始数据
    df = pd.read_csv(input_path, dtype={"股票代码": str})
    df["日期"] = pd.to_datetime(df["日期"])
    print(f"stock_data.csv: {len(df)} 行, {df['股票代码'].nunique()} 只股票")
    print(f"  日期范围: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # 从 test.csv 推断盲测日期区间
    test_df = pd.read_csv(test_csv_path, dtype={"股票代码": str})
    test_df["日期"] = pd.to_datetime(test_df["日期"])
    test_min = test_df["日期"].min()
    test_max = test_df["日期"].max()
    print(f"test.csv 盲测区间: {test_min.date()} ~ {test_max.date()} (保留不变)")

    # 排除盲测区间
    mask = (df["日期"] < test_min) | (df["日期"] > test_max)
    train_df = df.loc[mask].copy()
    train_df = train_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    train_df["日期"] = train_df["日期"].dt.strftime("%Y-%m-%d")

    # 统计
    excluded = len(df) - len(train_df)
    print(f"排除 {excluded} 行（盲测区间数据）")
    print(f"train.csv: {len(train_df)} 行, {train_df['股票代码'].nunique()} 只股票")

    train_min = train_df["日期"].min()
    train_max = train_df["日期"].max()
    print(f"  日期范围: {train_min} ~ {train_max}")
    if pd.to_datetime(train_max) < test_max:
        # train 末尾之后、test 之后有更新数据
        after_test = df[df["日期"] > test_max]
        if len(after_test) > 0:
            print(f"  含盲测后数据: {after_test['日期'].min().date()} ~ {after_test['日期'].max().date()} "
                  f"({len(after_test)} 行)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
