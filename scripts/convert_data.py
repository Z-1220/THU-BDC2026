"""Baostock CSV + 行业分类 → Qlib 二进制数据格式转换。

将行情数据与中证行业四级分类（编码为整数特征）一并写入 Qlib 结构：
    <qlib_dir>/
        calendars/day.txt
        instruments/all.txt
        features/<lowercase_code>/<field>.day.bin

特征列额外新增：
    sector_l1 .. sector_l4  （四级行业分类整数编码，无行业信息填 -1）
"""
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turn",
    "涨跌幅": "pctchg",
    "振幅": "amplitude",
    "涨跌额": "change",
}

SECTOR_LEVELS = [
    "中证一级行业分类代码",
    "中证二级行业分类代码",
    "中证三级行业分类代码",
    "中证四级行业分类代码",
]

# 获取 convert_data.py 所在目录 → scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
# 项目根目录 → scripts/ 的上一级
PROJECT_ROOT = SCRIPT_DIR.parent


def to_qlib_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "5", "9")):
        return f"SH{code}"
    return f"SZ{code}"


def _write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<f", float(start_index)))
        values.astype(np.float32).tofile(f)


def build_sector_features(sector_csv_path: str) -> dict[str, list[int]]:
    """从行业分类 CSV 构建股票→四级行业编码的字典。

    所有层级统一 factorize 为连续整数编码，若某股票缺失则返回 [-1,-1,-1,-1]。
    """
    if not os.path.exists(sector_csv_path):
        return {}

    ind_df = pd.read_csv(sector_csv_path, dtype={"证券代码": str})
    ind_df["证券代码"] = ind_df["证券代码"].str.zfill(6)
    ind_df["qlib_code"] = ind_df["证券代码"].apply(to_qlib_code)

    # 合并所有四个层级的字符串，统一 factorize
    all_values = []
    for col in SECTOR_LEVELS:
        if col in ind_df.columns:
            all_values.extend(ind_df[col].astype(str).tolist())
    codes, uniques = pd.factorize(all_values)  # 从0开始的整数

    # 切分回各层级，构建每只股票的(4,)整数列表
    n_stocks = len(ind_df)
    factorized = {}
    for i, row in ind_df.iterrows():
        qcode = row["qlib_code"]
        levels = []
        for col in SECTOR_LEVELS:
            if col in ind_df.columns:
                val = str(row[col])
                # 在 uniques 中查找索引
                idx = np.where(uniques == val)[0]
                if len(idx) > 0:
                    levels.append(int(idx[0]))
                else:
                    levels.append(-1)
            else:
                levels.append(-1)
        factorized[qcode] = levels
    return factorized


def convert_baostock_to_qlib(
    csv_path: str,
    qlib_dir: str,
    sector_csv_path: str | None = None,
) -> None:
    qlib_dir = Path(qlib_dir)
    feat_dir = qlib_dir / "features"
    cal_dir = qlib_dir / "calendars"
    inst_dir = qlib_dir / "instruments"
    for d in (feat_dir, cal_dir, inst_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. 读取行情数据 ---
    df = pd.read_csv(csv_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    df["instrument"] = df["股票代码"].apply(to_qlib_code)

    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df = df.rename(columns={"日期": "date"})

    if "amount" in df.columns and "volume" in df.columns:
        vol = df["volume"].astype(float).replace(0, np.nan)
        df["vwap"] = df["amount"].astype(float) / vol
        df["vwap"] = df["vwap"].fillna(df.get("close"))

    df["factor"] = 1.0

    feature_fields = [v for v in COLUMN_MAP.values() if v in df.columns]
    if "vwap" in df.columns:
        feature_fields.append("vwap")
    feature_fields.append("factor")

    # --- 2. 加载行业特征，并加入特征列表 ---
    sector_features = {}
    if sector_csv_path:
        sector_features = build_sector_features(sector_csv_path)
        if sector_features:  # 如果成功加载，则添加四个 sector 列
            feature_fields.extend(["sector_l1", "sector_l2", "sector_l3", "sector_l4"])

    keep = ["date", "instrument", *feature_fields]
    # 只保留行情数据已有的列（sector列暂时不在这里）
    available_cols = [c for c in keep if c in df.columns or c.startswith("sector")]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["date", "instrument"]).copy()
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

    # --- 3. 交易日历 ---
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    with open(cal_dir / "day.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_dates) + "\n")

    # --- 4. 按股票写 features ---
    inst_lines = []
    grouped = df.groupby("instrument", sort=True)
    n_all = len(all_dates)

    for instrument, group in grouped:
        group = group.drop_duplicates(subset=["date"]).sort_values("date")
        start_date = group["date"].iloc[0]
        end_date = group["date"].iloc[-1]
        start_idx = date_to_idx[start_date]
        end_idx = date_to_idx[end_date]

        sub_calendar = all_dates[start_idx : end_idx + 1]
        aligned = (
            group.set_index("date")
            .reindex(sub_calendar)[[f for f in feature_fields if f in group.columns]]
            .astype(float)
        )

        # 填充行业特征（静态值，每个日期都相同）
        sector_codes = sector_features.get(instrument, [-1, -1, -1, -1])
        for i, col in enumerate(["sector_l1", "sector_l2", "sector_l3", "sector_l4"]):
            if col in feature_fields:
                aligned[col] = sector_codes[i]

        code_dir = feat_dir / instrument.lower()
        for field in aligned.columns:
            _write_bin(code_dir / f"{field}.day.bin", start_idx, aligned[field].to_numpy())

        inst_lines.append(f"{instrument}\t{start_date}\t{end_date}")

    with open(inst_dir / "all.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(inst_lines) + "\n")

    print(f"[convert_data] 完成: {len(grouped)} 只股票, {n_all} 个交易日 -> {qlib_dir}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "model" / "data" / "train.csv"),
        help="行情 CSV 路径",
    )
    parser.add_argument(
        "--qlib_dir",
        default=str(PROJECT_ROOT / "model" / "qlib_data"),
        help="Qlib 数据输出目录",
    )
    parser.add_argument(
        "--sector_csv",
        default=str(PROJECT_ROOT / "resource" / "行业分类.csv"),
        help="中证行业分类 CSV 路径",
    )
    args = parser.parse_args()

    convert_baostock_to_qlib(
        csv_path=args.csv,
        qlib_dir=args.qlib_dir,
        sector_csv_path=args.sector_csv,
    )

if __name__ == "__main__":
    main()