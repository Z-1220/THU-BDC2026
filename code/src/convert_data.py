"""Baostock CSV -> Qlib 二进制数据格式转换。

直接写入 Qlib 期望的目录结构，避免调用外部脚本：
    <qlib_dir>/
        calendars/day.txt                 # 交易日历（YYYY-MM-DD，按升序）
        instruments/all.txt               # 股票列表 + 有效日期范围 (code, start, end)
        features/<lowercase_code>/<field>.day.bin  # 每只股票每个字段一份二进制

Qlib 二进制格式（日频）：
    - dtype = float32
    - 首个元素 = start_index（该股票首个交易日在全局日历中的索引，float32 cast）
    - 后续元素 = 按日历顺序对齐后的字段值，缺失日填 NaN

额外输出 data/sector_map.json：{QlibCode: 一级行业分类简称}
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd


# Baostock 原始列名 -> Qlib 标准字段名（统一小写、带 $ 前缀由 Qlib 表达式自动加）
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


def to_qlib_code(code: str) -> str:
    """A 股 6 位代码 -> Qlib 代码 (SHxxxxxx / SZxxxxxx)。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "5", "9")):
        return f"SH{code}"
    return f"SZ{code}"


def _write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    """写 Qlib 日频 .bin：首元素为 start_index（float32）+ 数据数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<f", float(start_index)))
        values.astype(np.float32).tofile(f)


def convert_baostock_to_qlib(
    csv_path: str,
    qlib_dir: str,
    sector_csv_path: str | None = None,
    sector_map_out: str | None = None,
) -> None:
    """把 Baostock CSV 转成 Qlib 二进制数据。"""
    qlib_dir = Path(qlib_dir)
    feat_dir = qlib_dir / "features"
    cal_dir = qlib_dir / "calendars"
    inst_dir = qlib_dir / "instruments"
    for d in (feat_dir, cal_dir, inst_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. 读取原始数据 ---
    df = pd.read_csv(csv_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    df["instrument"] = df["股票代码"].apply(to_qlib_code)

    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df = df.rename(columns={"日期": "date"})

    # 合成 vwap = amount / volume（Alpha158 许多因子会用到 $vwap）
    if "amount" in df.columns and "volume" in df.columns:
        vol = df["volume"].astype(float).replace(0, np.nan)
        df["vwap"] = df["amount"].astype(float) / vol
        df["vwap"] = df["vwap"].fillna(df.get("close"))

    # 因子标签 factor 恒为 1.0（未复权数据；Qlib 需要该字段计算成交金额等派生量）
    df["factor"] = 1.0

    # 只保留 Qlib 关心的列
    feature_fields = [v for v in COLUMN_MAP.values() if v in df.columns]
    if "vwap" in df.columns:
        feature_fields.append("vwap")
    feature_fields.append("factor")
    keep = ["date", "instrument", *feature_fields]
    df = df[keep].dropna(subset=["date", "instrument"]).copy()
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

    # --- 2. 写入全局交易日历 ---
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    with open(cal_dir / "day.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_dates) + "\n")

    # --- 3. 按股票写 features + instruments/all.txt ---
    inst_lines: list[str] = []
    grouped = df.groupby("instrument", sort=True)
    n_all = len(all_dates)

    for instrument, group in grouped:
        group = group.drop_duplicates(subset=["date"]).sort_values("date")
        start_date = group["date"].iloc[0]
        end_date = group["date"].iloc[-1]
        start_idx = date_to_idx[start_date]
        end_idx = date_to_idx[end_date]

        # 以 start_date..end_date 范围内的日历对齐
        sub_calendar = all_dates[start_idx : end_idx + 1]
        aligned = (
            group.set_index("date")
            .reindex(sub_calendar)[feature_fields]
            .astype(float)
        )

        code_dir = feat_dir / instrument.lower()
        for field in feature_fields:
            _write_bin(code_dir / f"{field}.day.bin", start_idx, aligned[field].to_numpy())

        inst_lines.append(f"{instrument}\t{start_date}\t{end_date}")

    with open(inst_dir / "all.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(inst_lines) + "\n")

    print(f"[convert_data] 完成: {len(grouped)} 只股票, {n_all} 个交易日 -> {qlib_dir}")

    # --- 4. 行业分类 -> sector_map.json ---
    if sector_csv_path and os.path.exists(sector_csv_path):
        ind_df = pd.read_csv(sector_csv_path, encoding="utf-8")
        code_col = "证券代码" if "证券代码" in ind_df.columns else ind_df.columns[0]
        sector_col = (
            "中证一级行业分类简称"
            if "中证一级行业分类简称" in ind_df.columns
            else ind_df.columns[1]
        )
        sector_map: dict[str, str] = {}
        for _, row in ind_df.iterrows():
            code = str(row[code_col]).zfill(6)
            qcode = to_qlib_code(code)
            sector = str(row[sector_col]) if pd.notna(row[sector_col]) else "未知"
            sector_map[qcode] = sector

        out_path = Path(sector_map_out) if sector_map_out else Path("data/sector_map.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sector_map, f, ensure_ascii=False, indent=2)
        print(f"[convert_data] sector_map.json 已写入: {out_path} ({len(sector_map)} 条)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/stock_data.csv", help="Baostock CSV 路径")
    parser.add_argument("--qlib_dir", default="./qlib_data/hs300_data", help="Qlib 数据输出目录")
    parser.add_argument(
        "--sector_csv",
        default="./data/行业分类.csv",
        help="中证行业分类 CSV 路径（可选）",
    )
    parser.add_argument(
        "--sector_map",
        default="./data/sector_map.json",
        help="输出的股票→行业映射 JSON 路径",
    )
    args = parser.parse_args()

    convert_baostock_to_qlib(
        csv_path=args.csv,
        qlib_dir=args.qlib_dir,
        sector_csv_path=args.sector_csv,
        sector_map_out=args.sector_map,
    )


if __name__ == "__main__":
    main()
