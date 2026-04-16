#!/usr/bin/env python3
"""
自动更新股票数据并重新划分训练/测试集的脚本
按照工作流执行以下步骤：
1. 更新 get_stock_data.py 中的 end_date 为今日
2. 运行数据获取脚本
3. 计算并更新 split_train_test.py 的参数
4. 执行数据切分
5. 验证输出文件
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def update_end_date(today_str: str) -> None:
    """更新 get_stock_data.py 中的 end_date"""
    file_path = "get_stock_data.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 替换 end_date
    pattern = r'end_date = ".*?"'
    replacement = f'end_date = "{today_str}"'
    new_content = re.sub(pattern, replacement, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"已更新 {file_path} 中的 end_date 为 {today_str}")


def run_data_fetch() -> None:
    """运行数据获取脚本"""
    print("开始运行数据获取脚本...")
    result = subprocess.run([sys.executable, "get_stock_data.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"数据获取失败: {result.stderr}")
        sys.exit(1)
    print("数据获取完成")


def update_split_params() -> None:
    """更新 split_train_test.py 的参数"""
    # 读取数据
    df = pd.read_csv("data/stock_data.csv")
    dates = sorted(pd.to_datetime(df["日期"].unique()))
    if len(dates) < 6:
        print("数据中的交易日不足，无法划分测试集")
        sys.exit(1)

    test_start = str(dates[-5].date())
    train_end = str(dates[-6].date())
    test_end = str(dates[-1].date())

    file_path = "split_train_test.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 更新默认值
    content = re.sub(
        r'default="[^"]*",\s*help="训练集结束日期"',
        f'default="{train_end}", help="训练集结束日期"',
        content,
    )
    content = re.sub(
        r'default="[^"]*",\s*help="测试集开始日期"',
        f'default="{test_start}", help="测试集开始日期"',
        content,
    )
    content = re.sub(
        r'default="[^"]*",\s*help="测试集结束日期"',
        f'default="{test_end}", help="测试集结束日期"',
        content,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"已更新 {file_path} 参数: train_end={train_end}, test_start={test_start}, test_end={test_end}")


def run_data_split() -> None:
    """运行数据切分脚本"""
    print("开始运行数据切分脚本...")
    result = subprocess.run([sys.executable, "split_train_test.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"数据切分失败: {result.stderr}")
        sys.exit(1)
    print("数据切分完成")


def validate_output() -> None:
    """验证输出文件"""
    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("输出文件不存在")
        sys.exit(1)

    train_size = train_path.stat().st_size
    test_size = test_path.stat().st_size
    train_mtime = datetime.fromtimestamp(train_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    test_mtime = datetime.fromtimestamp(test_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    print("输出文件验证:")
    print(f"  train.csv: {train_size} bytes, 修改时间: {train_mtime}")
    print(f"  test.csv: {test_size} bytes, 修改时间: {test_mtime}")


def main() -> None:
    """主函数"""
    print("开始自动更新股票数据和划分数据集...")

    # 获取今日日期
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    print(f"今日日期: {today_str}")

    # 步骤1: 更新 end_date
    update_end_date(today_str)

    # 步骤2: 运行数据获取
    run_data_fetch()

    # 步骤3: 更新划分参数
    update_split_params()

    # 步骤4: 运行数据切分
    run_data_split()

    # 步骤5: 验证输出
    validate_output()

    print("所有步骤完成！")


if __name__ == "__main__":
    main()