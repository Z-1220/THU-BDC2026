import os
import pandas as pd
import numpy as np
from config import config

def check_backtest_data():
    print("=" * 60)
    print("回测数据完整性检查")
    print("=" * 60)

    data_file = os.path.join(config['data_path'], 'train.csv')
    if not os.path.exists(data_file):
        print(f"[错误] 数据文件不存在: {data_file}")
        return

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 计算未来价格
    raw_df['open_t1'] = raw_df.groupby('股票代码')['开盘'].shift(-1)
    raw_df['open_t5'] = raw_df.groupby('股票代码')['开盘'].shift(-5)

    # 基础信息
    print(f"\n[基本信息]")
    print(f"数据总行数: {len(raw_df)}")
    print(f"股票数量: {raw_df['股票代码'].nunique()}")
    print(f"日期范围: {raw_df['日期'].min().date()} 至 {raw_df['日期'].max().date()}")

    # 检查关键列缺失
    required_cols = ['开盘', '股票代码', '日期']
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        print(f"\n[错误] 缺少必需列: {missing_cols}")
        return
    print(f"[通过] 必需列完整")

    # 检查空值
    print(f"\n[空值统计]")
    null_report = raw_df[['开盘', 'open_t1', 'open_t5']].isnull().sum()
    print(null_report)

    # 检查未来价格缺失对回测尾部的影响
    print(f"\n[尾部数据未来价格可用性]")
    # 按股票分组，看最后几天的 open_t1 和 open_t5 缺失情况
    tail_dates = raw_df['日期'].unique()[-10:]  # 最后10个交易日
    tail_data = raw_df[raw_df['日期'].isin(tail_dates)]
    tail_availability = tail_data.groupby('日期').agg({
        'open_t1': lambda x: x.isnull().mean(),
        'open_t5': lambda x: x.isnull().mean()
    }).round(4)
    print("最后10个交易日 open_t1 / open_t5 缺失比例:")
    print(tail_availability)

    # 检查是否有价格为0的情况（会导致收益计算除零错误）
    print(f"\n[价格为0检查]")
    zero_open = (raw_df['开盘'] == 0).sum()
    zero_t1 = (raw_df['open_t1'] == 0).sum()
    zero_t5 = (raw_df['open_t5'] == 0).sum()
    print(f"开盘价为0的行数: {zero_open}")
    print(f"open_t1为0的行数: {zero_t1}")
    print(f"open_t5为0的行数: {zero_t5}")
    if zero_t1 > 0 or zero_t5 > 0:
        print("[警告] 存在 open_t1 或 open_t5 为 0，会导致收益计算为无穷大或 nan，已在回测中跳过但可能影响结果。")
    else:
        print("[通过] 无价格为0情况")

    # 检查停牌/无交易导致的连续缺失
    print(f"\n[连续交易日缺失检查（停牌）]")
    # 统计每只股票相邻日期间隔 > 5 天的情况
    gaps = raw_df.groupby('股票代码')['日期'].diff().dt.days
    large_gaps = gaps[gaps > 5]
    if len(large_gaps) > 0:
        print(f"发现 {len(large_gaps)} 处相邻交易日间隔 > 5 天（可能停牌），将影响未来价格计算。")
        # 显示前5个示例
        gap_examples = large_gaps.reset_index()
        gap_examples['股票代码'] = gap_examples['index'].map(lambda i: raw_df.loc[i, '股票代码'])
        print("示例（前5条）:")
        print(gap_examples[['股票代码', '日期']].head())
    else:
        print("[通过] 无超过5天的交易日间隔")

    # 检查特征工程所需数据是否完整（依据 config['feature_num']）
    print(f"\n[特征列检查]")
    # 这里仅列出常用的几个特征列，具体可根据你的 config 调整
    possible_feature_cols = ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率']
    available = [c for c in possible_feature_cols if c in raw_df.columns]
    print(f"可用特征列: {available}")

    # 回测调仓日期模拟
    print(f"\n[回测调仓日期模拟]")
    all_dates = sorted(raw_df['日期'].unique())
    seq_len = config.get('sequence_length', 60)  # 默认60，请根据实际config调整
    start_idx = seq_len - 1
    rebalance_dates = all_dates[start_idx::5]
    print(f"序列长度: {seq_len}")
    print(f"回测起始可用日期: {all_dates[start_idx].date() if start_idx < len(all_dates) else '无'}")
    print(f"回测结束日期: {rebalance_dates[-1].date() if len(rebalance_dates) > 0 else '无'}")
    print(f"预计调仓次数: {len(rebalance_dates)}")

    # 检查每个调仓日的可交易股票数量（至少需要5只）
    print(f"\n[调仓日可交易股票数量检查]")
    min_count = float('inf')
    low_count_dates = []
    for d in rebalance_dates:
        # 当日有数据且 open_t1 和 open_t5 均非空的股票数
        day_data = raw_df[raw_df['日期'] == d]
        valid_stocks = day_data[day_data['open_t1'].notnull() & day_data['open_t5'].notnull()]
        count = len(valid_stocks)
        if count < min_count:
            min_count = count
        if count < 5:
            low_count_dates.append((d.date(), count))

    print(f"调仓日有效股票数最小值: {min_count}")
    if low_count_dates:
        print(f"[警告] 以下日期有效股票数不足5只:")
        for dt, cnt in low_count_dates[:10]:
            print(f"  {dt}: {cnt} 只")
    else:
        print("[通过] 所有调仓日有效股票数 >= 5")

    print("\n" + "=" * 60)
    print("检查完成。若存在警告或错误，请根据提示处理数据。")
    print("=" * 60)

if __name__ == "__main__":
    check_backtest_data()