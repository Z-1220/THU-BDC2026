import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from config import config

def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    sector_file = os.path.join(config['data_path'], 'hs300_stock_list_annotated.csv')
    output_dir = config['output_dir']

    print("1. 加载数据...")
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    sector_df = pd.read_csv(sector_file, dtype={'code_clean': str})
    sector_df['code_clean'] = sector_df['code_clean'].astype(str).str.zfill(6)
    sector_map = sector_df.set_index('code_clean')['行业'].to_dict()

    raw_df['行业'] = raw_df['股票代码'].map(sector_map).fillna('未知')

    raw_df['open_t1'] = raw_df.groupby('股票代码')['开盘'].shift(-1)
    raw_df['open_t5'] = raw_df.groupby('股票代码')['开盘'].shift(-5)
    raw_df['future_return'] = (raw_df['open_t5'] - raw_df['open_t1']) / raw_df['open_t1']
    raw_df['future_return'] = raw_df['future_return'].fillna(0.0)

    # 计算每只股票过去5天的收益率（用于惯性/反转信号）
    raw_df['open_t_5'] = raw_df.groupby('股票代码')['开盘'].shift(5)
    raw_df['past_5d_return'] = (raw_df['开盘'] - raw_df['open_t_5']) / raw_df['open_t_5']
    raw_df['past_5d_return'] = raw_df['past_5d_return'].fillna(0.0)

    print("2. 构建板块级别截面数据...")
    # 每个交易日、每个板块的平均收益率
    sector_daily = raw_df.groupby(['日期', '行业']).agg(
        future_return=('future_return', 'mean'),
        past_5d_return=('past_5d_return', 'mean'),
        stock_count=('股票代码', 'count')
    ).reset_index()

    all_dates = sorted(raw_df['日期'].unique())
    start_idx = config['sequence_length'] - 1
    rebalance_dates = all_dates[start_idx::5]

    print(f"回测区间: {rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}")
    print(f"调仓次数: {len(rebalance_dates)}")

    print("3. 测试简单板块轮动规则...")
    
    # 历史记录
    prev_best_sector = None
    prev_sector_ranks = None
    consecutive_best_count = 0

    results = {
        '惯性(过去第1选过去第1)': {'correct': 0, 'total': 0, 'returns': []},
        '反转(过去最后1选)': {'correct': 0, 'total': 0, 'returns': []},
        '连板惯性(连续≥2期第1选)': {'correct': 0, 'total': 0, 'returns': []},
        '连板反转(连续≥2期第1则回避选第2)': {'correct': 0, 'total': 0, 'returns': []},
        '随机(基线)': {'correct': 0, 'total': 0, 'returns': []},
    }

    for signal_date in tqdm(rebalance_dates, desc="Testing Rules"):
        current_data = sector_daily[sector_daily['日期'] == signal_date].copy()
        if current_data.empty:
            continue

        # 真实的未来板块排名
        current_data = current_data.sort_values('future_return', ascending=False)
        true_best_sector = current_data.iloc[0]['行业']
        all_sectors = current_data['行业'].tolist()

        # 随机基线：随机选一个板块
        random_pick = np.random.choice(all_sectors)
        random_return = current_data[current_data['行业'] == random_pick]['future_return'].iloc[0]
        results['随机(基线)']['total'] += 1
        results['随机(基线)']['correct'] += int(random_pick == true_best_sector)
        results['随机(基线)']['returns'].append(random_return)

        if prev_best_sector is None:
            prev_best_sector = true_best_sector
            prev_sector_ranks = {row['行业']: i for i, (_, row) in enumerate(current_data.iterrows())}
            continue

        # 规则1: 惯性 - 上期第1，这期还是第1？
        inertia_hit = (prev_best_sector == true_best_sector)
        inertia_return = current_data[current_data['行业'] == prev_best_sector]['future_return'].iloc[0]
        results['惯性(过去第1选过去第1)']['correct'] += int(inertia_hit)
        results['惯性(过去第1选过去第1)']['total'] += 1
        results['惯性(过去第1选过去第1)']['returns'].append(inertia_return)

        # 规则2: 反转 - 上期最后1名，这期变第1？
        prev_worst_sector = max(prev_sector_ranks, key=prev_sector_ranks.get)
        reversal_hit = (prev_worst_sector == true_best_sector)
        reversal_return = current_data[current_data['行业'] == prev_worst_sector]['future_return'].iloc[0]
        results['反转(过去最后1选)']['correct'] += int(reversal_hit)
        results['反转(过去最后1选)']['total'] += 1
        results['反转(过去最后1选)']['returns'].append(reversal_return)

        # 规则3: 连板惯性 - 如果上期已经是连续第1，继续选
        if consecutive_best_count >= 1:
            streak_hit = (prev_best_sector == true_best_sector)
            streak_return = inertia_return
        else:
            streak_hit = False
            streak_return = current_data.iloc[len(current_data)//2]['future_return']  # 选中间的
        results['连板惯性(连续≥2期第1选)']['correct'] += int(streak_hit)
        results['连板惯性(连续≥2期第1选)']['total'] += 1
        results['连板惯性(连续≥2期第1选)']['returns'].append(streak_return)

        # 规则4: 连板反转 - 连续第1太久了就回避
        if consecutive_best_count >= 2:
            # 选第2名
            second_sector = current_data.iloc[1]['行业']
            alt_hit = (second_sector == true_best_sector)
            alt_return = current_data.iloc[1]['future_return']
        else:
            alt_hit = inertia_hit
            alt_return = inertia_return
        results['连板反转(连续≥2期第1则回避选第2)']['correct'] += int(alt_hit)
        results['连板反转(连续≥2期第1则回避选第2)']['total'] += 1
        results['连板反转(连续≥2期第1则回避选第2)']['returns'].append(alt_return)

        # 更新状态
        if prev_best_sector == true_best_sector:
            consecutive_best_count += 1
        else:
            consecutive_best_count = 0
        prev_best_sector = true_best_sector
        prev_sector_ranks = {row['行业']: i for i, (_, row) in enumerate(current_data.iterrows())}

    # 输出结果
    print("\n" + "=" * 60)
    print("板块预测准确率对比 (随机基线 ≈ 9.1%)")
    print("=" * 60)
    
    for name, data in results.items():
        hit_rate = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        returns = np.array(data['returns'])
        avg_ret = np.mean(returns)
        cum_ret = np.cumprod(1 + returns)[-1] - 1 if len(returns) > 0 else 0
        
        print(f"\n【{name}】")
        print(f"  命中率: {hit_rate:.1f}% ({data['correct']}/{data['total']})")
        print(f"  平均板块收益: {avg_ret:.4f}")
        print(f"  累计收益: {cum_ret:.4f}")

    # 板块连续性统计
    print("\n" + "=" * 60)
    print("板块连续霸榜统计")
    print("=" * 60)
    print(f"板块最高连续霸榜期数: {max(consecutive_best_count, 0) if 'consecutive_best_count' in dir() else 'N/A'}")

if __name__ == "__main__":
    main()
