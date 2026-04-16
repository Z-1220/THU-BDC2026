import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import config

def calculate_metrics(returns):
    returns = np.array(returns)
    if len(returns) == 0:
        return {}
    cumulative_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cumulative_returns + 1)
    drawdown = (peak - (cumulative_returns + 1)) / peak
    max_drawdown = np.max(drawdown)
    win_rate = np.sum(returns > 0) / len(returns)
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    avg_loss = -np.mean(returns[returns < 0]) if np.any(returns < 0) else 1e-6
    profit_loss_ratio = avg_win / avg_loss
    return {
        '总调仓次数': len(returns),
        '最终累计收益率': cumulative_returns[-1],
        '最大回撤': max_drawdown,
        '胜率': win_rate,
        '盈亏比': profit_loss_ratio,
        '平均单期收益': np.mean(returns)
    }

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
    
    # 只保留行业列
    sector_map = sector_df.set_index('code_clean')['行业'].to_dict()

    print("2. 合并行业数据并计算未来收益...")
    raw_df['行业'] = raw_df['股票代码'].map(sector_map).fillna('未知')
    
    raw_df['open_t1'] = raw_df.groupby('股票代码')['开盘'].shift(-1)
    raw_df['open_t5'] = raw_df.groupby('股票代码')['开盘'].shift(-5)
    raw_df['future_return'] = (raw_df['open_t5'] - raw_df['open_t1']) / raw_df['open_t1']
    raw_df['future_return'] = raw_df['future_return'].fillna(0.0)

    print("3. 生成回测日期历...")
    all_dates = sorted(raw_df['日期'].unique())
    start_idx = config['sequence_length'] - 1
    rebalance_dates = all_dates[start_idx::5]
    
    print(f"回测区间: {rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}")

    print("4. 计算上帝视角：一级行业选股天花板...")
    sector_god_returns = []
    global_god_returns = [] # 对比全市场上帝视角
    valid_dates = []

    for signal_date in tqdm(rebalance_dates, desc="Sector God View"):
        daily_data = raw_df[raw_df['日期'] == signal_date].copy()
        if daily_data.empty:
            continue

        # --- 全市场上帝视角 (原理论最优) ---
        global_top5 = daily_data.nlargest(5, 'future_return')
        global_god_returns.append(global_top5['future_return'].mean())

        # --- 一级行业上帝视角 ---
        # 1. 找出未来5天板块平均涨幅第一的行业
        sector_avg_return = daily_data.groupby('行业')['future_return'].mean()
        best_sector = sector_avg_return.idxmax()
        
        # 2. 在该板块内，选未来5天涨幅最高的5只股票
        sector_stocks = daily_data[daily_data['行业'] == best_sector]
        
        if len(sector_stocks) >= 5:
            sector_top5 = sector_stocks.nlargest(5, 'future_return')
            sector_god_returns.append(sector_top5['future_return'].mean())
        else:
            # 如果该板块股票不足5只，全部买入
            sector_god_returns.append(sector_stocks['future_return'].mean())
            
        valid_dates.append(signal_date)

    print("\n5. 回测结果统计对比:")
    print("\n" + "="*40)
    print("全市场上帝视角 (无限制选股):")
    print("="*40)
    global_metrics = calculate_metrics(global_god_returns)
    for k, v in global_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n" + "="*40)
    print("一级行业上帝视角 (先选赛道再选车):")
    print("="*40)
    sector_metrics = calculate_metrics(sector_god_returns)
    for k, v in sector_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 绘制对比图
    try:
        plt.figure(figsize=(12, 6))
        global_cumulative = np.cumprod(1 + np.array(global_god_returns))
        sector_cumulative = np.cumprod(1 + np.array(sector_god_returns))
        
        plt.plot(valid_dates, global_cumulative, label='Global God View (No Restriction)', color='gray', linestyle='--')
        plt.plot(valid_dates, sector_cumulative, label='Sector God View (Top1 Industry -> Top5 Stocks)', color='purple', linewidth=2)
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.title('God View Comparison: Global vs Sector-First')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Wealth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sector_god_view_curve.png'))
        print(f"\n对比曲线图已保存至: {output_dir}/sector_god_view_curve.png")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
