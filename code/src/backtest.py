import os
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences


def calculate_metrics(returns):
    """计算回测指标"""
    returns = np.array(returns)
    if len(returns) == 0:
        return {}

    cumulative_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cumulative_returns + 1)
    drawdown = (peak - (cumulative_returns + 1)) / peak
    max_drawdown = np.max(drawdown)

    win_rate = np.sum(returns > 0) / len(returns)
    avg_win = float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0.0
    avg_loss = -float(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-6
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
    model_path = os.path.join(config['output_dir'], 'best_model.pth')
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    output_dir = config['output_dir']

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("未找到 best_model.pth 或 scaler.pkl，请先完成训练！")

    print("1. 加载数据与模型...")
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 预先计算未来价格
    raw_df['open_t1'] = raw_df.groupby('股票代码')['开盘'].shift(-1)
    raw_df['open_t5'] = raw_df.groupby('股票代码')['开盘'].shift(-5)

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_tmp, features = preprocess_predict_data(raw_df.copy(), stockid2idx)
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    print("2. 数据预处理...")
    processed, features = preprocess_predict_data(raw_df.copy(), stockid2idx)

    # 提取真实原始价格
    raw_prices = processed[['股票代码', '日期', '开盘']].copy()
    raw_prices['open_t1'] = raw_prices.groupby('股票代码')['开盘'].shift(-1)
    raw_prices['open_t5'] = raw_prices.groupby('股票代码')['开盘'].shift(-5)

    # 标准化特征
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[features] = scaler.transform(processed[features])

    print("3. 生成回测日期历...")
    all_dates = sorted(processed['日期'].unique())
    start_idx = config['sequence_length'] - 1
    # 确保最后调仓日后有至少5个交易日用于计算未来收益
    last_valid_date_index = len(all_dates) - 6  # 需要未来5日数据
    if last_valid_date_index < start_idx:
        raise ValueError("数据不足以进行回测：序列长度或未来交易日不足。")

    rebalance_dates = [d for d in all_dates[start_idx::5] if d <= all_dates[last_valid_date_index]]

    print(f"回测区间: {rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}")
    print(f"预计调仓次数: {len(rebalance_dates)}")

    print("4. 开始滚动回测...")
    baseline_returns = []
    theoretical_optimal_returns = []
    valid_dates = []

    # 计算Baseline收益
    print("   计算Baseline组合收益...")
    with torch.no_grad():
        for signal_date in tqdm(rebalance_dates, desc="Backtesting"):
            try:
                sequences_np, sequence_stock_ids = build_inference_sequences(
                    processed, features, config['sequence_length'], stock_ids, signal_date
                )
            except ValueError:
                continue

            x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()

            top5_indices = np.argsort(scores)[::-1][:5]
            top5_stock_ids = [sequence_stock_ids[i] for i in top5_indices]

            period_return = 0.0
            valid_stocks_count = 0

            for stock_id in top5_stock_ids:
                price_row = raw_prices[(raw_prices['股票代码'] == stock_id) & (raw_prices['日期'] == signal_date)]
                if price_row.empty:
                    continue

                p_t1 = price_row['open_t1'].iloc[0]
                p_t5 = price_row['open_t5'].iloc[0]

                if pd.isna(p_t1) or pd.isna(p_t5) or p_t1 == 0:
                    r_i = 0.0
                else:
                    r_i = (p_t5 - p_t1) / p_t1

                period_return += 0.2 * r_i
                valid_stocks_count += 1

            if valid_stocks_count > 0:
                baseline_returns.append(period_return)
                valid_dates.append(signal_date)

    # 计算理论最优收益（使用与Baseline相同的调仓日期）
    print("   计算理论最优组合收益...")
    for signal_date in valid_dates:
        daily_data = raw_df[raw_df['日期'] == signal_date].copy()
        if daily_data.empty:
            continue

        daily_data['future_return'] = (daily_data['open_t5'] - daily_data['open_t1']) / daily_data['open_t1']
        # 移除无穷值和空值
        daily_data['future_return'] = daily_data['future_return'].replace([np.inf, -np.inf], np.nan).dropna()
        if daily_data.empty:
            theoretical_optimal_returns.append(0.0)
            continue

        max_return = daily_data['future_return'].max()
        optimal_return = max(max_return, 0.0)
        theoretical_optimal_returns.append(optimal_return)

    print("\n5. 回测结果统计:")
    baseline_metrics = calculate_metrics(baseline_returns)
    theoretical_metrics = calculate_metrics(theoretical_optimal_returns)

    print("\nBaseline 结果:")
    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nTheoretical Optimal 结果:")
    for k, v in theoretical_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 保存结果
    result_df = pd.DataFrame({
        'date': valid_dates,
        'baseline_return': baseline_returns,
        'theoretical_return': theoretical_optimal_returns
    })
    result_df.to_csv(os.path.join(output_dir, 'backtest_comparison.csv'), index=False)

    # 绘制对比图
    # 绘制单期收益率对比图
    # 绘制单期收益率对比图
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(valid_dates, baseline_returns, label='Baseline Strategy (Period Return)', color='blue', marker='o',
                 markersize=3, linewidth=1.5)
        plt.plot(valid_dates, theoretical_optimal_returns, label='Theoretical Optimal (Best Single Stock Return)',
                 color='green', marker='s', markersize=3, linewidth=1.5)
        plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Backtest Period Return Comparison')
        plt.xlabel('Date')
        plt.ylabel('5-Day Holding Period Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'backtest_period_return_comparison.png'), dpi=150)
        print(f"\n单期收益率对比图已保存至: {output_dir}/backtest_period_return_comparison.png")
        plt.close()

        # 同时生成对数化后的累积收益图
        # 计算累积收益
        baseline_cumulative = np.cumprod(1 + np.array(baseline_returns))
        theoretical_cumulative = np.cumprod(1 + np.array(theoretical_optimal_returns))

        plt.figure(figsize=(12, 6))
        plt.plot(valid_dates, baseline_cumulative, label='Baseline Strategy', color='blue')
        plt.plot(valid_dates, theoretical_cumulative, label='Theoretical Optimal', color='green')
        plt.yscale('log')
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.title('Backtest Cumulative Return Comparison (Log Scale)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Wealth (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'backtest_comparison_log.png'))
        print(f"对数累积对比图已保存至: {output_dir}/backtest_comparison_log.png")
        plt.close()

    except ImportError:
        pass


if __name__ == "__main__":
    main()
