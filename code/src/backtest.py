# code/src/backtest.py
import os
import multiprocessing as mp
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

mp.set_start_method('spawn', force=True)


def calculate_metrics(returns):
    """计算回测指标"""
    returns = np.array(returns)
    if len(returns) == 0:
        return {}

    # 累计收益率
    cumulative_returns = np.cumprod(1 + returns) - 1

    # 最大回撤
    peak = np.maximum.accumulate(cumulative_returns + 1)
    drawdown = (peak - (cumulative_returns + 1)) / peak
    max_drawdown = np.max(drawdown)

    # 胜率 (按调仓周期算，收益>0即胜)
    win_rate = np.sum(returns > 0) / len(returns)

    # 盈亏比
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

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(input_dim=len(config['feature_num']) == 5 and 197 or 197, config=config,
                             num_stocks=len(stock_ids))

    # 这里为了获取准确的 input_dim，先跑一下 preprocess 拿到 features 列表
    processed_tmp, features = preprocess_predict_data(raw_df.copy(), stockid2idx)
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    print("2. 数据预处理与未来收益计算...")
    # 复用 predict 的预处理
    processed, features = preprocess_predict_data(raw_df.copy(), stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    processed[features] = scaler.transform(processed[features])

    # 【核心安全机制】使用 shift 提取未来开盘价，绝对不会引入前视偏差
    # shift(-1) 就是严格意义上的下一个交易日，shift(-5) 就是第5个交易日
    processed = processed.sort_values(['股票代码', '日期'])
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    print("3. 生成回测日期历...")
    all_dates = sorted(processed['日期'].unique())
    # 至少需要 sequence_length 的历史数据才能预测
    start_idx = config['sequence_length'] - 1
    # 步长为5，模拟真实的5天持有期调仓
    rebalance_dates = all_dates[start_idx::5]

    print(f"回测区间: {rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}")
    print(f"预计调仓次数: {len(rebalance_dates)}")

    print("4. 开始滚动回测...")
    portfolio_returns = []
    valid_dates = []

    with torch.no_grad():
        for signal_date in tqdm(rebalance_dates, desc="Backtesting"):
            # 构建当前日期的推理序列
            try:
                sequences_np, sequence_stock_ids = build_inference_sequences(
                    processed, features, config['sequence_length'], stock_ids, signal_date
                )
            except ValueError:
                continue  # 跳过没有足够序列的日期

            x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()

            # 获取 Top5
            top5_indices = np.argsort(scores)[::-1][:5]
            top5_stock_ids = [sequence_stock_ids[i] for i in top5_indices]

            # 计算这5只股票的真实收益
            period_return = 0.0
            valid_stocks_count = 0

            for stock_id in top5_stock_ids:
                stock_mask = (processed['股票代码'] == stock_id) & (processed['日期'] == signal_date)
                if not stock_mask.any():
                    continue

                row = processed[stock_mask].iloc[0]
                p_t1 = row['open_t1']
                p_t5 = row['open_t5']

                # 如果遇到停牌(T+1或T+5没有开盘价)，将该股票视作现金(收益为0)
                if pd.isna(p_t1) or pd.isna(p_t5) or p_t1 == 0:
                    r_i = 0.0
                else:
                    r_i = (p_t5 - p_t1) / p_t1

                # 按赛题规则，等权0.2
                period_return += 0.2 * r_i
                valid_stocks_count += 1

            # 如果5只股票全停牌等极端情况，收益记为0
            if valid_stocks_count > 0:
                portfolio_returns.append(period_return)
                valid_dates.append(signal_date)

    print("\n5. 回测结果统计:")
    metrics = calculate_metrics(portfolio_returns)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 保存结果
    result_df = pd.DataFrame({
        'date': valid_dates,
        'return': portfolio_returns
    })
    result_df.to_csv(os.path.join(output_dir, 'backtest_details.csv'), index=False)

    # 绘制累计收益曲线
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无头模式
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        cumulative = np.cumprod(1 + np.array(portfolio_returns))
        plt.plot(valid_dates, cumulative, label='Baseline Strategy', color='blue')
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.title('Backtest Cumulative Return (Baseline Model)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Wealth')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'backtest_curve.png'))
        print(f"\n细节数据已保存至: {output_dir}/backtest_details.csv")
        print(f"收益曲线图已保存至: {output_dir}/backtest_curve.png")
    except ImportError:
        print("\n未安装 matplotlib，跳过绘图。")


if __name__ == "__main__":
    main()
