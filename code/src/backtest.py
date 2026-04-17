"""精简回测脚本：模型推理 → Top K 选股 → 收益计算。

设计要点：
- 特征无关：通过 preprocessing.preprocess() + config['feature_scheme'] 自动处理
- 模型无关：通过 load_model() 动态导入模型类，类名从训练时保存的 config.json 中读取
- 推理序列构建：复用 predict.build_inference_sequences()
- 原始价格与标准化特征分离：raw_prices 在标准化之前提取
"""
import importlib
import json
import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from predict import build_inference_sequences
from preprocessing import preprocess


def load_model(model_path, input_dim, num_stocks, device):
    """根据 config 动态加载模型，不硬编码模型类。"""
    config_json_path = os.path.join(config['output_dir'], 'config.json')
    if os.path.exists(config_json_path):
        with open(config_json_path) as f:
            train_config = json.load(f)
    else:
        train_config = config

    model_class_name = train_config.get('model_class', 'StockTransformer')

    model_module = importlib.import_module('model')
    ModelClass = getattr(model_module, model_class_name)

    model = ModelClass(input_dim=input_dim, config=train_config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def calculate_metrics(returns):
    """计算回测绩效指标。"""
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
        '平均单期收益': np.mean(returns),
    }


def main():
    output_dir = config['output_dir']
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path = os.path.join(output_dir, 'best_model.pth')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    stockid2idx_path = os.path.join(output_dir, 'stockid2idx.pkl')

    for path, name in [(model_path, '模型'), (scaler_path, 'Scaler')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'未找到{name}文件: {path}')

    # 1. 加载原始数据
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 2. 构建股票索引映射（优先使用训练时保存的映射）
    if os.path.exists(stockid2idx_path):
        stockid2idx = joblib.load(stockid2idx_path)
        stock_ids = sorted(stockid2idx.keys())
    else:
        stock_ids = sorted(raw_df['股票代码'].unique())
        stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    # 3. 提取原始价格（在标准化之前，用于收益计算）
    raw_prices = raw_df[['股票代码', '日期', '开盘']].copy()
    raw_prices['open_t1'] = raw_prices.groupby('股票代码')['开盘'].shift(-1)
    raw_prices['open_t5'] = raw_prices.groupby('股票代码')['开盘'].shift(-5)

    # 4. 特征工程（自动根据 config['feature_scheme'] 选择方案）
    processed, features = preprocess(
        raw_df, stockid2idx, config['feature_scheme'],
        desc='回测特征工程', build_label=False,
    )
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 5. 标准化
    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    # 6. 加载模型（动态导入模型类）
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = load_model(model_path, input_dim=len(features), num_stocks=len(stockid2idx), device=device)

    # 7. 生成调仓日历
    all_dates = sorted(processed['日期'].unique())
    start_idx = config['sequence_length'] - 1
    last_valid_idx = len(all_dates) - 6  # 需要未来5个交易日
    if last_valid_idx < start_idx:
        raise ValueError('数据不足以进行回测')
    rebalance_dates = [d for d in all_dates[start_idx::5] if d <= all_dates[last_valid_idx]]

    print(f'回测区间: {rebalance_dates[0].date()} ~ {rebalance_dates[-1].date()}')
    print(f'调仓次数: {len(rebalance_dates)}')

    # 8. 滚动回测
    portfolio_returns = []
    valid_dates = []
    top_k = 5

    with torch.no_grad():
        for signal_date in tqdm(rebalance_dates, desc='Backtesting'):
            try:
                sequences_np, sequence_stock_ids = build_inference_sequences(
                    processed, features, config['sequence_length'], stock_ids, signal_date,
                )
            except ValueError:
                continue

            x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()

            top_indices = np.argsort(scores)[::-1][:top_k]
            top_stock_ids = [sequence_stock_ids[i] for i in top_indices]

            period_return = 0.0
            valid_count = 0
            weight = 1.0 / top_k

            for sid in top_stock_ids:
                row = raw_prices[(raw_prices['股票代码'] == sid) & (raw_prices['日期'] == signal_date)]
                if row.empty:
                    continue
                p_t1 = row['open_t1'].iloc[0]
                p_t5 = row['open_t5'].iloc[0]
                if pd.isna(p_t1) or pd.isna(p_t5) or p_t1 == 0:
                    r_i = 0.0
                else:
                    r_i = (p_t5 - p_t1) / p_t1
                period_return += weight * r_i
                valid_count += 1

            if valid_count > 0:
                portfolio_returns.append(period_return)
                valid_dates.append(signal_date)

    # 9. 输出结果
    metrics = calculate_metrics(portfolio_returns)
    print('\n回测结果:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    # 保存明细
    result_df = pd.DataFrame({'date': valid_dates, 'return': portfolio_returns})
    result_path = os.path.join(output_dir, 'backtest_results.csv')
    result_df.to_csv(result_path, index=False)
    print(f'\n明细已保存: {result_path}')

    # 绘制累计收益曲线（可选，matplotlib 不存在时跳过）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cumulative = np.cumprod(1 + np.array(portfolio_returns))
        plt.figure(figsize=(12, 6))
        plt.plot(valid_dates, cumulative, color='blue')
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.title('Backtest Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Wealth')
        plt.grid(True, alpha=0.3)
        chart_path = os.path.join(output_dir, 'backtest_curve.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f'收益曲线: {chart_path}')
    except ImportError:
        pass


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
