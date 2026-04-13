import pandas as pd
import os
import numpy as np
import joblib
from predict import preprocess_predict_data, build_inference_sequences
from model import StockTransformer
import torch
from config import config

# 加载股票名称列表
stock_list_path = os.path.join(config['data_path'], 'hs300_stock_list.csv')
stock_name_df = pd.read_csv(stock_list_path)
# 清理股票代码格式，统一为 '600000' 的形式
stock_name_df['code'] = stock_name_df['code'].str.replace('sh.', '', regex=False).str[-6:]
# 创建代码到名称的映射字典
code_to_name = dict(zip(stock_name_df['code'], stock_name_df['code_name']))

# 加载回测数据
backtest_df = pd.read_csv(os.path.join(config['output_dir'], 'backtest_details.csv'))
backtest_df['date'] = pd.to_datetime(backtest_df['date'])

# 找出亏损最大的 5 个日期
worst_days = backtest_df.sort_values('return').head(5)

# 加载模型和原始数据
model_path = os.path.join(config['output_dir'], 'best_model.pth')
scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
data_file = os.path.join(config['data_path'], 'train.csv')

raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
raw_df['日期'] = pd.to_datetime(raw_df['日期'])

stock_ids = sorted(raw_df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processed_tmp, features = preprocess_predict_data(raw_df.copy(), stockid2idx)
model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load(scaler_path)
processed, features = preprocess_predict_data(raw_df.copy(), stockid2idx)
processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
processed[features] = scaler.transform(processed[features])

print("亏损最严重的 5 个交易日及其选股:")
for _, row in worst_days.iterrows():
    signal_date = row['date']
    print(f"\n日期: {signal_date.date()}, 收益: {row['return']:.4f}")

    try:
        sequences_np, sequence_stock_ids = build_inference_sequences(
            processed, features, config['sequence_length'], stock_ids, signal_date
        )
    except ValueError:
        print("  [警告] 该日期无有效序列，跳过")
        continue

    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
    scores = model(x).squeeze(0).detach().cpu().numpy()

    top5_indices = np.argsort(scores)[::-1][:5]
    top5_stock_ids = [sequence_stock_ids[i] for i in top5_indices]

    print("  Top 5 股票代码及名称:")
    for code in top5_stock_ids:
        name = code_to_name.get(code, "未知")
        print(f"    - {code} ({name})")
