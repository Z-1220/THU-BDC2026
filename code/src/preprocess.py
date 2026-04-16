import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from config import config


def load_sector_map():
    """加载股票→行业映射表"""
    sector_file = os.path.join(config['data_path'], 'hs300_stock_list_annotated.csv')
    if not os.path.exists(sector_file):
        print(f"[WARNING] 未找到行业分类文件: {sector_file}, 将跳过板块特征构建")
        return {}
    sector_df = pd.read_csv(sector_file, dtype={'code_clean': str})
    sector_df['code_clean'] = sector_df['code_clean'].astype(str).str.zfill(6)
    return sector_df.set_index('code_clean')['行业'].to_dict()


def add_sector_features(df, sector_map):
    """
    在个股级别数据上添加板块和市场级别特征。
    这些特征让模型能感知板块趋势和市场环境，自动规避暴跌板块。
    
    新增特征：
      sector_mom_5d / 10d / 20d  : 所属板块过去N天平均涨幅
      sector_rank_5d              : 所属板块在所有行业中的动量排名
      sector_breadth_5d           : 板块内上涨股票占比
      sector_dispersion_5d        : 板块内收益分化度
      market_mom_5d / 10d         : 全市场过去N天平均涨幅
      market_breadth_1d           : 全市场当日涨跌家数比
      market_vol_5d               : 全市场波动率
      excess_mom_5d               : 板块超额动量 (sector_mom - market_mom)
    """
    if not sector_map:
        return df

    df = df.copy()
    df['行业'] = df['股票代码'].map(sector_map).fillna('未知')

    # ==================== 个股基础收益率 ====================
    df['daily_ret'] = df.groupby('股票代码')['收盘'].pct_change(1)
    for w in [5, 10, 20]:
        df[f'_stock_mom_{w}d'] = df.groupby('股票代码')['收盘'].pct_change(w)

    # ==================== 板块级别特征 ====================
    sector_agg = df.groupby(['日期', '行业']).agg(
        _sector_mom_5d=('_stock_mom_5d', 'mean'),
        _sector_mom_10d=('_stock_mom_10d', 'mean'),
        _sector_mom_20d=('_stock_mom_20d', 'mean'),
        _sector_breadth_5d=('_stock_mom_5d', lambda x: (x > 0).mean()),
        _sector_dispersion_5d=('_stock_mom_5d', 'std'),
    ).reset_index()

    # 板块截面排名
    sector_agg['_sector_rank_5d'] = sector_agg.groupby('日期')['_sector_mom_5d'].rank(ascending=False)

    df = df.merge(sector_agg, on=['日期', '行业'], how='left')

    # 重命名为最终特征名
    df['sector_mom_5d'] = df['_sector_mom_5d']
    df['sector_mom_10d'] = df['_sector_mom_10d']
    df['sector_mom_20d'] = df['_sector_mom_20d']
    df['sector_rank_5d'] = df['_sector_rank_5d']
    df['sector_breadth_5d'] = df['_sector_breadth_5d']
    df['sector_dispersion_5d'] = df['_sector_dispersion_5d']

    # ==================== 市场级别特征 ====================
    market_agg = df.groupby('日期').agg(
        _market_mom_5d=('_stock_mom_5d', 'mean'),
        _market_mom_10d=('_stock_mom_10d', 'mean'),
        _market_breadth_1d=('daily_ret', lambda x: (x > 0).mean()),
        _market_vol_5d=('daily_ret', 'std'),
    ).reset_index()

    df = df.merge(market_agg, on='日期', how='left')

    df['market_mom_5d'] = df['_market_mom_5d']
    df['market_mom_10d'] = df['_market_mom_10d']
    df['market_breadth_1d'] = df['_market_breadth_1d']
    df['market_vol_5d'] = df['_market_vol_5d']

    # ==================== 交互特征 ====================
    df['excess_mom_5d'] = df['sector_mom_5d'] - df['market_mom_5d']
    df['excess_mom_10d'] = df['sector_mom_10d'] - df['market_mom_10d']

    # ==================== 清理临时列 ====================
    drop_cols = [c for c in df.columns if c.startswith('_')]
    drop_cols += ['行业', 'daily_ret']
    df = df.drop(columns=drop_cols, errors='ignore')

    # 排名特征归一化到 [0, 1]
    n_sectors = df.groupby('日期')['sector_rank_5d'].transform('max')
    df['sector_rank_5d'] = 1.0 - df['sector_rank_5d'] / (n_sectors + 1)  # 越大越好

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def preprocess_train_data(raw_df, stockid2idx):
    """训练数据预处理：构建特征 + 标签"""
    sector_map = load_sector_map()
    print(f"  行业映射: {len(sector_map)} 只股票已标注")

    df = raw_df.copy()
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['stock_id'] = df['股票代码'].map(stockid2idx)

    # ==================== 个股技术特征 ====================
    # 动量
    for w in [3, 5, 10, 20]:
        df[f'mom_{w}d'] = df.groupby('股票代码')['收盘'].pct_change(w)

    # 波动率
    for w in [5, 10, 20]:
        df[f'vol_{w}d'] = df.groupby('股票代码')['收盘'].pct_change(1).rolling(w).std()
        df[f'vol_{w}d'] = df.groupby('股票代码')[f'vol_{w}d'].transform(lambda x: x)

    # 成交量变化
    for w in [5, 10]:
        df[f'vol_chg_{w}d'] = df.groupby('股票代码')['成交量'].pct_change(w)

    # 振幅
    df['amplitude_5d'] = df.groupby('股票代码').apply(
        lambda g: (g['最高'].rolling(5).max() - g['最低'].rolling(5).min()) / g['开盘'].shift(1)
    ).reset_index(level=0, drop=True)

    # RSI
    delta = df.groupby('股票代码')['收盘'].diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14d'] = 100 - 100 / (1 + rs)
    df['rsi_14d'] = df.groupby('股票代码')['rsi_14d'].transform(lambda x: x.fillna(50))

    # 均线偏离
    for w in [5, 10, 20]:
        ma = df.groupby('股票代码')['收盘'].transform(lambda x: x.rolling(w).mean())
        df[f'ma_bias_{w}d'] = (df['收盘'] - ma) / ma

    # ==================== 板块特征 ====================
    if sector_map:
        print("  添加板块/市场级别特征...")
        df = add_sector_features(df, sector_map)

    # ==================== 标签 ====================
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / df['open_t1']

    # ==================== 清理 ====================
    feature_candidates = [
        'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'vol_chg_5d', 'vol_chg_10d',
        'amplitude_5d',
        'rsi_14d',
        'ma_bias_5d', 'ma_bias_10d', 'ma_bias_20d',
    ]

    # 板块特征
    sector_features = [
        'sector_mom_5d', 'sector_mom_10d', 'sector_mom_20d',
        'sector_rank_5d', 'sector_breadth_5d', 'sector_dispersion_5d',
        'market_mom_5d', 'market_mom_10d', 'market_breadth_1d', 'market_vol_5d',
        'excess_mom_5d', 'excess_mom_10d',
    ]

    features = [f for f in feature_candidates if f in df.columns]
    features += [f for f in sector_features if f in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    label_valid_mask = df['label'].notna()

    return df, features, label_valid_mask


def preprocess_predict_data(raw_df, stockid2idx):
    """预测数据预处理：构建特征（无标签）"""
    sector_map = load_sector_map()

    df = raw_df.copy()
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['stock_id'] = df['股票代码'].map(stockid2idx)

    # ==================== 个股技术特征 ====================
    for w in [3, 5, 10, 20]:
        df[f'mom_{w}d'] = df.groupby('股票代码')['收盘'].pct_change(w)

    for w in [5, 10, 20]:
        df[f'vol_{w}d'] = df.groupby('股票代码')['收盘'].pct_change(1).rolling(w).std()
        df[f'vol_{w}d'] = df.groupby('股票代码')[f'vol_{w}d'].transform(lambda x: x)

    for w in [5, 10]:
        df[f'vol_chg_{w}d'] = df.groupby('股票代码')['成交量'].pct_change(w)

    df['amplitude_5d'] = df.groupby('股票代码').apply(
        lambda g: (g['最高'].rolling(5).max() - g['最低'].rolling(5).min()) / g['开盘'].shift(1)
    ).reset_index(level=0, drop=True)

    delta = df.groupby('股票代码')['收盘'].diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14d'] = 100 - 100 / (1 + rs)
    df['rsi_14d'] = df.groupby('股票代码')['rsi_14d'].transform(lambda x: x.fillna(50))

    for w in [5, 10, 20]:
        ma = df.groupby('股票代码')['收盘'].transform(lambda x: x.rolling(w).mean())
        df[f'ma_bias_{w}d'] = (df['收盘'] - ma) / ma

    # ==================== 板块特征 ====================
    if sector_map:
        df = add_sector_features(df, sector_map)

    # ==================== 清理 ====================
    feature_candidates = [
        'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'vol_chg_5d', 'vol_chg_10d',
        'amplitude_5d',
        'rsi_14d',
        'ma_bias_5d', 'ma_bias_10d', 'ma_bias_20d',
    ]

    sector_features = [
        'sector_mom_5d', 'sector_mom_10d', 'sector_mom_20d',
        'sector_rank_5d', 'sector_breadth_5d', 'sector_dispersion_5d',
        'market_mom_5d', 'market_mom_10d', 'market_breadth_1d', 'market_vol_5d',
        'excess_mom_5d', 'excess_mom_10d',
    ]

    features = [f for f in feature_candidates if f in df.columns]
    features += [f for f in sector_features if f in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)

    return df, features


def get_scaler_features(df, features):
    """从数据中提取用于标准化的特征值"""
    valid_mask = df[features].notna().all(axis=1)
    scaler = StandardScaler()
    scaler.fit(df.loc[valid_mask, features])
    return scaler


def main():
    """测试预处理流程"""
    data_file = os.path.join(config['data_path'], 'train.csv')
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    print("=" * 50)
    print("预处理流程测试")
    print("=" * 50)

    # 训练预处理
    print("\n[1] 训练数据预处理...")
    processed, features, label_mask = preprocess_train_data(raw_df.copy(), stockid2idx)
    print(f"  总行数: {len(processed)}")
    print(f"  有效标签行数: {label_mask.sum()}")
    print(f"  特征数: {len(features)}")
    print(f"  特征列表: {features}")

    if features:
        print(f"\n  特征统计:")
        print(processed[features].describe().round(4).to_string())

    # 预测预处理
    print("\n[2] 预测数据预处理...")
    processed_pred, features_pred = preprocess_predict_data(raw_df.copy(), stockid2idx)
    print(f"  总行数: {len(processed_pred)}")
    print(f"  特征数: {len(features_pred)}")
    assert features == features_pred, "训练和预测的特征列表不一致！"
    print("  特征列表一致性检查: PASS")

    # 标准化
    print("\n[3] 标准化测试...")
    scaler = get_scaler_features(processed, features)
    processed[features] = processed[features].fillna(0)
    processed[features] = scaler.transform(processed[features])
    print(f"  标准化后均值: {processed[features].mean().round(4).to_dict()}")
    print(f"  标准化后标准差: {processed[features].std().round(4).to_dict()}")

    # 保存 scaler
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"\n[4] Scaler 已保存至: {scaler_path}")


if __name__ == '__main__':
    main()
