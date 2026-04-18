"""统一的数据预处理管线，train.py 和 predict.py 共用。"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import features  # noqa: F401  触发特征方案注册
from config import config
from features import get_feature_engineer, get_feature_columns


CROSS_SECTIONAL_COLUMNS = [
    'sector_mom_5d', 'sector_mom_10d',
    'vs_sector_mom_5d', 'vs_sector_mom_10d',
    'market_mom_5d', 'market_mom_10d',
    'market_breadth_1d', 'market_dispersion',   # 原 market_vol_5d
    'excess_mom_5d', 'excess_mom_10d',
    'sector_rank_5d', 'sector_breadth',          # 原 sector_breadth_5d
    'cs_rank_return_5d', 'cs_zscore_return_5d',
]


def _add_cross_sectional_features(processed, data_path):
    """在所有股票合并后，按日期计算跨截面特征。"""
    processed = processed.copy()

    # --- 横截面相对强度（不依赖行业）---
    if 'return_5' in processed.columns:
        processed['cs_rank_return_5d'] = processed.groupby('日期')['return_5'].rank(pct=True)
        mean_r5 = processed.groupby('日期')['return_5'].transform('mean')
        std_r5 = processed.groupby('日期')['return_5'].transform('std')
        processed['cs_zscore_return_5d'] = (processed['return_5'] - mean_r5) / (std_r5 + 1e-12)
    else:
        processed['cs_rank_return_5d'] = 0.0
        processed['cs_zscore_return_5d'] = 0.0

    # --- 市场级特征（不依赖行业）---
    if 'return_5' in processed.columns:
        processed['market_mom_5d'] = processed.groupby('日期')['return_5'].transform('mean')
        processed['market_mom_10d'] = processed.groupby('日期')['return_10'].transform('mean')
    else:
        processed['market_mom_5d'] = 0.0
        processed['market_mom_10d'] = 0.0

    if 'return_1' in processed.columns:
        processed['market_breadth_1d'] = processed.groupby('日期')['return_1'].transform(lambda x: (x > 0).mean())
        processed['market_dispersion'] = processed.groupby('日期')['return_1'].transform('std')
    else:
        processed['market_breadth_1d'] = 0.0
        processed['market_dispersion'] = 0.0

    # --- 板块级特征（依赖行业分类文件）---
    sector_file = os.path.join(data_path, 'hs300_stock_list_annotated.csv')
    if os.path.exists(sector_file):
        sector_df = pd.read_csv(sector_file, dtype={'code_clean': str})
        sector_df['code_clean'] = sector_df['code_clean'].astype(str).str.zfill(6)
        sector_map = sector_df.set_index('code_clean')['行业'].to_dict()
        processed['_sector'] = processed['股票代码'].map(sector_map).fillna('未知')

        for suffix, col in [('5d', 'return_5'), ('10d', 'return_10')]:
            if col in processed.columns:
                sector_avg = processed.groupby(['日期', '_sector'])[col].transform('mean')
                processed[f'sector_mom_{suffix}'] = sector_avg
                processed[f'vs_sector_mom_{suffix}'] = processed[col] - sector_avg

                market_avg = processed.groupby('日期')[col].transform('mean')
                processed[f'excess_mom_{suffix}'] = sector_avg - market_avg
            else:
                processed[f'sector_mom_{suffix}'] = 0.0
                processed[f'vs_sector_mom_{suffix}'] = 0.0
                processed[f'excess_mom_{suffix}'] = 0.0

        if 'return_5' in processed.columns:
            processed['sector_rank_5d'] = processed.groupby('日期')['sector_mom_5d'].rank(pct=True)
        else:
            processed['sector_rank_5d'] = 0.0

        if 'return_1' in processed.columns:
            processed['sector_breadth'] = processed.groupby(['日期', '_sector'])['return_1'].transform(lambda x: (x > 0).mean())
        else:
            processed['sector_breadth'] = 0.0

        processed.drop(columns=['_sector'], inplace=True)
    else:
        # 行业文件不存在时，板块特征填 0
        for col in ['sector_mom_5d', 'sector_mom_10d', 'vs_sector_mom_5d', 'vs_sector_mom_10d',
                     'excess_mom_5d', 'excess_mom_10d', 'sector_rank_5d', 'sector_breadth']:
            processed[col] = 0.0

    processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed.fillna(0, inplace=True)
    return processed


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def preprocess(df, stockid2idx, feature_scheme, desc='特征工程', build_label=True, drop_small_open=True):
    """
    统一预处理函数。
    - feature_scheme: 特征方案名称，如 'technical', 'full'
    - build_label=True: 训练/验证时构建 label
    - build_label=False: 预测时不构建 label
    返回 (processed_df, feature_columns)
    """
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = get_feature_engineer(feature_scheme)
    feature_columns = get_feature_columns(feature_scheme)

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # --- 跨截面特征开关 ---
    processed['日期'] = pd.to_datetime(processed['日期'])
    if config.get('enable_cross_sectional_features', False):
        processed = _add_cross_sectional_features(processed, config['data_path'])
        feature_columns = feature_columns + CROSS_SECTIONAL_COLUMNS

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    if build_label:
        processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

    return processed, feature_columns
