"""统一的数据预处理管线，train.py 和 predict.py 共用。"""
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

import features  # noqa: F401  触发特征方案注册
from features import get_feature_engineer, get_feature_columns


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


def preprocess(df, stockid2idx, feature_num, desc='特征工程', build_label=True, drop_small_open=True):
    """
    统一预处理函数。
    - build_label=True: 训练/验证时构建 label
    - build_label=False: 预测时不构建 label
    返回 (processed_df, feature_columns)
    """
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = get_feature_engineer(feature_num)
    feature_columns = get_feature_columns(feature_num)

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

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    if build_label:
        processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    else:
        processed['日期'] = pd.to_datetime(processed['日期'])

    return processed, feature_columns
