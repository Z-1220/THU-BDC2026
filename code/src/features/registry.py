"""特征方案注册表。添加新特征方案只需在此注册，无需修改 train.py / predict.py。"""

# 全局注册表
_FEATURE_ENGINEER_MAP = {}   # name -> callable(df) -> df
_FEATURE_COLUMNS_MAP = {}    # name -> list[str]


def register_feature_scheme(name, engineer_func, columns):
    """注册一个特征方案。
    - name: 方案名称，如 '39', '158+39'
    - engineer_func: 特征工程函数，签名 (df: pd.DataFrame) -> pd.DataFrame
    - columns: 该方案产出的特征列名列表（含 'instrument' 和原始行情列）
    """
    _FEATURE_ENGINEER_MAP[name] = engineer_func
    _FEATURE_COLUMNS_MAP[name] = columns


def get_feature_engineer(name):
    assert name in _FEATURE_ENGINEER_MAP, (
        f"Unsupported feature_num: {name}. Available: {list(_FEATURE_ENGINEER_MAP.keys())}"
    )
    return _FEATURE_ENGINEER_MAP[name]


def get_feature_columns(name):
    assert name in _FEATURE_COLUMNS_MAP, (
        f"Unsupported feature_num: {name}. Available: {list(_FEATURE_COLUMNS_MAP.keys())}"
    )
    return _FEATURE_COLUMNS_MAP[name]
