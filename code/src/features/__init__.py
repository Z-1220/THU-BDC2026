"""特征工程模块。导入此包时自动注册所有特征方案。"""
from features.registry import get_feature_engineer, get_feature_columns, register_feature_scheme

# 导入各特征模块以触发注册
import features.technical39  # noqa: F401
import features.alpha158  # noqa: F401
