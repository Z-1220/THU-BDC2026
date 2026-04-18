# 配置参数
sequence_length = 60
feature_scheme = 'full'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_scheme': feature_scheme,
    'model_class': 'StockTransformer',
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    # 特征开关：逐组验证后再开启
    # Phase 2 (advanced ON) 回退：测试集加权收益从 +1.89e-03 翻负到 -2.77e-03，
    # 根因是 advanced 里若干 ratio 特征除以近零值产生尾部异常值污染 StandardScaler。
    # 此次进入 Phase 3 替代路径：关闭 advanced，只打开跨截面特征验证。
    'enable_cross_sectional_features': True,   # 14个跨截面特征

    'output_dir': f'./model/{sequence_length}_{feature_scheme}',
    'data_path': './data',
}
