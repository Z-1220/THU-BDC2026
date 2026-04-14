# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去一段时间（默认60个交易日）的量价与技术特征序列；
- 模型：`StockTransformer`，同时建模单股票时序模式与股票间交互；
- 输出：对同一天全部候选股票打分并排序，最终输出前5只股票（等权重0.2）。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（`data/stock_data.csv`）；
2. 做特征工程（39特征或`158+39`特征）；
3. 构建标签：未来收益率（代码中为 `open_t1` 到 `open_t5` 的相对收益）；
4. 按“日期”组织排序样本：每个样本是一日内多只股票的序列与目标；
5. 训练排序模型，监控 `final_score` 并保存最优权重；
6. 使用训练好的 `best_model.pth` + `scaler.pkl` 在最新日期上生成Top5选股结果。

---

## 2. 代码结构说明
### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（Baostock）：
- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式。
- 训练数据保存为`data/stock_data.csv`
- 最新的沪深300股票代码和名称数据保存在`data/hs300_stock_list.csv`

### [split_train_test.py](split_train_test.py)
数据分割脚本，按日期区间将股票数据切分为训练集和测试集。
- 训练集为`data/train.csv`
- 测试集为`data/test.csv`

### [config.py](code/src/config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60）；
- 模型超参数（`d_model`、`nhead`、`num_layers` 等）；
- 训练超参数（`batch_size`、`num_epochs`、`learning_rate`）；
- 排序损失权重参数（`pairwise_weight`、`top5_weight`、`base_weight`）；
- 数据路径和输出路径（默认输出到 `output/`）。

### [utils.py](code/src/utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [model.py](code/src/model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- `CrossStockAttention`：在同一交易日内建模股票间关系；
- `ranking_layers` + `score_head`：输出每只股票的排序分数。

输入形状：`[batch, num_stocks, seq_len, feature_dim]`  
输出形状：`[batch, num_stocks]`。

### [train.py](code/src/train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：按股票分组并行特征工程、股票ID映射、标签构建；
	- `split_train_val_by_last_month()`：按最后阶段数据切分训练/验证集，并保留序列上下文。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`WeightedRankingLoss`
	- 组合了 `listwise_loss` 与 `pairwise_loss`；
	- 对真实Top-k样本施加更高权重。
- 评估指标：`calculate_ranking_metrics()`
	- 计算 `pred_return_sum`、`max_return_sum`、`ratio_pred`、`final_score` 等；
	- 训练过程中以验证集 `final_score` 选择最优模型。

训练产物：
- `best_model.pth`：最佳模型参数；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：最佳分数记录；
- `log/`：TensorBoard日志。

### [predict.py](code/src/predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程；
3. 加载 `scaler.pkl` 进行特征标准化；
4. 用 `best_model.pth` 对全部可预测股票打分；
5. 按分数降序取前5只，输出到 `output.csv`：
	 - `stock_id`
	 - `weight`（固定 `0.2`）

### [check_backtest_data.py](code/src/check_backtest_data.py)
数据完整性诊断脚本，用于在执行回测前校验 `train.csv` 是否满足 `StockTransformer` 回测所需的数据条件，提前识别潜在问题。

检查项目：  
1. 基础信息  
   - 文件存在性、总行数、股票数量、日期范围、必需列（`开盘`、`股票代码`、`日期`）完整性。
2. 空值与零值检查  
   - 统计 `开盘`、`open_t1`、`open_t5` 的缺失数量。  
   - 检测价格为 0 的行数，避免收益计算出现除零或无穷大。
3. 尾部数据可用性  
   - 展示最后 10 个交易日 `open_t1` / `open_t5` 缺失比例，用于判断回测尾部截断是否合理。
4. 停牌与连续性检查  
   - 统计相邻交易日间隔超过 5 天的次数，并给出示例，提示停牌对未来价格计算的影响。
5. 特征列与配置兼容性  
   - 列出 `raw_df` 中实际可用的特征列，供与 `config['feature_num']` 对照。
6. 回测调仓模拟  
   - 基于 `sequence_length` 和 5 日调仓间隔，预估调仓次数与日期范围。  
   - 检查每个调仓日有效股票数（`open_t1`、`open_t5` 均非空）是否不少于 5 只，并报告不足日期。  

输出：  
- 控制台打印各项检查结果，标注 `[通过]`、`[警告]` 或 `[错误]`，便于快速定位数据问题。


### [backtest.py](code/src/backtest.py)
回测主脚本，用于在历史数据上滚动评估 `StockTransformer` 模型的策略表现，并与理论最优单股票收益进行对比。

计算回测绩效指标，返回字典包含：
  - 总调仓次数
  - 最终累计收益率
  - 最大回撤
  - 胜率
  - 盈亏比
  - 平均单期收益

流程：
1. 加载数据与模型 
   - 读取原始数据 `data/train.csv`，完成股票代码标准化与日期排序。  
   - 预计算未来价格列 `open_t1`、`open_t5`。  
   - 加载训练阶段保存的 `best_model.pth` 与 `scaler.pkl`，初始化 `StockTransformer` 模型并切换到评估模式。

2. 数据预处理  
   - 调用 `preprocess_predict_data` 生成特征集，并提取 `raw_prices` 用于后续收益计算。  
   - 使用 `scaler` 对特征进行标准化。

3. 生成回测日期序列  
   - 基于全部交易日与 `config['sequence_length']` 确定调仓起始点。  
   - 以每 5 个交易日为间隔生成调仓日期，并剔除未来数据不足（少于 5 日）的尾部日期。

4. 滚动回测计算  
   - Baseline 策略收益：  
     对每个调仓日，调用 `build_inference_sequences` 构建序列输入，模型输出股票分数，
     选取分数最高的前 5 只股票，以等权重计算该调仓期的 5 日收益率（基于 `open_t1` 与 `open_t5` 价格）。
   - 理论最优收益：  
     对每个有效调仓日，计算当日所有股票的未来 5 日收益率，取最大值或 0 作为理论最优单期收益。

依赖文件（训练产物）：  
- `output/best_model.pth`  
- `output/scaler.pkl`

输出产物（保存在 `config['output_dir']/` 下）：
- `backtest_comparison.csv`：每期收益率记录。 
- `backtest_period_return_comparison.png`：单期收益率对比折线图。 
- `backtest_comparison_log.png`：对数累积收益对比图。

控制台分别输出 Baseline 与理论最优策略的上述回测指标。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- output目录下 `result.csv`（由 `predict.py` 生成）。

---

## 4. 运行方法（推荐使用 uv）

1) 使用 `uv` 安装依赖
   `uv sync`
2) 激活虚拟环境
   `source .venv/bin/activate`
3) 训练模型
	```
	sh train.sh
	```
4) 生成预测结果
	```
	sh test.sh
	```

---

## 5. 常见问题

1) `TA-Lib` 安装失败
本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装Python包。
	```
	wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
		tar -xzf ta-lib-0.4.0-src.tar.gz && \
		cd ta-lib && \
		./configure --prefix=/usr && \
		make -j1 && \
		make install && \
		cd .. && \
		rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
	```
2) 多进程相关问题  
`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS下请保持通过脚本入口运行（不要在交互式环境里直接多进程调用主逻辑）。

3) GPU/CPU自动选择  
代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无GPU时可直接CPU运行。
