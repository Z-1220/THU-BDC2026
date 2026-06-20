# 代码说明

## 环境配置

- Python 3.12，PyTorch 2.11.0+cu128，pyqlib 0.9.7
- 包管理使用 uv（`uv sync` 安装依赖），禁止 pip/conda/poetry
- TA-Lib C 库在 Dockerfile 中编译安装，本地使用需系统级安装（见下文第 5 节常见问题）
- LightGBM 4.6.0，CatBoost 1.2.10，XGBoost 3.2.0

## 数据

使用 Baostock 公开数据（免费注册即可获取，http://baostock.com），通过 `scripts/get_stock_data.py` 下载沪深 300 成分股日线 OHLCV 数据（后复权 adjustflag="1"），数据时间范围为 2022-01 至最新。

辅助数据：
- `resource/行业分类.csv`：中证四级行业分类编码（sector_l1 ~ sector_l4），由 `scripts/convert_data.py` 编码为整数特征写入 Qlib 二进制
- `data/hs300_index.csv`：沪深 300 指数日线数据，用于市场状态诊断

数据切分：
- `data/stock_data.csv`：全量原始数据
- `data/train.csv`：训练数据（排除盲测区间 2026-04-13 ~ 2026-04-17）
- `data/test.csv`：主办方盲测集，不可修改
- `temp/qlib_data/`：Qlib 二进制缓存（573~1072 个交易日，2022-01-04 起），通过 `sh init.sh` 生成

## 预训练模型

使用 Kronos-small（NeurIPS 2025 / AAAI 2026 发表，2026 年 4 月 1 日前开源的非商业化模型），HuggingFace 地址：
- https://huggingface.co/NeoQuasar/Kronos-small
- https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base

通过 `python scripts/download_kronos_models.py` 下载（支持 HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`），权重保存至 `model/kronos_pretrained/`。

在 `code/models/Kronos/KronosModel.py` 中对 Kronos Tokenizer + Predictor 网络进行初始化，加载时使用 `local_files_only=True`，训练和推理过程无需联网。

## 算法

### 整体思路

基于 Kronos 金融 K 线基础模型，在 A 股数据上微调 Predictor（自回归 Transformer），利用其预测未来 5 日 OHLCV 序列，以预测的 T+5 开盘 vs T+1 开盘收益率作为选股排序分数。通过横截面 Z-score 标准化后，选 Top-3 等权作为最终组合。

### 方法细节

1. **预训练模型微调**：仅微调 Kronos Predictor（24.7M 参数），Tokenizer 保持冻结。训练目标是自回归 next-token prediction 的交叉熵损失。消融实验证实冻结 Tokenizer 优于微调，2024-2025 训练数据优于扩展至 2022。
2. **金融筛选（ScreenProcessor）**：在预测阶段剔除不合格股票（成交额排名后 30%、价格低于 MA60、20 日最大回撤超过 15%）。消融实验证实此筛选不可删除——去掉后累计收益从 +19.00% 暴跌至 +4.12%。
3. **横截面标准化**：Winsorize（1%/99% 截尾）→ 行业中性化（行业中位数减法）→ Z-score 标准化，统一跨周分数尺度。
4. **Top-3 等权组合**：选择 Kronos 分数最高的 3 只股票，各配 33.3% 权重。实验证实 Top3 贡献了 85%+ 的 Alpha，rank4-5 贡献不足 15% 且增加波动。
5. **实验证伪的方向**：规则型市场状态仓位（无效）、置信度仓位（退化为杠杆）、间距仓位（零信号）。

### 网络结构

- **Kronos Tokenizer**：Transformer 自编码器 + Binary Spherical Quantization (BSQ)，将 OHLCV 6 维数据量化为 20-bit 离散 token（s1 10-bit 粗粒度 + s2 10-bit 细粒度，码本大小 1024×1024）。3 层编码器 + 3 层解码器，d_model=256，4 头注意力。
- **Kronos Predictor**：Decoder-only Transformer（8 层，d_model=512，8 头），自回归预测 token 序列。使用 RoPE 位置编码、SwiGLU FFN、DualHead 分层预测（s1 → s2 通过 DependencyAwareLayer 交叉注意力）。
- **完整推理流水线**：历史 OHLCV → Tokenizer.encode() → s1/s2 tokens → Predictor 自回归生成 → Tokenizer.decode() → 未来 OHLCV → 排序分数

### 损失函数

- 微调阶段：层次化交叉熵损失（s1 + s2 子 token 的平均 CE）
- 推理阶段：无需训练，冻结权重前向传播。KronosModel.fit() 为 no-op。

### 模型集成

不涉及模型集成。实验计划中的 E6（Kronos + LightGBM 融合）和 E5c（Pairwise Ranking Head）均未执行。

### 算法的其他细节

- 排序分数计算：`score = (pred_open[T+5] - pred_open[T+1]) / pred_open[T+1]`，与 Qlib 的 LABEL0 定义一致
- 推理参数：pred_len=5，T=1.0，top_p=0.9，sample_count=1
- 固定随机种子 seed=42 保证可复现
- Tokenizer 保持预训练权重（冻结），不进行微调
- 数据划分：训练 2024-01 至 2025-12，验证 2026-01，测试 2026-02 至 2026-05

## 训练流程

1. `sh init.sh`：将 `data/train.csv` + `resource/行业分类.csv` 转为 Qlib 二进制数据（`temp/qlib_data/`），包括 calendars/day.txt、instruments/all.txt、features/<code>/*.day.bin
2. `sh train.sh`：调用 `code/src/train.py`，加载 `model/result_model.yaml` 配置，创建 KronosModel 并调用 `fit()`（Kronos 的 fit 为 no-op，使用预训练+微调权重），保存权重快照到 `model/result_model.pth`
3. 实际微调在 Docker 外完成（`scripts/finetune_kronos.py`），产物为 `model/kronos_finetuned/Kronos-small-lb60/`
4. 训练配置快照保存为 `model/config_snapshot.yaml`

## 推理流程

1. `sh test.sh`：调用 `code/src/commit.py`
2. 初始化 Qlib → 加载 Kronos 模型（`KronosTokenizer.from_pretrained` + `Kronos.from_pretrained`，`local_files_only=True`）
3. 构建测试数据集：`FridayFilterProcessor`（仅周五信号日）→ `Fillna`（填充缺失特征）→ `ScreenProcessor`（筛选合格股票）→ `DropnaLabel`（去除无标签样本）
4. Kronos 自回归预测合格股票的 5 日 OHLCV → 计算排序分数 → 横截面 Z-score 标准化 → Top-3 等权（各 33.3%）
5. 输出 `output/result.csv`（stock_id + weight）

## 代码结构说明

```
code/
├── handlers/stock_handler.py         # 自定义 DataHandler（Alpha158 + 额外表达式特征）
├── processors/custom_processor.py    # 12 个自定义 Processor（动量/波动/流动性/横截面/行业等）
├── models/
│   ├── Kronos/                        # Kronos Qlib 模型封装 + vendored 源码
│   │   ├── KronosModel.py             # Qlib Model 封装（fit no-op, predict 自回归推理）
│   │   └── kronos_src/                # Vendored Kronos 源码（Tokenizer, Predictor, BSQ）
│   ├── PointwiseStockTransformer/     # PyTorch Transformer 回归模型（基线对比）
│   └── LightGBM/                      # LightGBM 模型（基线对比）
├── PortfolioBuilder/
│   └── portfolio_strategy.py          # 组合优化策略（均值方差/最小方差/风险平价）
└── src/
    ├── train.py                       # 训练入口
    ├── commit.py                      # 推理入口 → output/result.csv
    └── run_all_model.py               # 调参评测引擎（单训 + 多周回测 + 排序/仓位指标）

scripts/
├── get_stock_data.py                  # Baostock 数据抓取 → data/stock_data.csv
├── update_train_data.py               # 生成 train.csv（排除盲测区间）
├── convert_data.py                    # CSV + 行业分类 → Qlib 二进制
├── finetune_kronos.py                 # Kronos Predictor 微调
├── finetune_kronos_tokenizer.py       # Kronos Tokenizer 微调
├── download_kronos_models.py          # 下载 Kronos 预训练权重
└── update_data.sh                     # 一键更新：Baostock → train.csv → Qlib 二进制

test/
├── score_self.py                      # 自评：output/result.csv vs data/test.csv
├── test.py                            # 批量 Docker 评分
└── score_docker.py                    # Docker 单次评分
```

## 其他注意事项

- 固定随机种子 seed=42 保证可复现
- 数据划分：训练 2024-01 至 2025-12，验证 2026-01，测试 2026-03-20 至 2026-06-12
- 筛选后股票池约 65-85 只（从 300 只沪深 300 成分股中筛选）
- Qlib 二进制缓存 `temp/qlib_data/` 被 `.gitignore` 排除，Docker 内需运行 `sh init.sh` 重建
- 推理时间约束 ≤ 5 分钟，训练时间约束 ≤ 8 小时，Docker 镜像 ≤ 10GB
- 所有训练和预测均离线运行，不得联网
- GPU/CPU 自动选择（CUDA → CPU）

### 常见问题

1) **TA-Lib 安装失败**：本项目的 MomentumProcessor / VolatilityProcessor / VolumeProcessor 依赖 TA-Lib。Dockerfile 中已内置编译步骤，本地需手动安装 C 库（见下方）。

```
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib && ./configure --prefix=/usr && make -j1 && make install
```

2) **Qlib 二进制数据缺失或过期**：运行 `sh init.sh` 重新生成。

3) **Kronos 预训练权重缺失**：运行 `python scripts/download_kronos_models.py` 下载（支持 `HF_ENDPOINT=https://hf-mirror.com` 镜像）。
