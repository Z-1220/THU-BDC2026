# 代码说明

## 环境配置

- Python 3.12, PyTorch 2.11.0+cu128, pyqlib 0.9.7
- 包管理使用 uv，依赖锁定在 uv.lock
- TA-Lib C 库在 Dockerfile 中编译安装

## 数据

使用 Baostock 公开数据（免费注册即可获取），通过 `scripts/get_stock_data.py` 下载沪深 300 成分股日线 OHLCV 数据（后复权），训练数据时间范围为 2024-01 至 2025-12。

辅助数据：`resource/行业分类.csv` 提供中证四级行业分类编码。

## 预训练模型

使用 Kronos-small（NeurIPS 2025 / AAAI 2026 发表），HuggingFace 地址：
https://huggingface.co/NeoQuasar/Kronos-small
https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base

下载脚本：`python scripts/download_kronos_models.py`（支持 HF 镜像）。

## 算法

### 整体思路

基于 Kronos 金融 K 线基础模型，在 A 股数据上微调 Predictor（自回归 Transformer），利用其预测未来 5 日 OHLCV 序列，以预测的 T+5 开盘 vs T+1 开盘收益率作为选股排序分数。选等权 Top-5 作为最终组合。

### 方法细节

1. **预训练模型微调**：仅微调 Kronos Predictor（24.7M 参数），Tokenizer 保持冻结。训练目标是自回归 next-token prediction 的交叉熵损失。
2. **金融筛选**：在预测阶段通过 ScreenProcessor 剔除不合格股票（成交额排名后30%、价格低于MA60、20日最大回撤超过15%）。
3. **行业中性化**：每只股票的得分减去其一级行业的中位数，消除行业系统性偏差。
4. **组合策略**：等权 Top-5（每只 0.2），累计权重 1.0。

### 网络结构

- Kronos Tokenizer：Transformer 自编码器 + Binary Spherical Quantization (BSQ)，将 OHLCV 6 维数据量化为离散 token
- Kronos Predictor：Decoder-only Transformer（12层，d_model=1024，16头），自回归预测 token 序列

### 损失函数

- 微调阶段：层次化交叉熵损失（s1 + s2 子 token 的平均 CE）
- 推理阶段：无需训练，冻结权重前向传播

## 训练流程

1. `sh init.sh`：将 `data/train.csv` 转为 Qlib 二进制数据
2. `sh train.sh`：加载 `model/result_model.yaml` 配置，创建 KronosModel 并调用 `fit()`（Kronos 的 fit 为 no-op，使用预训练+微调权重），保存权重快照
3. 实际微调在 Docker 外完成，产物为 `model/kronos_finetuned/Kronos-small-lb60/`

## 推理流程

1. `sh test.sh`：调用 `code/src/commit.py`
2. 从 `data/train.csv` 推断推理日期（2026-05-29，A2 提交窗口）
3. 初始化 Qlib → 加载 Kronos 模型 → 构建测试数据集（FridayFilter → Fillna → Screen → DropnaLabel）
4. Kronos 自回归预测 65 只合格股票的 5 日 OHLCV → 计算排序分数 → 等权 Top-5 → 输出 `output/result.csv`

## 其他注意事项

- 固定随机种子 seed=42 保证可复现
- Tokenizer 保持预训练权重（冻结），不进行微调
- 数据划分：训练 2024-01 至 2025-12，验证 2026-01，测试 2026-02 至 2026-05
- 筛选后股票池约 65-85 只（从 300 只中筛选）
