# 代码说明

## 环境配置

- Python 3.12（`.python-version` 锁定），包管理仅用 **uv**（禁止 pip/conda/poetry）
- PyTorch 2.11.0+cu128，pyqlib 0.9.7，pandas 2.3.x，numpy 2.x
- 直接依赖仅保留最终提交运行时闭包：numpy、pandas、torch、pyqlib、pyyaml、huggingface_hub、safetensors、einops、tqdm、scipy
- 安装依赖：`uv sync`；激活环境：`source .venv/bin/activate`

## 数据

- 使用 **Baostock 公开行情数据**（http://baostock.com），沪深 300 成分股日线 OHLCV（后复权 adjustflag="1"），
  时间范围 2022-01-04 ~ 2026-07-31
- 获取方式：`uv run python scripts/get_stock_data.py`（通过 START_DATE/END_DATE 环境变量控制区间），输出至 `model/data/stock_data.csv`
- 在训练（微调 Kronos + 学习排序头）时使用；辅助数据：`resource/行业分类.csv`（中证四级行业编码）、
  `data/hs300_index.csv`（市场状态诊断）
- 数据切分：`model/data/train.csv`（训练，排除盲测区间 2026-04-13 ~ 04-17）、`data/test.csv`（主办方盲测集，仅本地自评用，不可修改）、
  `model/qlib_data/`（Qlib 二进制缓存，`sh init.sh` 生成）
- **挂载规则**：最终验证时赛事方会挂载并覆盖 `data/`、`temp/`、`output/`；自有数据全部打包在非挂载目录 `model/data/` 下，运行时不依赖挂载内容

## 预训练模型

- 使用 **Kronos-small**（NeurIPS 2025 / AAAI 2026 发表的开源金融 K 线基础模型，24.7M 参数）与
  **Kronos-Tokenizer-base**
- 获取方式：`uv run python scripts/download_kronos_models.py`（支持 HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`），
  保存至 `model/kronos_pretrained/`
- 对 `code/models/Kronos/KronosModel.py` 中的 **KronosPredictor** 网络在 A 股数据上微调（Tokenizer 冻结），
  产物 `model/kronos_finetuned/Kronos-small-lb60`
- 加载使用 `local_files_only=True`，训练与推理全程离线

## 算法

### 整体思路介绍

微调 Kronos 自回归预测未来 5 日 OHLCV → 计算 5 日开盘收益分数
`(open[T+5]-open[T+1])/open[T+1]` → 学习排序头（Context Transformer，NDCG 损失，
输入 = Kronos 分数 + 20 日反转 + 行业动量 + 横截面统计 + 流动性 + 市场环境）精排 →
市场状态分类（科技/老经济 × 趋势/轮动）自动选择持仓数 K → **Top-K 等权**输出。

### 方法的创新点（如果有）

1. **状态条件化自动激进度**：按市场状态自动选择持仓数 K（科技类状态 K=2、当前老经济-轮动 K=3），
   见 `model/auto_k_table.json` 与 `docs/auto_aggressiveness_20260801.md`。
2. **学习排序头融合短期信号**：把 20 日反转、行业动量、横截面统计、流动性、市场环境作为特征，
   用 NDCG 端到端学习选股（非手写规则），10 周样本外周均超额 +3.89%（最佳单头）。
3. **特征管线统一**：训练与生产使用同一套特征计算（代码规范化 + rev20 zfill 修复），
   保证"生产选股 = 验证选股"。
4. **研究证伪支撑设计**：CardNN 端到端可微配权在真 Kronos 评分上未超过 Top-3 等权；
   20 日波动/回撤作为注意力头输入反而下降——这些消融结论直接指导了最终方案。

### 网络结构

- **Kronos Tokenizer**：Transformer 自编码器 + BSQ（Binary Spherical Quantization），
  OHLCV 量化为 20-bit 离散 token（s1/s2 各 10-bit，码本 1024×1024），3 层编码器 + 3 层解码器，d_model=256。
- **Kronos Predictor**：decoder-only Transformer（8 层，d_model=512，8 头），RoPE 位置编码、SwiGLU FFN、
  DualHead 分层预测（s1→s2 通过 DependencyAwareLayer）；微调仅 Predictor。
- **学习排序头 ContextTransformer**：d_model=32、2 层、4 头；`[CLS]` 市场 token + 股票 token
  （7 维特征）self-attention → 每股精排分数。

### 损失函数

- Kronos 微调：层次化交叉熵（s1 + s2 子 token 的平均 CE）
- 学习排序头：NDCG approximation（σ=1.0, k=5），per-date-group 训练
- 研究过的替代损失（MSE / Pairwise / ListMLE / 组合收益损失）在真 Kronos 评分上均未优于 NDCG

### 数据扩增

- 滑动窗口采样：每只股票以 60 日 lookback → 未来 5 日生成多个训练样本
- 特征归一化（窗口内标准化、Winsorize 1%/99%）；未使用生成式扩增

### 模型集成

- 最终提交**不使用集成**（研究显示 3-seed 分数平均会稀释最佳单头收益，且当前状态统计不支持）
- 曾探索：多 seed 排序分数平均、CardNN（Gumbel-Sinkhorn）端到端可微配权——均已通过消融验证
  不进入最终方案，详见 `docs/cardnn_e2e_report_20260801.md`

### 算法的其他细节

- 排序分数与赛题标签一致：`(open[T+5]-open[T+1])/open[T+1]`
- 推理参数：pred_len=5、T=1.0、top_p=0.9、sample_count=1、seed=42
- 金融筛选 ScreenProcessor：成交额前 70%、收盘价 > MA60、20 日最大回撤 < 15%
- 自动 K：`commit.py` 的 `exposure_mode: auto` 按状态查表（`model/auto_k_table.json`）决定持仓数
- 固定随机种子 seed=42，两次运行输出完全一致（可复现性已验证）

## 训练流程

`sh train.sh`（等价 `uv run code/src/train.py`），逐步说明：

1. 初始化 Qlib：读取 `model/result_model.yaml` 的 `qlib_init`（数据路径 `temp/qlib_data`）。
2. 通过配置创建模型：`KronosContextHeadModel`（Kronos 微调权重 + 学习排序头）。
3. `model.fit(dataset)`：加载排序头权重（`model/context_head/head_s2024.pt`）；
   排序头的 NDCG 训练由研究脚本 `scripts/train_context_head_d.py` 完成（离线、GPU）。
4. 保存产物：`model/result_model.pth`（排序头权重）+ `model/config_snapshot.yaml`（配置快照）。

前置步骤（一次性）：
- `sh init.sh`：`model/data/train.csv` + 行业分类 → Qlib 二进制（`model/qlib_data`）
- `uv run python scripts/finetune_kronos.py`：微调 Kronos Predictor（2024-01~2025-12）
- `uv run python scripts/train_context_head_d.py`：训练学习排序头（3 seeds，保存 head_s*.pt）

## 推理流程

`sh test.sh`（等价 `uv run code/src/commit.py`），逐步说明：

1. `set_seed(42)`，加载 `model/result_model.yaml`。
2. 初始化 Qlib（`model/qlib_data`）；从 `model/data/train.csv` 推断测试日期（最新交易日）。
3. `build_model`：构造模型并加载 `model/result_model.pth`（排序头权重，`_net` 注入）。
4. `build_test_dataset`：单日测试集（FridayFilter → Fillna → ScreenProcessor 筛选）。
5. `model.predict`：Kronos 打分 → 上下文特征 → 排序头精排 → 每股分数。
6. `exposure_mode: auto`：`classify_market_state` 判定状态 → `auto_k_table.json` 选 K →
   Top-K 等权配权。
7. 输出 `output/result.csv`（stock_id + weight，≤5 行、权重和 ≤ 1）。

## 其他注意事项

- **验证/测试划分**：训练 2024-01-01 ~ 2026-04-17（排除盲测相邻周 04-10/04-17）、
  验证 2026-04-24 ~ 05-08、测试最近 10 周 2026-05-15 ~ 07-24；研究窗口细节见 `docs/`。
- `data/test.csv`（2026-04-13 ~ 04-17）为主办方盲测集，**不可修改、不可训练**。
- 硬性约束：推理 ≤ 5 分钟、训练 ≤ 8 小时、Docker 镜像 ≤ 10GB、全程离线。
- 权重/数据不入 git（`.gitignore`：model/kronos_*、model/context_head、model/data/、model/qlib_data/）；
  Docker 镜像内 `data/` 由赛事方挂载（最终验证会覆盖），自有数据在 `model/data/`、权重在 `model/` 随镜像打包。
- 预训练模型已按 7/18 要求报备（开源链接 + md5，见 `报备邮件.txt` 模板）。
- 可复现性：固定 seed=42，两次 `sh test.sh` 输出一致（已实测验证）。
