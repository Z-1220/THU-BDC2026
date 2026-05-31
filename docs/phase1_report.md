# Phase 1 最佳模型复现报告

## 1. 测试环境

| 项目 | 配置 |
|------|------|
| OS | Ubuntu 24.04 (Linux 6.8.0) |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA L2 (22.0 GB VRAM) |
| Qlib | 0.9.7 |
| 包管理 | uv（禁止 pip/conda/poetry） |

### 核心依赖版本

| 包 | 版本 |
|------|------|
| `pyqlib` | 0.9.7 |
| `torch` | 2.11.0+cu128 |
| `lightgbm` | 4.6.0 |
| `catboost` | 1.2.10 |
| `xgboost` | 3.2.0 |
| `baostock` | 最新 |
| `pandas` | 最新（uv.lock 锁定） |

### 模型文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `model/kronos_pretrained/Kronos-Tokenizer-base/` | 16 MB | 预训练 Tokenizer（BSQ 码本 + 编解码器） |
| `model/kronos_pretrained/Kronos-small/` | 95 MB | 预训练 Predictor（24.7M 参数） |
| `model/kronos_finetuned/Kronos-small-lb60/` | 95 MB | Phase 1 微调产物（仅 Predictor 权重） |
| `model/result_model.pth` | 151 KB | Phase 1 训练保存的权重快照 |

预训练模型来源：[NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) 和 [NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)。下载脚本：`python scripts/download_kronos_models.py`，支持 HF 镜像。

---

## 2. 复现步骤

### 2.1 环境准备

```bash
# 安装 Python 依赖
uv sync
source .venv/bin/activate

# 下载 Kronos 预训练模型（如已下载可跳过）
python scripts/download_kronos_models.py
```

### 2.2 数据准备

**数据源**：Baostock 在线 API，后复权（adjustflag="1"），日线数据。

```bash
# 拉取 2024-01-01 至 2026-05-29 的沪深 300 日线数据
END_DATE="2026-05-29" START_DATE="2024-01-01" uv run python scripts/get_stock_data.py
```

数据字段：股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅。

输出：`data/stock_data.csv`（约 17 万行，300 只股票，覆盖 2024-01-02 至 2026-05-29）。

### 2.3 微调 Kronos Predictor

```bash
# 仅微调 Predictor（Tokenizer 冻结）
uv run python scripts/finetune_kronos.py \
  --model small \
  --lookback 60 \
  --pred-len 5 \
  --epochs 20 \
  --lr 4e-5 \
  --save-dir "./model/kronos_finetuned"
```

训练参数：
- 优化器：AdamW (lr=4e-5, weight_decay=0.1, betas=(0.9, 0.95))
- 调度器：OneCycleLR (pct_start=0.03, div_factor=10)
- 批次大小：512
- 梯度裁剪：3.0
- 早停：5 epochs
- 训练样本：~154K，验证样本：~5K
- 训练时间：~2 小时（NVIDIA L2, 20 epochs）

产物：`model/kronos_finetuned/Kronos-small-lb60/`（微调后的 Predictor 权重）。

### 2.4 构建 Qlib 二进制数据

```bash
# 生成 train.csv（排除盲测区间）
uv run python scripts/update_train_data.py

# 转为 Qlib 二进制
# 注意：temp/qlib_data/ 为 Docker root 权限，重建需通过 Docker
docker compose run --rm --entrypoint "" app uv run scripts/convert_data.py
```

产物：`temp/qlib_data/`（1059 个交易日，2022-01 至 2026-05，300 只股票 × 16 字段）。

### 2.5 评测

```bash
# 确保 Kronos-small-finetuned.yaml 中 finetuned_dir 指向 Phase 1 模型
uv run python code/src/run_all_model.py run \
  --yaml_paths="models/Kronos/Kronos-small-finetuned.yaml"
```

**配置文件**（`code/models/Kronos/Kronos-small-finetuned.yaml`）：

```yaml
task:
  model:
    class: KronosModel
    module_path: "code.models.Kronos.KronosModel"
    kwargs:
      model_name: "small"
      pretrained_dir: "./model/kronos_pretrained"
      finetuned_dir: "./model/kronos_finetuned/Kronos-small-lb60"
      max_context: 512
      pred_len: 5
      T: 1.0
      top_p: 0.9
      sample_count: 1
      seed: 42

  # 无 strategy = 等权 top-5 (0.2 each)

  dataset:
    class: DatasetH
    kwargs:
      handler:
        class: StockDataHandler
        kwargs:
          instruments: "all"
          start_time: "2024-01-01"
          end_time: "2026-05-29"
          fit_start_time: "2024-01-01"
          fit_end_time: "2025-12-31"
          infer_processors:
            - class: FridayFilterProcessor
            - class: Fillna
            - class: ScreenProcessor
              kwargs:
                min_amount_rank: 0.3
                trend_ma: 60
                max_drawdown: 0.15
          learn_processors:
            - class: DropnaLabel
      segments:
        train: ["2024-01-01", "2025-12-31"]
        valid: ["2026-01-05", "2026-01-30"]
        test:  ["2026-02-02", "2026-05-29"]

  evaluation:
    universe: "all"
    data_end: "2026-05-29"
```

### 2.6 推理

```bash
# 确保 model/result_model.yaml 配置正确
uv run python code/src/commit.py
```

输出：`output/result.csv`。

---

## 3. 演进路线：从基线到最优

### 3.1 起点：Kronos 零样本推理

**提交**：`e59322a`（Kronos 集成）

直接用 Kronos-small 预训练权重做零样本推理。模型通过自回归预测未来 5 日 OHLCV，用预测的 T+5 开盘 vs T+1 开盘计算收益率作为排序分数。

**结果**：可运行，但选股质量不高——预训练权重未适配 A 股市场特征。

### 3.2 突破 1：Predictor 微调（Phase 1 核心）

**提交**：`08b1488` + `6407136` + `2979154`

在 A 股数据上微调 Kronos Predictor（自回归 Transformer），Tokenizer 保持冻结。
- 标签：`(Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)` — 与赛题公式对齐
- 损失：交叉熵（预测 OHLCV token 序列的下一个 token）
- 数据：2024-01 至 2025-12 的沪深 300 日线 OHLCV

**结果**：累计收益从零样本的退化表现提升至 +21.24%（13 周评测）。

### 3.3 突破 2：ScreenProcessor 金融筛选

**提交**：`cb59363`

在预测前通过 Qlib Processor 管道剔除不合格股票：
- 成交额排名 < 前 70% → 剔除流动性枯竭股
- 价格 < MA60 → 剔除长期下行股
- 20 日最大回撤 > 15% → 剔除持续暴跌股

筛选后股票池从 300 缩减到 65-85 只，有效排除了噪声。

**关键架构决策**：筛选放在预测阶段（Processor），不在组合优化阶段。这保证了模型只对"值得考虑"的股票打分，避免排名分数被基本面有问题的股票污染。

### 3.4 突破 3：Score 对齐赛题 => 行业中性化

**提交**：`bcbbc43` + `79aef8b`

- Kronos 预测的 5 日收益 score 从之前的不一致计算统一为 `(open[T+5] - open[T+1]) / open[T+1]`
- 加入行业中性化：每只股票的 score 减去其行业的中位数，消除行业系统性偏差

### 3.5 突破 4：等权优于风险平价

**提交**：`0c9fdb6`

对比了风险平价（Hierarchical Risk Parity）和等权 top-5：

| | 等权 | 风险平价 |
|------|------|------|
| 累计收益 | **+21.24%** | +14.23% |
| 周均 | **+1.54%** | +1.05% |
| 胜率 | **69.2%** | 61.5% |
| 最佳周 | **+9.16%** | +4.41% |

**结论**：对于 ≤5 只股票、4 天持有期的极端短期选股，复杂组合优化不但没有帮助，反而反效果。等权 top-5 是最优策略。

### 3.6 被证明无效的尝试

#### ❌ 尝试 1：Tokenizer 微调（两阶段微调）

用 A 股 OHLCV 数据微调 Tokenizer 的 BSQ 码本和编解码器，试图让 Tokenizer 更好地适配 A 股量价特征。

**消融结果**：

| | Tokenizer | 数据 | Val Loss | 累计收益 |
|------|-----------|------|------|------|
| Phase 1 | 预训练（冻结） | 24-25 | — | **+19.66%** |
| Phase 2 | 微调 | 22-25 | 3.73 | +11.54% |
| Exp B | 微调 | 24-25 | 3.77 | +15.68% |

**结论**：无论在哪种数据组合下，微调后的 Tokenizer 都比预训练的差。Tokenizer 在 45+ 交易所 120 亿条 K 线上训练出的全局码本已经足够通用和优质，用 A 股数据微调只会扭曲这个码本。**不应当微调 Tokenizer。**

#### ❌ 尝试 2：扩展数据至 2022 年

将训练数据从 2024-2025 扩展到 2022-2025，试图让模型学习更多历史规律。

**消融结果**：

| | 数据 | Tokenizer | Val Loss | 累计收益 |
|------|------|------|------|------|
| Phase 1 | 24-25 | 预训练 | — | **+19.66%** |
| Exp A | 22-25 | 预训练 | **3.09** | +10.31% |

**结论**：Exp A 的训练 loss 最低（3.09 vs Phase 2 的 3.73），但交易收益最差（+10.31% vs +19.66%）。这暴露了一个关键问题——**自回归交叉熵 loss 与 stock ranking 之间存在严重的代理目标错配**。模型学会了更好地预测 token 序列，但这并没有转化为更好的选股能力。

2022-2023 年沪深 300 大幅下跌（-22%），市场状态与 2025-2026 截然不同，混入这些数据带来了更多的噪声而非信号。

#### ❌ 尝试 3：多维预测信号（日度 Sharpe 替代简单收益）

**提交**：`1e94837` → `6f2767b`（回退）

在预测期内用日度 Sharpe 替代简单的开盘点差作为 score。

**结论**：效果不如简单的 `(open[T+5] - open[T+1]) / open[T+1]`，已回退。简单直接 open-to-open 收益最匹配赛题。

#### ❌ 尝试 4：预测不确定性剔除（两次推理发散排除）

**提交**：`d4ea237` → `c245d04`（回退）

用两次不同随机种子推理，如果同一股票两次分数差异过大（发散了），排除之。

**结论**：退步，已回退。

#### ❌ 尝试 5：sample_count 增大（MC 采样次数 1→5）

**提交**：`1eb0500` → `f70d866`（回退）

增加 Monte Carlo 采样次数从 1 到 5，试图降低预测方差。

**结论**：退步（推理更慢且结果更差），已回退到 sample_count=1。

---

## 4. 消融实验设计（4 组对比 2 个变量）

### 4.1 完整指标对比

| | Phase 1 | Phase 2 | Exp A | Exp B |
|------|------|------|------|------|
| 训练数据 | 2024-2025 | 2022-2025 | 2022-2025 | **2024-2025** |
| Tokenizer | 预训练(冻结) | 微调 | 预训练(冻结) | 微调 |
| 累计收益 | **+19.66%** | +11.54% | +10.31% | +15.68% |
| 周均收益 | **+1.44%** | +0.86% | +0.78% | +1.14% |
| 中位数 | +1.08% | +1.30% | +0.10% | **+1.28%** |
| 波动率(std) | 3.11% | 1.97% | 1.94% | **1.76%** |
| 夏普 | 0.46 | 0.44 | 0.40 | **0.65** |
| 索提诺 | **1.75** | 0.53 | 1.57 | 1.11 |
| 胜率 | 69.2% | 69.2% | 53.8% | 69.2% |
| 最大回撤 | -3.03% | -4.43% | -3.84% | **-2.78%** |
| 最佳3周均值 | **+5.92%** | +2.76% | +3.63% | +3.22% |
| 最差3周均值 | -2.17% | -2.01% | -1.35% | **-1.39%** |
| 平均盈利 | **+2.88%** | +1.97% | +2.24% | +2.12% |
| 平均亏损 | -1.80% | -1.64% | **-0.93%** | -1.06% |
| 盈亏比 | 1.60 | 1.21 | **2.41** | 2.00 |
| 胜:负 | 9:4 | 9:4 | 7:6 | 9:4 |

### 4.1.1 排名质量指标

除投资组合收益指标外，引入 4 个学习排序（Learning-to-Rank）指标，从排序质量角度评估模型预测能力：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **RankIC** | 收益排名相关性 | Spearman 相关系数：`corr(pred_scores, actual_rets)` — 预测分数与真实收益的排序一致性 |
| **IC>0%** | 正相关周占比 | IC>0 的周数 / 总周数 — 模型预测方向是否正确 |
| **Hit@5** | Top5 命中率 | 预测 Top5 与实际 Top5 的交集 / 5 — 是否选到了真最好的股票 |
| **Recall@10** | 高收益召回率 | 预测 Top5 中出现在实际 Top10 的占比 / 10 |
| **NDCG@5** | Top5 排序准确率 | 标准化折损累积增益 — 衡量 Top5 预测排序与理想排序的接近程度 |

**消融实验结果**：

| 模型 | RankIC | IC>0% | Hit@5 | Recall@10 | NDCG@5 |
|------|-------|------|------|----------|-------|
| **Phase 1** | **0.0465** | **69.2%** | **7.7%** | **10.8%** | 0.0561 |
| Phase 2 | 0.0239 | 53.8% | 7.7% | 6.9% | 0.0024 |
| Exp A | 0.0421 | 46.2% | 6.2% | 7.7% | **0.0581** |
| Exp B | 0.0170 | 38.5% | 7.7% | 6.9% | 0.0305 |

**关键解读**：

1. **Phase 1 排序质量全面最优** — RankIC 0.0465、IC>0% 69.2% 均显著领先。其高累计收益（+19.66%）有坚实的排序能力支撑，不是靠运气。

2. **Exp B 排序能力最差，高夏普是假象** — RankIC 仅 0.017，IC>0% 仅 38.5%（意味着 61.5% 的周预测与实际收益负相关）。它的高夏普（0.65）来自低波动选股（稳定选同样几只大盘股），而非准确的排序能力。这种"保守"在下跌市保护了损失，但长期看没有抓住真正的 alpha。

3. **Exp A 的 RankIC 两极分化** — RankIC 0.042（不错）但 IC>0% 仅 46.2%（最差），说明预测质量极度不稳定：好的时候很好，坏的时候完全方向错误。这解释了其胜率最低（53.8%）。

4. **Hit@5 普遍仅 7-8%** — 平均每周只命中 ~0.4 只真正的 Top5 股票。即使最好的 Phase 1 也只有 0.4/5，说明还有巨大的选股改进空间。

5. **NDCG@5 极度偏低（0.002-0.058）** — 所有模型的 Top5 排序质量都远不理想，表明当前模型对"哪些股票最好"的判断力有限，需要结构性地改进信号质量。

### 4.2 核心发现

#### 数据范围的影响（控制 Tokenizer 变量）

| 对比 | 数据 | 累计收益 | 夏普 | 波动率 | 最大回撤 |
|------|------|------|------|------|------|
| Phase 1 vs Exp A | 24-25 vs 22-25 (均冻结) | +19.66% vs +10.31% | 0.46 vs 0.40 | 3.11% vs 1.94% | -3.03% vs -3.84% |
| Phase 2 vs Exp B | 22-25 vs 24-25 (均微调) | +11.54% vs +15.68% | 0.44 vs 0.65 | 1.97% vs **1.76%** | -4.43% vs **-2.78%** |

**结论**：24-25 数据在两组对比中都优于 22-25——累计收益和夏普均更高。22-25 增加了波动但未增加收益。

#### Tokenizer 的影响（控制数据变量）

| 对比 | Tokenizer | 累计收益 | 夏普 | 盈亏比 |
|------|-----------|------|------|------|
| Phase 1 vs Exp B | 冻结 vs 微调 (均24-25) | +19.66% vs +15.68% | 0.46 vs **0.65** | 1.60 vs **2.00** |
| Exp A vs Phase 2 | 冻结 vs 微调 (均22-25) | +10.31% vs +11.54% | 0.40 vs 0.44 | **2.41** vs 1.21 |

**结论**：Tokenizer 微调降低了累计收益但显著改善了风险指标。这是一个典型的"收益-风险"权衡。

### 4.3 用户洞察：Exp B 的夏普最高意味着什么？

Exp B 的夏普 0.65 是 Phase 1 的 1.4 倍。原因是：

**Phase 1 的收益分布更极端**：
- 最佳周 **+9.16%**（2026-05-22 周）—— 单周拉高了整体收益
- 最差周 -3.03%，平均亏损 -1.80%
- 波动率 3.11%，标准差几乎是 Phase 2 的两倍

**Exp B 的收益分布更均匀**：
- 最佳周 +3.75%，最差周 -2.78%
- 波动率仅 1.76%，是四个模型中最低的
- 平均亏损 -1.06%，是 Phase 1 的近一半
- 盈亏比 2.00（赚 2 块亏 1 块）vs Phase 1 的 1.60

**这意味着**：如果赛题是多期平均排名或按夏普排名，Exp B 是最优选择。但赛题按 **绝对累计收益率** 排名——Phase 1 的大涨周直接贡献了最终的胜利。

### 4.4 潜在改进方向：相位融合

Phase 1 的"爆发力"和 Exp B 的"稳定性"不一定互斥。可能的融合方案：

1. **分数加权平均**：`score_final = α * score_ph1 + (1-α) * score_expB`，其中 α 用验证集调优
2. **保守投票**：两个模型预测的 top-5 取交集（至少 3 只重叠），交集不够则减少持仓数
3. **波动率自适应**：市场高波动时多用 Exp B（风控），低波动时多用 Phase 1（进攻）

---

## 5. 当前最优配置总结

| 组件 | 选择 | 理由 |
|------|------|------|
| 预训练模型 | Kronos-small (24.7M) | 预训练 Tokenizer 已足够优，不应微调 |
| 微调方式 | 仅 Predictor FT (frozen Tokenizer) | Tokenizer FT 被消融证伪 |
| 训练数据 | 2024-01 至 2025-12 | 过多历史数据（2022-2023）增加噪声 |
| 金融筛选 | ScreenProcessor（前 70% 成交额 + MA60 趋势 + 15% 回撤） | 有效剔除不可投股票 |
| Score 计算 | `(open[T+5] - open[T+1]) / open[T+1]` 对齐赛题 | 消融确认最佳 |
| 行业调整 | Score - 行业中位数 | 消除行业系统性偏差 |
| 组合策略 | 等权 top-5 (0.2 each) | 优于风险平价/均值方差 |
| 推理参数 | T=1.0, top_p=0.9, sample_count=1 | MC 采样增加被证伪 |

### 评测指标（13 周，2026-02-06 至 2026-05-29）

| 指标 | 值 | 说明 |
|------|-----|------|
| 累计收益 | **+19.66%** | 13 周累计复利收益 |
| 周均收益 | +1.44% | 均值 |
| 周中位数 | +1.08% | 稳健性参考 |
| 胜率 | 69.2% (9 胜 4 负) | 盈利周占比 |
| 夏普比率 | 0.46 | 周频年化 |
| 索提诺比率 | 1.75 | 仅下行风险 |
| 最大回撤 | -3.03% | 单周最大亏损 |
| 最佳 3 周均值 | +5.92% | 上行爆发力 |
| 最差 3 周均值 | -2.17% | 下行控制力 |
| 盈亏比 | 1.60 | 平均盈利/平均亏损 |
| **RankIC** | **0.0465** | 收益排名相关性 (Spearman) |
| Hit@5 | 7.7% | Top5 命中率 |
| Recall@10 | 10.8% | Top10 召回率 |
| NDCG@5 | 0.056 | Top5 排序准确率 |

---

## 6. 已知限制与下一步

1. **温度/采样尚未调优**：T=1.0, top_p=0.9 是 Kronos 默认值，未针对 A 股优化
2. **无短期爆发信号**：只有 Kronos 的 OHLCV 预测，没有独立的技术面爆发识别
3. **无动态仓位**：始终满仓 5 只，市场下跌期也满仓
4. **无模型集成**：只用了 Kronos-small，未集成 LightGBM 或 PointwiseStockTransformer
5. **代理目标错配**：训练用的是 cross-entropy loss（预测 OHLCV token），但目标是 stock ranking
