# THU-BDC2026 研究与探索报告

## 一、竞赛背景与硬约束

**任务**：T 日（周五信号）→ T+1 开盘买入 → T+5 开盘卖出，从沪深 300 中选 ≤5 只股票，权重和 ≤1（剩余现金，收益为 0）。

**核心矛盾**：极度短期（4 个交易日）、小组合（≤5 只）、可空仓 — 本质是"短期爆发识别 + 极端收益预测"，而非传统多因子选股。

**硬约束**：Docker ≤10GB、预测 ≤5 分钟、训练 ≤8 小时、离线、可复现。

**关键时间节点**：A3 窗口 6/27-28，B 窗口 8/1-2（最终排名），模型/数据报备 7/18。

---

## 二、Phase 1 最佳配置与消融实验

### 2.1 当前最优配置

| 组件 | 配置 |
|------|------|
| 基模型 | Kronos-small (24.7M)，Kronos-Tokenizer-base |
| 微调策略 | 仅微调 Predictor，Tokenizer 冻结 |
| 训练数据 | 2024-01 至 2025-12，2026-01 验证 |
| 推理 | pred_len=5，T=1.0，top_p=0.9，sample_count=1 |
| 筛选 | ScreenProcessor（成交额后 30%、跌破 MA60、回撤 >15% 剔除） |
| 后处理 | 行业中性化（sector median 减法） |
| 组合 | 等权 Top-5（每只 0.2） |
| 测试周期 | 2026-02-02 至 2026-05-29（约 17 周） |

### 2.2 消融实验：4 配置 × 2 变量

**实验设计**：数据范围（2024-2025 vs 2022-2025）× Tokenizer 状态（冻结 vs 微调）

| 配置 | 数据范围 | Tokenizer | 累计收益 | Sharpe | RankIC | IC>0% | 备注 |
|------|---------|-----------|---------|--------|--------|-------|------|
| Phase 1 | 2024-2025 | 冻结 | **+19.66%** | 0.52 | 0.047 | 53.8% | **最优** |
| Exp A | 2022-2025 | 冻结 | +10.31% | 0.28 | 0.035 | 50.0% | 验证损失最低(3.088)但交易最差 |
| Exp B | 2024-2025 | 微调 | +15.68% | 0.65 | 0.017 | 38.5% | Sharpe 最高但 RankIC 最差 |
| Phase 2 | 2022-2025 | 微调 | +11.54% | 0.41 | 0.012 | 42.3% | 两项改动叠加，综合最差 |

### 2.3 消融核心发现

1. **冻结 Tokenizer 优于微调**：冻结平均 +14.99%，微调平均 +13.61%，差距约 1.4 个百分点
2. **2024-2025 数据优于扩展至 2022**：短数据平均 +17.67%，长数据平均 +10.93%，差距约 6.7 个百分点
3. **验证损失（交叉熵）与交易表现不相关**：Exp A 的 val_loss 最低(3.088)但累计收益最低(+10.31%)
4. **Sharpe 与 RankIC 可能背离**：Exp B 的 Sharpe 最高(0.65)但 RankIC 最低(0.017)，Sharpe 可能被少数大收益周拉高
5. **交叉熵损失与股票排序之间存在目标 mismatch**：Kronos 优化的是 token 预测准确率，而赛题需要的是排序质量

### 2.4 A2 提交结果

| 项目 | 详情 |
|------|------|
| 模型 | Phase 1 配置 |
| 入选股票 | 600011, 600150, 600176, 600188, 600460（各 0.2） |
| 实际得分 | +1.67% |
| 关键贡献 | 兖矿能源 +20.66% 撑起组合收益 |

---

## 三、改进方向梳理

### 3.1 原始改进方案（8 个方向，按优先级）

**P0（ROI 最高）**：
1. **动态仓位管理**：市场状态检测（涨跌 + 波动率）→ 四象限 → 决定投几只股票、多少仓位
2. **横截面 Z-Score 后处理**：去极端值 → 行业中性化 → Z-score 标准化

**P1（A3 窗口可做）**：
3. **温度/采样校准**：网格搜索 Kronos 的 T、top_p、sample_count 最优值
4. **短期爆发信号融合**：Bollinger 挤压突破、52 周高点动量、成交量异常等独立技术信号
5. **市场环境特征注入**：通过 prefix conditioning 或后处理注入市场涨跌、波动率、宽度信息

**P2（B 窗口前）**：
6. **Tokenizer 行业信息微调**：用更温和的方式在 Tokenizer 中注入行业信息
7. **周频 Kronos 集成**：周线 Kronos (pred_len=1) + 日线 Kronos 分数加权

**P3（最后考虑）**：
8. **模型集成**：Kronos + LightGBM + PointwiseStockTransformer 分数集成

### 3.2 优先级判断依据

ROI（收益/工作量）：动态仓位 > 后处理 > 温度校准 > 爆发信号 > 市场特征 > Tokenizer 微调 > 周频 Kronos > 模型集成

---

## 四、Kronos 架构深度分析

### 4.1 完整数据流

```
原始OHLCV (T, 6)
  → 逐列标准化 (x-mean)/std → clip(-5,5)
  → Tokenizer.encode()
    → embed: Linear(6→256)
    → Encoder: 3×TransformerBlock (d_model=256, 4 heads, RoPE, causal)
    → quant_embed: Linear(256→20)
    → BSQuantizer: 符号二值化 → 拆分为 s1(前10bit) + s2(后10bit)
    → 输出 s1_indices ∈ [0,1023], s2_indices ∈ [0,1023]
  → Predictor (Decoder-only Transformer, 8 layers, d_model=512, 8 heads)
    → HierarchicalEmbedding(s1, s2) + TemporalEmbedding
    → decode_s1(): 因果自注意力 → DualHead.proj_s1 → logits → 采样 s1
    → decode_s2(): DependencyAwareLayer(cross-attn) → DualHead.proj_s2 → 采样 s2
    → 自回归重复 pred_len=5 次
  → Tokenizer.decode()
    → indices_to_bits() → 重建 20-bit 量化向量
    → post_quant_embed: Linear(20→256)
    → Decoder: 3×TransformerBlock
    → head: Linear(256→6)
  → 反标准化 → 预测的未来 OHLCV (pred_len, 6)
  → score = (open[T+5] - open[T+1]) / open[T+1]
```

### 4.2 Tokenizer 的本质

Tokenizer 是 VQ-VAE 风格的神经压缩模型，**不是 K 线形态检测器**。

- 输入：6 维 OHLCV（open, high, low, close, volume, amount）
- 输出：每时间步 2 个离散 token（s1 ∈ [0,1023], s2 ∈ [0,1023]），共 20-bit 码字
- BSQ (Binary Spherical Quantization)：符号二值化 + Straight-Through Estimator + 球面归一化 + 熵正则化
- 解码器可近无损重建原始 OHLCV
- **它学的是"如何用 1024×1024 个离散码字覆盖所有可能的 K 线状态空间"**，而非识别人类可理解的形态

关键约束：`d_in=6` 硬编码在预训练权重中（config.json）。无法直接扩展输入维度而不从头预训练。

### 4.3 Predictor 与 Tokenizer 的关系

**紧密耦合，不可分割。** 推理时 Tokenizer 参与两次：

```
历史OHLCV → Tokenizer.encode() → tokens → Predictor自回归 → 新tokens → Tokenizer.decode() → 未来OHLCV
```

不存在可以单独拿出来用的"预测模块" — 只能拿最终的排序分数。

### 4.4 Kronos 不是排序模型

Kronos 的输出是未来 5 天的 6 维 OHLCV 序列。排序是外部强加的 — 在 KronosModel 中对预测结果做 `(open[T+5] - open[T+1]) / open[T+1]` 后排序。Kronos 本身不知道它在参加选股比赛，它只知道"预测未来 K 线"。这解释了为什么交叉熵损失（预测 token 准确率）和排序质量（RankIC）之间存在 mismatch。

### 4.5 微调时特征正确性

验证了全部 4 个环节，Kronos 微调和推理流水线**均只输入 6 维 OHLCV**，未混入行业分类、技术指标、估值因子等额外特征。时间特征（weekday, day, month）通过独立的 `stamp` 张量传递，不混入价格量数据。

---

## 五、股票筛选策略分析

### 5.1 当前方案

```
300只沪深300 → ScreenProcessor（剔除成交额后30%、跌破MA60、回撤>15%）→ 约70-80只 → Kronos预测 → Top-5
```

### 5.2 预筛选的问题

1. **Signal 剪除**：ScreenProcessor 的 MA60/回撤条件恰恰是 Kronos 可能给高分的场景（超跌反弹、均值回归）。在全球训练的 Kronos 分布中，跌破 MA60 且回撤 >15% 的股票是最典型的反转信号源。提前排除 = 砍掉了 Kronos 最高置信度的信号。

2. **排序池缩小**：从 80 只选 Top-5 vs 从 300 只选 Top-5。在单股票独立预测的框架下，池子大小不影响单只股票分数质量，但决定了候选集的期望真收益上限。增大候选集 3.75 倍，噪声期望值仅增加约 15%（$\sigma\sqrt{2\log N}$），但候选集扩大了 275%，收益远大于风险。

3. **行业中性化退化**：80 只股票中某些行业可能只剩 1-2 只，行业中位数估计极不稳定。300 只的行业中位数更可靠。

4. **Kronos 是单股票独立预测**：Kronos 对每只股票独立做自回归推理，股票 A 的分数与股票 B 不在池子里无关。不存在"让模型看到更多 A 股模式"的跨股票学习效应。

### 5.3 替代方案

```
300只沪深300 → Kronos预测 → 后筛选/后加权 → Top-5
```

后筛选可以在 Kronos 评分后用更灵活的加权方式替代硬阈值（如在最终分数中惩罚不合格股票而非直接剔除）。计算开销从 80 到 300 增加约 3.75 倍，在可接受范围内。

### 5.4 是否需要喂全 A 股（5000+ 只）

不需要。Kronos 对每只股票独立预测，喂更多股票不会帮助模型"学到"什么。且最终选股仍限制在沪深 300 内，全 A 股的额外计算开销（5000/300 ≈ 17 倍）没有对应收益。

---

## 六、周线数据分析

### 6.1 周线定义

每根周 K 线代表一周的交易（以周五为采样点）：
- open = 周一开盘价
- close = 周五收盘价
- high/low = 周内最高/最低
- volume/amount = 周内累计

周线是日线的 5:1 降采样，天然更平滑，滤除日间噪声。

### 6.2 与赛题的对齐问题

- 赛题：T+1（周一）开盘买 → T+5（下周五）开盘卖
- 周线预测的 open = 下周一开盘 ≈ T+1 开盘 ✓
- 周线预测的 close = 下周五收盘 ≠ T+5 开盘（差了周五的日内波动）

这个 mismatch 意味着周线预测对应的是"周一开盘买 → 周五收盘卖"，与赛题"周一开盘买 → 周五开盘卖"差一个交易日的日内波动。实际影响可能不大（开盘价与收盘价高度相关），但需注意。

### 6.3 节假日不完整周

Baostock 的周线按日历周聚合，不完整周（如国庆前后只有 3 个交易日）仍输出一根 K 线，但波动范围偏小、日历对齐偏移。如需精确对齐，应自己用日线聚合（按实际交易日数聚合，以周五为锚定点）。

### 6.4 验证路径

先直接用 Baostock 原始周线快速测试效果（Kronos pred_len=1 on weekly）→ 如果效果好，再手动用日线聚合精确对齐赛题。

---

## 七、K 线特征与 Kronos 的关系

### 7.1 信息论角度

技术指标（RSI、MACD、KDJ、ATR 等）是 OHLCV 的确定性函数。根据数据处理不等式：

$$I(\text{returns}; \text{RSI}) \leq I(\text{returns}; \text{OHLCV})$$

即技术指标不可能包含 Kronos 的输入（OHLCV）中不存在的信息。

### 7.2 统计学习角度（技术指标仍有价值）

**第一层 — 归纳偏置**：Kronos-small 只有 24.7M 参数，通过反向传播从零"发现"RSI 的计算方式需要消耗宝贵的参数容量。显式提供 RSI = 为模型节省容量用于更重要的模式学习。从 NTK（Neural Tangent Kernel）视角，神经网络的 NTK 核不一定包含"RSI 核"，显式提供相当于给特征空间加了一个有强先验的基函数。

**第二层 — 任务 gap**：Kronos 的训练目标（重建未来 OHLCV）和选股任务（排序未来收益）之间存在 mismatch。技术指标可以帮助下游模型学到"什么样的 Kronos 预测模式对应好的买入机会"。

**第三层（最强）— 横截面信息**：`CrossSectionalRankProcessor`、`MarketLevelProcessor`、`SectorLevelProcessor` 生成的横截面特征（行业排名、市场宽度、行业超额收益等）计算的是跨股票的相对值。由于 Kronos 对每只股票独立预测，**它完全无法获取这些横截面信息**。这是 Kronos 的盲区。

### 7.3 特征贡献排序预测

| 特征类别 | Kronos 能否学到 | 增量贡献 | 原因 |
|---------|----------------|---------|------|
| 横截面（行业排名、市场宽度） | 完全不能 | 最高 | Kronos 独立预测，无跨股票信息 |
| 量价结合（OBV, Amihud） | 隐式可能 | 中高 | 成交量和价格的交互，Kronos 编码不够显式 |
| 波动率结构（ATR, 波幅比） | 隐式可能 | 中 | 波动率在 Kronos 编码中是隐式的 |
| 动量类（RSI, MACD, KDJ） | 最可能学到 | 低 | 与 Kronos 预测的共线性最高 |

### 7.4 与 NLP 的类比

NLP 中 BERT embedding + 手工特征（TF-IDF, POS tags）通常只带来微小提升，但金融中情况不同：
- BERT 的预训练任务（MLM）与下游分类任务高度相关；Kronos 的预训练（OHLCV 重建）与选股排序的 gap 远大于此
- Kronos 无法获取跨截面信息，而技术指标中的横截面特征提供了互补信息
- A 股有独特的微观结构（涨跌停板、T+1、散户主导），Kronos"见过"的全球分布可能与 A 股动态存在偏差

---

## 八、数据源分析

### 8.1 当前已用数据

- Baostock 日线：12 列（open, high, low, close, volume, amount, 振幅, 涨跌额, 换手率, 涨跌幅 + 股票代码, 日期）
- 行业分类：resource/行业分类.csv（中证四级行业编码 sector_l1~l4）
- Qlib Alpha158 因子 + 自定义表达式特征 + 12 个 Processor（约 197 维）
- 均已使用后复权（adjustflag="1"）

### 8.2 Baostock 已有但未使用的数据（免费，零合规风险）

| 优先级 | 数据 | 接口 | 获取成本 |
|--------|------|------|---------|
| 最高 | 日频估值（peTTM, pbMRQ, psTTM, pcfNcfTTM） | `query_history_k_data_plus` 加 fields | 已在 API 返回中，只需修改 fields 参数 |
| 最高 | 是否 ST（isST） | 同上 | 同上 |
| 高 | 季频财务（ROE, 净利率, EPS, 毛利率） | `query_profit_data()` | 新 API 调用 |
| 高 | 成长能力（营收/利润同比增长） | `query_growth_data()` | 新 API 调用 |
| 高 | 业绩预告/快报 | `query_forecast_report()` / `query_express_report()` | 新 API 调用 |
| 中 | 偿债能力（资产负债率, 流动/速动比率） | `query_balance_data()` | 新 API 调用 |
| 中 | 现金流（经营活动现金流净额） | `query_cash_flow_data()` | 新 API 调用 |
| 中 | 货币供应量（M1/M2） | `query_money_supply_data_month()` | 新 API 调用 |
| 低 | 杜邦分析、存款/贷款利率等 | 各接口 | 与短期选股相关性弱 |

### 8.3 Tushare Pro 独特数据（需注册 + 积分）

| 数据 | 接口 | 积分门槛 | 独特价值 |
|------|------|---------|---------|
| 个股资金流向（大/中/小单分类） | `moneyflow()` | 2000 | 判断主力动向，与量价因子低相关 |
| 沪深港通资金流向 | `moneyflow_hsgt()` | 2000 | 外资偏好/北向资金因子 |
| 龙虎榜席位明细 | `top_list()` | 2000 | 席位买卖明细，短线爆发信号 |
| 综合财务指标 | `fina_indicator()` | 2000 | 比 Baostock 财务接口字段更全 |
| 同花顺概念板块 | `ths_index/member()` | 6000 | 主题投资，行业分类的补充 |
| 筹码分布 | `cyq_chips()` | 2000 | 持仓成本分布，支撑/压力位 |
| 每日涨跌停信息 | `limit_list()` | 基础 | 涨跌停板因子 |

### 8.4 超短期预测中情绪因子的价值评估

对于 4 天持有期的超短期预测：
- **资金流向（大单净买入）**：主力建仓信号，短期有一定有效性，沪深 300 大盘股上更可靠（排除散户噪声）
- **龙虎榜**：机构席位跟随效应，但沪深 300 很少上龙虎榜
- **北向资金**：外资持续买入的短期动量效应有一定有效性
- **新闻舆情/热度榜**：4 天窗口太短，市场可能未充分定价；沪深 300 几乎不受主题炒作影响
- **散户情绪**：在沪深 300 大盘股上基本是噪声或反向指标

---

## 九、模型设计空间

### 9.1 端到端 vs 两阶段

| 维度 | 两阶段（分数预测 → 组合优化） | 端到端（决策聚焦学习） |
|------|-------------------------------|----------------------|
| 实现难度 | 低（模块化，分步调优） | 高（需可微优化层，如 Sinkhorn/Perturbed Optimizer） |
| 训练稳定性 | 高 | 中低（梯度可能爆炸/消失） |
| 调参难度 | 低 | 高（多个松耦合超参） |
| 理论上限 | 中 | 高（消除预测-决策错配，模型知道哪些预测错误对组合影响大） |
| 实证表现 | Margin Loss 年化 16.23%, 夏普 0.75 (S&P500) | DFL 在 S&P100/DOW30 优于 baseline |
| 调试能力 | 强（每步可检查） | 弱（黑盒） |
| 比赛适配 | 当前阶段更务实 | B 阶段值得尝试 |

核心区别：端到端的价值在于"模型知道哪些预测错误对最终组合影响最大，从而战略性地分配学习能力"。在传统两阶段中，模型对所有预测错误一视同仁；在端到端中，模型会优先保证 Top 排名股票的预测质量。

### 9.2 排序损失函数对比

| 损失函数 | 类型 | 年化收益 | 夏普比率 | 计算复杂度 | 排序感知 | Top-K 区分 |
|---------|------|---------|---------|-----------|---------|-----------|
| MSE | Pointwise | 14.78% | 0.66 | O(N) | 否 | 否 |
| Margin Loss | Pairwise | **16.23%** | **0.75** | O(N²) | 是 | 部分 |
| ListNet | Listwise | 16.00% | 0.74 | O(N) | 是 | 部分 |
| BPR | Pairwise | 15.74% | 0.72 | O(N²) | 是 | 部分 |
| Hinge | Pairwise | 15.06% | 0.69 | O(N²) | 是 | 部分 |
| LambdaRank | Pairwise+ | — | — | O(N²) | 是 | 是(NDCG加权) |

数据来源：Kwiatkowski & Chudziak (2024), arXiv:2510.14156

### 9.3 专为此任务设计的损失函数：TopK-Weighted NDCG Loss

#### Part 1: 排序损失（修正的 ListNet Top-K）

$$\mathcal{L}_{rank} = -\sum_{i=1}^{K} \frac{K+1-i}{\sum_{j=1}^K (K+1-j)} \cdot \log\left( \frac{\exp(s_{\pi(i)}/\tau)}{\sum_{j=i}^{N} \exp(s_{\pi(j)}/\tau)} \right)$$

- $\pi$ 是按分数降序排列的索引
- 第 1 名权重最高，第 5 名权重最低（因为第 1 名对组合贡献最大）
- 排名靠后的股票也收到梯度（被推出前 K），不像 Sparsemax 那样梯度为零
- 在分数 s 上是凸的（softmax 函数的对数），保证 SGD 收敛

#### Part 2: 权重分配损失（修正的 NDCG）

$$\mathcal{L}_{weight} = 1 - \frac{\sum_{i=1}^{K} \max(0, r_{\pi(i)}/|r_{\pi(i)}|) \cdot w_{\pi(i)}}{\sum_{i=1}^{K} |\max(0, r_{\pi(i)})| / K}$$

- 只关心正收益股票（不做空）
- 惩罚"高权重配给低收益股票"

#### Part 3: 稀疏性正则化

$$\mathcal{L}_{sparse} = \lambda \cdot \sum_i (1 - \exp(-\alpha \cdot w_i)), \quad \alpha=100$$

- 对任何非零权重施加惩罚（近似 L0），鼓励主动空仓/少选

#### 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{rank} + \beta \cdot \mathcal{L}_{weight} + \gamma \cdot \mathcal{L}_{sparse}$$

- 超参数 $\beta, \gamma$ 可通过 Uncertainty Weighting（Kendall et al., 2018）自动学习：$\mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$

### 9.4 可微 Top-K 选择与权重约束

约束条件：股票数 ≤ 5，权重和 ≤ 1（剩余现金），可空仓。

#### 方案对比

| 方案 | 约束满足 | 梯度质量 | 实现难度 | 理论基础 |
|------|---------|---------|---------|---------|
| Gumbel-Softmax Top-K + STE | 良好（基数） | 中等（STE有偏） | 低 | Jang et al. 2016 |
| Gumbel-Sinkhorn (CardNN) | 优秀（基数+预算） | 良好 | 中等 | Wang et al. 2023, 夏普 1.1→2.0 |
| Perturbed Optimizer | 完美 | 差（高方差） | 高 | Berthet et al. NeurIPS 2020 |
| Sparsemax | 差（不支持基数约束） | 差（支撑集外零梯度） | 低 | Martins & Astudillo 2016 |
| Differentiable Knapsack DP | 完美（基数+预算） | 良好 | 高 | Vivier-Ardisson et al. 2026 |

#### 推荐方案

**Gumbel-Sinkhorn / CardNN** — 在基数约束投资组合优化中实证最有效（Wang et al. 2023, OpenReview）。通过 Sinkhorn 算子将离散 Top-K 松弛为双随机矩阵，Gumbel 噪声使采样可微。

**空仓实现**：加一个虚拟"现金资产"，其收益恒为 0，权重 $w_{cash} = 1 - \sum w_i$。模型可自然学到大比例配给现金 = 空仓。

**降级方案（实现更简单）**：Gumbel-Softmax Top-K 产生可微的 K-hot 掩码 → 选中的股票 softmax 归一化 → 乘以 budget（≤1.0）。梯度通过 STE 反向传播。

### 9.5 端到端选股+配权模型架构（概念）

```
输入: 每只股票 [K线序列(60天×6) + Alpha158(158维) + 基本面(PE/PB/ROE等)]

特征提取器: GRU/LSTM (时序) + MLP (截面因子) → concat → 股票embedding
  或: Transformer Encoder (300只股票做self-attention → 天然捕获截面互动)

评分头: MLP → 标量分数 (每只股票)
  损失: ListNet / Margin Loss

可微Top-K选择: Gumbel-Sinkhorn / Gumbel-Softmax → 选出K只股票

配权头: MLP → softmax → 权重 (和为1)
  或: 传统优化器 (MVO / min-variance)

组合损失: L_rank + λ1 * L_portfolio + λ2 * L_diversity
```

若用 Transformer Encoder 让股票之间做 self-attention，模型能学到"选了茅台就别选五粮液"的替代关系 — 这是纯 Pointwise 方法做不到的。

---

## 十、微调知识体系

### 10.1 Conditioning 技术

"Conditioning" 指在微调时给模型注入额外的条件信息，让它知道"现在是什么环境"。

| 方式 | 原理 | 计算开销 | 适用场景 |
|------|------|---------|---------|
| Prefix conditioning | 输入序列前加可学习的条件 token | 低 | 离散条件（如市场状态分类） |
| Cross-attention | 额外 cross-attn 层，条件作为 K/V | 中 | 连续条件（如波动率数值） |
| Feature modulation (FiLM) | 条件信息缩放/偏移隐藏状态 h' = γ(c)*h + β(c) | 低 | 轻量注入 |
| Token concatenation | 条件 token 拼入序列参与 self-attn | 低 | 最简单实现 |

对于 Kronos，最自然的是 **prefix conditioning**：在 OHLCV 编码后的 token 序列前拼接"市场状态 token"，其 embedding 由市场特征通过小 MLP 生成。

### 10.2 参数高效微调方法

| 方法 | 原理 | 可训参数占比 | 论文 |
|------|------|------------|------|
| Full Fine-tuning | 所有参数参与训练 | 100% | (当前方案) |
| LoRA | 在 Attention 权重上加低秩矩阵 A·B | ~1-5% | Hu et al. 2021 |
| Prefix Tuning | 学一组虚拟 token 拼在输入前 | <1% | Lester et al. 2021 |
| Adapter | 在 Transformer 层间插入瓶颈层 | ~3-8% | Houlsby et al. 2019 |
| QLoRA | LoRA + 4-bit 量化 | ~1-5% | Dettmers et al. 2023 |

---

## 十一、关键问题 Q&A 索引

以下汇总跨多轮对话中的关键问题与结论。

### Q1: Kronos 的 Tokenizer 只是在学 K 线形态吗？
不是。Tokenizer 是 VQ-VAE 风格的神经压缩模型，它学的是"如何用离散码字覆盖可能的 OHLCV 状态空间"，而非识别人类可理解的形态（如头肩顶、三只乌鸦）。

### Q2: Kronos 的 d_in=6 是否硬约束？
是。`d_in=6` 在预训练权重 config.json 中硬编码。微调时也只能输入 6 维 OHLCV。已验证当前流水线 4 个环节均严格遵守此约束。

### Q3: 预筛选（ScreenProcessor 在 Kronos 之前）是改善还是削弱？
理论上更差。ScreenProcessor 的 MA60/回撤条件可能排除了 Kronos 会给高反转信号的股票。应改为"300 只全送 Kronos → 后筛选/后加权"。

### Q4: 周线预测与赛题是否对齐？
不完全对齐。周线 close 是周五收盘，赛题用 T+5 开盘价。差了一天的日内波动。但可以先快速测试效果，效果好再手动聚合日线精确对齐。

### Q5: 端到端 vs 两阶段怎么选？
两阶段在当前阶段更务实。端到端上限更高但实现难度大，可以在 B 阶段（8 月初，时间充裕）尝试。

### Q6: 可微权重约束（≤5 只，权重和 ≤1）怎么实现？
推荐 Gumbel-Sinkhorn/CardNN。空仓通过虚拟"现金资产"实现。

### Q7: 损失函数能不能针对此任务专门设计？
可以。设计了 TopK-Weighted NDCG Loss，包含 L_rank（位置加权排序损失）+ L_weight（权重-收益对齐）+ L_sparse（稀疏性正则化）。超参数可通过 Uncertainty Weighting 自动学习。

### Q8: 技术指标 + Kronos 是噪声还是有用信息？
有用。特别是横截面特征（Kronos 完全不可得）和量价结合特征。动量类指标与 Kronos 共线性最高，价值最低。但精确归因需要在工程上做成组消融实验。

### Q9: Tushare 情绪因子对超短期有用吗？
对沪深 300 的 4 天持有期，情绪因子的作用有限。资金流向（主力动向）有一定价值，但新闻舆情/热度榜/散户情绪在此时的沪深 300 上基本是噪声。建议优先做 Baostock 免费估值因子。

### Q10: 微调时如何注入市场环境信息？
最自然的是 prefix conditioning（在 token 序列前加市场状态 token），也可通过后处理（市场状态 → 调整仓位/分数权重）。

### Q11: Kronos 分数能否作为因子给 LightGBM？
可以。KronosModel 返回的就是标量分数，直接作为 LightGBM 的一个特征列。这是最务实的模型融合路径。但需注意 Kronos 推理速度（300 只 × 自回归生成）。

### Q12: 多损失函数权重如何确定？
可以用 Uncertainty Weighting（Kendall et al., 2018）：每个损失项配一个可学习参数 σ_i，L_total = Σ(1/(2σ_i²) * L_i + log σ_i)。log σ_i 项防止 σ_i 趋于无穷。不需要手动调。

---

## 十二、参考文献

1. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
2. George, T. J., & Hwang, C. Y. (2004). The 52-Week High and Momentum Investing. *Journal of Finance*, 59(5), 2145-2176.
3. Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*, 72(4), 1611-1644.
4. Kwiatkowski & Chudziak (2024). On Evaluating Loss Functions for Stock Ranking. arXiv:2510.14156.
5. Linger et al. (2024). Enhancing Long-Short Portfolios: A Refined Approach Using Learn-to-Rank Algorithms. *Journal of Financial Data Science*.
6. Luo, S., et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. *NeurIPS 2025 / AAAI 2026*. arXiv:2508.02739.
7. Jang, Gu, Poole (2016). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017.
8. Mena et al. (2018). Learning Latent Permutations with Gumbel-Sinkhorn. ICLR 2018.
9. Berthet et al. (2020). Learning with Differentiable Perturbed Optimizers. NeurIPS 2020.
10. Wang et al. (2023). On Solving Cardinality Constrained Combinatorial Optimization with One-shot Learning. OpenReview.
11. Vivier-Ardisson et al. (2026). Differentiable Knapsack and Top-k Operators via Dynamic Programming. arXiv:2601.21775.
12. Lee, Tae, Lee (2024). Anatomy of Machines for Markowitz: Decision-Focused Learning for Mean-Variance Portfolio Optimization. arXiv:2409.09684.
13. Kendall, Gal, Cipolla (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. CVPR 2018.
14. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357-384.
15. Anis & Kwon (2025). End-to-End, Decision-Based, Cardinality-Constrained Portfolio Optimization. *European Journal of Operational Research*.
