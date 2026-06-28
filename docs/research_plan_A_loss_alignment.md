# 研究计划 A：损失函数结构对齐研究

## A1. 研究目的

**不是"找一个更强的排序 loss"，而是回答：对于 Kronos 这种 coarse-to-fine、tokenizer + autoregressive predictor 的架构，什么样的下游损失函数才是结构一致的？**

Kronos 的核心设计哲学：
1. Tokenizer 把每个 K-line 映射成 coarse/fine 两个 subtokens；
2. 收益是把大词表预测拆成两个较小词表的顺序预测，降低计算与参数复杂度；
3. 训练目标不是单纯拟合一个数值，而是让目标和架构分解方式一致；
4. "先粗后细"的层次结构应反映到下游 ranking/decision loss 里。

## A2. 核心研究问题

### Q1：常规排序损失是否足够？

| 损失 | 类型 | 说明 |
|------|------|------|
| Pairwise Margin Loss | Pairwise | 标准成对排序 |
| ListNet / ListMLE / NDCG surrogate | Listwise | 列表级排序 |
| RankIC loss | 轻量 | 最大化 Spearman 相关 |

目的：不考虑 Kronos 结构，只用标准排序监督，建立基准线。

### Q2：结构对齐的排序损失是否更适合？

#### 方案 A：Coarse-to-Fine Ranking

拆成两层：
1. **Coarse layer**：判断股票属于"明显强 / 中等 / 弱"哪个候选层
2. **Fine layer**：仅在 coarse 候选层内部做精排

与 Kronos tokenizer coarse/fine 两级设计同构。

#### 方案 B：候选层 + 组内排序层

1. **候选召回**：是否进入 Top-M 候选
2. **组内排序**：在 Top-M 内部排序 Top-K

更贴近已证实的"粗粒度筛选器 + 头部权重分配"结构。

#### 方案 C：结构一致性正则

给排序 head 增加约束：
- 预测分数不脱离 Kronos 原始 score 太远
- Coarse 分支保留原始 score 的整体方向
- Fine 分支只负责局部修正

目的：防止下游损失冲掉 Kronos 的层次表示。

## A3. 实验矩阵

### A3.1 基线组

| Exp | 描述 |
|-----|------|
| A-E0 | CE 基线：冻结 tokenizer，微调 predictor/head，沿用现有 Kronos score |
| A-E1 | Pairwise baseline：只换成 pairwise ranking loss |
| A-E2 | Listwise baseline：只换成 listwise loss |
| A-E3 | NDCG surrogate baseline：直接优化排序质量 |

### A3.2 结构对齐组

| Exp | 描述 |
|-----|------|
| A-E4 | Coarse-to-fine ranking |
| A-E5 | 候选层 + 组内排序层 |
| A-E6 | 结构一致性正则 + 排序损失 |
| A-E7 | 结构对齐 + rank-weighted allocation proxy |

### A3.3 负对照组

| Exp | 描述 |
|-----|------|
| A-E8 | 随机打乱标签顺序 |
| A-E9 | 随机打乱股票顺序 |
| A-E10 | 只保留 coarse，不做 fine |
| A-E11 | 只保留 fine，不做 coarse |

**负对照的判定价值**：如果结构对齐损失真的有效，打乱顺序、去掉 coarse/fine 任一层应导致性能明显下降。若无下降，说明结构设计未被模型真正利用。

## A4. 评价指标

### 交易类
- Cum Return, Sharpe, Sortino, MaxDD, Win Rate, G/L Ratio

### 排序类
- RankIC, IC>0%, Hit@5, Recall@10, NDCG@5

### 结构类
- Top1/Top3/Top5 平均收益
- Top3→Top5 衰减梯度
- Top-K 贡献占比
- 分数分布集中度

### 诊断类
- 常规 loss vs 结构化 loss 在不同窗口的对比较
- 强势市 / 弱势市差异
- 训练更好但收益更差的错配检测

## A5. Gate 规则

### 每个实验
- Cum Return 提升 > 1pp，或 RankIC 提升 > 10% → KEEP
- 否则 DROP

### 结构对齐整体判定
- 若 A-E4~A-E7 中 ≥ 2 个优于 A-E0 → 结构对齐有效
- 若全部劣于 A-E0 但优于 A-E8~A-E11 → 结构方向正确但实现需调整
- 若 A-E10/A-E11（单层）优于完整结构 → Kronos 层次偏置不适合此任务

## A6. 解释规则

### 若结构对齐 loss 优于常规 loss
1. Kronos 的层次归纳偏置可迁移到下游
2. 任务本质是"粗召回 + 细排序"的复合任务
3. 后续 Phase B/C/D 应继续沿结构化方向

### 若常规 loss 反而更好

先排查：
1. 样本量是否足够支撑 coarse/fine 学习（~100 周）
2. 训练窗口是否太短
3. 股票分布是否过于稀疏或偏态
4. Top-K 标签是否噪声过大
5. Coarse/fine 切分是否适合金融数据

全部排查后才能说：Kronos 的层次结构对下游排序并非最优归纳偏置。
