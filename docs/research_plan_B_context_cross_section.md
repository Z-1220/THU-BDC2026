# 研究计划 B：上下文、横截面与配权研究

## B0. 前置条件

**本计划的损失函数假设来自研究计划 A 的结论。** 计划 A 先回答"损失怎么写才符合 Kronos 的架构归纳偏置"，计划 B 再在此 loss 上做上下文 Transformer、横截面建模和配权层。

## B1. 研究目的

验证三个核心假设：

1. **H1：外部上下文在特定行情下提供增益。** 市场宽度、指数状态、行业相对强弱等 Kronos 天然看不到的信息是否有预测价值。
2. **H2：横截面 Transformer 优于单票独立预测。** "股票之间相对位置"比"单股票绝对预测值"更关键。
3. **H3：配权层比复杂预测头更容易带来真实收益。** rank-weighted / Top3-only 已有效，配权优化可能比加大预测模型更有用。

## B2. 研究假设详述

### H1：外部上下文有用，但只在特定行情下有效

Kronos 原始输入是单股票 OHLCVA（6 维），天然看不到：
- 市场宽度（breadth）
- 行业轮动（sector rotation）
- 全市场情绪（dispersion）

提供这些上下文的前提是：**它们提供的增量信息 ≥ 引入的噪声**。

### H2：横截面 Transformer 可能比单票独立预测更适合

赛题不是预测单票，而是从同一周 300 只股票里挑 Top-K。股票间的相对位置关系比单股票绝对预测值更关键。Transformer 的 self-attention 让股票之间可以隐式建模"替代关系"。

### H3：配权层比复杂预测头更容易带来真实收益

实验已证实：动态仓位规则（E3）全失败，置信度仓位（E4）全失败，但 rank-weighted / Top3-only 有效。说明**不改排序、只改权重分配**的方向比"改进排序质量"更容易产生可测量的收益。

## B3. 实验模块

---

### B3.1 上下文模块：Context Transformer

#### 输入结构

每个调仓日输入：
- Kronos predictor 的 score 或 hidden state（per-stock）
- 市场环境特征（hs300_20d_return, hs300_20d_vol, market_breadth, dispersion）
- 行业特征（sector_l1 one-hot or embedding）
- 流动性特征（amount_rank, turnover）
- 横截面统计特征（CS z-score rank, sector relative strength）

#### 模型结构

小 Transformer 或 Set-Aware Transformer：
- 一个 `[CLS_env]` context token（汇总全局信息）
- 多个 stock token（per-stock embedding）
- 可选的 sector token（行业级汇总）
- 输出：修正后的 per-stock score

#### 对照实验

| Exp | 描述 |
|-----|------|
| B-C1 | 无上下文基线（仅 Kronos score） |
| B-C2 | 仅加市场上下文 |
| B-C3 | 仅加行业上下文 |
| B-C4 | 仅加横截面统计上下文 |
| B-C5 | 全量上下文 |
| B-C6 | 上下文打乱（shuffle）对照 |
| B-C7 | 上下文置零对照 |

#### 目标指标
- RankIC 提升
- Hit@5 / Recall@10 改善
- 真实组合收益改善
- 上下文类型贡献分解

---

### B3.2 横截面模块：Cross-Sectional Transformer

#### 输入

同一调仓日的所有候选股票作为一个 set（非序列）。**不对股票强加顺序**，使用 permutation-invariant 或 weakly-ordered 结构。

#### 模型结构

- 每只股票 = 一个 token
- Self-attention across stocks（让股票间隐式交互）
- 无 position encoding（或使用 score-based ordering 作为弱位置信号）
- 输出：横截面修正后的 per-stock score

#### 关键设计原则

股票之间没有天然顺序。Trivially sorting by code or market cap introduces artificial bias. 应：
- 使用 permutation-equivariant 结构
- 或使用 score rank 作为弱有序信号

#### 对照实验

| Exp | 描述 |
|-----|------|
| B-X1 | 单票独立打分（baseline） |
| B-X2 | 横截面 Transformer |
| B-X3 | 随机打乱股票顺序 |
| B-X4 | 固定顺序 vs 随机顺序对比 |
| B-X5 | 加入行业 token |
| B-X6 | 不加行业 token |

#### 目标

验证"同周股票间的相对关系"能否改善 Top-K 命中和 NDCG。

---

### B3.3 配权模块：Rank-Weighted → CardNN

#### 第一层：Rank-Weighted（已验证有效，直接复用）

- Top3-only [33, 33, 34]
- 或略微倾斜的强权重方案

#### 第二层：可微约束配权（条件执行）

只在 rank-weighted 稳定有效后才尝试：

| 方法 | 原理 | 复杂度 |
|------|------|--------|
| Gumbel-Softmax Top-K | 可微 Top-K 选择 + STE | 低 |
| Sparsemax | 稀疏概率分布 | 低 |
| Differentiable Knapsack DP | 基数 + 预算联合约束 | 高 |
| CardNN (Gumbel-Sinkhorn) | OT 框架，精确基数约束 | 高 |

#### 对照实验

| Exp | 描述 |
|-----|------|
| B-W1 | 等权 Top-5（E0 baseline） |
| B-W2 | Top3-only（当前 champion） |
| B-W3 | Rank-weighted [35,25,18,12,10] |
| B-W4 | Rank-weighted + CardNN |
| B-W5 | CardNN-only |

#### 目标

判断 CardNN 是"更复杂的权重分配器"还是能在此基础上进一步提高收益。

---

## B4. Phase A 完整实验顺序

所有阶段都使用 A 计划确定的 loss 函数。

```
Stage A1: 无上下文 + 无横截面 + 无 CardNN（最小基线）
Stage A2: + 上下文（环境信息）
Stage A3: + 横截面 Transformer（股票间交互）
Stage A4: + rank-weighted（改配权）
Stage A5: rank-weighted + 上下文
Stage A6: rank-weighted + 横截面
Stage A7: rank-weighted + 上下文 + 横截面
Stage A8: + CardNN（仅当前面全部有增益时）
```

## B5. 对照体系

### 上下文实验
- Base vs Base + Context
- Base + Context vs Base + Context (shuffle)
- 单个 context 类型互相对照

### 横截面实验
- Base vs Base + Cross-Sectional Transformer
- 顺序固定 vs 顺序打乱
- 有行业 token vs 无行业 token

### 配权实验
- 等权 vs Top3-only vs rank-weighted
- rank-weighted vs rank-weighted + CardNN
- CardNN vs simple softmax allocation

## B6. Gate 规则

| Stage | Gate 条件 | 不通过则 |
|-------|----------|---------|
| A2 | 任一上下文的 RankIC 或 Cum Ret 提升 | 跳过 A2，进入 A3 |
| A3 | 横截面 Transformer > 单票独立 | 跳过 A3，进入 A4 |
| A4 | rank-weighted ≥ 等权（预期通过） | 重新检查权重分配 |
| A5 | 上下文改善权重结构 | 回退到 A4 |
| A6 | 横截面改善权重结构 | 回退到 A4 |
| A7 | 两者互补 | 取最优单方案 |
| A8 | 前面全部有增益 | 跳过 CardNN |

## B7. 结果解释原则

### 1. 数据量是否足够
样本按周计算（~100 周训练 + 10-20 周测试）。若某结构仅少数周有效，可能是偶然。

### 2. 分布是否变化
训练周和测试周是否处于同一市场状态。市场风格切换时表现不同是正常的。

### 3. 输入是否真的表达了理论变量
- 上下文是否真的是可预测的信息
- 横截面 token 是否真的保留了同周相对关系
- CardNN 约束是否真的和比赛约束一致

### 4. 结构是否过拟合
若复杂模块只在一个窗口上表现好 → 优先怀疑过拟合，而非理论成立。

## B8. 两计划关系

```
计划 A（损失函数结构对齐）
    │
    ├── 确定下游 loss 设计原则
    │
    ▼
计划 B（上下文 / 横截面 / 配权）
    │
    ├── B3.1 上下文 Transformer
    ├── B3.2 横截面 Transformer
    ├── B3.3 配权模块
    │
    ▼
Phase C/D（CardNN / 端到端）
```

- **A 是前提**：先定训练目标
- **B 是应用**：再做结构改动
- **A 的结论影响 B 的训练方式**
- **B 的结果反向验证 A 的 loss 设计**
