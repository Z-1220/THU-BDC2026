# 研究计划 C：Context Transformer 生产集成与 B 榜准备

## C0. 前置条件

- ✅ Plan A：NDCG 损失已确认为最佳结构化损失
- ✅ Plan B：B-C5 (Context Transformer + Full Context) 为代理评分冠军 (Sharpe 5.47)
- ⏳ Kronos 真实嵌入上的 B-C5 尚未验证

## C1. 研究目的

将 Plan B 产出的 Context Transformer（B-C5）**从研究原型提升为生产可用模型**，用于 8/1-8/2 B 榜最终提交。

## C2. 核心任务

### 阶段 1：真 Kronos 嵌入验证

**目标**：确认 B-C5 在真实 Kronos 评分上是否保持优势。

| 步骤 | 操作 | 验证标准 |
|------|------|---------|
| C1.1 | 预计算 Kronos 评分缓存（训练+验证+测试区间全部信号日） | 缓存文件生成成功 |
| C1.2 | 用真 Kronos 评分复现 Plan A 关键实验 (A-E0 MSE, A-E3 NDCG) | RankIC 方向一致 |
| C1.3 | 用真 Kronos 评分复现 Plan B B-C5 | Sharpe > 纯 Kronos 基线 |

**风险**：代理评分（动量）与 Kronos 评分的分布特征不同，B-C5 增益可能不 transfer。

**缓解**：若 B-C5 不显著，回退到 Kronos 纯排序 + NDCG 头部（Plan A 最优）。最坏情况：保持当前 Kronos + CS Z-score + Top-3。

### 阶段 2：Context Transformer 生产化

**目标**：将 B-C5 集成到 `commit.py` 推理流程。

| 步骤 | 操作 | 验证标准 |
|------|------|---------|
| C2.1 | 训练最终 Context Transformer 头部（全量训练数据，固定 seed） | 训练收敛，保存权重 |
| C2.2 | 修改 `commit.py` 支持 `KronosContextModel` 推理路径 | `sh test.sh` 成功生成 result.csv |
| C2.3 | 验证推理时间 ≤ 5 分钟（含 Kronos + Context Transformer） | Docker 环境计时 |
| C2.4 | 将训练好的 Context Transformer 权重加入 Docker 镜像 | 镜像大小 ≤ 10GB |

**关键约束**：
- 推理时延：Kronos 推理 ~2-3 分钟，Context Transformer forward ~毫秒级，总时间仍可控
- 镜像大小：Context Transformer 权重 ~50KB，可忽略
- 可复现性：固定 seed=42，保存训练脚本 + 配置

### 阶段 3：Context 特征生产化

**目标**：确保 `commit.py` 中能高效计算 B-C5 所需的上下文特征。

B-C5 需要 6 维股票特征 + 5 维市场特征：

| 特征 | 来源 | 计算复杂度 |
|------|------|-----------|
| Kronos 分数 | `KronosModel.predict()` | 已有 |
| 行业动量 (sector_mom_5) | `stock_data.csv` 聚合 | O(N) |
| CS 排名 (cs_rank) | 截面排序 | O(N log N) |
| CS Z-score (cs_zscore) | 截面标准化 | O(N) |
| 对数成交额 (amount_log) | 60日均值 | O(N) |
| 对数成交量 (turnover_log) | 60日均值 | O(N) |
| 市场动量/波动/广度/离散度 | 全市场聚合 | O(N) |

全部特征可在秒级内从 `stock_data.csv` 计算，不影响 5 分钟推理限制。

### 阶段 4：对照实验与消融

**目标**：确认每个组件在真实 Kronos 上的贡献。

| 实验 | 配置 | 对比基线 |
|------|------|---------|
| C-E1 | Kronos 纯分数 (baseline) | — |
| C-E2 | Kronos + NDCG MLP | vs C-E1 |
| C-E3 | Kronos + Context Transformer (score only) | vs C-E2 |
| C-E4 | Kronos + Context Transformer + Full Context (**B-C5**) | vs C-E3 |
| C-E5 | Kronos + Context Transformer + Full Context + CardNN | vs C-E4 |

**Gate 规则**：
- C-E2 > C-E1 → NDCG 头部有价值
- C-E4 > C-E2 → Context Transformer 架构有价值
- C-E4 > C-E3 → 上下文特征有价值
- C-E5 > C-E4 → 可微分配权有价值（仅在前面全部通过时执行）

## C3. 时间线

| 日期 | 里程碑 | 产出 |
|------|--------|------|
| **7/1 – 7/5** | 阶段 1: 真 Kronos 缓存 + 验证 | 缓存文件 + 对照结果 |
| **7/6 – 7/12** | 阶段 2+3: 生产化 Context Transformer | 训练好的权重 + 修改后的 commit.py |
| **7/13 – 7/18** | 阶段 4: 消融实验 | 各组件贡献量化 |
| **7/18** | 模型/数据报备截止 | 报备邮件（Kronos 开源链接 + md5） |
| **7/19 – 7/25** | Docker 打包 + 完整验证 | 可提交的 .tar 文件 |
| **7/26 – 7/31** | 最终测试 + 缓冲 | 确认 5 分钟推理 / 10GB 限制 |
| **8/1 – 8/2** | 🔴 **B 榜提交** | result.csv + Docker 镜像 |

## C4. 成功标准

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| B-C5 在真 Kronos 上的 Sharpe | > 纯 Kronos 基线的 1.3x | `run_all_model.py` 评测 |
| 推理时间（Docker） | ≤ 5 分钟 | `time docker compose up` |
| Docker 镜像大小 | ≤ 10GB | `docker image ls` |
| 可复现性 | seed=42 重跑一致 | 两次独立运行结果一致 |
| 超越基准程序 | ✅ 必须 | 主办方评分 |

## C5. 回退计划

若阶段 1 验证失败（B-C5 在真 Kronos 上无增益），执行以下回退：

| 优先级 | 方案 | 说明 |
|--------|------|------|
| P0 | Kronos + NDCG MLP 头部 (C-E2) | Plan A 最优，结构简单 |
| P1 | Kronos + CS Z-score + Top-3 | 当前冠军配置，A1/A2/A3 已验证 |
| P2 | Kronos + Context Transformer (score only, C-E3) | 去掉上下文，仅用 Transformer 架构 |

任一方案须在 7/25 前完成 Docker 打包验证。

## C6. 与 Plan A/B 的关系

```
Plan A (损失函数) ──→ 确定 NDCG 为最佳损失
        │
Plan B (上下文/横截面) ──→ 确定 B-C5 为最佳架构
        │
Plan C (本计划) ──→ 将 A+B 的结论集成到生产环境
        │
        ├── 真 Kronos 验证（消除代理评分偏差）
        ├── 生产化（训练/推理/Docker）
        └── 消融确认（每个组件的真实贡献）
                │
                ▼
        B 榜提交 (8/1-8/2)
```
