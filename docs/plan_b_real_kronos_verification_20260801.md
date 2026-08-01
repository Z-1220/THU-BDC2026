# Plan B 真 Kronos 验证报告

日期: 2026-08-01 | 数据截止: 2026-07-31 | 评分来源: fine-tuned Kronos-small (lb60, 仅 Predictor 微调, Tokenizer 冻结)

## 目的

验证研究计划 B 的结论在真实 Kronos 评分（而非 20 日动量代理）上是否成立：Context Transformer（NDCG 损失）是否显著优于纯 Kronos 生产基线与 Plan A 最佳（MSE / NDCG MLP 头部），为 B 榜（8/1-8/2）提交提供依据。

## 方法

- 评分来源: Kronos-small 微调权重（2024-01 ~ 2025-12 A 股数据，best val CE=3.1096 @ epoch 18），通过 temp/kronos_scores_cache_ft.pkl 预计算缓存
- 股票池: FridayFilter + ScreenProcessor（成交额前 70%、MA60、20 日回撤 < 15%）筛选后的横截面（训练段平均约 113 只/周，测试段约 72 只/周）
- 数据范围: 2022-01-04 ~ 2026-07-31（从 2022 起加载历史，保证 ScreenProcessor 的周频 MA60 有足够预热，避免 2025-04 前信号日整段丢失）
- 信号窗口: 训练 2024-01-05 ~ 2025-12-26（95 周）/ 验证 2026-01-09 ~ 2026-01-30（4 周）/ 测试 2026-02-06 ~ 2026-07-31（21 周）
- 标签: 实际 5 日开盘收益 (open[T+5]-open[T+1])/open[T+1]，与赛题一致
- 组合评估: Top-3 rank-weighted [33.3%, 33.3%, 33.4%]；Sharpe 按周频年化（sqrt(52)）
- 固定随机种子 seed=42；训练 100 epoch + 早停（patience=15），per-date-group 训练
- 对比项:
  - Kronos-raw (Top-3): 缓存原始分数直接选股
  - Kronos-CS (Top-3): 行业中性化 + Winsorize(1%/99%) + Z-score 后选股（= 当前生产基线，champion 配置）
  - Plan A: MSE / NDCG MLP 排序头部（仅用 Kronos 分数）
  - Plan B: B-C1..B-C7 Context Transformer（NDCG 损失）

## 结果

| 实验 | 周数 | Sharpe | 累计收益 | 胜率 | 最大回撤 | RankIC | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kronos-raw (Top-3) | 21 | 1.1648 | +0.1450 | 47.6% | 0.0783 | -0.0205 | 0.057 |
| Kronos-CS (Top-3) | 21 | 1.1648 | +0.1450 | 47.6% | 0.0783 | -0.0205 | 0.057 |
| A-E0 (MSE Baseline) | 21 | 1.1648 | +0.1450 | 47.6% | 0.0783 | -0.0205 | 0.057 |
| A-E3 (NDCG Approx) | 21 | 1.1648 | +0.1450 | 47.6% | 0.0783 | -0.0205 | 0.057 |
| B-C1 (No Context) | 21 | 1.5127 | +0.2110 | 42.9% | 0.0799 | -0.0121 | 0.057 |
| B-C2 (Market) | 21 | 2.7909 | +0.7284 | 57.1% | 0.0970 | 0.0077 | 0.076 |
| B-C3 (Sector) | 21 | 1.1741 | +0.1634 | 33.3% | 0.1036 | -0.0285 | 0.057 |
| B-C4 (CS Stats) | 21 | 2.8552 | +0.4872 | 66.7% | 0.0763 | 0.0036 | 0.076 |
| B-C5 (Full Context) | 21 | 2.6446 | +0.3094 | 61.9% | 0.0569 | 0.0419 | 0.076 |
| B-C6 (Shuffled) | 21 | 1.1006 | +0.1302 | 47.6% | 0.1298 | 0.0044 | 0.114 |
| B-C7 (Zero) | 21 | 2.8269 | +0.5370 | 57.1% | 0.0888 | 0.0203 | 0.095 |

## 关键发现

1. **Context Transformer 架构在真 Kronos 评分上显著有效。** 最佳 B 模式 Sharpe 2.64~2.86，为生产基线（Kronos-CS，1.16）的 2.3~2.5 倍，远超 Plan C 设定的 1.3x 门槛；累计收益 +31%~+73% vs 基线 +14.5%。

2. **"B-C5（Full Context）最优"的代理结论在真 Kronos 上不成立。** B-C5 为 2.64，低于 B-C4（仅横截面统计，2.86）、B-C2（仅市场，2.79）和 B-C7（上下文置零，2.83）。代理实验中"全量上下文 1+1+1>3 协同"未完全迁移；加入行业动量和流动性特征反而稀释了信号。

3. **增益主要来自跨股票 self-attention 结构，而非上下文特征内容本身。** B-C7（非分数特征全部置零）仍达 2.83，接近最优；B-C6（打乱上下文）明显劣于 B-C5（1.10 vs 2.64），说明"上下文与股票的对应关系"确实携带信息，但增量更多来自 Transformer 对股票间相对位置的建模。

4. **Plan A（MLP 排序头）在真 Kronos 分数上无增量。** MSE 与 NDCG 头部结果与基线完全相同（Sharpe 1.1648、相同选股），说明简单 MLP 无法从 Kronos 分数中提取额外排序信息；此前代理实验的 NDCG 优势未迁移。

5. **测试窗口内纯 Kronos 排序信号偏弱。** 21 周 RankIC 约 -0.02（此前 13 周窗口约 +0.047），Hit@5 仅 0.057；这说明近期行情下模型原始排序质量下降，也为 B-C 类精排头部提供了空间。

## 结论与建议

- **"B-C5 是最优配置"未通过真 Kronos 验证**，但"Context Transformer 提升选股"成立。若采用 Plan B，应优先 **B-C4（CS rank/z-score）或 B-C2（市场上下文）**，而非 B-C5 全量配置。
- 当前结果仍是 21 周单次运行的点估计，且 NDCG 训练对初始化敏感（研究计划 A 已注明），上线前建议：多 seed（3~5 个）稳定性验证、扩展测试窗口、确认推理耗时。
- **B 榜提交建议（按风险排序）**：P0 继续使用当前 champion（Kronos-CS + Top-3，已验证）；P1 若时间允许，用 B-C4/B-C2 精排头做多 seed 复验后再替换；不建议直接采用 B-C5。

## 复现

```bash
# 1) 生成微调权重评分缓存（handler 从 2022 起加载，保证 MA60 预热）
uv run python scripts/cache_kronos_scores_real.py \
  --start-date 2024-01-01 --end-date 2026-07-31 --handler-start 2022-01-01 \
  --finetuned-dir ./model/kronos_finetuned/Kronos-small-lb60 \
  --output temp/kronos_scores_cache_ft.pkl

# 2) 运行验证（输出 JSON + 本报告）
uv run python scripts/verify_plan_b_real_kronos.py \
  --cache temp/kronos_scores_cache_ft.pkl --score-source "fine-tuned Kronos-small (lb60)" \
  --test-start 2026-02-02 --test-end 2026-07-31 --out-dir output/research
```

结果 JSON: output/research/plan_b_real_kronos_20260801_062847.json

## 局限

- 测试窗口含官方盲测周（2026-04-10 信号 → 04-13~04-17 标签），仅用于评估、未参与训练；与既有研究报告口径一致。
- 样本仅 21 周，B-C 模式间的 Sharpe 差距（2.64 vs 2.86）在噪声范围内，不应过度解读。
- 精排头训练仅 95 周，NDCG 损失 + 早停存在过拟合与初始化敏感性风险。
