# THU-BDC2026 竞赛改进方案深度分析

## 赛题重述与核心挑战

**目标**：每个提交窗口，基于沪深300历史数据，预测未来一周（T+1开盘→T+5开盘）收益最大的≤5只股票组合，权重和≤1。

**核心矛盾**：这是一个**极度短期（4个交易日）、小组合（≤5只）、允许空仓**的选股问题。传统量化多因子选股面向月度以上持仓、分散化组合（50-200只），直接套用会出现方法论错配。需要的不是"哪只股票好"，而是"哪几只股票未来4天最可能大涨"——这是**短期爆发识别 + 极端收益预测**问题。

---

## 一、横截面后处理（Cross-Sectional Post-Processing）

### 1.1 问题诊断

当前 KronosModel 的 `_predict_signal_date()` 对每只股票独立预测 OHLCV 序列，然后计算预测收益 `score = (open[T+5] - open[T+1]) / open[T+1]`。这等价于一个未经校准的原始 alpha 信号。问题在于：

- **系统性偏差**：Kronos 预测的收益可能存在系统性正偏或负偏（例如对所有股票都倾向预测正收益）
- **行业聚集**：同一行业的股票分数可能整体偏高，导致选股集中在某个行业
- **时间不稳定**：Alpha 的 scale 随时间变化（高波动期分数范围大，低波动期小）

### 1.2 方案：分层横截面标准化

对每个预测日的所有股票分数做三步后处理：

**Step 1: 去极端值（Winsorize）**
```
score_clipped = clip(score, q01, q99)  # 1%/99% 分位数截断
```

**Step 2: 行业中性化（Sector Neutralization）**
```
score_neutral = score - median(score | sector)  # 行业内去中心
```
这已在 KronosModel 中实现（`_predict_signal_date` 第 334-343 行），但做的是简单的 sector median 减法。更严谨的做法是用线性回归：
```
score = α + β₁·sector₁ + β₂·sector₂ + ... + ε
residual = ε  # 这就是行业中性化后的信号
```
当前实现已够用，可以先保持。

**Step 3: 横截面 Z-Score**
```
score_final = (score_neutral - mean(score_neutral)) / std(score_neutral)
```
将分数转为横截面 Z-score，均值 0，标准差 1。这消除了市场整体偏置，使分数在不同周之间可比。

### 1.3 理论依据

Grinold & Kahn (2000) *Active Portfolio Management* 中的核心原则：alpha 信号应该标准化为"预期残差收益的 Z-score"。这确保了：
- 信息系数（IC）的估计不受 alpha scale 影响
- 跨时间的信号强度可比
- 组合构建时的权重分配更合理

### 1.4 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 极低（5行代码） |
| 计算开销 | 几乎为零 |
| 预期收益提升 | 中等（减少选股集中度偏差） |
| 风险 | 无（纯后处理，不改变排序） |

### 1.5 进一步优化：Cross-Sectional Rank

某些情况下，Z-score 受到尾部极端值影响。更稳健的替代方案是直接用分位数 rank：
```
score_final = rank(score) / N  # 0到1之间的百分位
```
这等价于用 Spearman 而非 Pearson 视角看信号。对于≤5只的选股，rank 和 z-score 的 top-5 基本一致，但 rank 对异常值更稳健。

---

## 二、温度与采样参数校准

### 2.1 问题诊断

KronosPredictor 有三个关键参数影响预测质量和多样性：

| 参数 | 当前值 | 作用 |
|------|--------|------|
| `T` (温度) | 1.0 | 控制 softmax 输出的锐度：T→0 确定性最强，T→∞ 均匀分布 |
| `top_p` | 0.9 | 核采样：只从累积概率 ≥ top_p 的最小 token 集合中采样 |
| `sample_count` | 1 | Monte Carlo 采样次数，多次采样取平均 |

当前 `T=1.0, top_p=0.9, sample_count=1` 是默认值，未针对 A 股微调。

### 2.2 理论背景

**温度的作用**：在自回归生成中，每个 token 的 logits 除以 T 后再 softmax：
```
p(token_i) = softmax(logits / T)
```
- T < 1：分布更"尖锐"，模型更倾向于高概率 token，预测更确定但多样性低
- T > 1：分布更"平滑"，增加预测多样性但可能引入噪声

对于收益预测问题，较低的 T 可能更好——我们希望预测最可能的路径，而非探索各种可能性。

**top_p 的作用**：Nucleus sampling 截断尾部低概率 token。对于 2^10=1024 的词汇表，top_p=0.9 意味着只保留约 100-200 个高概率 token。降低 top_p 使预测更集中。

**sample_count 的作用**：多次采样取平均可以降低采样噪声。`sample_count=1` 只有一次前向传播，结果可能不稳定。增加到 3-5 可以平滑预测，代价是 3-5 倍的推理时间。

### 2.3 校准策略

**建议的搜索空间**：

| 参数 | 搜索范围 | 步长 |
|------|----------|------|
| T | 0.3 ~ 2.0 | 0.1 |
| top_p | 0.5 ~ 1.0 | 0.05 |
| sample_count | 1 ~ 5 | 1 |

**校准方法**：用历史验证集（如 2026-01-05 到 2026-04-24 的 9-13 周），网格搜索最大化 Sharpe ratio 或周均收益。注意：
- sample_count > 1 时推理时间倍增，需确保 ≤ 5 分钟
- 温度过低保保守，温度过高引入噪声

### 2.4 Sampling 校准的深层改变

当前 KronosModel 的 `_predict_signal_date()` 直接使用固定的 T/top_p。更激进的方案是：

**基于市场波动率的自适应温度**：
```
T_adaptive = T_base * (current_volatility / historical_median_volatility)
```
高波动期用更高温度（更不确定），低波动期用更低温度（更确定）。

### 2.5 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 中等（参数化 YAML 配置 + 验证脚本） |
| 计算开销 | sample_count 每增加 1 推理时间翻倍 |
| 预期收益提升 | 中等偏高 |
| 风险 | sample_count 过高可能超时 |

---

## 三、Kronos Tokenizer 微调 + 行业信息注入

### 3.1 问题诊断

当前 Kronos 微调（`scripts/finetune_kronos.py`）只微调 Predictor（自回归 Transformer），**Tokenizer 保持冻结**（`p.requires_grad = False`）。这意味着：

- Tokenizer 的 BSQ 码本基于全球 45+ 交易所数据训练，对 A 股的量价分布可能不最优
- Tokenizer 只看到 6 维 OHLCV 数据（open/high/low/close/volume/amount），**完全没有行业信息**
- 同一行业中两只股票的 K 线形态可能相似但被映射到不同的 token

### 3.2 Tokenizer 微调方案

Kronos 官方支持两阶段微调（见 `finetune/train_tokenizer.py`）：
1. **Stage 1**：微调 Tokenizer，调整码本以适应目标市场
2. **Stage 2**：微调 Predictor，学习目标市场的 token 序列分布

当前我们跳过了 Stage 1。加入 Tokenizer 微调的好处：
- 码本向量适应 A 股的波动率特征（A 股涨跌停 10%/20% 限制与其他市场不同）
- Quantization 边界重新校准，减少量化误差

### 3.3 行业信息注入方案

Tokenizer 的标准输入是 6 维 OHLCV。有几种注入行业信息的方式：

**方案 A：扩展输入维度（推荐）**
将 `(open, high, low, close, volume, amount)` 扩展为 `(open, high, low, close, volume, amount, sector_embedding)`，其中 sector_embedding 是 4 维 one-hot（sector_l1 到 l4 编码为 4 个整数特征）。Tokenizer 重新训练以处理 10 维输入。

**方案 B：条件 Tokenizer**
在 Tokenizer encoder 中注入行业 embedding 作为条件向量（类似 Stable Diffusion 的 text conditioning），通过 cross-attention 或 FiLM 层影响量化过程。

**方案 C：后融合**
不改 Tokenizer，在 Predictor 的输入中加入行业 token（类似 BERT 的 segment embedding），与 OHLCV token 拼接后送入 Transformer。

### 3.4 实现路径

最务实的路线是**方案 A（扩展输入维度）**：
1. 改编 `KlineFinetuneDataset` 使其返回 10 维数据（6 维 OHLCV + 4 维 sector）
2. 使用 Kronos 官方的 `finetune/train_tokenizer.py` 微调 Tokenizer
3. 然后用新 Tokenizer 微调 Predictor

### 3.5 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 中高（需要深入 Kronos 源码修改 Tokenizer 架构） |
| 训练时间 | Tokenizer 微调 + Predictor 微调 ≈ 数小时（需 GPU） |
| 预期收益提升 | 高（结构性地提升模型对 A 股的理解） |
| 风险 | 修改 Tokenizer 架构可能不兼容现有 Predictor 权重 |

**Kronos Tokenizer 输入维度修改的技术要点**：
- Tokenizer encoder 的第一层线性投影从 `(6 → d_model)` 改为 `(10 → d_model)`
- Tokenizer decoder 的最后一层从 `(d_model → 6)` 改为 `(d_model → 6)`（重建仍只重建 OHLCV，行业是条件信息）
- 需要从头初始化新增的参数，其他参数从预训练权重加载

---

## 四、周频预测 Kronos 集成

### 4.1 问题诊断

当前 Kronos 的 `pred_len=5` 生成 5 根日 K 线，然后用 `(open[4] - open[0]) / open[0]` 计算周收益。但这是一种间接方法——模型预测了 5 天的完整 OHLCV 序列，而我们只关心首尾开盘价。预测中间 4 天增加了误差。

### 4.2 方案：直接训练周频 Kronos

回看数据：Baostock 提供的不只是日线，还有**周线**（`frequency="w"` 或直接用 `get_stock_data.py` 改为周频）。

**思路**：
1. 用周线数据训练一个"周频 Kronos"——每个 token 代表一周的 K 线
2. 预测 `pred_len=1`（未来 1 周），直接用预测周线的 `(close - open) / open` 作为分数
3. 与日频 Kronos（pred_len=5）的分数做加权平均

### 4.3 为什么有效

- **减少累积误差**：日频模型需要 5 步自回归，每步都有误差。周频只需 1 步。
- **不同视角**：周线与日线捕捉不同的市场模式（周线更注重趋势，日线更注重波动）
- **集成互补**：两个模型的误差来源不同，集成后能降低方差

### 4.4 替代方案：用现有日频 Kronos 做多步预测平均

如果重新训练周频 Kronos 成本太高，可以用现有日频 Kronos 做变通：
- `pred_len=1` 预测一天后收益
- `pred_len=5` 预测五天后收益
- `pred_len=10` 预测十天后收益
- 三个预测的信号取加权平均

### 4.5 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 高（需要重新获取周线数据、重新微调两个模型） |
| 计算开销 | 翻倍（两个模型推理） |
| 预期收益提升 | 高（集成效应 + 减少自回归误差） |
| 风险 | 周线样本少（5年=~250根周线 vs ~1200根日线） |

---

## 五、市场环境特征注入

### 5.1 问题诊断

当前 Kronos 只看到单只股票的 OHLCV 数据，**完全缺失市场层面的环境信息**：
- 不知道当前是牛市还是熊市
- 不知道市场整体波动率水平
- 不知道资金流向和情绪

市场环境信息对短期选股至关重要。例如：
- 在强势牛市中，动量策略有效；在震荡市中，反转策略更好
- 市场高波动时，股票同涨同跌（相关性上升），选股区分度下降
- 低波动环境更适合精选个股

### 5.2 方案：Market Context Features via Processor

利用现有的 `MarketLevelProcessor` 和 `SectorLevelProcessor` 生成的因子（已经在 StockDataHandler 中可用），作为 Kronos 的**辅助输入**。

但问题是：KronosModel 的 `predict()` 绕过了 TSDataHandler，直接从 `stock_data.csv` 读 OHLCV，不经过 Processor 链。

**解决方案**：在 KronosModel 推理时额外计算市场环境特征：

```python
# 在 _predict_signal_date() 中
market_features = {
    "market_momentum_5": close_df.pct_change(5).mean(axis=1).iloc[-1],
    "market_volatility": close_df.pct_change().std(axis=1).rolling(20).mean().iloc[-1],
    "market_breadth": (close_df > close_df.shift(60)).mean(axis=1).iloc[-1],
    "vix_proxy": close_df.pct_change().std(axis=1).iloc[-1],
    "dispersion": close_df.pct_change().std(axis=1).rolling(20).mean().iloc[-1],
}
```

然后用于：
1. **调整 Kronos 分数**：市场弱势时对整体分数打折
2. **调整仓位数量**：高波动/下跌市场减少持股数
3. **风格切换**：不同市场环境下侧重不同的选股逻辑

### 5.3 理论依据

**波动率聚类**（Mandelbrot, 1963; Engle, 1982 ARCH）：金融市场波动率呈现持续性——高波动期倾向于持续，低波动期也是如此。这意味着当前市场状态对未来 1-2 周有预测能力。

**市场状态转换**（Hamilton, 1989）：市场的牛市/熊市/震荡状态通过 Markov 过程转换，状态持续时间通常为几周到几个月。识别当前状态有助于确定最优策略。

### 5.4 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 低（在 predict 时额外计算几个标量指标） |
| 计算开销 | 极低 |
| 预期收益提升 | 中等（市场状态信息对 4 天持有期帮助有限） |
| 风险 | 无 |

---

## 六、短期爆发型股票识别

### 6.1 问题诊断

赛题目标是找"未来一周收益最大的股票"，而非"长期表现好的股票"。这两类股票的特征可能截然不同：

- **长期好股票**：ROE 高、利润增长稳定、估值合理 = 基本面驱动
- **短期爆发股票**：技术面突破、事件催化、资金流入 = 动量/情绪驱动

Kronos 作为自回归 OHLCV 预测模型，自然倾向于预测"趋势延续"——即已经涨的股票继续涨。但真正的短期爆发可能来自：
- **反转爆发**：超跌反弹（之前跌过头了）
- **突破爆发**：横盘整理后放量突破阻力位
- **事件驱动**：财报、政策、行业利好
- **资金驱动**：龙虎榜大单、北向资金流入

### 6.2 方案：多维度爆发信号

在 Kronos 分数的基础上，叠加独立的"爆发潜力"信号：

**信号 1：压缩-突破模式（Volatility Squeeze）**
```
compression_score = 1 - (BB_width_current / BB_width_20d_max)
breakout_score = volume_current / volume_20d_avg
squeeze_signal = compression_score * breakout_score
```
Bollinger Band 收窄后放量突破是经典的短期爆发前兆（Bollinger, 2002）。

**信号 2：异常收益检测**
```
abnormal_return = ret_5d - beta * market_ret_5d
# beta 用过去 60 天日收益对市场收益回归得到
```
正的异常收益意味着股票独立于市场走强——可能是有基本面或资金面的独立催化。

**信号 3：RSI 黄金交叉**
```
rsi_14_above_50 = (RSI14 > 50)  # 多头区域
rsi_just_crossed = (RSI14_prev < 50) & (RSI14 > 50)  # 刚刚突破
```
RSI 从 50 以下上穿 50 是短期动能的经典技术信号（Wilder, 1978）。

**信号 4：成交量异常**
```
volume_surge = volume_current / volume_20d_median
# > 2.0 表示异常放量，可能伴随消息面变化
```

### 6.3 融合方式

将这些信号与 Kronos 分数融合，而非替代：

```
final_score = kronos_score_zscore * w_kronos
            + squeeze_signal * w_squeeze
            + abnormal_return * w_abnormal
            + rsi_signal * w_rsi
            + volume_surge * w_volume
```

权重可以通过验证集上的网格搜索确定。

### 6.4 理论依据

- **52 周高点动量效应**（George & Hwang, 2004, *Journal of Finance*）：接近 52 周高点的股票未来有更高的正收益——这比传统的 6 个月动量效应更强，且持有期更短（适合本赛题）
- **短期反转效应**（Jegadeesh, 1990; Lehmann, 1990）：1 周内的超跌存在反弹
- **波动率挤压**（Bollinger, 2002）：低波动压缩后往往迎来方向性突破

### 6.5 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 中等（需要修改 score 计算逻辑） |
| 计算开销 | 低（只需要 pandas TA-Lib 计算） |
| 预期收益提升 | 中高（直接针对"短期爆发"场景） |
| 风险 | 过度依赖技术信号可能导致过拟合 |

---

## 七、动态仓位管理（最重要）

### 7.1 问题诊断

赛题的关键约束被低估了：**权重和 ≤ 1，不到 1 的部分持有现金，现金收益率为 0**。

这意味着：
- 如果市场整体下跌 2%，持有现金（收益 0%）优于满仓（收益 -2%）
- 赛题是按绝对收益率排名，不是相对收益（没有 benchmark 跟踪误差概念）
- 规避下跌周的重要性 ≥ 捕捉上涨周

当前等权 top-5 策略的问题是**永远满仓**——无论市场好坏，总是投 1.0 的权重。这在下跌市中会造成不必要的亏损。

### 7.2 方案：市场状态驱动的动态仓位

**Step 1：市场状态判定**

用两个简单指标：
```
market_trend = (close_index[-1] / close_index[-20] - 1)  # 过去1个月市场涨跌
market_vol = std(daily_returns[-20:])  # 过去1个月波动率
```

状态分类：
| 状态 | 条件 | 含义 |
|------|------|------|
| 牛市+低波 | trend > 0, vol < median | 理想环境，满仓 |
| 牛市+高波 | trend > 0, vol > median | 机会多但风险大，中等仓位 |
| 熊市+低波 | trend < 0, vol < median | 缓跌，轻仓防御 |
| 熊市+高波 | trend < 0, vol > median | 暴跌风险，空仓或极轻仓 |

**Step 2：仓位规则**

| 状态 | top-K 数量 | 总权重 |
|------|-----------|--------|
| 牛市+低波 | 5 | 1.0 |
| 牛市+高波 | 4 | 0.8 |
| 熊市+低波 | 3 | 0.6 |
| 熊市+高波 | 2 | 0.4 |

**Step 3：个股权重约束**

即使市场好，单只股票权重也设上限（如 0.3），防止单只暴雷造成过大亏损。

### 7.3 为什么风险平价/均值方差不适用

用户正确观察到风险平价/均值方差有问题。原因是：

1. **样本太短**：均值方差需要估计协方差矩阵（300×300），用 60 天数据估计极不稳定。
2. **收益估计噪声**：均值方差的期望收益非常敏感，稍有误差就导致极端权重。
3. **风险平价过于保守**：强制等风险贡献会稀释高分股票的权重，与赛题目标（找最好的股票）矛盾。
4. **没有做空机制**：标准优化允许做空，约束 long-only 后往往退化为 corner solution（少数股票拿大部分权重）。

**结论**：对于 4 天持有期 + ≤5 只股票 + 允许空仓的赛题，简单的基于状态的仓位规则优于复杂的优化方法。

### 7.4 进一步优化：基于预测置信度的仓位

如果能够量化 Kronos 预测的"置信度"（例如预测的 5 天收益序列的波动性），可以更精细地调整：

```
confidence = 1 / (1 + std(predicted_5day_returns))  # 预测路径波动越小，置信度越高
position_size = base_position * confidence
```

### 7.5 可行性评估

| 维度 | 评估 |
|------|------|
| 实现难度 | 低（修改 `optimize_portfolio` 函数） |
| 计算开销 | 极低 |
| 预期收益提升 | **高**（最直接有效的改进） |
| 风险 | 状态判定需要足够健壮，避免频繁切换 |

### 7.6 理论依据

**波动率目标（Volatility Targeting）**是业界标准做法（Moreira & Muir, 2017, *Journal of Finance*）：根据当前波动率调整风险暴露，在波动率高时降低仓位，长期能显著改善 Sharpe Ratio。

**市场择时（Market Timing）**的学术证据：虽然长期择时困难，但短期（1-2周）的可预测性更强（Campbell & Thompson, 2008, *Review of Financial Studies*）——估值比率和波动率指标对短期收益有一定预测力。

---

## 八、优先级排序与实施路线图

### Phase 1：立即实施（A2/A3 窗口可完成）

| 优先级 | 改进项 | 预期提升 | 实现工作量 | 风险 |
|--------|--------|----------|-----------|------|
| **P0** | 动态仓位管理 | +3~8% 累计 | 2-3h | 低 |
| **P0** | 横截面 Z-Score 后处理 | +1~2% 累计 | 30min | 极低 |
| **P1** | 温度/采样校准 | +2~5% 累计 | 2-4h | 中（推理时间） |
| **P1** | 短期爆发信号融合 | +1~3% 累计 | 3-5h | 中（过拟合） |

### Phase 2：短期实施（A3/B 窗口可完成）

| 优先级 | 改进项 | 预期提升 | 实现工作量 | 风险 |
|--------|--------|----------|-----------|------|
| **P1** | 市场环境特征注入 | +1~2% 累计 | 2-3h | 低 |
| **P2** | Tokenizer 行业信息微调 | +3~5% 累计 | 1-3 天 | 中高 |

### Phase 3：中长期（B 窗口前完成）

| 优先级 | 改进项 | 预期提升 | 实现工作量 | 风险 |
|--------|--------|----------|-----------|------|
| **P2** | 周频 Kronos 集成 | +3~5% 累计 | 2-5 天 | 高 |
| **P3** | 模型集成 (Kronos + LightGBM + Transformer) | +2~4% 累计 | 1-2 天 | 低 |

### 优先级判断依据

1. **ROI（收益/工作量）**：动态仓位 > 后处理 > 温度校准 > 爆发信号 > 市场特征 > Tokenizer 微调 > 周频 Kronos > 模型集成
2. **确定性与风险**：后处理最安全，Tokenier 微调/周频 Kronos 风险最高
3. **与赛题的匹配度**：动态仓位和爆发信号最直接针对赛题的"短期、小组合、可空仓"特性

---

## 九、关键提醒

### 关于"超越基准"的要求

官方规则：**总投资组合收益率低于基准程序的，不能获得任何名次奖励**。基准程序（https://github.com/Sherlock1956/THU-BDC2026）是 LightGBM + 等权 top-5 的简单方案。我们的 Kronos-finetuned 等权方案（+21.24% 已超过基准）满足此要求。

### 关于月月星（A2 窗口）

当前窗口（5/30-31）的月月星奖金（800 元）对当前排名有帮助。建议：
1. 先提交当前 Kronos-finetuned 等权方案（确保有分数）
2. 然后快速实施 P0 改进（仓位+后处理），在窗口关闭前提交第二版

### 关于 Docker 10GB 限制

Docker 镜像预估：
- PyTorch + CUDA：~5GB
- Qlib + 其他依赖：~2GB
- Kronos 模型权重：~1.1GB
- 数据（train/test.csv）：~60MB
- 代码：< 10MB
- **总计：~8.2GB**，在 10GB 以内 ✓

---

## 参考文献

1. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
2. George, T. J., & Hwang, C. Y. (2004). The 52-Week High and Momentum Investing. *Journal of Finance*, 59(5), 2145-2176.
3. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357-384.
4. Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*, 72(4), 1611-1644.
5. Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.
6. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
7. Campbell, J. Y., & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? *Review of Financial Studies*, 21(4), 1509-1531.
8. Luo, S., et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. *NeurIPS 2025 / AAAI 2026*. arXiv:2508.02739.
