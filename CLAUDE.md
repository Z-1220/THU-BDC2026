# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Git Version Management (MANDATORY)

**Every meaningful change must be committed to git.** This is non-negotiable.

### When to commit
- **Before and after each experiment**: `E0 baseline`, `E1 ScreenProcessor removal`, etc.
- **After any config change**: YAML edits, model parameter changes, new experiment configs
- **After any code change**: model logic, evaluation code, data pipeline
- **After any documentation change**: README, CLAUDE.md, research reports
- **Before any destructive operation**: deleting files, resetting, rebasing

### How to commit
- **One logical change per commit.** Don't batch unrelated changes.
- **Descriptive messages** in English or Chinese, prefix with stage/experiment ID:
  - `feat: E5b rank-weighted portfolio (+4.63pp vs equal-weight)`
  - `fix: commit.py support rank_weights from evaluation config`
  - `docs: merge README.md, update to champion Top3-only config`
  - `cleanup: remove 46 stale experiment runs from run_all_model/`
- **Never commit**: `.venv/`, `__pycache__/`, `*.pth`, `*.tar`, `data/stock_data.csv`, `data/train.csv`, `temp/`, `output/`, `.env`, secrets, tokens
- **Check .gitignore before committing.** If a new pattern should be ignored, add it.

### Before committing
```bash
git status          # What changed?
git diff --stat     # How much changed?
git diff            # What exactly changed? Review every hunk.
```

### After committing
```bash
git status          # Clean?
git log --oneline -5  # Verify commit history
```

### Branches
- `main` is the source of truth. Always commit to `main` directly for this project.
- If experimenting on a branch, merge back to `main` after validation.
- Never leave `main` in a broken state.

### Recovery
- If a commit is wrong, `git revert` (don't `git reset --hard` on shared history).
- Uncommitted work is at risk. Commit early, commit often.

---

## Project context

THU-BDC2026 比赛：学习沪深300成分股排序，输出 5 只股票及权重（权重和 ≤1），5 日持有期。基于 Qlib 框架，YAML 驱动流程。

## Qlib 版本与 Python 环境

- `pyqlib==0.9.7`，Python 3.12，包管理**仅用 uv**（禁止 pip/conda/poetry）
- PyTorch 2.11.0+cu128，LightGBM 4.6.0，CatBoost 1.2.10，XGBoost 3.2.0
- TA-Lib C 库在 Dockerfile 中从源码编译；本地使用需系统级安装（见 README.md 第 196-207 行）
- `uv sync` 安装依赖，激活 `.venv/bin/activate`

## 数据源与数据缓存

**上游数据源**：
- Baostock 在线 API → `scripts/get_stock_data.py` → 保存为 `data/stock_data.csv`（量价日线）
- `resource/行业分类.csv` → 中证四级行业编码（`sector_l1` ~ `sector_l4`）

**数据切分**：
- `data/stock_data.csv`：全量原始数据 → `scripts/split_train_test.py` 按日期切分
- `data/train.csv`：训练数据（排除 test.csv 日期区间 2026-04-13 ~ 2026-04-17），由 `scripts/update_train_data.py` 生成
- `data/test.csv`：**主办方盲测集，不可修改**

**Qlib 二进制缓存**：`temp/qlib_data/`（`.gitignore` 已排除）
- `scripts/convert_data.py` 将 `data/train.csv` + 行业分类转为 Qlib 二进制
- 结构：`calendars/day.txt`（573 个交易日，2024-01-02 ~ 2026-05-27）、`instruments/all.txt`（300 只股票）、`features/<code>/*.day.bin`（每只 16 个字段：open/close/high/low/volume/amount/turn/pctchg/amplitude/change/vwap/factor + sector_l1..l4）
- `temp/qlib_data/` 为 Docker root 权限，重建需通过 Docker
- 若 `temp/qlib_data/` 不存在或过期，运行 `sh init.sh` 重新生成

**数据更新 Skill**：`.claude/skills/update-data.md` — 说"更新数据"时自动拉取 Baostock → 切分 → 重建 Qlib 二进制。一键脚本：`sh scripts/update_data.sh`

**MLflow 缓存**：`temp/mlruns/`（已清空，.gitignore 排除），训练日志不保留在此

## 模型目录结构与 Workflow 配置文件

```
code/models/
  <ModelName>/
    __init__.py
    <ModelName>.py          # qlib.model.base.Model 子类
    <ModelName>.yaml        # 全流程配置（模型、数据集、策略、评测）
```

YAML 结构（以 `PointwiseStockTransformer.yaml` 为例）：
```yaml
qlib_init: {provider_uri, region}                        # Qlib 初始化参数
task:
  model:      {class, module_path, kwargs}               # 模型类+超参
  dataset:    {class, module_path, kwargs}               # handler + segments + processors
  strategy:   {class, module_path, kwargs}               # 组合优化策略
  evaluation: {universe, data_end}                       # 评测参数（非 Qlib 原生，run_all_model.py 读取）
```

**两个关键 YAML 路径约定**：
- `model/result_model.yaml`：最终提交用的最优模型配置 — `train.py` 读取它进行训练，`commit.py` 读取它进行推理
- `code/models/<Name>/<Name>.yaml`：调参实验用的配置 — `run_all_model.py` 读取它进行单次训练+多周评测

**产物路径**：
- `model/result_model.pth`：训练保存的最优权重（由 `train.py` 输出，`commit.py` 加载）
- `model/config_snapshot.yaml`：训练时的配置快照
- `output/result.csv`：推理输出（stock_id + weight）

## 自定义 DataHandler：StockDataHandler

定义在 [code/handlers/stock_handler.py](code/handlers/stock_handler.py)，继承 Qlib 的 `Alpha158`：

1. **特征**：Alpha158 内置 158 因子 + 额外 Qlib 表达式（EMA12/26/60、MACD_LINE、Bollinger、量价变化、HL/OC 价差、RET1/5/10）+ 行业分类（sector_l1~l4）+ 基础字段（CLOSE0/HIGH0/LOW0/VOLUME0/AMOUNT0）
2. **标签**：`LABEL0 = (Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)` — 5 日开盘收益率
3. **Processor 注入**：全部通过 YAML 的 `infer_processors` 列表注入，不在代码中硬编码。共 12 个自定义 Processor 定义在 [code/processors/custom_processor.py](code/processors/custom_processor.py)

## 自定义 Processor（特征工程，code/processors/custom_processor.py）

每个 Processor 独立负责一类特征，通过 YAML 的 `infer_processors` 列表按需启用：

| Processor | 输出特征 | 实现方式 |
|-----------|---------|---------|
| MomentumProcessor | MACD_SIGNAL, MACD_HIST, RSI14, KDJ_K/D/J | TA-Lib |
| VolatilityProcessor | ATR14 | TA-Lib |
| VolumeProcessor | OBV | TA-Lib |
| MomentumQualityProcessor | ADV_MOM_QUALITY_20 | pandas 滚动 |
| TailRiskProcessor | ADV_DOWNSIDE_VOL_20, ADV_MAX_DD_20 | pandas 滚动 |
| DistributionProcessor | ADV_SKEW_20, ADV_KURT_20 | pandas 滚动 |
| LiquidityProcessor | ADV_AMIHUD_20, ADV_TURNOVER_RATIO_20_60 | pandas 滚动 |
| PricePositionProcessor | ADV_PRICE_POS_60, ADV_DIST_TO_HIGH/LOW_60 | pandas 滚动 |
| VolatilityStructureProcessor | ADV_VOL_RATIO_5_60, ADV_HL_RANGE_MEAN_20 | pandas 滚动 |
| CrossSectionalRankProcessor | CS_RANK_RET5, CS_ZSCORE_RET5 | 横截面 groupby |
| MarketLevelProcessor | MARKET_MOM_5/10, MARKET_BREADTH_1, MARKET_DISPERSION | 全市场 groupby |
| SectorLevelProcessor | SECTOR_MOM_5/10, VS_SECTOR_MOM, EXCESS_SECTOR_MOM, SECTOR_RANK_5, SECTOR_BREADTH_1 | 行业内 groupby |
| FridayFilterProcessor | （仅保留周五样本） | 日期过滤 |

注：前三个依赖 TA-Lib；所有 Processor 的 `fields_group` 默认为 `"feature"`，操作 Qlib 的 MultiIndex 列结构 `(fields_group, field_name)`。

## 自定义 Model：PointwiseStockTransformer

定义在 [code/models/PointwiseStockTransformer/PointwiseStockTransformer.py](code/models/PointwiseStockTransformer/PointwiseStockTransformer.py)：

- **`PointwiseStockTransformer`** (nn.Module)：input_proj → PositionalEncoding → TransformerEncoder → FeatureAttention（时序池化）→ MLP head → 标量分数
- **`PointwiseTransformerModel`** (qlib.model.base.Model)：Qlib 模型封装，实现 `fit()` 和 `predict()`。使用 `TSDatasetH` 产生 (N, seq_len, feat_dim) 序列。predict 返回带 (datetime, instrument) MultiIndex 的 pd.Series
- 训练：AdamW + CosineAnnealingLR + early stop（监控验证 MSE），梯度裁剪

**实现新模型的关键契约**：
- 继承 `qlib.model.base.Model`，重写 `fit(dataset)` 和 `predict(dataset)` → pd.Series
- `__init__` 的参数名必须与 YAML kwargs 一致
- `predict` 返回的 pd.Series index 必须是 (datetime, instrument) MultiIndex

## 自定义 Model：Kronos（零样本推理）

Kronos 是清华开源的金融 K 线基础模型，decoder-only Transformer，在 45+ 全球交易所 120 亿条 OHLCV 数据上预训练。通过 KronosTokenizer 将 OHLCV 量化为离散 token，然后自回归生成未来 K 线序列。

**文件**：[code/models/Kronos/](code/models/Kronos/)：
- `KronosModel.py`：Qlib Model 封装，`fit()` 为 no-op（冻结预训练权重），`predict()` 使用 KronosPredictor 自回归生成未来 OHLCV 并转为排序分数。绕过 TSDatasetH，直接从 `data/stock_data.csv` 读取 OHLCV
- [kronos_src/](code/models/Kronos/kronos_src/)：vendored Kronos 源码（导入修复为相对导入）

**预训练权重**：
- `Kronos-small` (24.7M) 和 `Kronos-base` (102.3M)，均使用 `Kronos-Tokenizer-base`，上下文 512
- 下载脚本：`python scripts/download_kronos_models.py` → `model/kronos_pretrained/`（已 gitignore）
- 支持 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com python scripts/download_kronos_models.py`
- 加载时使用 `local_files_only=True`，训练过程无需联网

**运行**：
```bash
python code/src/run_all_model.py run \
  --yaml_paths="models/Kronos/Kronos-small.yaml,models/Kronos/Kronos-base.yaml"
```

## 组合优化策略（code/PortfolioBuilder/portfolio_strategy.py）

`PyPortfolioOptStrategy`：独立类（不依赖 Qlib backtest 框架），通过 `set_price_data(close_df)` 注入价格数据。

支持的优化器（YAML 中 `strategy.kwargs.optimizer` 指定）：
- `mean_variance`：均值-方差优化（分数作预期收益），优选 max_sharpe，降级 min_volatility
- `min_variance`：最小方差优化
- `risk_parity`：层级风险平价 (HRP)

默认降级策略：分数最高的 top-5 等权（0.2）。

## 调参与评测（code/src/run_all_model.py）

CLI 入口：`python code/src/run_all_model.py run --yaml_paths="models/A/A.yaml,models/B/B.yaml"`

核心流程：
1. 读 YAML → 初始化 Qlib → 应用补丁（修复 `D.features()` 的 freq/inst_processors 参数冲突）
2. 训练模型（`model.fit(dataset)`）
3. 预测（`model.predict(dataset)`）
4. 从 `handler._data` 提取 open/close 价格（彻底绕过 Qlib `D.features()`）
5. 按 5 日非重叠周期评测：signal_day → buy_day（t+1）→ sell_day（t+5）
6. 每期计算组合收益率，汇总指标：周数、均值收益、标准差、Sharpe、胜率、最大亏损、累计收益

输出：`output/run_all_model/<timestamp>/summary.md` + 每模型 `_weekly.csv`

## 推理流程（训练→预测→打包→提交）

```
sh train.sh              # uv run code/src/train.py   → model/result_model.pth + config_snapshot.yaml
sh test.sh               # uv run code/src/test.py    → output/result.csv
docker buildx build ...  # 打包 docker
docker compose up        # 验证 docker 可运行
python test/test.py      # 批量评分所有提交的 .tar 文件
```

- `train.py`：读 `model/result_model.yaml` → 训练 → 存 `result_model.pth` + `config_snapshot.yaml`
- `commit.py`：读 `model/result_model.yaml` + `model/result_model.pth` → 推理 → `output/result.csv`
- `test.sh` 调用 `code/src/test.py`（git status 显示该文件已删除 `D`），`commit.py` 可能是其重命名替代版

## 自评与批量评分

- `python test/score_self.py`：将 `output/result.csv` 与 `data/test.csv` 对比，计算加权收益率，保存到 `temp/tmp.csv`
- `python test/test.py`：批量 Docker 评分 — 读 `test/tar_files_list.txt` → 加载 Docker 镜像 → `docker compose up` → 等容器退出 → 复制 `result.csv` → 调 `score_docker.py` 算分 → 汇总到 `test/result.csv`

## 比赛规则与时间线

### 赛题

基于沪深 300 成分股历史数据，预测未来一周收益最大的 ≤5 只股票组合（权重和 ≤1，剩余为现金）。

### 评估公式（竞赛官方）

T 为信号日（周五），T+1 开盘买入，T+5 开盘卖出：

$$R_i = \frac{P_{i,T+5}^{open} - P_{i,T+1}^{open}}{P_{i,T+1}^{open}} \qquad R_{total} = \sum_{i=1}^{n} w_i \times R_i$$

- 现金权重 $w_{cash} = 1 - \sum w_i$，现金收益率为 0
- **必须超越基准程序**（https://github.com/Sherlock1956/THU-BDC2026）才能获得名次和奖项

### 时间线

| 阶段 | 日期 | 说明 |
|------|------|------|
| 报名组队 | 3/26 – **7/15 12:00** | 实名认证，截止后不可更改成员 |
| 线上赛 A1 | **4/25 8:00 – 4/26 23:59** | 已过 |
| 线上赛 A2 | **5/30 8:00 – 5/31 23:59** | **本次提交窗口** |
| 线上赛 A3 | 6/27 8:00 – 6/28 23:59 | |
| 线上赛 B | 8/1 8:00 – 8/2 23:59 | 最终排名（B 榜前 3 名在校生队直接晋级决赛） |
| 模型/数据报备截止 | 7/18 | 开源模型和数据的链接+md5 报备到 data@tsinghua.edu.cn |
| 决赛 | 8 月中下旬 | 现场答辩，清华 |

### 提交方式

每个提交窗口内，在竞赛平台（https://www.heywhale.com/home/）上传：
1. `result.csv` 文件
2. Docker 镜像导出为 `队伍名称.tar`，上传至夸克网盘并提交分享链接

### 硬性约束

| 约束 | 值 | 说明 |
|------|-----|------|
| Docker 文件总大小 | **≤ 10GB** | 不可压缩，含环境+代码+数据+模型 |
| 预测时间 | **≤ 5 分钟** | 从 test.sh 开始到 result.csv 生成 |
| 训练时间 | **≤ 8 小时** | i7-1365U/16GB/4060 8GB/50GB 磁盘 |
| 离线运行 | **必须** | 训练和预测均不得联网 |
| 可复现 | **必须** | 固定随机种子，从训练开始完整复现，结果与提交一致 |

### Docker 文件结构（比赛要求）

```
/app
├── code/           # 运行代码
│   └── src/
├── data/           # 数据（docker-compose.yml 挂载）
│   ├── test.csv
│   └── train.csv
├── model/          # 训练模型和其它数据（docker 内）
├── output/         # 计算结果（docker-compose.yml 挂载）
│   └── result.csv
├── temp/           # 中间结果（docker-compose.yml 挂载）
├── init.sh         # 必选
├── train.sh        # 必选
├── test.sh         # 必选
└── readme.md       # 必选
```

评审方使用 `docker load -i 队伍名称.tar` 加载，再用下发的 `docker-compose.yml` 运行。

### 预训练模型规则

- 必须是 **2026 年 4 月 1 日前**开源的非商业化模型，或已发表的学术模型（arxiv 不算）
- 需在 **7/18 前**向 data@tsinghua.edu.cn 报备（主题："团队名称 + 模型数据报备"），内容包含开源链接和 md5
- Kronos 满足此条件（2025 年 NeurIPS 发表，模型已开源）

### 代码审核

主办方检查：可复现性（固定 seed 重跑一致）、主要贡献为 ML 方法、时间/大小合规、数据/模型来源合规。

### 奖项（总额 5.48 万元）

- 决赛 1/2/3 名：2 万/1 万/0.8 万，4-6 名各 0.4 万
- 在校生全国一等奖 5 名（学生排名 1-5）、二等奖 10 名（6-15）、三等奖 15 名（16-30）
- 月月星：A 阶段每次提交窗口，在校生第一+在职第一各 800 元

## 常用命令速查

```bash
uv sync                                          # 安装依赖
source .venv/bin/activate                        # 激活环境
sh init.sh                                       # CSV → Qlib 二进制 (temp/qlib_data/)
sh train.sh                                      # 训练 → model/result_model.pth
sh test.sh                                       # 推理 → output/result.csv
python code/src/run_all_model.py run \
  --yaml_paths="models/LightGBM/LightGBM.yaml"   # 单训多周评测
python test/score_self.py                        # 自评分数
docker compose up                                # Docker 验证
```
