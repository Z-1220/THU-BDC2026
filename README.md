# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**学习排序选股**方案，基于 Qlib 框架，全流程 YAML 驱动：
- 输入：每只股票过去 60 个交易日的量价与技术特征序列（Alpha158 + 自定义因子）；
- 模型：`PointwiseStockTransformer`（PyTorch Transformer 回归），以及 LightGBM 等对比模型；
- 输出：预测分数最高的 5 只股票及其权重（权重和 ≤ 1），5 日持有期。

---

## 1. 项目目标与整体流程

核心目标是学习"未来 5 日哪些股票的开盘收益率最高"的排序函数。

训练与推理主流程：
1. `scripts/convert_data.py` 将 `data/train.csv` + `resource/行业分类.csv` 转为 Qlib 二进制数据（`temp/qlib_data/`）；
2. `StockDataHandler`（继承 Alpha158）通过 Qlib 表达式引擎计算基础特征和标签；
3. YAML 配置的 Processor 链注入额外技术指标（MACD/RSI/ATR 等 12 类）；
4. `DatasetH`（表格模型）或 `TSDatasetH`（序列模型）组织训练/验证/测试样本；
5. 训练模型，监控验证集 loss，early stop 保存最优权重到 `model/result_model.pth`；
6. `commit.py` 加载最优模型在最新日期上推理，输出 `output/result.csv`。

标签定义：`LABEL0 = (Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)`，即未来第 5 个交易日开盘相对未来第 1 个交易日开盘的收益率。

---

## 2. 代码结构说明

```
code/
├── handlers/stock_handler.py         # 自定义 DataHandler，继承 Alpha158，追加额外表达式特征
├── processors/custom_processor.py    # 12 个自定义 Processor（动量/波动/流动性/横截面/行业等）
├── models/
│   ├── PointwiseStockTransformer/    # PyTorch Transformer 回归模型 (.py + .yaml)
│   └── LightGBM/                     # LightGBM 表格模型 (.yaml，使用 Qlib 内置 LGBModel)
├── PortfolioBuilder/
│   └── portfolio_strategy.py         # 组合优化策略（均值方差/最小方差/风险平价）
└── src/
    ├── train.py                      # 训练入口：读 model/result_model.yaml → 训练 → 保存权重
    ├── commit.py                     # 推理入口：加载模型 → 预测 → output/result.csv
    └── run_all_model.py              # 调参评测引擎：单次训练 + 多周非重叠周期回测

scripts/
├── get_stock_data.py                 # Baostock 数据抓取 → data/stock_data.csv
├── split_train_test.py               # 按日期切分训练/测试集 → data/train.csv, data/test.csv
└── convert_data.py                   # CSV + 行业分类 → Qlib 二进制 (temp/qlib_data/)

test/
├── score_self.py                     # 自评：将 output/result.csv 与 data/test.csv 对比
├── test.py                           # 批量 Docker 评分
└── score_docker.py                   # Docker 单次评分
```

**训练产物**：
- `model/result_model.pth`：最优模型权重
- `model/config_snapshot.yaml`：训练配置快照

**预测输出**：
- `output/result.csv`：stock_id + weight（由 `commit.py` 生成）

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv` — 训练输入，并作为 `scripts/convert_data.py` 的数据源

关键字段：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅`

行业数据：
- `resource/行业分类.csv` — 中证四级行业编码，由 `scripts/convert_data.py` 编码为整数特征写入 Qlib 二进制

---

## 4. 运行方法

依赖管理使用 uv，禁止 pip/conda/poetry。

```bash
uv sync                       # 安装依赖
source .venv/bin/activate     # 激活虚拟环境
sh init.sh                    # 生成 Qlib 二进制数据 (temp/qlib_data/)
sh train.sh                   # 训练 → model/result_model.pth + config_snapshot.yaml
sh test.sh                    # 推理 → output/result.csv
```

调参评测（不修改 model/ 目录）：

```bash
python code/src/run_all_model.py run \
  --yaml_paths="models/LightGBM/LightGBM.yaml"
```

Docker 打包与验证：

```bash
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .
docker compose up             # 验证容器可运行
python test/test.py           # 批量评分提交的 .tar 文件
```

---

## 5. 常见问题

1) **TA-Lib 安装失败**

本项目部分 Processor 依赖 TA-Lib，需先安装系统级 C 库：

```
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

Dockerfile 中已内置 TA-Lib 编译步骤，Docker 环境无需额外安装。

2) **Qlib 二进制数据缺失**

若 `temp/qlib_data/` 不存在或数据过期，运行 `sh init.sh` 重新生成。

3) **GPU/CPU 自动选择**

代码按 `CUDA → CPU` 顺序自动选择设备；无 GPU 时可直接 CPU 运行。
