# 自定义 DataHandler 集成

继承 Qlib 内置的 `Alpha158`，在此基础上追加额外技术指标与自定义标签，
同时通过 YAML 配置注入 Processor 列表，实现数据处理管道的完全配置驱动。

具体继承方式见 https://qlib.readthedocs.io/en/latest/component/data.html#data-handler 中的
`DataHandlerLP` 及其子类 `Alpha158`。

## Qlib DataHandler 文档部分

### Data Handler 数据处理器

Qlib 中的 **Data Handler** 模块用于处理大多数模型将使用到的常见数据处理方法。

用户可以通过 `qrun` 以自动工作流程的方式使用 Data Handler，更多详情请参考 [Workflow: Workflow Management](#)。

#### DataHandlerLP

除了在自动工作流程中使用 Data Handler 之外，Data Handler 也可以作为一个独立模块使用，用户可以通过它轻松地预处理数据（标准化、去除 NaN 等）并构建数据集。

为了实现这一点，Qlib 提供了一个基础类 `qlib.data.dataset.DataHandlerLP`。这个类的核心思想是：我们将有一些可学习的 **Processors**，它们可以学习数据处理的参数（例如，z-score 归一化的参数）。当新数据到来时，这些训练好的 Processors 可以处理新数据，从而使得高效地实时数据处理成为可能。关于 Processors 的更多信息将在下一小节中列出。

#### 接口 Interface

以下是一些 `DataHandlerLP` 提供的重要接口：

```python
class qlib.data.dataset.handler.DataHandlerLP(
    instruments=None,
    start_time=None,
    end_time=None,
    data_loader: dict | str | DataLoader | None = None,
    infer_processors: List = [],
    learn_processors: List = [],
    shared_processors: List = [],
    process_type='append',
    drop_raw=False,
    **kwargs
)
```
1. **动机 Motivation**

   - 在我们希望为 **学习（learning）** 和 **推理（inference）** 使用不同的处理器工作流程的情况下。

2. **说明 Description**

   此处理器将以 `pd.DataFrame` 格式生成三份数据：

   - **DK_R / `self._data`**: 从加载器中加载的原始数据  
   - **DK_I / `self._infer`**: 用于推理的数据  
   - **DK_L / `self._learn`**: 用于学习模型的数据  

3. **使用不同处理器工作流程进行学习和推理的动机示例：**

   - 学习和推理所用的股票池（instrument universe）可能不同。
   - 某些样本的处理可能依赖于标签（例如，某些达到涨跌停的样本可能需要额外处理或被丢弃）。
   - 这些处理器仅适用于学习阶段。

4. **数据处理器使用提示 Tips for data handler**

   - **为了降低内存成本**：
     - `drop_raw=True`：这将直接在原始数据上进行修改。
   - **注意**：像 `self._infer` 或 `self._learn` 这样的处理数据与 Qlib 数据集中的 `"train"` 和 `"test"` 等 **时间分段（segments）** 的概念是不同的：
     - `self._infer` / `self._learn` 是使用不同处理器处理后的底层数据。
     - `"train"` / `"test"` 仅表示查询数据时的时间范围（通常 `"train"` 在时间序列上早于 `"test"`）。
     - 例如，你可以查询在 `"train"` 时间分段中由 `infer_processors` 处理过的 `data._infer`。

5. **参数 Parameters**

   - **`infer_processors` (list)**:
     - 用于生成推理数据的处理器 `<description info>` 列表。
     - `<description info>` 示例：
       1. 类名与参数：
          ```python
          {
              "class": "MinMaxNorm",
              "kwargs": {
                  "fit_start_time": "20080101",
                  "fit_end_time": "20121231"
              }
          }
          ```
       2. 仅类名：
          ```python
          "DropnaFeature"
          ```
       3. Processor 的实例对象。

   - **`learn_processors` (list)**:
     - 类似于 `infer_processors`，但用于为学习模型生成数据。

   - **`process_type` (str)**:
     - `PTYPE_I = 'independent'`（独立）:
       - `self._infer` 将由 `infer_processors` 处理
       - `self._learn` 将由 `learn_processors` 处理
     - `PTYPE_A = 'append'`（追加）:
       - `self._infer` 将由 `infer_processors` 处理
       - `self._learn` 将由 `infer_processors + learn_processors` 处理

   - **`drop_raw` (bool)**:
     - 是否丢弃原始数据。

6. **方法 Methods**

   - **`fit()`**
     - 拟合数据，但不进行实际处理。

   - **`fit_process_data()`**
     - 拟合并处理数据。
     - 拟合的输入将是前一个处理器的输出。

   - **`process_data(with_fit: bool = False)`**
     - 处理数据。如有必要，调用 `processor.fit()`。
     - **符号说明**：`(数据) [处理器]`
     - **数据处理流程**：
       - 当 `self.process_type == DataHandlerLP.PTYPE_I`（独立）时：
         ```
         (self._data) -[shared_processors]-> (_shared_df)
           ├── [learn_processors] -> (_learn_df)
           └── [infer_processors] -> (_infer_df)
         ```
       - 当 `self.process_type == DataHandlerLP.PTYPE_A`（追加）时：
         ```
         (self._data) -[shared_processors]-> (_shared_df) -[infer_processors]-> (_infer_df) -[learn_processors]-> (_learn_df)
         ```

   - **`config(processor_kwargs: dict | None = None, **kwargs)`**
     - 配置数据（指定从数据源加载哪些数据）。
     - 当从数据集加载 pickled handler 时使用，数据将使用不同的时间范围初始化。

   - **`setup_data(init_type: str = 'fit_seq', **kwargs)`**
     - 在多次运行初始化时设置数据。
     - `enable_cache` (bool): 默认为 `False`。
       - 若为 `True`，处理后的数据将保存在磁盘上，下次调用 `init` 时直接加载缓存。

   - **`fetch(selector: Timestamp | slice | str = slice(None, None, None), level: str | int = 'datetime', col_set='__all', data_key: Literal['raw', 'infer', 'learn'] = 'infer', squeeze: bool = False, proc_func: Callable | None = None) → DataFrame`**
     - 从底层数据源获取数据。
     - 参数说明：
       - `selector`: 描述如何通过索引选择数据。
       - `level`: 选择数据的索引级别。
       - `col_set`: 选择一组有意义的列（如特征列）。
       - `data_key`: 要获取的数据类型（`'raw'`, `'infer'`, `'learn'`）。

   - **`get_cols(col_set='__all', data_key: Literal['raw', 'infer', 'learn'] = 'infer') → list`**
     - 获取列名列表。

   - **`classmethod cast(handler: DataHandlerLP) → DataHandlerLP`**
     - **动机**：用户在其自定义包中创建了一个 datahandler，希望在不引入包依赖和复杂逻辑的情况下与其他用户共享处理后的 handler。
     - 此方法将子类转换为 `DataHandlerLP`，并仅保留处理后的数据。

   - **`classmethod from_df(df: DataFrame) → DataHandlerLP`**
     - **动机**：当用户想要快速获取一个数据处理器时。
     - 创建的 handler 仅包含一个共享的 DataFrame，不含任何处理器。
     - 典型用法：
       ```python
       from qlib.data.dataset import DataHandlerLP
       dh = DataHandlerLP.from_df(df)
       dh.to_pickle(fname, dump_all=True)
       ```
     - **TODO**: 静态数据加载器（StaticDataLoader）相当慢，其实无需再次复制数据。

> 如果用户希望通过配置加载特征和标签，可以定义一个新的 handler，并调用 `qlib.contrib.data.handler.Alpha158` 的静态方法 `parse_config_to_fields`。  
> 此外，用户还可以将 `qlib.contrib.data.processor.ConfigSectionProcessor`（为配置定义的特征提供预处理方法）传入新的 handler。

## 本目录模块

### StockDataHandler

**路径**：`code/handlers/stock_handler.py`

**继承**：`qlib.contrib.data.handler.Alpha158`

**职责**：在 `Alpha158` 内置 158 因子的基础上，追加可用 Qlib 表达式引擎直接描述
的额外技术指标，并替换默认标签为比赛要求的 5 日开盘价差收益率。

**特征扩展**（`EXTRA_EXPR_FEATURES`）：

| 类别 | 特征名 | Qlib 表达式 |
| :--- | :--- | :--- |
| 均线 / MACD 线 | `EMA12`, `EMA26`, `EMA60` | `EMA($close, N)` |
| | `MACD_LINE` | `EMA($close, 12) - EMA($close, 26)` |
| Bollinger | `BOLL_MID`, `BOLL_STD` | `Mean($close, 20)` / `Std($close, 20)` |
| 成交量 | `VOL_CHANGE` | `($volume - Ref($volume, 1)) / (Ref($volume, 1) + 1e-12)` |
| | `VOL_RATIO` | `Mean($volume, 5) / (Mean($volume, 20) + 1e-12)` |
| K 线价差 | `HL_SPREAD`, `OC_SPREAD`, `HC_SPREAD`, `LC_SPREAD` | `$high - $low` 等 |
| 基础收益率 | `RET1`, `RET5`, `RET10` | `$close / Ref($close, N) - 1` |

**标签**：`LABEL0 = (Ref($open, -5) - Ref($open, -1)) / (Ref($open, -1) + 1e-12)`

**设计原则**：Handler 只负责定义特征公式和标签公式，**不在代码中组装 Processor
列表**。所有 Processor（包括自定义的复杂指标处理器和标准化的 `RobustZScoreNorm`、
`Fillna` 等）均通过 YAML 配置文件中的 `infer_processors` 字段注入。