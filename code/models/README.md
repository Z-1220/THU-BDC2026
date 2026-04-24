# model目录放置内容

以“模型/模型类”和“模型/YAML配置”两个文件作为基础结构。

## 模型类应该是qlib.model.base.Model的子类，即继承qlib.model.base.Model并重写方法。

### Custom Model Class  自定义模型类
The Custom models need to inherit qlib.model.base.Model and override the methods in it.
自定义模型需要继承 qlib.model.base.Model 并重写其中的方法。

Override the __init__ method
Qlib passes the initialized parameters to the __init__ method.
Qlib 将初始化参数传递给 __init__ 方法。

The hyperparameters of model in the configuration must be consistent with those defined in the __init__ method.
配置文件中模型的超参数必须与 __init__ 方法中定义的一致。

Code Example: In the following example, the hyperparameters of model in the configuration file should contain parameters such as loss:mse.
代码示例：在以下示例中，配置文件中模型的超参数应包含如 loss:mse 等参数。

```python
def __init__(self, loss='mse', **kwargs):
    if loss not in {'mse', 'binary'}:
        raise NotImplementedError
    self._scorer = mean_squared_error if loss == 'mse' else roc_auc_score
    self._params.update(objective=loss, **kwargs)
    self._model = None
```

Override the fit method
Qlib calls the fit method to train the model.
Qlib 调用 fit 方法来训练模型。

The parameters must include training feature dataset, which is designed in the interface.
参数必须包括训练特征数据集，该数据集在接口中设计。

The parameters could include some optional parameters with default values, such as num_boost_round = 1000 for GBDT.
参数可以包括一些带默认值的可选参数，例如 GBDT 的 num_boost_round = 1000。

Code Example: In the following example, num_boost_round = 1000 is an optional parameter.
代码示例：在以下示例中，num_boost_round = 1000 是一个可选参数。

```python
def fit(self, dataset: DatasetH, num_boost_round = 1000, **kwargs):

    # prepare dataset for lgb training and evaluation
    df_train, df_valid = dataset.prepare(
        ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )
    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]

    # Lightgbm need 1D array as its label
    if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
    else:
        raise ValueError("LightGBM doesn't support multi-label training")

    dtrain = lgb.Dataset(x_train.values, label=y_train)
    dvalid = lgb.Dataset(x_valid.values, label=y_valid)

    # fit the model
    self.model = lgb.train(
        self.params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        evals_result=evals_result,
        **kwargs
    )
```

Override the predict method
The parameters must include the parameter dataset, which will be used to get the test dataset.
参数必须包含参数数据集，该数据集将用于获取测试数据集。

Return the prediction score.
返回预测分数。

Please refer to Model API for the parameter types of the fit method.
请参考 Model API 了解 fit 方法的参数类型。

```
Model 模型:
classqlib.model.base.BaseModel
Modeling things  建模事物

abstractpredict(*args, **kwargs)→ object
Make predictions after modeling things
建模后进行预测

classqlib.model.base.Model
Learnable Models  可学习模型

fit(dataset: Dataset, reweighter: Reweighter)
Learn model from the base model
从基础模型学习模型

Note  注意

The attribute names of learned model should not start with ‘_’. So that the model could be dumped to disk.
学习模型的属性名称不应以‘_’开头。这样模型才能被保存到磁盘。

The following code example shows how to retrieve x_train, y_train and w_train from the dataset:
以下代码示例展示了如何从数据集中获取 x_train、y_train 和 w_train：

# get features and labels
df_train, df_valid = dataset.prepare(
    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
)
x_train, y_train = df_train["feature"], df_train["label"]
x_valid, y_valid = df_valid["feature"], df_valid["label"]

# get weights
try:
    wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"],data_key=DataHandlerLP.DK_L)
    w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
except KeyError as e:
    w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
    w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

Parameters:
dataset (Dataset) – dataset will generate the processed data from model training.
dataset (Dataset) – 数据集将生成模型训练的加工数据。

abstractpredict(dataset: Dataset, segment: str | slice = 'test')→ object
give prediction given Dataset
给出基于数据集的预测

Parameters:
dataset (Dataset) – dataset will generate the processed dataset from model training.
dataset (Dataset) – dataset 将根据模型训练生成处理后的数据集。

segment (Text or slice) – dataset will use this segment to prepare data. (default=test)
分段（文本或切片）——数据集将使用此分段来准备数据。（默认=测试）

Return type:
Prediction results with certain type such as pandas.Series.
预测结果以特定类型呈现，如 pandas.Series。

classqlib.model.base.ModelFT
Model (F)ine(t)unable  模型（微调）

abstractfinetune(dataset: Dataset)
finetune model based given dataset
基于给定数据集微调模型

A typical use case of finetuning model with qlib.workflow.R
使用 qlib.workflow.R 对模型进行微调的典型用例

# start exp to train init model
with R.start(experiment_name="init models"):
    model.fit(dataset)
    R.save_objects(init_model=model)
    rid = R.get_recorder().id

# Finetune model based on previous trained model
with R.start(experiment_name="finetune model"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="init models")
    model = recorder.load_object("init_model")
    model.finetune(dataset, num_boost_round=10)
Parameters:
dataset (Dataset) – dataset will generate the processed dataset from model training.
dataset (Dataset) – dataset 将根据模型训练生成处理后的数据集。
```

Code Example: In the following example, users need to use LightGBM to predict the label(such as preds) of test data x_test and return it.
代码示例：在以下示例中，用户需要使用 LightGBM 来预测测试数据 x_test 的标签（例如 preds），并将其返回。

```python
def predict(self, dataset: DatasetH, **kwargs)-> pandas.Series:
    if self.model is None:
        raise ValueError("model is not fitted yet!")
    x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    return pd.Series(self.model.predict(x_test.values), index=x_test.index)
Override the finetune method (Optional)
```

This method is optional to the users. When users want to use this method on their own models, they should inherit the ModelFT base class, which includes the interface of finetune.
此方法对用户是可选的。当用户希望在他们的模型上使用此方法时，他们应该继承包含 finetune 接口的 ModelFT 基础类。

The parameters must include the parameter dataset.
参数必须包含参数数据集。

Code Example: In the following example, users will use LightGBM as the model and finetune it.
代码示例：在以下示例中，用户将使用 LightGBM 作为模型并进行微调。

```python
def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
    # Based on existing model and finetune by train more rounds
    dtrain, _ = self._prepare_data(dataset)
    self.model = lgb.train(
        self.params,
        dtrain,
        num_boost_round=num_boost_round,
        init_model=self.model,
        valid_sets=[dtrain],
        valid_names=["train"],
        verbose_eval=verbose_eval,
    )
```

YAML配置至少需要包含当前目录下模型的参数配置和其它必要的workflow配置，该文件将直接被传入`code\src\run_all_model.py`进行训练。