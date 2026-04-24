# 源代码目录
该目录下存放最简的代码工作流，包括以下文件（路径都为项目根目录的相对路径）：

1. `featurework.py`文件。用以将`data/train.csv`和`resource/行业分类.csv`两份数据文件生成Qlib所需的数据格式，保存至`temp/qlib_data/`目录下。该部分通过辅助脚本`scripts/convert_data.py`实现。
2. `run_all_model.py`文件。用以对比调优模型以及超参数。该文件接受YAML配置文件。在该项目中，它们应该在`code/models/`下，不同模型中的目录中找到。
3. `test.py`文件。用以将人工输入的最优YAML配置文件(该文件将放置在`model/result_model.yaml`中)进行推理，并输出模型文件到`model/result_model.pth`中，输出预测的不多于5只，权重和不大于1的股票组合到`output/result.csv`中。