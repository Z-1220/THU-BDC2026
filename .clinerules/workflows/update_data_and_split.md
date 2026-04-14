# 更新股票数据并重新划分训练/测试集 (Windows版)

本工作流自动完成以下操作：
1. 将 `get_stock_data.py` 中的 `end_date` 更新为今日日期
2. 运行数据获取脚本（增量更新至今日）
3. 调整 `split_train_test.py` 的参数，使测试集为数据集的最后 5 个交易日
4. 执行数据切分

## Step 1: 获取今日日期并更新脚本中的 end_date
使用 PowerShell 获取今日日期，并直接替换 `get_stock_data.py` 中的对应行。

<execute_command>
<command>PowerShell -Command "$today = Get-Date -Format 'yyyy-MM-dd'; (Get-Content get_stock_data.py -Encoding UTF8) -replace 'end_date = \".*\"', ('end_date = \"' + $today + '\"') | Set-Content get_stock_data.py -Encoding UTF8"</command>
<requires_approval>true</requires_approval>
</execute_command>

## Step 2: 运行数据获取脚本（增量更新至今日）
使用 uv 执行脚本，无需用户批准。

<execute_command>
<command>uv run python get_stock_data.py</command>
<requires_approval>false</requires_approval>
</execute_command>

## Step 3: 计算新的训练/测试集划分日期
通过 uv 运行 Python 一行脚本，从 `data/stock_data.csv` 中提取所有唯一交易日，自动确定：
- `TEST_START`：倒数第 5 个交易日（测试集起始）
- `TRAIN_END`：倒数第 6 个交易日（训练集结束）
- `TEST_END`：最后一个交易日（测试集结束）

然后，该脚本会直接修改 `split_train_test.py` 文件中 `parse_args()` 函数的三个 `default` 值，保持文件其他部分不变。

<execute_command>
<command>uv run python -c "import re, pandas as pd; df = pd.read_csv('data/stock_data.csv'); dates = sorted(df['日期'].unique()); test_start, train_end, test_end = dates[-5], dates[-6], dates[-1]; content = open('data/split_train_test.py', encoding='utf-8').read(); content = re.sub(r'default=\"[0-9\-]+\",?\s*help=\"训练集结束日期\"', f'default=\"{train_end}\", help=\"训练集结束日期\"', content); content = re.sub(r'default=\"[0-9\-]+\",?\s*help=\"测试集开始日期\"', f'default=\"{test_start}\", help=\"测试集开始日期\"', content); content = re.sub(r'default=\"[0-9\-]+\",?\s*help=\"测试集结束日期\"', f'default=\"{test_end}\", help=\"测试集结束日期\"', content); open('data/split_train_test.py', 'w', encoding='utf-8').write(content); print(f'参数已更新: train_end={train_end}, test_start={test_start}, test_end={test_end}')"</command>
<requires_approval>true</requires_approval>
</execute_command>

## Step 4: 执行数据切分
使用 uv 运行切分脚本。

<execute_command>
<command>uv run python split_train_test.py</command>
<requires_approval>false</requires_approval>
</execute_command>

## Step 5: 验证输出文件
使用 PowerShell 列出生成的两个 CSV 文件的基本信息。

<execute_command>
<command>PowerShell -Command "Get-ChildItem data/train.csv, data/test.csv | Format-Table Name, Length, LastWriteTime"</command>
</execute_command>