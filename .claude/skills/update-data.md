---
name: update-data
description: 从 Baostock 拉取最新股票数据，更新 train.csv 并重建 Qlib 二进制缓存
---

# 数据更新 Skill

更新项目训练数据到最新日期。不会影响 `data/test.csv`（主办方盲测集）。

## 触发条件

当用户说以下任何一句时调用本 skill：
- "更新数据"
- "更新股票数据"
- "拉取最新数据"
- "刷新数据"
- "/update-data"

## 操作步骤

### 步骤 1：更新 get_stock_data.py 的日期范围

修改 `scripts/get_stock_data.py` 第 223-224 行：
```python
start_date = "2024-01-01"
end_date = "<今天的日期 YYYY-MM-DD>"
```

### 步骤 2：拉取 Baostock 数据

```bash
source .venv/bin/activate
uv run python scripts/get_stock_data.py
```

约需 5-10 分钟（300 只股票，Baostock API 限速）。

### 步骤 3：生成新的 train.csv

排除 test.csv 的日期区间（2026-04-13 ~ 2026-04-17），生成新 train.csv：

```bash
source .venv/bin/activate
uv run python scripts/update_train_data.py
```

### 步骤 4：重建 Qlib 二进制缓存

旧 `temp/qlib_data/` 为 Docker root 权限，需通过 Docker 重建：

```bash
docker compose run --rm --entrypoint "" app uv run scripts/convert_data.py
```

或使用脚本一键完成步骤 3+4：

```bash
sh scripts/update_data.sh
```

## 不会变动的文件

- `data/test.csv` — 主办方盲测集，不可修改
- `model/result_model.yaml` — 训练配置
- `code/` 下所有代码

## 更新后的数据概览

此命令运行后：
- `data/stock_data.csv`：原始 Baostock 全量数据
- `data/train.csv`：训练用数据（排除盲测区间 2026-04-13 ~ 2026-04-17）
- `temp/qlib_data/`：Qlib 二进制缓存（需配合 Docker 重建）

## 后续操作建议

更新数据后，应调整 YAML 中 `segments` 的时间范围，扩大 valid 和 test 段的覆盖。当前数据截止日期为最新 Baostock 可用日期。
