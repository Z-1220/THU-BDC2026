#!/bin/bash
set -e
# Baostock CSV -> Qlib 二进制格式 + sector_map.json
uv run update_data_and_split.py
uv run annotate_stocks.py
uv run code/src/convert_data.py \
    --csv ./data/stock_data.csv \
    --qlib_dir ./qlib_data/hs300_data \
    --sector_csv ./doc/行业分类.csv \
    --sector_map ./data/sector_map.json
# Qlib workflow：训练 + 评估
uv run code/src/workflow.py --config code/src/config.yaml
