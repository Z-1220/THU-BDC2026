#!/bin/bash
# 一键更新训练数据：拉取 Baostock → 生成 train.csv → 重建 Qlib 二进制
# 注意：不会修改 data/test.csv（主办方盲测集）
set -e

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  数据更新: Baostock → train.csv → Qlib"
echo "  目标截止日期: $TODAY"
echo "=========================================="

cd "$PROJECT_ROOT"

# ---- 1. 更新 get_stock_data.py 的日期范围 ----
echo ""
echo "[1/4] 更新 get_stock_data.py 截止日期为 $TODAY ..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/end_date = \"[0-9-]*\"/end_date = \"$TODAY\"/" scripts/get_stock_data.py
else
    sed -i "s/end_date = \"[0-9-]*\"/end_date = \"$TODAY\"/" scripts/get_stock_data.py
fi

# ---- 2. 拉取 Baostock 数据 ----
echo ""
echo "[2/4] 从 Baostock 拉取最新数据（约5-10分钟）..."
source .venv/bin/activate
uv run python scripts/get_stock_data.py

# ---- 3. 生成新 train.csv ----
echo ""
echo "[3/4] 生成新的 train.csv（排除盲测区间 2026-04-13 ~ 2026-04-17）..."
uv run python scripts/update_train_data.py

# ---- 4. 重建 Qlib 二进制 ----
echo ""
echo "[4/4] 通过 Docker 重建 Qlib 二进制数据..."
docker compose run --rm --entrypoint "" app uv run scripts/convert_data.py

echo ""
echo "=========================================="
echo "  数据更新完成"
echo "  - data/stock_data.csv: Baostock 全量"
echo "  - data/train.csv:      训练数据"
echo "  - data/test.csv:       盲测集（未修改）"
echo "  - temp/qlib_data/:     Qlib 二进制"
echo "=========================================="
echo ""
echo "提示: 数据更新后，考虑调整 YAML 中的 segments 时间范围。"
