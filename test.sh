#!/bin/bash
set -e
# 生成比赛提交文件（最新交易日 Top-5 等权）
uv run code/src/submit.py --config code/src/config.yaml
uv run test/score_self.py
