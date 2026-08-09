#!/bin/bash
set -e
cd /app
export UV_OFFLINE=1
bash /app/init.sh
bash /app/train.sh
bash /app/test.sh
