#!/bin/bash
set -e
cd /app
bash /app/init.sh
bash /app/train.sh
bash /app/test.sh