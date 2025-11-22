#!/usr/bin/env bash
set -euo pipefail
python3 data/generate_multivariate_ts.py --out data/ts_multivariate.csv --n_periods 1200 --freq D
python3 src/train_transformer.py --data data/ts_multivariate.csv --output results --epochs 5 --seq_len 90 --pred_len 30
echo 'Done. Results in results/'
