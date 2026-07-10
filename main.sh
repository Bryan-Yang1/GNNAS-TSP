#!/usr/bin/env bash
set -euo pipefail

TIME_LIMIT="${TIME_LIMIT:-60}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs}"
DEVICE="${DEVICE:-auto}"

python main.py \
  --time_limit "${TIME_LIMIT}" \
  --data_root "${DATA_ROOT}" \
  --results_root "${RESULTS_ROOT}" \
  --device "${DEVICE}" \
  --num_neighbors -1 \
  --hidden_dim 128 \
  --num_layers 3 \
  --node_limit 1000 \
  --learning_rate 0.001 \
  --decay_rate 1.2 \
  --max_epochs 30 \
  --decay_every 5 \
  --rank_method dense \
  --cost_loss MSE \
  --rank_loss ListNet \
  --loss_weight 0.5
