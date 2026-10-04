#!/usr/bin/env bash
set -euo pipefail
ROOT=/nas_ds925/tyj/IEVE
OUT=${UNIGOAL_MULTI_OUTPUT_ROOT:-"$ROOT/multi_iin_v0/results/experiments/unigoal_multi_graphshare_full100_20261004"}
SESSION_PREFIX=${UNIGOAL_MULTI_SESSION_PREFIX:-unigoal_graphshare}
mkdir -p "$OUT"
for gpu in $(seq 0 7); do
  start=$((gpu * 13))
  count=13
  if [ "$gpu" -eq 7 ]; then count=9; fi
  tmux new-session -d -s "${SESSION_PREFIX}_${gpu}" \
    "cd '$ROOT' && UNIGOAL_MULTI_OUTPUT_ROOT='$OUT' '$ROOT/baselines/unigoal_multi_iin/run_full100_worker.sh' '$gpu' '$start' '$count' > '$OUT/worker_gpu${gpu}.log' 2>&1"
done
