#!/usr/bin/env bash
set -euo pipefail
ROOT=/nas_ds925/tyj/IEVE
UNI=/code1/tyj/UniGoal
GPU=${1:?gpu}
START=${2:?episode_start}
COUNT=${3:?episode_count}
PORT=$((11440 + GPU))
OUT=${UNIGOAL_MULTI_OUTPUT_ROOT:-"$ROOT/multi_iin_v0/results/experiments/unigoal_multi_graphshare_full100_20261004"}
export CUDA_VISIBLE_DEVICES="$GPU"
export OLLAMA_HOST="127.0.0.1:${PORT}"
export UNIGOAL_OLLAMA_BASE_URL="http://127.0.0.1:${PORT}/v1/"
export UNIGOAL_DUMP_LOCATION="$OUT"
export UNIGOAL_EXPERIMENT_ID="ep$(printf '%03d' "$START")_gpu${GPU}"
export PYTHONPATH="$UNI:$UNI/third_party/Grounded-Segment-Anything:$UNI/third_party/Grounded-Segment-Anything/segment-anything:$ROOT/baselines/unigoal_multi_iin:${PYTHONPATH:-}"
mkdir -p "$OUT"
ollama serve >"$OUT/ollama_gpu${GPU}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf --max-time 2 "http://127.0.0.1:${PORT}/api/tags" >/dev/null && break; sleep 1; done
cd "$UNI"
/code1/tyj/anaconda3/envs/uninav/bin/python -u "$ROOT/baselines/unigoal_multi_iin/main_unigoal_multi.py" \
  --config-file "$ROOT/baselines/unigoal_multi_iin/config_habitat_unigoal_multi.yaml" \
  --multi_manifest "$ROOT/multi_iin_v0/manifests/miin_v0_val_k2.json" \
  --multi_episode_start "$START" --multi_num_episodes "$COUNT" \
  --multi_goal_step_budget "${MULTI_GOAL_STEP_BUDGET:-1000}"
STATUS=$?
kill "$OLLAMA_PID" 2>/dev/null || true
wait "$OLLAMA_PID" 2>/dev/null || true
exit "$STATUS"
