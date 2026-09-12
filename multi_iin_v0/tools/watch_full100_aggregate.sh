#!/usr/bin/env bash
set -uo pipefail

task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_episode_count="${MULTI_IIN_EPISODE_COUNT:-100}"
task_python="${MULTI_IIN_PYTHON:-/code1/tyj/anaconda3/envs/ieve/bin/python}"
task_aggregator="${task_repo}/multi_iin_v0/tools/aggregate_results.py"
task_log="${task_root}/worker_logs/aggregate_monitor.log"
mkdir -p "${task_root}/worker_logs"

while true; do
  "${task_python}" "${task_aggregator}" "${task_root}" \
    --output "${task_root}/aggregate" >>"${task_log}" 2>&1 || true
  completed="$(find "${task_root}" -mindepth 2 -maxdepth 2 -name full_result.json -type f | wc -l)"
  echo "AGGREGATE completed=${completed}/${task_episode_count} $(date '+%F %T')" | tee -a "${task_log}"
  if (( completed >= task_episode_count )); then
    break
  fi
  sleep 300
done
