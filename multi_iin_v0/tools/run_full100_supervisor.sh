#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID EPISODE_INDEX [EPISODE_INDEX ...]" >&2
  exit 2
fi

task_gpu="$1"
shift
task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_worker="${task_repo}/multi_iin_v0/tools/run_full100_worker.sh"
task_log="${task_root}/worker_logs/supervisor_gpu${task_gpu}.log"
mkdir -p "${task_root}/worker_logs"

round=0
while true; do
  round=$((round + 1))
  echo "SUPERVISOR gpu=${task_gpu} round=${round} $(date '+%F %T')" | tee -a "${task_log}"
  "${task_worker}" "${task_gpu}" "$@" >>"${task_log}" 2>&1
  worker_code=$?
  if (( worker_code == 0 )); then
    break
  fi
  echo "SUPERVISOR gpu=${task_gpu} worker_exit=${worker_code}; retry queue in 60s" | tee -a "${task_log}"
  sleep 60
done

echo "SUPERVISOR gpu=${task_gpu} complete $(date '+%F %T')" | tee -a "${task_log}"
