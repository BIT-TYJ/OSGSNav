#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID EPISODE_INDEX [EPISODE_INDEX ...]" >&2
  exit 2
fi

task_gpu="$1"
shift
task_root="multi_iin_v0/results/experiments/frame_planner_ablation_10ep_20260908"
supervisor_log="${task_root}/worker_logs/supervisor_gpu${task_gpu}.log"
mkdir -p "${task_root}/worker_logs"

is_complete() {
  local episode_index mode episode_label
  for episode_index in "$@"; do
    episode_label="$(printf '%03d' "${episode_index}")"
    for mode in off on; do
      if [[ ! -s "${task_root}/ep${episode_label}_${mode}_gpu${task_gpu}/full_result.json" ]]; then
        return 1
      fi
    done
  done
  return 0
}

round=0
while ! is_complete "$@"; do
  round=$((round + 1))
  echo "SUPERVISOR gpu=${task_gpu} round=${round} start $(date '+%F %T')" | tee -a "${supervisor_log}"
  multi_iin_v0/tools/run_frame_planner_ablation_worker.sh "${task_gpu}" "$@" \
    >>"${supervisor_log}" 2>&1
  worker_code=$?
  if is_complete "$@"; then
    break
  fi
  echo "SUPERVISOR gpu=${task_gpu} worker_exit=${worker_code}; wait 30s" | tee -a "${supervisor_log}"
  sleep 30
done

echo "SUPERVISOR gpu=${task_gpu} complete $(date '+%F %T')" | tee -a "${supervisor_log}"
