#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GPU:EPISODE:PYTHON_PID:WORKER_PID [...]" >&2
  exit 2
fi

task_session="${MULTI_IIN_TMUX_SESSION:-miin_full100}"
task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_variant="${MULTI_IIN_VARIANT:-Continuous-Shared}"
task_episode_count="${MULTI_IIN_EPISODE_COUNT:-100}"
task_new_gpus="${MULTI_IIN_NEW_GPUS:-5 6 7}"
task_queue_worker="${task_repo}/multi_iin_v0/tools/run_full100_queue_worker.sh"
task_claims="${task_root}/.episode_claims"
task_log="${task_root}/worker_logs/queue_handoff.log"
mkdir -p "${task_claims}" "${task_root}/worker_logs"

declare -a gpu_ids episode_ids python_pids worker_pids
for spec in "$@"; do
  IFS=: read -r gpu_id episode_id python_pid worker_pid extra <<< "${spec}"
  if [[ -n "${extra:-}" ]] || ! [[ "${gpu_id}" =~ ^[0-9]+$ && "${episode_id}" =~ ^[0-9]+$ && "${python_pid}" =~ ^[0-9]+$ && "${worker_pid}" =~ ^[0-9]+$ ]]; then
    echo "invalid handoff specification: ${spec}" >&2
    exit 2
  fi
  cmdline="$(tr '\0' ' ' < "/proc/${python_pid}/cmdline" 2>/dev/null || true)"
  actual_parent="$(ps -o ppid= -p "${python_pid}" | tr -d ' ' || true)"
  if [[ "${actual_parent}" != "${worker_pid}" ]] || [[ "${cmdline}" != *"--multi_episode_start ${episode_id}"* ]] || [[ "${cmdline}" != *"--sim_gpu_id ${gpu_id}"* ]]; then
    echo "handoff validation failed for ${spec}: parent=${actual_parent} cmd=${cmdline}" >&2
    exit 1
  fi
  gpu_ids+=("${gpu_id}")
  episode_ids+=("${episode_id}")
  python_pids+=("${python_pid}")
  worker_pids+=("${worker_pid}")
done

for index in "${!gpu_ids[@]}"; do
  claim_dir="${task_claims}/ep$(printf '%03d' "${episode_ids[index]}")"
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    echo "claim already exists for active episode ${episode_ids[index]}" >&2
    exit 1
  fi
done

for index in "${!gpu_ids[@]}"; do
  kill -STOP "${worker_pids[index]}"
  echo "HANDOFF armed gpu=${gpu_ids[index]} episode=${episode_ids[index]} python_pid=${python_pids[index]} worker_pid=${worker_pids[index]} $(date '+%F %T')" | tee -a "${task_log}"
done

queue_command() {
  local gpu_id="$1"
  printf 'cd %s && exec env MULTI_IIN_REPO=%s MULTI_IIN_TASK_ROOT=%s MULTI_IIN_VARIANT=%s MULTI_IIN_EPISODE_COUNT=%s %s %s' \
    "${task_repo}" "${task_repo}" "${task_root}" "${task_variant}" "${task_episode_count}" "${task_queue_worker}" "${gpu_id}"
}

for gpu_id in ${task_new_gpus}; do
  tmux new-window -t "${task_session}" -n "gpu${gpu_id}" "$(queue_command "${gpu_id}")"
  echo "HANDOFF queue started gpu=${gpu_id} $(date '+%F %T')" | tee -a "${task_log}"
done

remaining="${#gpu_ids[@]}"
while (( remaining > 0 )); do
  for index in "${!gpu_ids[@]}"; do
    [[ "${python_pids[index]}" == "done" ]] && continue
    process_state="$(ps -o stat= -p "${python_pids[index]}" | tr -d ' ' || true)"
    if [[ -n "${process_state}" && "${process_state}" != Z* ]]; then
      continue
    fi
    gpu_id="${gpu_ids[index]}"
    episode_id="${episode_ids[index]}"
    rmdir "${task_claims}/ep$(printf '%03d' "${episode_id}")" 2>/dev/null || true
    tmux respawn-window -k -t "${task_session}:gpu${gpu_id}" "$(queue_command "${gpu_id}")"
    python_pids[index]="done"
    remaining=$((remaining - 1))
    echo "HANDOFF queue replaced legacy worker gpu=${gpu_id} after episode=${episode_id} state=${process_state:-exited} $(date '+%F %T')" | tee -a "${task_log}"
  done
  (( remaining > 0 )) && sleep 10
done

echo "HANDOFF complete: all legacy workers moved to atomic queue $(date '+%F %T')" | tee -a "${task_log}"
