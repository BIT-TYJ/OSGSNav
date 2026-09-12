#!/usr/bin/env bash
set -euo pipefail

task_session="${MULTI_IIN_TMUX_SESSION:-miin_full100}"
task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_variant="${MULTI_IIN_VARIANT:-Continuous-Shared}"
task_episode_count="${MULTI_IIN_EPISODE_COUNT:-100}"
read -r -a task_gpus <<< "${MULTI_IIN_GPUS:-0 1 2 3 4 5 6 7}"
task_supervisor="${task_repo}/multi_iin_v0/tools/ablation/run_reconstructed_supervisor.sh"

case "${task_variant}" in
  Independent-Reset|Continuous-Clear|Continuous-Shared) ;;
  *)
    echo "invalid MULTI_IIN_VARIANT: ${task_variant}" >&2
    exit 2
    ;;
esac
if (( ${#task_gpus[@]} == 0 )); then
  echo "MULTI_IIN_GPUS must list at least one GPU" >&2
  exit 2
fi
if ! [[ "${task_episode_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MULTI_IIN_EPISODE_COUNT must be a positive integer" >&2
  exit 2
fi

if tmux has-session -t "${task_session}" 2>/dev/null; then
  echo "tmux session already exists: ${task_session}" >&2
  exit 1
fi

worker_count="${#task_gpus[@]}"
for worker_index in "${!task_gpus[@]}"; do
  gpu_id="${task_gpus[worker_index]}"
  if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU id in MULTI_IIN_GPUS: ${gpu_id}" >&2
    exit 2
  fi
  episode_indices=()
  for ((episode_index=worker_index; episode_index<task_episode_count; episode_index+=worker_count)); do
    episode_indices+=("${episode_index}")
  done
  command="cd ${task_repo} && exec env MULTI_IIN_REPO=${task_repo} MULTI_IIN_TASK_ROOT=${task_root} MULTI_IIN_VARIANT=${task_variant} MULTI_IIN_EXTRA_ARGS='${MULTI_IIN_EXTRA_ARGS:-}' ${task_supervisor} ${gpu_id} ${episode_indices[*]}"
  if (( worker_index == 0 )); then
    tmux new-session -d -s "${task_session}" -n "gpu${gpu_id}" "${command}"
  else
    tmux new-window -t "${task_session}" -n "gpu${gpu_id}" "${command}"
  fi
done

tmux select-window -t "${task_session}:gpu${task_gpus[0]}"
tmux new-window -t "${task_session}" -n aggregate \
  "cd ${task_repo} && exec env MULTI_IIN_REPO=${task_repo} MULTI_IIN_TASK_ROOT=${task_root} MULTI_IIN_EPISODE_COUNT=${task_episode_count} ${task_repo}/multi_iin_v0/tools/ablation/watch_reconstructed_aggregate.sh"
tmux select-window -t "${task_session}:gpu${task_gpus[0]}"
echo "started ${task_session}: variant=${task_variant} episodes=${task_episode_count} gpus=${task_gpus[*]} root=${task_root}"
