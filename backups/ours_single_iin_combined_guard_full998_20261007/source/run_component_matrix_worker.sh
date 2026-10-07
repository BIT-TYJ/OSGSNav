#!/usr/bin/env bash
set -euo pipefail

# Run one disjoint share of the 4-profile x 9-episode screen. Start workers
# 0..7 in separate tmux windows; each worker owns exactly one GPU.
WORKER_INDEX="${1:?usage: $0 WORKER_INDEX WORKER_COUNT}"
WORKER_COUNT="${2:?usage: $0 WORKER_INDEX WORKER_COUNT}"
[[ "${WORKER_INDEX}" =~ ^[0-9]+$ && "${WORKER_COUNT}" =~ ^[0-9]+$ ]] || exit 2
(( WORKER_COUNT > 0 && WORKER_INDEX < WORKER_COUNT )) || exit 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/nas_ds925/tyj/IEVE}"
LOG_ROOT="${LOG_ROOT:-/nas_ds925/tyj/IEVE/logs/ours_component_ablation_20261005}"
RESULT_ROOT="${NAS_RESULTS:-/nas_ds925/tyj/IEVE/results/ours_component_ablation_20261005}"
# The evaluator has OpenCV visualization enabled. Fresh tmux servers do not
# inherit the interactive VNC display, so provide the validated server default.
export DISPLAY="${DISPLAY:-:78}"
export XAUTHORITY="${XAUTHORITY:-/home/tyj/.Xauthority}"

EPISODES=(4 25 61 212 249 517 581 614 635)
PROFILES=(baseline phase2_400 downstairs_guard stuck_recovery)

task_index=0
for profile in "${PROFILES[@]}"; do
    case "${profile}" in
        baseline) phase2=500; downstairs=0; stuck=0 ;;
        phase2_400) phase2=400; downstairs=0; stuck=0 ;;
        downstairs_guard) phase2=500; downstairs=1; stuck=0 ;;
        stuck_recovery) phase2=500; downstairs=0; stuck=1 ;;
        *) exit 2 ;;
    esac
    for episode in "${EPISODES[@]}"; do
        if (( task_index % WORKER_COUNT == WORKER_INDEX )); then
            tag="${profile}_ep$(printf '%04d' "${episode}")"
            echo "[TASK] worker=${WORKER_INDEX} profile=${profile} episode=${episode}"
            EPISODE_IDS="${episode}" EXPECTED_EPISODE_COUNT=1 \
            ONEPASS_RUN_TAG="${tag}" LOG_ROOT="${LOG_ROOT}" NAS_RESULTS="${RESULT_ROOT}" \
            PHASE2_START_STEP="${phase2}" PREVENT_DOWNSTAIRS="${downstairs}" \
            STUCK_RECOVERY_ENABLED="${stuck}" CUDA_VISIBLE_DEVICES="${WORKER_INDEX}" \
            "${SCRIPT_DIR}/run_component_screen.sh" 0 1
        fi
        task_index=$((task_index + 1))
    done
done
