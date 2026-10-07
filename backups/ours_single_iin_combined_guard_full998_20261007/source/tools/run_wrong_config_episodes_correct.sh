#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-416-430,796-836,838-999}"
SIM_GPU_ID="${SIM_GPU_ID:-0}"
SEM_GPU_ID="${SEM_GPU_ID:--1}"
NAS_RESULTS="${NAS_RESULTS:-/nas_ds925/IEVE/results}"
LOG_ROOT="${LOG_ROOT:-/nas_ds925/IEVE/logs}"
RUN_TAG="${RUN_TAG:-rerun_wrong_config_correct}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}}"
SUMMARY_NAME="${SUMMARY_NAME:-autoskip_summary.json}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

RUNNER_ARGS=(
    --episodes "${EPISODES}"
    --log-dir "${LOG_DIR}"
    --exp-name-prefix "${RUN_TAG}"
    --summary-name "${SUMMARY_NAME}"
    --resume
)

if [[ "${DRY_RUN}" == "1" ]]; then
    RUNNER_ARGS+=(--dry-run)
fi

python tools/run_eval_skip_cuda_crash.py \
    "${RUNNER_ARGS[@]}" \
    -- \
    python -u main.py \
        --eval 1 \
        --num_processes 1 \
        --task_config tasks/instance_imagenav.yaml \
        --split val \
        --version v3 \
        --load pretrained_models/sem_exp.pth \
        --dump_location "${NAS_RESULTS}" \
        --minav_results_dir "${NAS_RESULTS}" \
        --auto_gpu_config 0 \
        --sim_gpu_id "${SIM_GPU_ID}" \
        --sem_gpu_id "${SEM_GPU_ID}" \
        --enable_3dgs 1 \
        --use_minav_confirmation_localization 1 \
        --minav_initial_spin 1 \
        --exploration_policy frontier \
        --print_images 1 \
        --visualize 1 \
        --target_instance_debug_dump 1 \
        --minav_planner_debug 1
