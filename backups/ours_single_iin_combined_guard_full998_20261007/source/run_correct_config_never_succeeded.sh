#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

SHARD_INDEX="${1:-${SHARD_INDEX:-0}}"
SHARD_COUNT="${2:-${SHARD_COUNT:-1}}"

ID_LOG="${ID_LOG:-/nas_ds925/IEVE/logs/correct_config_never_succeeded_episode_ids.log}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-337}"
SIM_GPU_ID="${SIM_GPU_ID:-0}"
SEM_GPU_ID="${SEM_GPU_ID:--1}"
NAS_RESULTS="${NAS_RESULTS:-/nas_ds925/IEVE/results}"
LOG_ROOT="${LOG_ROOT:-/nas_ds925/IEVE/logs}"
RUN_TAG="${RUN_TAG:-rerun_correct_config_never_succeeded_current}"
SUMMARY_NAME="${SUMMARY_NAME:-autoskip_summary.json}"
PYTHON_BIN="${PYTHON_BIN:-/code1/tyj/anaconda3/envs/ieve/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || die "SHARD_INDEX must be a non-negative integer"
[[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || die "SHARD_COUNT must be a positive integer"
(( SHARD_COUNT > 0 && SHARD_INDEX < SHARD_COUNT )) || \
    die "invalid shard: index=${SHARD_INDEX}, count=${SHARD_COUNT}"

# Most IEVE/3DGS tensors use logical cuda:0.  For simple multi-shard launches,
# expose one physical GPU per shard so every child still uses logical GPU 0.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SHARD_INDEX}"
    GPU_ASSIGNMENT_SOURCE="shard_index"
else
    GPU_ASSIGNMENT_SOURCE="environment"
fi

[[ "${EXPECTED_EPISODE_COUNT}" =~ ^[0-9]+$ ]] || die "EXPECTED_EPISODE_COUNT must be an integer"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ -f "${ID_LOG}" ]] || die "ID log not found: ${ID_LOG}"
[[ -f main.py ]] || die "missing main.py under repo root: ${REPO_ROOT}"
[[ -f envs/habitat/configs/tasks/instance_imagenav.yaml ]] || \
    die "missing task config under repo root: ${REPO_ROOT}"
[[ -f pretrained_models/sem_exp.pth ]] || die "missing semantic policy checkpoint under repo root: ${REPO_ROOT}"
[[ -f tools/run_eval_skip_cuda_crash.py ]] || die "missing runner: tools/run_eval_skip_cuda_crash.py"

EPISODE_LINE="$(
    awk '
        /机器可读单行/ {
            if (getline > 0) print
            exit
        }
    ' "${ID_LOG}" | tr -d '[:space:]'
)"
[[ -n "${EPISODE_LINE}" ]] || die "could not parse the machine-readable ID line from ${ID_LOG}"

IFS=',' read -r -a EPISODE_IDS <<< "${EPISODE_LINE}"
(( ${#EPISODE_IDS[@]} == EXPECTED_EPISODE_COUNT )) || \
    die "parsed ${#EPISODE_IDS[@]} IDs; expected ${EXPECTED_EPISODE_COUNT}"

declare -A SEEN_EPISODE_IDS=()
PREVIOUS_ID=-1
for EPISODE_ID in "${EPISODE_IDS[@]}"; do
    [[ "${EPISODE_ID}" =~ ^[0-9]+$ ]] || die "invalid episode ID: ${EPISODE_ID}"
    (( EPISODE_ID >= 0 && EPISODE_ID <= 999 )) || die "episode ID outside 0-999: ${EPISODE_ID}"
    (( EPISODE_ID > PREVIOUS_ID )) || \
        die "episode IDs must be unique and strictly increasing: ${PREVIOUS_ID}, ${EPISODE_ID}"
    [[ -z "${SEEN_EPISODE_IDS[${EPISODE_ID}]+x}" ]] || die "duplicate episode ID: ${EPISODE_ID}"
    SEEN_EPISODE_IDS["${EPISODE_ID}"]=1
    PREVIOUS_ID="${EPISODE_ID}"
done

TOTAL="${#EPISODE_IDS[@]}"
BASE=$(( TOTAL / SHARD_COUNT ))
REMAINDER=$(( TOTAL % SHARD_COUNT ))
if (( SHARD_INDEX < REMAINDER )); then
    SHARD_SIZE=$(( BASE + 1 ))
    SHARD_OFFSET=$(( SHARD_INDEX * SHARD_SIZE ))
else
    SHARD_SIZE="${BASE}"
    SHARD_OFFSET=$(( REMAINDER * (BASE + 1) + (SHARD_INDEX - REMAINDER) * BASE ))
fi
(( SHARD_SIZE > 0 )) || die "shard ${SHARD_INDEX}/${SHARD_COUNT} has no episodes"

SHARD_EPISODES=("${EPISODE_IDS[@]:SHARD_OFFSET:SHARD_SIZE}")
EPISODE_SPEC="$(IFS=,; echo "${SHARD_EPISODES[*]}")"
FIRST_EPISODE="${SHARD_EPISODES[0]}"
LAST_EPISODE="${SHARD_EPISODES[$(( SHARD_SIZE - 1 ))]}"

LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}_shard${SHARD_INDEX}of${SHARD_COUNT}}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-${RUN_TAG}_s${SHARD_INDEX}of${SHARD_COUNT}}"
mkdir -p "${LOG_DIR}"

echo "[INFO] id_log=${ID_LOG}"
echo "[INFO] episode_numbering=official_habitat_episode_id"
echo "[INFO] no ID conversion is performed; listed ID N runs Habitat episode_id N"
echo "[INFO] total=${TOTAL} shard=${SHARD_INDEX}/${SHARD_COUNT} shard_count=${SHARD_SIZE} first=${FIRST_EPISODE} last=${LAST_EPISODE}"
echo "[INFO] sim_gpu_id=${SIM_GPU_ID} sem_gpu_id=${SEM_GPU_ID} log_dir=${LOG_DIR}"
echo "[INFO] repo_root=${REPO_ROOT} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} gpu_assignment=${GPU_ASSIGNMENT_SOURCE}"

RUNNER_ARGS=(
    --episodes "${EPISODE_SPEC}"
    --log-dir "${LOG_DIR}"
    --exp-name-prefix "${EXP_NAME_PREFIX}"
    --summary-name "${SUMMARY_NAME}"
    --resume
)
if [[ "${DRY_RUN}" == "1" ]]; then
    RUNNER_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" tools/run_eval_skip_cuda_crash.py \
    "${RUNNER_ARGS[@]}" \
    -- \
    "${PYTHON_BIN}" -u main.py \
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
        --minav_target_detect_window 2 \
        --minav_enable_strategy_transition 1 \
        --minav_strategy_phase2_start_step 500 \
        --minav_strategy_phase3_start_step 800 \
        --minav_strategy_use_area_growth_trigger 0 \
        --minav_phase2_arrival_threshold 0.35 \
        --minav_phase2_standoff_retry_limit 3 \
        --minav_phase2_standoff_stuck_patience 8 \
        --minav_phase2_standoff_retry_block_radius 0.45 \
        --minav_phase2_no_reachable_cooldown_steps 40 \
        --minav_phase2_no_reachable_retry_limit 3 \
        --minav_phase3_no_candidate_cooldown_steps 40 \
        --minav_phase3_navigation_stuck_patience 20 \
        --minav_phase3_navigation_progress_epsilon_m 0.02 \
        --minav_phase3_navigation_stuck_tolerance_m 0.1 \
        --minav_targeted_quick_lock_max_dist_m 1.5 \
        --minav_navigation_arrival_threshold 1.0 \
        --minav_strategy_phase3_fallback_stop_radius 0.5 \
        --minav_buffer_goal_dilation_radius 10 \
        --minav_navigation_goal_dilation_radius 5 \
        --exploration_policy frontier \
        --print_images 1 \
        --visualize 1 \
        --target_instance_debug_dump 1 \
        --minav_planner_debug 1
