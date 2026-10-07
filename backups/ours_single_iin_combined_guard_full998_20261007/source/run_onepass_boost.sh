#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/nas_ds925/tyj/IEVE}"
cd "${RUNTIME_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}:${RUNTIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SHARD_INDEX="${1:-${SHARD_INDEX:-0}}"
SHARD_COUNT="${2:-${SHARD_COUNT:-1}}"

ID_LOG="${ID_LOG:-/nas_ds925/IEVE/logs/all_998_official_episode_ids_no774_778.log}"
EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-998}"
SIM_GPU_ID="${SIM_GPU_ID:-0}"
SEM_GPU_ID="${SEM_GPU_ID:--1}"
NAS_RESULTS="${NAS_RESULTS:-/nas_ds925/tyj/IEVE/results}"
LOG_ROOT="${LOG_ROOT:-/nas_ds925/tyj/IEVE/logs}"
# Use a scoped override so an exported RUN_TAG from another experiment cannot
# silently redirect this run into stale logs and resume corrupted results.
RUN_TAG="${ONEPASS_RUN_TAG:-ours_onepass_boost_20261005}"
SUMMARY_NAME="${SUMMARY_NAME:-autoskip_summary.json}"
PYTHON_BIN="${PYTHON_BIN:-/code1/tyj/anaconda3/envs/ieve/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED="${GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED:-1}"
GAUSSMI_EXPLORATION_ENABLED="${GAUSSMI_EXPLORATION_ENABLED:-1}"
MINAV_FINE_LOCALIZATION_ENABLED="${MINAV_FINE_LOCALIZATION_ENABLED:-1}"
SECONDARY_EXPLORATION_ENABLED="${SECONDARY_EXPLORATION_ENABLED:-1}"
EXCLUDE_EPISODES="${EXCLUDE_EPISODES:-}"
EXCLUDE_REASON="${EXCLUDE_REASON:-manual exclusion requested}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || die "SHARD_INDEX must be a non-negative integer"
[[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || die "SHARD_COUNT must be a positive integer"
(( SHARD_COUNT > 0 && SHARD_INDEX < SHARD_COUNT )) || \
    die "invalid shard: index=${SHARD_INDEX}, count=${SHARD_COUNT}"
[[ "${EXPECTED_EPISODE_COUNT}" =~ ^[0-9]+$ ]] || \
    die "EXPECTED_EPISODE_COUNT must be a non-negative integer"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
for FLAG_NAME in \
    GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED \
    GAUSSMI_EXPLORATION_ENABLED \
    MINAV_FINE_LOCALIZATION_ENABLED \
    SECONDARY_EXPLORATION_ENABLED; do
    FLAG_VALUE="${!FLAG_NAME}"
    [[ "${FLAG_VALUE}" == "0" || "${FLAG_VALUE}" == "1" ]] || \
        die "${FLAG_NAME} must be 0 or 1"
done

# Each shard exposes one physical GPU. All CUDA users inside the child process
# therefore continue to use logical cuda:0.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="${SHARD_INDEX}"
    GPU_ASSIGNMENT_SOURCE="shard_index"
else
    GPU_ASSIGNMENT_SOURCE="environment"
fi

[[ -f "${ID_LOG}" ]] || die "ID log not found: ${ID_LOG}"
[[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -f "${SOURCE_ROOT}/main.py" ]] || die "missing backup main.py: ${SOURCE_ROOT}/main.py"
[[ -f envs/habitat/configs/tasks/instance_imagenav.yaml ]] || \
    die "missing task config under repo root: ${REPO_ROOT}"
[[ -f pretrained_models/sem_exp.pth ]] || \
    die "missing semantic policy checkpoint under repo root: ${REPO_ROOT}"
[[ -f "${SOURCE_ROOT}/tools/run_eval_skip_cuda_crash.py" ]] || \
    die "missing backup runner: ${SOURCE_ROOT}/tools/run_eval_skip_cuda_crash.py"

# Prefer the first integer-list line after the machine-readable heading. As a
# fallback, accept the last line containing only comma-separated integer IDs.
EPISODE_LINE="$(
    awk '
        /机器可读单行|machine-readable single line/ {
            want_machine_line = 1
            next
        }
        {
            candidate = $0
            gsub(/[`[:space:]]/, "", candidate)
            if (candidate ~ /^[0-9]+(,[0-9]+)*$/) {
                if (want_machine_line) {
                    print candidate
                    found = 1
                    exit
                }
                last_candidate = candidate
            }
        }
        END {
            if (!found && last_candidate != "") print last_candidate
        }
    ' "${ID_LOG}"
)"
[[ -n "${EPISODE_LINE}" ]] || \
    die "could not parse a comma-separated episode ID line from ${ID_LOG}"

IFS=',' read -r -a EPISODE_IDS <<< "${EPISODE_LINE}"
if (( EXPECTED_EPISODE_COUNT > 0 )); then
    (( ${#EPISODE_IDS[@]} == EXPECTED_EPISODE_COUNT )) || \
        die "parsed ${#EPISODE_IDS[@]} IDs; expected ${EXPECTED_EPISODE_COUNT}"
fi

declare -A SEEN_EPISODE_IDS=()
PREVIOUS_ID=-1
for EPISODE_ID in "${EPISODE_IDS[@]}"; do
    [[ "${EPISODE_ID}" =~ ^[0-9]+$ ]] || die "invalid episode ID: ${EPISODE_ID}"
    (( EPISODE_ID >= 0 && EPISODE_ID <= 999 )) || \
        die "episode ID outside 0-999: ${EPISODE_ID}"
    (( EPISODE_ID != 774 && EPISODE_ID != 778 )) || \
        die "historical resource/infrastructure-abnormal episode found in 998-episode list: ${EPISODE_ID}"
    (( EPISODE_ID > PREVIOUS_ID )) || \
        die "episode IDs must be unique and strictly increasing: ${PREVIOUS_ID}, ${EPISODE_ID}"
    [[ -z "${SEEN_EPISODE_IDS[${EPISODE_ID}]+x}" ]] || \
        die "duplicate episode ID: ${EPISODE_ID}"
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
echo "[INFO] total=${TOTAL} shard=${SHARD_INDEX}/${SHARD_COUNT} shard_size=${SHARD_SIZE} first=${FIRST_EPISODE} last=${LAST_EPISODE}"
echo "[INFO] source_root=${SOURCE_ROOT} runtime_root=${RUNTIME_ROOT} log_dir=${LOG_DIR}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} gpu_assignment=${GPU_ASSIGNMENT_SOURCE} sim_gpu_id=${SIM_GPU_ID} sem_gpu_id=${SEM_GPU_ID}"
echo "[INFO] ablation_flags=semantic_mapping:${GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED},gaussmi_exploration:${GAUSSMI_EXPLORATION_ENABLED},fine_localization:${MINAV_FINE_LOCALIZATION_ENABLED},secondary_exploration:${SECONDARY_EXPLORATION_ENABLED}"

RUNNER_ARGS=(
    --episodes "${EPISODE_SPEC}"
    --log-dir "${LOG_DIR}"
    --exp-name-prefix "${EXP_NAME_PREFIX}"
    --summary-name "${SUMMARY_NAME}"
    --resume
)
if [[ -n "${EXCLUDE_EPISODES}" ]]; then
    RUNNER_ARGS+=(
        --exclude-episodes "${EXCLUDE_EPISODES}"
        --exclude-reason "${EXCLUDE_REASON}"
    )
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    RUNNER_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" "${SOURCE_ROOT}/tools/run_eval_skip_cuda_crash.py" \
    "${RUNNER_ARGS[@]}" \
    -- \
    "${PYTHON_BIN}" -u "${SOURCE_ROOT}/main.py" \
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
        --collision_threshold 0.2 \
        --prevent_downstairs 1 \
        --downstairs_height_threshold 0.2 \
        --downstairs_min_observations 2 \
        --enable_3dgs 1 \
        --3dgs_semantic_guided_optimization_enabled "${GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED}" \
        --use_minav_confirmation_localization 1 \
        --minav_fine_localization_enabled "${MINAV_FINE_LOCALIZATION_ENABLED}" \
        --minav_initial_spin 1 \
        --gaussmi_exploration_enabled "${GAUSSMI_EXPLORATION_ENABLED}" \
        --minav_target_detect_window 2 \
        --minav_secondary_exploration_enabled "${SECONDARY_EXPLORATION_ENABLED}" \
        --minav_strategy_phase2_start_step 400 \
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
        --minav_strategy_phase3_fallback_stop_radius 0.5 \
        --minav_targeted_quick_lock_max_dist_m 1.5 \
        --minav_navigation_arrival_threshold 1.0 \
        --minav_navigation_preferred_stop_radius 0.5 \
        --minav_navigation_staging_max_radius 2.0 \
        --minav_phase3_navigation_staging_max_radius 1.2 \
        --minav_navigation_reachable_ring_step 0.25 \
        --minav_navigation_staging_replan_limit 1 \
        --minav_navigation_execution_goal_dilation_radius 1 \
        --minav_buffer_goal_dilation_radius 10 \
        --minav_navigation_goal_dilation_radius 5 \
        --minav_cluster_recovery_enabled 1 \
        --minav_cluster_recovery_trigger_score 100.0 \
        --minav_cluster_recovery_max_radius_m 3.0 \
        --minav_cluster_recovery_waypoints 12 \
        --minav_cluster_recovery_goal_dilation_radius 2 \
        --minav_cluster_recovery_timeout 200 \
        --minav_cluster_recovery_min_remaining_steps 50 \
        --minav_cluster_recovery_yaw_range_deg 45.0 \
        --minav_cluster_recovery_yaw_step_deg 5.0 \
        --minav_stg_angle_fix 0 \
        --minav_sweep_turn_in_place_fix 1 \
        --stuck_recovery_enabled 1 \
        --stuck_recovery_patience 8 \
        --stuck_recovery_move_eps 0.03 \
        --stuck_recovery_stg_eps 1.0 \
        --stuck_recovery_block_radius 3 \
        --stuck_recovery_backward_steps 2 \
        --stuck_recovery_turn_steps 4 \
        --stuck_recovery_forward_steps 3 \
        --exploration_policy frontier \
        --frontier_visited_path_block_radius_m 0.5 \
        --frontier_debug_log 1 \
        --print_images 1 \
        --visualize 1 \
        --target_instance_debug_dump 1 \
        --minav_planner_debug 1
