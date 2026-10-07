#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/nas_ds925/tyj/IEVE/.runtime_cache/combined_guard_full998_20261007}"
export TMPDIR="${RUNTIME_CACHE_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
export TORCH_EXTENSIONS_DIR="${RUNTIME_CACHE_ROOT}/torch_extensions"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TORCH_EXTENSIONS_DIR}"

# Union of the independently useful/recovered settings. Phase 2 remains at
# the restored 500-step setting; the 400-step trial was an ablation regression.
export ONEPASS_RUN_TAG="${ONEPASS_RUN_TAG:-combined_guard_full998_20261007}"
export SUMMARY_NAME="${SUMMARY_NAME:-combined_guard_full998_summary.json}"
export EXPECTED_EPISODE_COUNT="${EXPECTED_EPISODE_COUNT:-998}"
export PREVENT_DOWNSTAIRS="${PREVENT_DOWNSTAIRS:-1}"
export PHASE2_START_STEP="${PHASE2_START_STEP:-500}"
export STUCK_RECOVERY_ENABLED="${STUCK_RECOVERY_ENABLED:-1}"
export GS_SEMANTIC_GUIDED_OPTIMIZATION_ENABLED=1
export GAUSSMI_EXPLORATION_ENABLED=1
export MINAV_FINE_LOCALIZATION_ENABLED=1
export SECONDARY_EXPLORATION_ENABLED=1

# Metrics-only storage profile for the long run. This disables bulky PLY and
# visualization dumps, not online Gaussian mapping or metric collection.
export GS_SAVE_INTERVAL="${GS_SAVE_INTERVAL:-0}"
export GS_RENDER_DEBUG="${GS_RENDER_DEBUG:-0}"
export GS_SAVE_FINAL="${GS_SAVE_FINAL:-0}"
export SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"

# Build the complete official 0..999 list locally and omit the two historical
# resource-abnormal tasks. A caller may still override EPISODE_IDS for smoke.
if [[ -z "${EPISODE_IDS:-}" ]]; then
    episode_ids=()
    for ((episode_id=0; episode_id<1000; episode_id++)); do
        if (( episode_id == 774 || episode_id == 778 )); then
            continue
        fi
        episode_ids+=("${episode_id}")
    done
    export EPISODE_IDS="$(IFS=,; echo "${episode_ids[*]}")"
fi

exec "${SOURCE_ROOT}/run_component_screen.sh" "$@"
