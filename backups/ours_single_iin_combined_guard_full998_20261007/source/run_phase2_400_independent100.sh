#!/usr/bin/env bash
# Strict paired validation of the earlier Phase-2 transition on the fixed 100 set.
set -euo pipefail

SHARD_INDEX="${1:?shard index required}"
SHARD_COUNT="${2:-8}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDS_FILE="${SOURCE_ROOT}/../stuck_recovery_independent100_seed20261006.txt"

export EPISODE_IDS="$(tail -n 1 "${IDS_FILE}")"
export EXPECTED_EPISODE_COUNT=100
export ONEPASS_RUN_TAG="phase2_400_independent100_20261007"
export PHASE2_START_STEP=400
export PREVENT_DOWNSTAIRS=0
export STUCK_RECOVERY_ENABLED=0
export DISPLAY="${DISPLAY:-:78}"
export XAUTHORITY="${XAUTHORITY:-/home/tyj/.Xauthority}"

exec "${SOURCE_ROOT}/run_component_screen.sh" "${SHARD_INDEX}" "${SHARD_COUNT}"
