#!/usr/bin/env bash
# Re-run only the 19 paired success-flip episodes with trigger auditing.
set -euo pipefail

SHARD_INDEX="${1:?shard index required}"
SHARD_COUNT="${2:-8}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPISODE_IDS="34,40,43,67,86,102,129,170,190,218,237,242,298,370,679,691,843,926,993"
export EXPECTED_EPISODE_COUNT=19
export ONEPASS_RUN_TAG="stuck_recovery_flip19_audit_20261006"
export STUCK_RECOVERY_ENABLED=1
export PREVENT_DOWNSTAIRS=0
export PHASE2_START_STEP=500
export DISPLAY="${DISPLAY:-:78}"
export XAUTHORITY="${XAUTHORITY:-/home/tyj/.Xauthority}"

exec "${SOURCE_ROOT}/run_component_screen.sh" "${SHARD_INDEX}" "${SHARD_COUNT}"
