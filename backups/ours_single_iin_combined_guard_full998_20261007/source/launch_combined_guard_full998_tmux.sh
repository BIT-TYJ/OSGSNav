#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-osgs_full998_guard_20261007}"
SHARD_COUNT="${SHARD_COUNT:-8}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

for ((shard=0; shard<SHARD_COUNT; shard++)); do
    command="CUDA_VISIBLE_DEVICES=${shard} ${SOURCE_ROOT}/run_combined_guard_full998.sh ${shard} ${SHARD_COUNT}"
    if (( shard == 0 )); then
        tmux new-session -d -s "${SESSION_NAME}" -n "gpu${shard}" "${command}"
    else
        tmux new-window -t "${SESSION_NAME}" -n "gpu${shard}" "${command}"
    fi
done

echo "started ${SESSION_NAME} with ${SHARD_COUNT} shards"
tmux list-windows -t "${SESSION_NAME}" -F '#{window_index}:#{window_name} dead=#{window_dead} pid=#{pane_pid}'
