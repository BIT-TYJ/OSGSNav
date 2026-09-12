#!/usr/bin/env bash
set -uo pipefail

task_session="miin_ablation_10ep"
task_root="/nas_ds925/tyj/IEVE/multi_iin_v0/results/experiments/frame_planner_ablation_10ep_20260908"
task_supervisor="/nas_ds925/tyj/IEVE/multi_iin_v0/tools/run_frame_planner_ablation_supervisor.sh"
task_log="${task_root}/worker_logs/rebalance_5gpu.log"

pair_complete() {
  local episode_label="$1"
  local gpu_id="$2"
  [[ -s "${task_root}/ep${episode_label}_off_gpu${gpu_id}/full_result.json" ]] &&
    [[ -s "${task_root}/ep${episode_label}_on_gpu${gpu_id}/full_result.json" ]]
}

echo "WAIT gpu7 ep000 pair $(date '+%F %T')" >>"${task_log}"
while ! pair_complete 000 7; do
  sleep 10
done
echo "REBALANCE gpu7 -> episodes 0,2 $(date '+%F %T')" >>"${task_log}"
tmux respawn-window -k -t "${task_session}:gpu7" \
  "cd /nas_ds925/tyj/IEVE && exec ${task_supervisor} 7 0 2"

echo "WAIT gpu6 ep001 pair $(date '+%F %T')" >>"${task_log}"
while ! pair_complete 001 6; do
  sleep 10
done
echo "REBALANCE gpu6 -> episodes 1,3 $(date '+%F %T')" >>"${task_log}"
tmux respawn-window -k -t "${task_session}:gpu6" \
  "cd /nas_ds925/tyj/IEVE && exec ${task_supervisor} 6 1 3"

echo "DONE $(date '+%F %T')" >>"${task_log}"
