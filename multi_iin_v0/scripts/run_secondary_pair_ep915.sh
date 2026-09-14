#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: $0 enabled|disabled GPU_ID OUTPUT_ROOT [EPISODE_ID] [CASE_NAME] [PHASE2_STEP]}"
gpu_id="${2:?missing GPU_ID}"
output_root="${3:?missing OUTPUT_ROOT}"
episode_id="${4:-914}"
case_name="${5:-ep915}"
phase2_step="${6:-400}"

case "$mode" in
  enabled) secondary=1 ;;
  disabled) secondary=0 ;;
  *) echo "mode must be enabled or disabled" >&2; exit 2 ;;
esac

mkdir -p "$output_root/results"
cd /code3/tyj/IEVE
export CUDA_VISIBLE_DEVICES="$gpu_id"
export CUDA_LAUNCH_BLOCKING=0
export DISPLAY=:78
export XAUTHORITY=/home/tyj/.Xauthority

/code1/tyj/anaconda3/envs/ieve/bin/python -u main.py \
  --eval 1 --num_processes 1 \
  --task_config tasks/instance_imagenav.yaml --split val --version v3 \
  --load pretrained_models/sem_exp.pth \
  --dump_location "$output_root/results" --minav_results_dir "$output_root/results" \
  --auto_gpu_config 0 --sim_gpu_id 0 --sem_gpu_id -1 \
  --enable_3dgs 1 --use_minav_confirmation_localization 1 \
  --minav_initial_spin 1 --minav_target_detect_window 2 \
  --minav_enable_strategy_transition "$secondary" \
  --minav_strategy_phase2_start_step "$phase2_step" --minav_strategy_phase3_start_step 800 \
  --minav_strategy_use_area_growth_trigger 0 \
  --minav_phase2_arrival_threshold 0.35 \
  --minav_phase2_standoff_retry_limit 3 --minav_phase2_standoff_stuck_patience 8 \
  --minav_phase2_standoff_retry_block_radius 0.45 \
  --minav_phase2_no_reachable_cooldown_steps 40 --minav_phase2_no_reachable_retry_limit 3 \
  --minav_strategy_phase3_fallback_stop_radius 0.5 \
  --minav_targeted_quick_lock_max_dist_m 1.5 \
  --minav_navigation_arrival_threshold 1.0 \
  --minav_buffer_goal_dilation_radius 10 --minav_navigation_goal_dilation_radius 5 \
  --minav_stg_angle_fix 0 --minav_sweep_turn_in_place_fix 1 \
  --exploration_policy frontier --frontier_visited_path_block_radius_m 0.5 \
  --frontier_debug_log 1 --print_images 1 --visualize 1 \
  --target_instance_debug_dump 1 --minav_planner_debug 1 \
  --eval_habitat_episode_id "$episode_id" --num_eval_episodes 1 \
  --exp_name "paper_${case_name}_secondary_${mode}" \
  2>&1 | tee "$output_root/run.log"
