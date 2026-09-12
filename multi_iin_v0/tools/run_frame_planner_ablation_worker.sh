#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID EPISODE_INDEX [EPISODE_INDEX ...]" >&2
  exit 2
fi

task_gpu="$1"
shift
task_root="multi_iin_v0/results/experiments/frame_planner_ablation_10ep_20260908"
task_python="/code1/tyj/anaconda3/envs/ieve/bin/python"
mkdir -p "${task_root}/worker_logs"

run_one() {
  local episode_index="$1"
  local mode="$2"
  local enabled=0
  if [[ "${mode}" == "on" ]]; then
    enabled=1
  fi
  local episode_label
  episode_label="$(printf '%03d' "${episode_index}")"
  local run_name="ep${episode_label}_${mode}_gpu${task_gpu}"
  local run_root="${task_root}/${run_name}"
  local result_json="${run_root}/full_result.json"
  local launcher_log="${task_root}/worker_logs/${run_name}.launcher.log"
  if [[ -s "${result_json}" ]]; then
    echo "SKIP ${run_name}: result exists"
    return 0
  fi
  mkdir -p "${run_root}"
  local attempt=1
  local max_attempts=5
  local exit_code=1
  local transient_gpu_failure=0
  while (( attempt <= max_attempts )); do
    echo "START ${run_name} attempt=${attempt}/${max_attempts} $(date '+%F %T')"
    "${task_python}" -u multi_iin_v0/runtime/main_multi_iin.py \
    --eval 1 --split val --exploration_policy frontier \
    --num_processes 1 --num_eval_episodes 1 --num_training_frames 2500 \
    --eval_skip_episodes 0 \
    --multi_manifest multi_iin_v0/manifests/miin_v0_val_k2.json \
    --multi_episode_start "${episode_index}" --multi_num_episodes 1 \
    --multi_goal_step_budget 1000 --multi_variant Continuous-Shared \
    --multi_result_json "${result_json}" \
    --enable_3dgs 1 --use_minav_confirmation_localization 1 \
    --minav_initial_spin 0 --goal_cat_source oracle --auto_gpu_config 0 \
    --sim_gpu_id "${task_gpu}" --sem_gpu_id "${task_gpu}" \
    --dump_location "${run_root}" --exp_name "${run_name}" \
    --minav_results_dir "${run_root}/minav" \
    --multi_compact_results_dir "${run_root}/compact" \
    --multi_save_compact_results 0 --minav_save_stage_visualizations 0 \
    --print_images 0 --quiet_log 1 --log_interval 1000 \
    --3dgs_render_debug 0 --frontier_debug_log 0 \
    --minav_confirm_match_threshold 20 \
    --minav_localization_match_threshold 40 \
    --minav_mr_closed_loop_threshold 40 \
    --multi_minav_reuse_frame_maps "${enabled}" \
      --multi_fmm_exact_cache "${enabled}" \
      >"${launcher_log}" 2>&1
    exit_code=$?
    if (( exit_code == 0 )); then
      break
    fi
    if rg -qi \
      'no EGL devices|NVIDIA.*driver|CUDA.*driver|cudaGetDeviceCount|libcuda' \
      "${launcher_log}"; then
      transient_gpu_failure=1
    else
      transient_gpu_failure=0
    fi
    if (( attempt >= max_attempts )) || (( transient_gpu_failure == 0 )); then
      break
    fi
    echo "RETRY ${run_name}: transient GPU driver/EGL failure"
    sleep 15
    attempt=$((attempt + 1))
  done
  echo "END ${run_name} exit=${exit_code} attempts=${attempt} $(date '+%F %T')"
  if (( exit_code != 0 && transient_gpu_failure == 1 )); then
    return 75
  fi
  return "${exit_code}"
}

worker_failures=0
for episode_index in "$@"; do
  if (( episode_index % 2 == 0 )); then
    task_modes=(off on)
  else
    task_modes=(on off)
  fi
  for mode in "${task_modes[@]}"; do
    run_one "${episode_index}" "${mode}"
    run_code=$?
    if (( run_code != 0 )); then
      worker_failures=$((worker_failures + 1))
      # Let the persistent supervisor wait for driver/EGL recovery instead of
      # hammering every remaining episode while the device is unavailable.
      if (( run_code == 75 )); then
        exit 75
      fi
    fi
  done
done

exit "${worker_failures}"
