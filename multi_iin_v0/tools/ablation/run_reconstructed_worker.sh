#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID EPISODE_INDEX [EPISODE_INDEX ...]" >&2
  exit 2
fi

task_gpu="$1"
shift
task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
task_root="${MULTI_IIN_TASK_ROOT:-/nas_ds925/tyj/IEVE/multi_iin_v0/results/experiments/full100_compact_correctness_20260908}"
task_variant="${MULTI_IIN_VARIANT:-Continuous-Shared}"
task_python="${MULTI_IIN_PYTHON:-/code1/tyj/anaconda3/envs/ieve/bin/python}"
read -r -a task_extra_args <<< "${MULTI_IIN_EXTRA_ARGS:-}"

case "${task_variant}" in
  Independent-Reset|Continuous-Clear|Continuous-Shared) ;;
  *)
    echo "invalid MULTI_IIN_VARIANT: ${task_variant}" >&2
    exit 2
    ;;
esac

mkdir -p "${task_root}/worker_logs"

result_complete() {
  local result_json="$1"
  [[ -s "${result_json}" ]] || return 1
  "${task_python}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert len(d.get("episodes",[]))==1 and len(d["episodes"][0].get("goals",[]))==2' \
    "${result_json}" >/dev/null 2>&1
}

gpu_ready() {
  nvidia-smi -i "${task_gpu}" --query-gpu=index --format=csv,noheader,nounits \
    >/dev/null 2>&1
}

run_one() {
  local episode_index="$1"
  local episode_label
  episode_label="$(printf '%03d' "${episode_index}")"
  local run_name="ep${episode_label}_gpu${task_gpu}"
  local run_root="${task_root}/${run_name}"
  local result_json="${run_root}/full_result.json"
  local launcher_log="${task_root}/worker_logs/${run_name}.launcher.log"
  if result_complete "${result_json}"; then
    echo "SKIP ${run_name}: complete result exists"
    return 0
  fi
  mkdir -p "${run_root}"

  local attempt=0
  while true; do
    while ! gpu_ready; do
      echo "WAIT ${run_name}: GPU ${task_gpu} driver/device unavailable $(date '+%F %T')"
      sleep 30
    done
    attempt=$((attempt + 1))
    echo "START ${run_name} attempt=${attempt} $(date '+%F %T')"
    "${task_python}" -u "${task_repo}/multi_iin_v0/runtime/main_multi_iin.py" \
      --eval 1 --split val --exploration_policy frontier \
      --num_processes 1 --num_eval_episodes 1 --num_training_frames 2500 \
      --eval_skip_episodes 0 \
      --multi_manifest "${task_repo}/multi_iin_v0/manifests/miin_v0_val_k2.json" \
      --multi_episode_start "${episode_index}" --multi_num_episodes 1 \
      --multi_goal_step_budget 1000 --multi_variant "${task_variant}" \
      --multi_result_json "${result_json}" \
      --enable_3dgs 1 --use_minav_confirmation_localization 1 \
      --minav_initial_spin 0 --goal_cat_source oracle --auto_gpu_config 0 \
      --sim_gpu_id "${task_gpu}" --sem_gpu_id "${task_gpu}" \
      --dump_location "${run_root}" --exp_name "${run_name}" \
      --minav_results_dir "${run_root}/minav" \
      --multi_compact_results_dir "${run_root}/compact" \
      --multi_save_compact_results 1 --minav_save_stage_visualizations 1 \
      --multi_save_repeated_full_gaussian_snapshots 0 \
      --print_images 0 --quiet_log 1 --log_interval 1000 \
      --3dgs_render_debug 0 --frontier_debug_log 0 \
      --minav_confirm_match_threshold 20 \
      --minav_localization_match_threshold 40 \
      --minav_mr_closed_loop_threshold 40 \
      --multi_minav_reuse_frame_maps 1 --multi_fmm_exact_cache 1 \
      --multi_phase3_identity_first 1 \
      "${task_extra_args[@]}" \
      >"${launcher_log}" 2>&1
    local exit_code=$?
    if (( exit_code == 0 )) && result_complete "${result_json}"; then
      echo "END ${run_name} complete attempt=${attempt} $(date '+%F %T')"
      return 0
    fi
    if rg -qi \
      'no EGL devices|EGL_NOT_INITIALIZED|NVIDIA.*driver|Driver Not Loaded|/dev/nvidia|CUDA.*driver|cudaGetDeviceCount|libcuda|out of memory|CUDA error' \
      "${launcher_log}"; then
      echo "RETRY ${run_name}: transient GPU/CUDA/EGL failure exit=${exit_code}; wait 30s"
      sleep 30
      continue
    fi
    echo "FAIL ${run_name}: non-transient exit=${exit_code}; see ${launcher_log}"
    return "${exit_code:-1}"
  done
}

failures=0
for episode_index in "$@"; do
  run_one "${episode_index}" || failures=$((failures + 1))
done
exit "${failures}"
