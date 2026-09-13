#!/usr/bin/env bash
set -euo pipefail

task_repo="${MULTI_IIN_REPO:-/nas_ds925/tyj/IEVE}"
export MULTI_IIN_REPO="${task_repo}"
export MULTI_IIN_TMUX_SESSION="${MULTI_IIN_TMUX_SESSION:-pre99_three_part100}"
export MULTI_IIN_TASK_ROOT="${MULTI_IIN_TASK_ROOT:-${task_repo}/multi_iin_v0/results/experiments/ablation_wo_9p9_three_components_full100_20260912}"
export MULTI_IIN_EPISODE_COUNT="${MULTI_IIN_EPISODE_COUNT:-100}"
export MULTI_IIN_GPUS="${MULTI_IIN_GPUS:-0 1 2 3 4 5 6 7}"

# Only ablate the three behavior groups documented as added on 2026-09-09:
# 1) Phase-3 target-directed frontier exploration;
# 2) stable candidate retention / identity-first history ordering;
# 3) detection-anchor-first target-center selection (plant investigation).
# The target-center/distance audit fields are passive diagnostics and remain.
export MULTI_IIN_EXTRA_ARGS="--multi_phase3_target_frontier 0 --multi_phase3_identity_first 0 --multi_stable_detection_memory 0 --multi_confirm_anchor_first 0"

exec "${task_repo}/multi_iin_v0/tools/ablation/launch_reconstructed_tmux.sh"
