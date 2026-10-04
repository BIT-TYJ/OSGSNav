#!/usr/bin/env bash
set -euo pipefail
ROOT=/nas_ds925/tyj/IEVE
UNI=/code1/tyj/UniGoal
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="$UNI:$UNI/third_party/Grounded-Segment-Anything:$UNI/third_party/Grounded-Segment-Anything/segment-anything:$ROOT/baselines/unigoal_multi_iin:${PYTHONPATH:-}"
# graph.py resolves this dependency with a relative path.
cd "$UNI"
exec /code1/tyj/anaconda3/envs/uninav/bin/python -u "$ROOT/baselines/unigoal_multi_iin/main_unigoal_multi.py" \
  --config-file "$ROOT/baselines/unigoal_multi_iin/config_habitat_unigoal_multi.yaml" \
  --multi_manifest "$ROOT/multi_iin_v0/manifests/miin_v0_val_k2.json" \
  --multi_episode_start "${MULTI_EPISODE_START:-0}" \
  --multi_num_episodes "${MULTI_NUM_EPISODES:-1}" \
  --multi_goal_step_budget "${MULTI_GOAL_STEP_BUDGET:-1000}"
