# UniGoal-M reference adapter

Independent **two-goal and five-goal Habitat adapters** for UniGoal. This directory does not implement a robot backend. Do not use `--real_world` on the copied entry as proof of real hardware support: the entry still creates a Habitat environment.

## Source and changes

- Dependency checkout: https://github.com/BIT-TYJ/omnigoal
- Pinned revision: `6e581d269e246d60cb1276b9af812f4d864eb2a4`
- K=2 source: original `baselines/unigoal_multi_iin/` in the experiment workspace.
- K=5 source: original `five_goal_benchmark_20261026/runtime/unigoal_*.py`.
- `adapter_provenance.json` records original and packaged hashes; `portability_changes.diff` shows namespacing and path-only changes. Effective `turn_angle` remains 15 degrees; duplicate YAML key is collapsed without changing its effective value.
- `upstream.lock.json` records actual dependency Python-source hashes. No upstream algorithm modifications or navigation reruns were performed for this upload.

The dependency is the actual fork used in the local experiments, not a claim that this is an unmodified official UniGoal release. Follow its upstream notices and installation instructions; model/data assets are not bundled.

## Install and preflight

Use a separate UniGoal-compatible environment; the OSGSNav environment is not an independently verified replacement. Install the upstream dependencies, Habitat, vision models and LLM/VLM services according to the pinned checkout. `requirements-upstream-snapshot.txt` is an upstream requirements snapshot, not a tested environment lock.

```bash
git clone https://github.com/BIT-TYJ/omnigoal.git external/UniGoal
git -C external/UniGoal checkout 6e581d269e246d60cb1276b9af812f4d864eb2a4
# Install upstream dependencies and required model weights separately.
python baselines/unigoal_m/run.py --unigoal-root external/UniGoal \
  --goals 5 --gpu 0 --episode 0 --output results/unigoal_m_ep000
```

Without `--execute` this verifies the dependency revision/source hashes and prints the resolved command; it imports no Habitat/CUDA/model modules and does not start navigation. After assets and environment setup, append `--execute` to run a reference simulation. `--goals 2` selects the original two-goal manifest. Output must be fresh. The launcher exposes only the requested GPU and retains the upstream working directory for its relative assets.

## Memory and evaluation contract

At a target switch, preserve pose, BEV/collision/visited map and observed scene-graph nodes/objects/edges. Reset current-goal matching, temporary navigation targets, graph target-conditioned scores/subgraphs and per-query budget. Advance to the next goal even if the previous one fails. Only the current goal image is delivered to the policy. All targets are different physical instances; same-category transitions are allowed.

The simulation adapter supplies task-provided categories, as did the reference comparison. For hardware, explicitly choose the same category-input protocol for both methods. Do not provide oracle object location, instance mask or goal depth to one method only.

## LLM/VLM

The reference YAML preserves `llama3.2:3b`, `llama3.2-vision:latest`, local Ollama placeholder key and localhost endpoint. These are historical defaults, not a frozen model-weight release. Before quantitative hardware runs pin model digests, prompts, quantization, request timeouts and retry rules. The endpoint can be overridden with `UNIGOAL_OLLAMA_BASE_URL`; retain model timing and include requests/network waits in task wall time.

## Tests and hardware handoff

`test_state.py` retains the original CPU tests for graph retargeting and display-map correction; they do not validate navigation or robot integration. The display correction does not mutate the planner's live map.

For Codex on the robot computer, start with [CODEX_REAL_ROBOT_README.md](../../docs/real_robot/CODEX_REAL_ROBOT_README.md) and the linked quantitative-experiment checklist. The existing dependency `real_world_env.py` is a TODO/zero-observation placeholder at the pinned source snapshot; real RGB-D, time-aligned pose, motion execution and independent evaluation remain to be integrated.
