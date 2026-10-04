# UniGoal sequential Multi-IIN adapter

This adapter leaves `/code1/tyj/UniGoal` unchanged and runs two ordered image
goals in one Habitat episode.

At `target1 -> target2`, it retains the simulator pose, BEV map, collision and
visited planning memory, and observed scene graph. It resets only goal-specific
matching state and scene-graph scores, then forces an immediate replan for the
new goal. The visualization trajectory is cleared only for display, so target2
terminal maps contain the shared map but only the target2 path.

Each target terminal state exports both a fixed 720x720 global-coordinate map
and a tight paper-friendly crop under `map_snapshots/<episode_id>/`.

Run one episode:

```bash
UNIGOAL_MULTI_OUTPUT_ROOT=/path/to/output \
  baselines/unigoal_multi_iin/run_full100_worker.sh 0 20 1
```

Run all 100 episodes on eight GPUs:

```bash
UNIGOAL_MULTI_OUTPUT_ROOT=/path/to/output \
  baselines/unigoal_multi_iin/launch_full100_tmux.sh
```
