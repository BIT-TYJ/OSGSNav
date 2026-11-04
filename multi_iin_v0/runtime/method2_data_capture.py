"""Optional read-only FMM diagnostics for method-figure data collection."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from multi_iin_v0.runtime.multi_env import _make_multi_env as make_original_env


def make_env_with_fmm_capture(args, config_env, rank, specs):
    env = make_original_env(args, config_env, rank, specs)
    output = Path(os.environ['METHOD2_FMM_CAPTURE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    original = env._get_stg

    def observed_get_stg(grid, start, goal, planning_window, *extra, **kwargs):
        result = original(grid, start, goal, planning_window, *extra, **kwargs)
        cache = getattr(env, '_multi_fmm_exact_cache', None)
        if isinstance(cache, dict) and cache.get('fmm_dist') is not None:
            gx1,gx2,gy1,gy2 = [int(x) for x in planning_window]
            step = int(env.timestep)
            np.savez_compressed(
                output / f'fmm_before_action_step{step:04d}.npz',
                fmm_distance=cache['fmm_dist'], traversible=cache['traversible'],
                goal=cache['goal'], start_local_rc=np.asarray(start),
                planning_window_full_rc=np.asarray(planning_window),
                obstacle_input=np.asarray(grid),
                collision_map=env.collision_map[gx1:gx2,gy1:gy2],
                visited=env.visited[gx1:gx2,gy1:gy2],
                actual_short_term_goal_rc=np.asarray(result[0]),
                stop=np.asarray(result[1]),
            )
            with (output / 'fmm_trace.jsonl').open('a') as handle:
                handle.write(json.dumps({
                    'before_action_env_step': step, 'start_local_rc': np.asarray(start).tolist(),
                    'short_term_goal_local_rc': np.asarray(result[0]).tolist(),
                    'frontier_active_goal': getattr(env, 'frontier_active_goal', None),
                    'padded_grid_border_cells': 1,
                    'note': 'Exact distance field used by the existing FMM controller, not an extra planning call',
                }) + '\n')
        return result

    env._get_stg = observed_get_stg
    return env
