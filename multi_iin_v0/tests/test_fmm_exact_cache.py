from types import SimpleNamespace

import numpy as np
from skimage.morphology import disk

# Initialize the repository's env package through the same import order as the
# runtime; importing agents.instance_exp first triggers the legacy envs/agents
# circular package initializer.
from ieve_minav.manager import IEVETargetMode  # noqa: F401
from agents.instance_exp import Instance_Exp_Env_Agent


def _agent(cache_enabled=True):
    agent = Instance_Exp_Env_Agent.__new__(Instance_Exp_Env_Agent)
    agent.selem = disk(1)
    agent.visited = np.zeros((31, 31), dtype=np.float32)
    agent.collision_map = np.zeros((31, 31), dtype=np.float32)
    agent.frontier_representative_points = []
    agent.frontier_active_goal = None
    agent.global_goal = None
    agent.temp_goal = None
    agent.stuck_goal = None
    agent.args = SimpleNamespace(
        map_resolution=5,
        multi_fmm_exact_cache=int(cache_enabled),
    )
    agent._multi_fmm_exact_cache = None
    agent._multi_fmm_cache_hits = 0
    agent._multi_fmm_cache_misses = 0
    agent._last_multi_fmm_cache_hit = False
    return agent


def _plan(agent, grid, goal, start):
    return agent._get_stg(
        grid,
        start,
        goal,
        [0, grid.shape[0], 0, grid.shape[1]],
        goal_dilation_radius=0,
    )


def test_exact_fmm_cache_reuses_only_identical_grids_and_preserves_result():
    grid = np.zeros((31, 31), dtype=np.float32)
    grid[5:25, 16] = 1.0
    grid[15, 16] = 0.0
    goal = np.zeros_like(grid)
    goal[15, 26] = 1.0

    cached = _agent(cache_enabled=True)
    uncached = _agent(cache_enabled=False)

    first = _plan(cached, grid, goal, [15, 5])
    reference_first = _plan(uncached, grid, goal, [15, 5])
    assert first == reference_first
    assert cached._multi_fmm_cache_misses == 1
    assert cached._multi_fmm_cache_hits == 0

    second = _plan(cached, grid.copy(), goal.copy(), [15, 6])
    reference_second = _plan(uncached, grid, goal, [15, 6])
    assert second == reference_second
    assert cached._last_multi_fmm_cache_hit is True
    assert cached._multi_fmm_cache_hits == 1

    changed_goal = np.zeros_like(goal)
    changed_goal[14, 26] = 1.0
    changed = _plan(cached, grid, changed_goal, [15, 6])
    reference_changed = _plan(uncached, grid, changed_goal, [15, 6])
    assert changed == reference_changed
    assert cached._last_multi_fmm_cache_hit is False
    assert cached._multi_fmm_cache_misses == 2
