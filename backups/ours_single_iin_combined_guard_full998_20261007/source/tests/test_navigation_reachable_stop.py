import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from skimage.morphology import disk

from ieve_minav.manager import IEVETargetMode, MinAVConfirmationLocalizationManager
from ieve_minav.localizer.localization_system import LocalizationSystem
from agents.instance_exp import Instance_Exp_Env_Agent
from envs.utils.ieve_reachability import select_reachable_goal_ieve


def _obstacle_disk(shape, center, radius_cells):
    rows, cols = np.indices(shape)
    return (
        np.hypot(rows - center[0], cols - center[1]) <= radius_cells
    ).astype(np.float32)


def _completion_manager(*, phase3=False, patience=2):
    manager = MinAVConfirmationLocalizationManager.__new__(
        MinAVConfirmationLocalizationManager
    )
    manager.args = SimpleNamespace(map_resolution=5.0, success_dist=1.0, turn_angle=30.0)
    manager.mode = IEVETargetMode.NAVIGATION
    manager.localized_target_position = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    manager.confirmed_target_position = None
    manager.navigation_goal_raw_world = manager.localized_target_position.copy()
    manager.navigation_goal_world = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
    manager.navigation_goal_tier = "preferred"
    manager.navigation_effective_stop_radius_m = 0.5
    manager.navigation_may_terminal_stop = True
    manager.navigation_complete = False
    manager.navigation_fallback_force_stop = False
    manager.navigation_fallback_force_stop_reason = None
    manager.navigation_staging_observation_pending = False
    manager.navigation_staging_replan_count = 0
    manager.navigation_staging_replan_limit = 1
    manager.navigation_execution_goal_dilation_radius = 1
    manager.navigation_phase3_tolerance_hold = False
    manager.navigation_goal_local_cell = (10, 10)
    manager.navigation_goal_full_cell = (100, 116)
    manager.navigation_raw_target_full_cell = (100, 100)
    manager.navigation_goal_adjusted = True
    manager.navigation_arrival_threshold = 1.0
    manager.localization_stage = (
        "phase3_best_evidence_navigation" if phase3 else "localized"
    )
    manager.strategy_phase3_fallback_stop_radius = 0.5
    manager.phase3_navigation_stuck_tolerance_m = 0.1
    manager.phase3_navigation_progress_epsilon_m = 0.02
    manager.phase3_navigation_stuck_patience = int(patience)
    manager.phase3_navigation_best_distance = float("inf")
    manager.phase3_navigation_stuck_frames = 0
    manager.phase3_navigation_stuck_tolerance_active = False
    manager._trace = lambda *_args, **_kwargs: None
    return manager


class ReachableGoalSelectorTest(unittest.TestCase):
    def test_open_target_selects_preferred_goal(self):
        result = select_reachable_goal_ieve(
            np.zeros((81, 81), dtype=np.float32),
            (40, 10),
            (40, 40),
            map_resolution=5.0,
        )
        self.assertTrue(result["selected"], result)
        self.assertEqual(result["goal_tier"], "preferred")
        self.assertEqual(result["goal_cell"], (40, 40))
        self.assertEqual(result["effective_stop_radius_m"], 0.5)
        self.assertTrue(result["may_terminal_stop"])
        self.assertTrue(np.isfinite(result["obstacle_clearance_m"]))

    def test_blocked_half_meter_selects_fallback_goal(self):
        shape = (101, 101)
        result = select_reachable_goal_ieve(
            _obstacle_disk(shape, (50, 50), 10),
            (50, 10),
            (50, 50),
            map_resolution=5.0,
        )
        self.assertTrue(result["selected"], result)
        self.assertEqual(result["goal_tier"], "fallback")
        self.assertGreater(result["target_center_distance_m"], 0.5)
        self.assertLessEqual(result["target_center_distance_m"], 1.0)
        self.assertEqual(result["effective_stop_radius_m"], 1.0)

    def test_only_outer_ring_selects_non_terminal_staging_goal(self):
        shape = (121, 121)
        result = select_reachable_goal_ieve(
            _obstacle_disk(shape, (60, 60), 20),
            (60, 10),
            (60, 60),
            map_resolution=5.0,
            max_radius_m=2.0,
            ring_step_m=0.25,
        )
        self.assertTrue(result["selected"], result)
        self.assertEqual(result["goal_tier"], "staging")
        self.assertGreater(result["target_center_distance_m"], 1.0)
        self.assertIsNone(result["effective_stop_radius_m"])
        self.assertFalse(result["may_terminal_stop"])

    def test_disconnected_target_region_is_not_selected(self):
        grid = np.zeros((61, 61), dtype=np.float32)
        grid[:, 30] = 1.0
        result = select_reachable_goal_ieve(
            grid,
            (30, 5),
            (30, 50),
            map_resolution=5.0,
            preferred_radius_m=0.25,
            fallback_radius_m=0.5,
            max_radius_m=0.5,
        )
        self.assertFalse(result["selected"], result)
        self.assertEqual(result["reason"], "target_region_disconnected")

    def test_collision_map_can_block_the_target_region(self):
        shape = (61, 61)
        result = select_reachable_goal_ieve(
            np.zeros(shape, dtype=np.float32),
            (30, 5),
            (30, 45),
            collision_map=_obstacle_disk(shape, (30, 45), 10),
            map_resolution=5.0,
            preferred_radius_m=0.25,
            fallback_radius_m=0.5,
            max_radius_m=0.5,
        )
        self.assertFalse(result["selected"], result)
        self.assertEqual(result["reason"], "target_region_blocked_by_collision_map")

    def test_collision_map_path_cut_is_reported_separately(self):
        shape = (61, 61)
        collision = np.zeros(shape, dtype=np.float32)
        collision[:, 30] = 1.0
        result = select_reachable_goal_ieve(
            np.zeros(shape, dtype=np.float32),
            (30, 5),
            (30, 50),
            collision_map=collision,
            map_resolution=5.0,
            preferred_radius_m=0.25,
            fallback_radius_m=0.5,
            max_radius_m=0.5,
        )
        self.assertFalse(result["selected"], result)
        self.assertEqual(result["reason"], "path_blocked_by_collision_map")


class ReachableGoalExecutorConstraintTest(unittest.TestCase):
    def test_goal_dilation_is_clipped_to_high_level_goal_region(self):
        captured = {}

        class _Planner:
            def __init__(self, traversible):
                self.fmm_dist = np.ones_like(traversible, dtype=np.float32)

            def set_multi_goal(self, goal):
                captured["goal"] = np.asarray(goal).copy()
                return True

            def get_short_term_goal(self, state, *_args):
                return float(state[0]), float(state[1]), False, False

        agent = Instance_Exp_Env_Agent.__new__(Instance_Exp_Env_Agent)
        agent.selem = disk(1)
        agent.visited = np.zeros((21, 21), dtype=np.float32)
        agent.collision_map = np.zeros((21, 21), dtype=np.float32)
        agent.frontier_representative_points = []
        agent.frontier_active_goal = None
        agent.global_goal = None
        agent.temp_goal = None
        agent.stuck_goal = None
        agent.args = SimpleNamespace(map_resolution=5)
        goal = np.zeros((21, 21), dtype=np.float32)
        goal[10, 15] = 1.0
        allowed = np.zeros_like(goal, dtype=bool)
        allowed[10, 15] = True

        with patch("agents.instance_exp.FMMPlanner", _Planner):
            agent._get_stg(
                np.zeros_like(goal),
                [10, 5],
                goal,
                [0, 21, 0, 21],
                goal_dilation_radius=3,
                goal_region_constraint=allowed,
            )

        self.assertEqual(int(np.count_nonzero(captured["goal"])), 1)
        self.assertEqual(float(captured["goal"][11, 16]), 1.0)


class LocalizerRawTargetTest(unittest.TestCase):
    def test_localizer_returns_raw_cluster_center_separately(self):
        localizer = LocalizationSystem.__new__(LocalizationSystem)
        localizer.refined_pose = np.eye(4, dtype=np.float64)
        localizer.best_pose = None
        localizer.state_history = []
        localizer._select_target_gaussian_cluster_from_ref_view = lambda _pose: (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            {
                "center": np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                "source": "test_cluster",
                "reachable": True,
            },
        )
        localizer._select_target_object_fallback_position = lambda: (None, None)

        with contextlib.redirect_stdout(io.StringIO()):
            result = localizer._make_localization_complete_result("test")

        np.testing.assert_allclose(result["target_position"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            result["target_object_center"], [2.0, 0.0, 0.0]
        )


class ReachableNavigationCompletionTest(unittest.TestCase):
    def test_normal_completion_uses_raw_target_not_execution_goal(self):
        manager = _completion_manager()
        manager.navigation_goal_world = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
        manager._update_navigation_completion(
            {
                "step": 1,
                "agent_position": np.asarray([100.0, 0.0, 100.0]),
                "pose_pred": np.asarray(
                    [5.45, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "map_resolution_cm": 5.0,
            }
        )
        self.assertTrue(manager.navigation_complete)

    def test_staging_arrival_requests_replan_without_completing(self):
        manager = _completion_manager()
        manager.navigation_goal_tier = "staging"
        manager.navigation_effective_stop_radius_m = None
        manager.navigation_may_terminal_stop = False
        manager.navigation_goal_world = np.asarray([1.25, 0.0, 0.0], dtype=np.float32)
        manager._update_navigation_completion(
            {
                "step": 1,
                "agent_position": np.asarray([1.25, 0.0, 0.0]),
                "pose_pred": np.asarray(
                    [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "map_resolution_cm": 5.0,
                "minav_fmm_stop": True,
            }
        )
        self.assertFalse(manager.navigation_complete)
        self.assertTrue(manager.navigation_staging_observation_pending)

    def test_staging_replan_still_staging_emits_explicit_failure_stop(self):
        manager = _completion_manager()
        manager.navigation_goal_tier = "staging"
        manager.navigation_may_terminal_stop = False
        manager.navigation_staging_observation_pending = True
        manager._navigation_staging_face_action = lambda *_args, **_kwargs: None
        manager._select_reachable_navigation_goal = lambda *_args, **_kwargs: {
            "selected": True,
            "goal_tier": "staging",
        }
        result = manager._get_reachable_navigation_goal_override({"step": 2})
        self.assertTrue(result["force_stop"])
        self.assertEqual(
            result["force_stop_reason"],
            "localized_target_unreachable_fallback_stop",
        )
        self.assertFalse(manager.navigation_complete)

    def test_goal_selection_can_complete_without_an_extra_action_frame(self):
        manager = _completion_manager()
        goal_map = np.zeros((5, 5), dtype=np.float32)
        goal_map[2, 2] = 1.0
        manager._select_reachable_navigation_goal = lambda *_args, **_kwargs: {
            "selected": True,
            "raw_target_world": np.asarray([0.0, 0.0, 0.0]),
            "execution_goal_world": np.asarray([0.2, 0.0, 0.0]),
            "goal_cell": (2, 2),
            "full_cell": (2, 2),
            "raw_full_cell": (2, 2),
            "goal_map": goal_map,
            "goal_region_mask": np.ones_like(goal_map, dtype=bool),
            "goal_tier": "preferred",
            "effective_stop_radius_m": 0.5,
            "may_terminal_stop": True,
        }
        result = manager._get_reachable_navigation_goal_override(
            {
                "step": 1,
                "agent_position": np.asarray([0.4, 0.0, 0.0]),
                "pose_pred": np.asarray(
                    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                "map_resolution_cm": 5.0,
            }
        )
        self.assertTrue(manager.navigation_complete)
        self.assertEqual(result["goal_source"], "MINAV_NAVIGATION_REACHABLE_PREFERRED")

    def test_phase3_fallback_holds_until_stuck_tolerance_activates(self):
        manager = _completion_manager(phase3=True, patience=2)
        manager.navigation_goal_tier = "fallback"
        manager.navigation_goal_world = np.asarray([0.55, 0.0, 0.0], dtype=np.float32)
        payload = {
            "agent_position": np.asarray([0.55, 0.0, 0.0]),
            "pose_pred": np.asarray(
                [5.55, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "map_resolution_cm": 5.0,
            "minav_fmm_stop": True,
        }

        manager._update_navigation_completion(dict(payload, step=1))
        self.assertFalse(manager.navigation_complete)
        self.assertTrue(manager.navigation_phase3_tolerance_hold)

        manager._update_navigation_completion(dict(payload, step=2))
        self.assertFalse(manager.navigation_complete)
        manager._update_navigation_completion(dict(payload, step=3))
        self.assertTrue(manager.phase3_navigation_stuck_tolerance_active)
        self.assertTrue(manager.navigation_complete)
        self.assertFalse(manager.navigation_phase3_tolerance_hold)


if __name__ == "__main__":
    unittest.main()
