from types import SimpleNamespace
import unittest

import numpy as np

from ieve_minav.localizer.localization_system import (
    LocalizationState,
    LocalizationSystem,
)
from ieve_minav.localizer.novel_view_renderer import NovelViewRenderer
from ieve_minav.manager import MinAVConfirmationLocalizationManager
from agents.instance_exp import Instance_Exp_Env_Agent


class _GridAdapter:
    def __init__(self, center_cell=70, resolution_m=0.05):
        self.center_cell = int(center_cell)
        self.resolution_m = float(resolution_m)

    def update(self, _payload):
        return None

    def map_to_world(self, col, row):
        return (
            (float(col) - self.center_cell) * self.resolution_m,
            (float(row) - self.center_cell) * self.resolution_m,
        )


def _route_manager():
    manager = MinAVConfirmationLocalizationManager.__new__(
        MinAVConfirmationLocalizationManager
    )
    manager.args = SimpleNamespace(map_resolution=5)
    manager.cluster_recovery_max_radius_m = 3.0
    manager.cluster_recovery_waypoint_count = 12
    manager.cluster_recovery_approach_inner_radius_m = 2.5
    manager.cluster_recovery_approach_outer_radius_m = 2.7
    manager.cluster_recovery_cell_arrival_radius_m = 0.15
    manager.occupancy_adapter = _GridAdapter()

    def project(world, _payload):
        world = np.asarray(world, dtype=np.float32)
        cell = (
            int(round(70 + float(world[2]) / 0.05)),
            int(round(70 + float(world[0]) / 0.05)),
        )
        return {"raw_full_cell": cell, "target_full_cell": cell}

    manager._buffer_project_world_to_local = project
    return manager


def _pose_pred_for_cell(row, col, resolution_m=0.05):
    return np.asarray(
        [
            float(col) * resolution_m,
            float(row) * resolution_m,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


class ClusterRecoveryRouteTest(unittest.TestCase):
    def test_fmm_route_is_closed_winds_once_and_stays_within_three_meters(self):
        manager = _route_manager()
        shape = (141, 141)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 110),
            "map_resolution_cm": 5.0,
        }
        plan = manager._plan_cluster_recovery_closed_route(
            payload, np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )

        self.assertTrue(plan.get("success"), plan)
        route = np.asarray(plan["route_cells"], dtype=np.float32)
        self.assertTrue(np.array_equal(route[0], route[-1]))
        self.assertAlmostEqual(abs(float(plan["winding"])), 1.0, delta=0.2)
        radii_m = np.hypot(route[:, 0] - 70, route[:, 1] - 70) * 0.05
        self.assertLessEqual(float(radii_m.max()), 3.0 + 1e-6)
        self.assertEqual(len(plan["route_segments"]), 12)
        self.assertEqual(len(plan["waypoint_cells"]), 12)
        self.assertTrue(plan["closed_curve_validated"])

    def test_closed_route_validator_rejects_open_or_non_winding_paths(self):
        closed = {
            "center_cell": (10, 10),
            "route_cells": [(5, 10), (10, 15), (15, 10), (10, 5), (5, 10)],
        }
        self.assertTrue(
            MinAVConfirmationLocalizationManager
            ._validate_cluster_recovery_closed_plan(closed)["valid"]
        )

        open_route = dict(closed)
        open_route["route_cells"] = closed["route_cells"][:-1]
        result = (
            MinAVConfirmationLocalizationManager
            ._validate_cluster_recovery_closed_plan(open_route)
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "route_not_closed")

        no_winding = dict(closed)
        no_winding["route_cells"] = [
            (5, 10), (10, 15), (5, 10), (10, 5), (5, 10)
        ]
        result = (
            MinAVConfirmationLocalizationManager
            ._validate_cluster_recovery_closed_plan(no_winding)
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "route_winding_not_one")

    def test_route_is_rejected_when_start_is_outside_three_meter_domain(self):
        manager = _route_manager()
        shape = (161, 161)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 140),
            "map_resolution_cm": 5.0,
        }
        plan = manager._plan_cluster_recovery_closed_route(
            payload, np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertFalse(plan.get("success"))
        self.assertEqual(plan.get("reason"), "start_outside_3m_constraint")

    def test_fmm_closed_route_deforms_around_obstacles(self):
        manager = _route_manager()
        shape = (141, 141)
        obstacle = np.zeros(shape, dtype=np.float32)
        obstacle[38:65, 88:101] = 1.0
        payload = {
            "full_map_pred": obstacle,
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 110),
            "map_resolution_cm": 5.0,
        }
        plan = manager._plan_cluster_recovery_closed_route(
            payload, np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(plan.get("success"), plan)
        route = np.asarray(plan["route_cells"], dtype=np.int32)
        self.assertFalse(np.any(obstacle[route[:, 0], route[:, 1]] > 0))
        self.assertAlmostEqual(abs(float(plan["winding"])), 1.0, delta=0.2)
        self.assertLessEqual(float(plan["max_node_radius_m"]), 3.0 + 1e-6)

    def test_route_failure_cache_matches_only_same_cluster_until_map_changes(self):
        manager = _route_manager()
        manager.cluster_recovery_failure_cluster_center_tolerance_m = 0.25
        manager.cluster_recovery_failure_cluster_overlap_ratio = 0.60
        manager.cluster_recovery_failure_map_change_ratio = 0.05
        manager.cluster_recovery_route_failure_cache = []
        shape = (141, 141)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.zeros(shape, dtype=np.float32),
            "map_resolution_cm": 5.0,
        }
        signature = {
            "center": np.zeros(3, dtype=np.float32),
            "voxels": frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
            "gaussian_count": 30,
        }
        manager._cache_cluster_recovery_route_failure(
            payload, signature, "missing_angular_anchor", 10
        )
        self.assertIsNotNone(
            manager._cached_cluster_recovery_route_failure(payload, signature)
        )

        nearby_different_cluster = {
            "center": np.asarray([0.05, 0.0, 0.05], dtype=np.float32),
            "voxels": frozenset({(20, 0, 20), (21, 0, 20), (22, 0, 20)}),
            "gaussian_count": 30,
        }
        self.assertIsNone(
            manager._cached_cluster_recovery_route_failure(
                payload, nearby_different_cluster
            )
        )

        changed_payload = dict(payload)
        changed_payload["full_exp_pred"] = np.ones(shape, dtype=np.float32)
        self.assertIsNone(
            manager._cached_cluster_recovery_route_failure(
                changed_payload, signature
            )
        )

    def test_changed_second_cache_entry_is_deleted_without_numpy_equality(self):
        manager = _route_manager()
        manager.cluster_recovery_failure_cluster_center_tolerance_m = 0.25
        manager.cluster_recovery_failure_cluster_overlap_ratio = 0.60
        manager.cluster_recovery_failure_map_change_ratio = 0.05
        manager.cluster_recovery_route_failure_cache = []
        shape = (141, 141)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.zeros(shape, dtype=np.float32),
            "map_resolution_cm": 5.0,
        }
        first_signature = {
            "center": np.asarray([5.0, 0.0, 5.0], dtype=np.float32),
            "voxels": frozenset({(50, 0, 50), (51, 0, 50), (52, 0, 50)}),
            "gaussian_count": 30,
        }
        matching_signature = {
            "center": np.zeros(3, dtype=np.float32),
            "voxels": frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
            "gaussian_count": 30,
        }
        manager._cache_cluster_recovery_route_failure(
            payload, first_signature, "first_failure", 5
        )
        manager._cache_cluster_recovery_route_failure(
            payload, matching_signature, "matching_failure", 10
        )
        self.assertEqual(len(manager.cluster_recovery_route_failure_cache), 2)

        changed_payload = dict(payload)
        changed_payload["full_exp_pred"] = np.ones(shape, dtype=np.float32)
        self.assertIsNone(
            manager._cached_cluster_recovery_route_failure(
                changed_payload, matching_signature
            )
        )
        self.assertEqual(len(manager.cluster_recovery_route_failure_cache), 1)
        self.assertEqual(
            manager.cluster_recovery_route_failure_cache[0]["reason"],
            "first_failure",
        )

    def test_runtime_replan_installs_an_independently_valid_closed_route(self):
        manager = _route_manager()
        manager.cluster_recovery_center = np.zeros(3, dtype=np.float32)
        manager.cluster_recovery_replan_count = 0
        manager.cluster_recovery_max_replans = 3
        manager.cluster_recovery_active_signature = None
        manager.cluster_recovery_route_failure_cache = []
        manager.cluster_recovery_replan_requested_reason = None
        manager._trace = lambda *_args, **_kwargs: None
        manager._skip_cluster_recovery_to_mr = lambda *_args, **_kwargs: self.fail(
            "valid closed-route replan must not fall back"
        )
        shape = (141, 141)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 110),
            "map_resolution_cm": 5.0,
        }
        self.assertTrue(
            manager._replan_cluster_recovery_closed_route(
                payload, 25, "waypoint_no_progress_20"
            )
        )
        route = np.asarray(manager.cluster_recovery_route_cells)
        self.assertTrue(np.array_equal(route[0], route[-1]))
        self.assertEqual(manager.cluster_recovery_replan_count, 1)

    def test_outside_start_selects_reachable_ingress_using_estimated_map_pose(self):
        manager = _route_manager()
        shape = (181, 181)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 140),
            "agent_position": np.asarray([100.0, 0.0, 100.0], dtype=np.float32),
            "map_resolution_cm": 5.0,
        }
        approach = manager._select_cluster_recovery_approach_goal(
            payload, np.zeros(3, dtype=np.float32)
        )
        self.assertTrue(approach.get("success"), approach)
        self.assertTrue(approach.get("needed"), approach)
        self.assertAlmostEqual(approach["start_radius_m"], 3.5, places=4)
        self.assertGreaterEqual(approach["goal_radius_m"], 2.5)
        self.assertLessEqual(approach["goal_radius_m"], 2.7)

    def test_start_outside_constraint_is_not_a_cached_no_loop_failure(self):
        manager = _route_manager()
        self.assertFalse(
            manager._cluster_recovery_failure_proves_no_closed_route(
                "start_outside_3m_constraint"
            )
        )
        self.assertTrue(
            manager._cluster_recovery_failure_proves_no_closed_route(
                "missing_angular_anchor"
            )
        )

    def test_replan_outside_three_meters_enters_map_based_approach(self):
        manager = _route_manager()
        manager.cluster_recovery_center = np.zeros(3, dtype=np.float32)
        manager.cluster_recovery_replan_count = 0
        manager.cluster_recovery_max_replans = 3
        manager.cluster_recovery_route_cells = [(1, 1)]
        manager.cluster_recovery_route_segments = [[(1, 1)]]
        manager.cluster_recovery_waypoint_cells = [(1, 1)]
        manager.cluster_recovery_waypoints_world = []
        manager._trace = lambda *_args, **_kwargs: None
        manager._skip_cluster_recovery_to_mr = lambda *_args, **_kwargs: self.fail(
            "a reachable map ingress must be attempted before MR fallback"
        )
        shape = (181, 181)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 140),
            "map_resolution_cm": 5.0,
        }
        self.assertTrue(
            manager._replan_cluster_recovery_closed_route(
                payload, 30, "waypoint_no_progress_20"
            )
        )
        self.assertTrue(manager.cluster_recovery_approach_active)
        self.assertTrue(manager.cluster_recovery_approach_for_replan)
        self.assertEqual(manager.cluster_recovery_replan_count, 0)
        goal = manager.cluster_recovery_approach_goal_cell
        radius = np.hypot(goal[0] - 70, goal[1] - 70) * 0.05
        self.assertGreaterEqual(radius, 2.5)
        self.assertLessEqual(radius, 2.7)

    def test_arriving_at_map_ingress_plans_a_complete_closed_route(self):
        manager = _route_manager()
        manager.args = SimpleNamespace(map_resolution=5, max_episode_length=1000)
        manager.cluster_recovery_center = np.zeros(3, dtype=np.float32)
        manager.cluster_recovery_active_signature = None
        manager.cluster_recovery_route_failure_cache = []
        manager.cluster_recovery_min_remaining_steps = 50
        manager.cluster_recovery_approach_active = True
        manager.cluster_recovery_approach_goal_cell = (70, 122)
        manager.cluster_recovery_approach_for_replan = False
        manager.cluster_recovery_approach_reason = "initial_start_outside_3m"
        manager.cluster_recovery_replan_count = 0
        manager._trace = lambda *_args, **_kwargs: None
        manager._skip_cluster_recovery_to_mr = lambda *_args, **_kwargs: self.fail(
            "a valid closed route after ingress must not fall back"
        )
        shape = (181, 181)
        payload = {
            "full_map_pred": np.zeros(shape, dtype=np.float32),
            "full_exp_pred": np.ones(shape, dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 122),
            "map_resolution_cm": 5.0,
        }
        self.assertTrue(manager._finish_cluster_recovery_approach(payload, 40))
        self.assertFalse(manager.cluster_recovery_approach_active)
        route = np.asarray(manager.cluster_recovery_route_cells)
        self.assertTrue(np.array_equal(route[0], route[-1]))
        self.assertAlmostEqual(
            abs(
                manager._validate_cluster_recovery_closed_plan(
                    {
                        "center_cell": manager.cluster_recovery_center_full_cell,
                        "route_cells": manager.cluster_recovery_route_cells,
                    }
                )["winding"]
            ),
            1.0,
            delta=0.2,
        )

    def test_orbit_goal_and_progress_use_map_cells_not_agent_world_position(self):
        manager = _route_manager()
        manager.cluster_recovery_approach_active = False
        manager.cluster_recovery_waypoint_index = 0
        manager.cluster_recovery_waypoint_cells = [(70, 110)]
        manager.cluster_recovery_route_segments = [[(70, 110)]]
        manager.cluster_recovery_center_full_cell = (70, 100)
        manager._raw_ieve_goal_from_world = lambda *_args: self.fail(
            "cluster recovery must not project a world-coordinate waypoint"
        )
        context = {
            "step": 12,
            "pose_pred": np.asarray(
                [5.5, 3.5, 0.0, 50.0, 0.0, 90.0, 0.0],
                dtype=np.float32,
            ),
            "map_pred": np.zeros((41, 41), dtype=np.float32),
            "map_resolution_cm": 5.0,
            "agent_position": np.asarray([100.0, 0.0, 100.0], dtype=np.float32),
        }
        goal_world, goal_map, _, hold = manager._get_cluster_recovery_goal_override(
            context
        )
        self.assertIsNone(goal_world)
        self.assertFalse(hold)
        self.assertEqual(float(goal_map[20, 20]), 1.0)

        full_payload = {
            "full_map_pred": np.zeros((141, 141), dtype=np.float32),
            "pose_pred": _pose_pred_for_cell(70, 108),
            "map_resolution_cm": 5.0,
            "agent_position": np.asarray([-100.0, 0.0, -100.0], dtype=np.float32),
        }
        self.assertAlmostEqual(
            manager._cluster_recovery_map_distance_to_cell(
                full_payload, (70, 110)
            ),
            0.1,
            places=5,
        )

    def test_executor_fmm_cannot_plan_to_goal_outside_constraint(self):
        from skimage.morphology import disk

        agent = Instance_Exp_Env_Agent.__new__(Instance_Exp_Env_Agent)
        agent.selem = disk(1)
        agent.visited = np.zeros((31, 31), dtype=np.float32)
        agent.collision_map = np.zeros((31, 31), dtype=np.float32)
        agent.frontier_representative_points = []
        agent.frontier_active_goal = None
        agent.global_goal = None
        agent.temp_goal = None
        agent.stuck_goal = None
        agent.args = SimpleNamespace(map_resolution=5)
        grid = np.zeros((31, 31), dtype=np.float32)
        goal = np.zeros_like(grid)
        goal[15, 25] = 1
        rows, cols = np.indices(grid.shape)
        constraint = np.hypot(rows - 15, cols - 15) <= 5

        stg, stop = agent._get_stg(
            grid,
            [15, 15],
            goal,
            [0, 31, 0, 31],
            goal_dilation_radius=0,
            traversible_constraint=constraint,
        )
        self.assertEqual(stg, (15, 15))
        self.assertFalse(stop)
        self.assertTrue(agent._last_fmm_replan)


class ClusterRecoveryTriggerTest(unittest.TestCase):
    @staticmethod
    def _localizer(tc_score):
        localizer = LocalizationSystem.__new__(LocalizationSystem)
        localizer.config = {
            "cluster_recovery_enabled": True,
            "cluster_recovery_trigger_score": 100.0,
        }
        localizer.cluster_recovery_used = False
        localizer.cluster_recovery_requested = False
        localizer.cluster_recovery_tc_evidence = {
            "score": float(tc_score),
            "pose": np.eye(4, dtype=np.float64),
            "center": np.zeros(3, dtype=np.float32),
        }
        localizer.fem_best_score = 80.0
        localizer.best_pose = np.eye(4, dtype=np.float64)
        localizer.current_state = LocalizationState.REF
        return localizer

    def test_any_score_strictly_above_100_requests_recovery_once(self):
        localizer = self._localizer(100.01)
        result = localizer._maybe_request_cluster_recovery(
            90.0, np.eye(4, dtype=np.float64)
        )
        self.assertEqual(result["status"], "cluster_recovery_requested")
        self.assertEqual(result["trigger_source"], "target_confirmation")
        self.assertEqual(localizer.current_state, LocalizationState.RECOVERY_WAIT)

        localizer.mark_cluster_recovery_started()
        self.assertIsNone(
            localizer._maybe_request_cluster_recovery(
                150.0, np.eye(4, dtype=np.float64)
            )
        )

    def test_score_equal_to_100_does_not_trigger(self):
        localizer = self._localizer(100.0)
        self.assertIsNone(
            localizer._maybe_request_cluster_recovery(
                99.0, np.eye(4, dtype=np.float64)
            )
        )


class ClusterRecoveryRenderingTest(unittest.TestCase):
    def test_relative_yaw_rendering_generates_19_views(self):
        renderer = NovelViewRenderer.__new__(NovelViewRenderer)
        renderer._batch_render = lambda poses: [
            np.zeros((8, 8, 3), dtype=np.uint8) for _ in poses
        ]
        reference_center = np.zeros(3, dtype=np.float64)
        reference_pose = NovelViewRenderer._look_at_w2c(
            np.asarray([1.0, 0.3, 0.0]), reference_center
        )
        candidate_center = np.asarray([2.0, 0.0, 1.0])

        result = renderer.render_relative_yaw_views(
            candidate_center,
            reference_center,
            reference_pose,
            yaw_range_deg=45.0,
            yaw_step_deg=5.0,
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["rendered_images"]), 19)
        self.assertEqual(result["yaw_offsets_deg"][0], -45.0)
        self.assertEqual(result["yaw_offsets_deg"][-1], 45.0)
        center_h = np.concatenate([candidate_center, [1.0]])
        for view in result["rendered_images"]:
            center_camera = np.asarray(view["pose"]) @ center_h
            self.assertGreater(float(center_camera[2]), 0.0)
            self.assertAlmostEqual(float(center_camera[0]), 0.0, places=5)
            self.assertAlmostEqual(float(center_camera[1]), 0.0, places=5)

    def test_center_object_match_count_requires_both_masks(self):
        target_mask = np.zeros((20, 20), dtype=np.uint8)
        render_mask = np.zeros((20, 20), dtype=np.uint8)
        target_mask[4:10, 4:10] = 1
        render_mask[7:13, 7:13] = 1
        cache = {
            "kpts_ref": np.asarray([[5, 5], [6, 6], [15, 15]], dtype=np.float32),
            "kpts_query": np.asarray([[8, 8], [16, 16], [9, 9]], dtype=np.float32),
        }
        count = MinAVConfirmationLocalizationManager._count_center_object_matches(
            cache, target_mask, render_mask
        )
        self.assertEqual(count, 1)

    def test_candidate_selection_prioritizes_center_matches_before_total_score(self):
        class _NVSM:
            def render_relative_yaw_views(self, object_center_3d, **_kwargs):
                cluster_id = int(round(float(object_center_3d[0])))
                return {
                    "success": True,
                    "rendered_images": [
                        {
                            "image": np.zeros((20, 20, 3), dtype=np.uint8),
                            "pose": np.eye(4, dtype=np.float64),
                            "yaw_offset_deg": float(offset),
                            "cluster_id": cluster_id,
                        }
                        for offset in range(-45, 50, 5)
                    ],
                }

        class _FEM:
            def __init__(self):
                self._last_batch_match_caches = []

            def match_rendered_views(self, views, feature_mode=None):
                del feature_mode
                cluster_id = views[0]["cluster_id"]
                center_matches = 1 if cluster_id == 0 else 2
                score = 150.0 if cluster_id == 0 else 120.0
                target = np.asarray([[5.0, 5.0]] * center_matches)
                query = np.asarray([[5.0, 5.0]] * center_matches)
                self._last_batch_match_caches = [
                    {"kpts_ref": target, "kpts_query": query} for _ in views
                ]
                return {"all_scores": [score] * len(views)}

        manager = MinAVConfirmationLocalizationManager.__new__(
            MinAVConfirmationLocalizationManager
        )
        manager.cluster_recovery_base_pose = np.eye(4, dtype=np.float64)
        manager.cluster_recovery_center = np.zeros(3, dtype=np.float32)
        manager.cluster_recovery_yaw_range_deg = 45.0
        manager.cluster_recovery_yaw_step_deg = 5.0
        manager.localization_match_threshold = 40.0
        manager.goal_center_mask = np.ones((20, 20), dtype=np.uint8)
        manager.cluster_recovery_candidates = [
            {
                "center_3d": np.asarray([0.0, 0.0, 0.0]),
                "global_indices": np.asarray([0]),
                "support_count": 10,
            },
            {
                "center_3d": np.asarray([1.0, 0.0, 0.0]),
                "global_indices": np.asarray([1]),
                "support_count": 1,
            },
        ]
        manager._ensure_goal_image_masks = lambda *_args: None
        manager._get_gaussians_for_confirmation = lambda: (object(), None)
        manager._ensure_target_confirm_nvsm_fem = lambda: (_NVSM(), _FEM(), None)
        manager._project_cluster_to_render_mask = lambda *_args: np.ones(
            (20, 20), dtype=np.uint8
        )
        manager._trace = lambda *_args, **_kwargs: None

        result = manager._evaluate_cluster_recovery_candidates({}, 10)
        self.assertTrue(result["success"])
        self.assertEqual(result["center_match_count"], 2)
        self.assertEqual(float(result["winner_cluster"]["center_3d"][0]), 1.0)
        self.assertEqual(result["best_score"], 120.0)


if __name__ == "__main__":
    unittest.main()
