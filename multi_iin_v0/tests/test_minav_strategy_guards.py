import unittest
from collections import deque

import numpy as np

from multi_iin_v0.runtime.minav_strategy_guards import (
    _compact_center_prior_mask,
    _refresh_fem_foreground_features,
    _select_centered_goal_instance_mask,
    _target_directed_frontier_goal,
    install_minav_strategy_guards,
)


class _Args:
    multi_failure_soft_threshold = 3
    multi_failure_soft_cooldown_steps = 80
    minav_confirm_match_threshold = 20.0
    minav_confirm_min_support = 2
    success_dist = 1.0
    map_resolution = 5.0
    multi_navigation_official_stop_guard = 0
    multi_navigation_perception_stop_guard = 1
    multi_navigation_min_standoff_arrivals = 2
    multi_navigation_retry_block_radius_m = 0.45
    multi_navigation_retry_memory = 8
    multi_navigation_stuck_patience = 20
    multi_navigation_stuck_tolerance_m = 0.03
    multi_goal_foreground_mask_guard = 1
    multi_goal_compact_center_prior = 1
    multi_confirm_anchor_first = 1
    multi_confirm_coarse_views_per_scale = 8
    multi_confirm_coarse_min_score = 5.0
    multi_confirm_full_candidate_limit = 0


class _Manager:
    args = _Args()
    strategy_phase3_start_step = 800
    targeted_phase2_force_phase3 = False
    targeted_phase = 2
    phase2_no_reachable_retry_limit = 3
    blacklist_radius = 0.5
    blacklist_cooldown = 50
    confirmation_failure_blacklist_threshold = 10
    confirm_match_threshold = 20.0
    min_support_count = 2
    phase1_confirmation_evidence = []
    target_category = "bed"

    def _desired_strategy_phase(self, step):
        return 3 if self.targeted_phase2_force_phase3 else 2

    def _strategy_phase3_active(self, step):
        return True

    def _run_targeted_update(self, payload, step):
        self.phase2_no_reachable_retry_limit_seen = self.phase2_no_reachable_retry_limit
        self.targeted_phase2_force_phase3 = True

    def _maybe_activate_strategy_phase3_fallback_stop(self, step, reason):
        return True

    def _record_confirmation_failure_for_blacklist(self, result, step):
        return (self.blacklist_radius, self.confirmation_failure_blacklist_threshold, self.blacklist_cooldown)

    def _confirmation_failure_anchor(self, result):
        return np.asarray(result.get("anchor", [0.0, 0.0, 0.0]), dtype=np.float32)

    def _is_blacklisted(self, position, step):
        return False

    def _category_matches(self, detected, target):
        return str(detected).lower() == str(target).lower()

    def _filter_and_rank_target_confirm_candidates(self, clusters):
        return list(clusters)

    def _target_confirm_pass_rank(self, cluster, match):
        return (int(cluster.get("support_count", 1)), float(match.get("score", 0.0)))

    def reset_episode(self, *args, **kwargs):
        self.phase1_confirmation_evidence = []

    def _trace(self, *args, **kwargs):
        self.last_trace = args[0]


class _Mode:
    value = "navigation"


class _NavigationManager(_Manager):
    mode = _Mode()
    navigation_complete = False
    navigation_goal_world = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    navigation_goal_full_cell = (10, 10)
    navigation_goal_local_cell = (10, 10)
    navigation_raw_target_full_cell = (20, 20)
    navigation_goal_tier = "preferred"
    navigation_may_terminal_stop = True

    def update_frame(self, payload):
        self._update_navigation_completion(payload)
        return {"navigation_complete": self.navigation_complete}

    def _update_navigation_completion(self, payload):
        self.navigation_complete = bool(payload.get("internal_complete", False))

    def _navigation_map_distance_m(self, payload, **kwargs):
        if kwargs.get("full_cell") == self.navigation_raw_target_full_cell:
            return float(payload.get("raw_target_distance", 0.4))
        return float(payload.get("execution_distance", 0.1))

    def _invalidate_navigation_execution_goal(self):
        self.navigation_goal_world = None
        self.navigation_goal_full_cell = None
        self.navigation_goal_local_cell = None
        self.navigation_goal_tier = None
        self.navigation_may_terminal_stop = False

    def get_goal_override(self, context):
        # Model the upstream behavior that can set completion again while
        # rebuilding a goal from a nearby estimated target center.
        self.navigation_complete = bool(context.get("internal_complete", False))
        return {
            "navigation_complete": self.navigation_complete,
            "found_goal": self.navigation_complete,
            "force_stop": False,
            "force_stop_reason": None,
        }

    def _switch_to_navigation_from_localization(
        self, step, from_mode, target_position, loc_result
    ):
        self.localized_target_position = np.asarray(
            loc_result["target_object_center"], dtype=np.float32
        )
        self.navigation_localizer_suggested_goal = np.asarray(
            target_position, dtype=np.float32
        )


class StrategyGuardTest(unittest.TestCase):
    def test_target_directed_frontier_progresses_toward_unreachable_evidence(self):
        class _FrontierManager:
            args = _Args()

            def _buffer_project_world_to_local(self, target, context):
                del target, context
                return {
                    "ok": True,
                    "start_cell": (15, 4),
                    "raw_local_cell": (15, 29),
                    "target_local_cell": (15, 34),
                    "map_resolution_cm": 5.0,
                    "gx1": 0,
                    "gy1": 0,
                }

            def _local_cell_to_world_from_context(self, row, col, context):
                del context
                return np.asarray([col * 0.05, 0.0, row * 0.05])

        explored = np.zeros((30, 30), dtype=np.float32)
        explored[:, :15] = 1.0
        selected = _target_directed_frontier_goal(
            _FrontierManager(),
            np.asarray([2.0, 0.0, 0.75]),
            {
                "map_pred": np.zeros((30, 30), dtype=np.float32),
                "exp_pred": explored,
            },
        )
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(selected["goal_cell"][1], 12)
        self.assertEqual(int(np.sum(selected["goal_map"])), 1)
        self.assertGreater(selected["information_gain_cells"], 0)

    def test_stable_low_confidence_plant_observations_accumulate(self):
        class _StableManager(_Manager):
            target_category = "plant"

            def __init__(self):
                self.conf_history = deque(maxlen=2)
                self.frame_window = deque(maxlen=2)

            def _record_detection(self, payload):
                del payload
                self.conf_history.clear()
                self.frame_window.clear()

            def _ready_to_confirm(self):
                return False

            def _clear_detection_window(self):
                self.conf_history.clear()
                self.frame_window.clear()

            def _estimate_detection_position(self, payload, instances):
                del instances
                return np.asarray(payload["test_position"], dtype=np.float32)

            def _latest_slam_camera(self):
                return None

            def _copy_target_instances(self, instances):
                return list(instances)

        manager = _StableManager()
        install_minav_strategy_guards(manager)
        manager._record_detection({
            "step": 1,
            "test_position": [1.0, 0.0, 2.0],
            "minav_target_instances": [{"confidence": 0.20}],
        })
        self.assertFalse(manager._ready_to_confirm())
        manager._record_detection({"step": 2, "minav_target_instances": []})
        manager._record_detection({
            "step": 4,
            "test_position": [1.1, 0.0, 2.1],
            "minav_target_instances": [{"confidence": 0.22}],
        })
        self.assertTrue(manager._ready_to_confirm())
        self.assertEqual(len(manager.frame_window), 2)
        manager._clear_detection_window()
        self.assertFalse(manager._ready_to_confirm())

    def test_stable_detection_memory_cannot_revive_blacklisted_target(self):
        class _StableManager(_Manager):
            target_category = "plant"
            detect_window = 2

            def __init__(self):
                self.conf_history = deque(maxlen=2)
                self.frame_window = deque(maxlen=2)

            def _record_detection(self, payload):
                del payload
                self.conf_history.clear()
                self.frame_window.clear()

            def _ready_to_confirm(self):
                return False

            def _clear_detection_window(self):
                pass

            def _estimate_detection_position(self, payload, instances):
                del payload, instances
                return np.asarray([1.0, 0.0, 2.0], dtype=np.float32)

            def _latest_slam_camera(self):
                return None

            def _copy_target_instances(self, instances):
                return list(instances)

            def _is_blacklisted(self, position, step):
                del position, step
                return True

        manager = _StableManager()
        install_minav_strategy_guards(manager)
        for step in (1, 2):
            manager._record_detection({
                "step": step,
                "minav_target_instances": [{"confidence": 0.25}],
            })
        self.assertFalse(manager._ready_to_confirm())
        self.assertEqual(manager._multi_stable_detection_memory, [])

    def test_ordinary_confirmation_coarse_rejects_without_full_render(self):
        class _NVSM:
            num_views = 16

        class _ConfirmManager(_Manager):
            target_confirm_nvsm = _NVSM()

            def _ensure_target_confirm_nvsm_fem(self):
                return self.target_confirm_nvsm, object(), None

            def _match_candidate_with_nvsm_views(
                self, center, indices, stage=None, step=0, metadata=None
            ):
                self.view_calls = getattr(self, "view_calls", []) + [
                    (self.target_confirm_nvsm.num_views, stage)
                ]
                return {
                    "available": True,
                    "score": 4.0 if self.target_confirm_nvsm.num_views == 8 else 30.0,
                    "match_count": 10,
                    "rendered_view_count": self.target_confirm_nvsm.num_views * 5,
                }

        manager = _ConfirmManager()
        install_minav_strategy_guards(manager)
        result = manager._match_candidate_with_nvsm_views(
            np.zeros(3), np.arange(20), stage="target_confirmation", step=3,
            metadata={"candidate_rank": 0},
        )
        self.assertEqual(manager.view_calls, [(8, None)])
        self.assertEqual(result["confirmation_pass"], "coarse")
        self.assertLess(result["score"], manager.confirm_match_threshold)

    def test_ordinary_confirmation_promising_candidate_keeps_full_check(self):
        class _NVSM:
            num_views = 16

        class _ConfirmManager(_Manager):
            target_confirm_nvsm = _NVSM()

            def _ensure_target_confirm_nvsm_fem(self):
                return self.target_confirm_nvsm, object(), None

            def _match_candidate_with_nvsm_views(
                self, center, indices, stage=None, step=0, metadata=None
            ):
                self.view_calls = getattr(self, "view_calls", []) + [
                    (self.target_confirm_nvsm.num_views, stage)
                ]
                return {
                    "available": True,
                    "score": 9.0 if self.target_confirm_nvsm.num_views == 8 else 24.0,
                    "match_count": 30,
                    "rendered_view_count": self.target_confirm_nvsm.num_views * 5,
                }

        manager = _ConfirmManager()
        install_minav_strategy_guards(manager)
        result = manager._match_candidate_with_nvsm_views(
            np.zeros(3), np.arange(20), stage="target_confirmation", step=3,
            metadata={"candidate_rank": 0},
        )
        self.assertEqual(manager.view_calls, [(8, None), (16, "target_confirmation")])
        self.assertEqual(result["confirmation_pass"], "full")
        self.assertEqual(result["score"], 24.0)
        self.assertEqual(result["coarse_score"], 9.0)

    def test_nonordinary_confirmation_is_unchanged(self):
        class _NVSM:
            num_views = 16

        class _ConfirmManager(_Manager):
            target_confirm_nvsm = _NVSM()

            def _ensure_target_confirm_nvsm_fem(self):
                return self.target_confirm_nvsm, object(), None

            def _match_candidate_with_nvsm_views(
                self, center, indices, stage=None, step=0, metadata=None
            ):
                self.view_calls = getattr(self, "view_calls", []) + [
                    (self.target_confirm_nvsm.num_views, stage)
                ]
                return {"available": True, "score": 12.0, "rendered_view_count": 80}

        manager = _ConfirmManager()
        install_minav_strategy_guards(manager)
        manager._match_candidate_with_nvsm_views(
            np.zeros(3), np.arange(20), stage="target2_map_prevalidation", step=0
        )
        self.assertEqual(manager.view_calls, [(16, "target2_map_prevalidation")])

    def test_promoted_candidate_reuses_coarse_views_and_renders_only_complement(self):
        class _NVSM:
            num_views = 16

            def __init__(self):
                self.batch_sizes = []

            def _batch_render(self, poses):
                self.batch_sizes.append(len(poses))
                return ["rgb-{}".format(pose) for pose in poses]

            def render_novel_views(self, *args, **kwargs):
                del args, kwargs
                poses = list(range(self.num_views * 5))
                images = self._batch_render(poses)
                entries = []
                for index, image in enumerate(images):
                    if image is None:
                        continue
                    entries.append(
                        {
                            "image": image,
                            "scale_index": index // self.num_views,
                            "local_index": index % self.num_views,
                        }
                    )
                return {"success": True, "rendered_images": entries}

        class _ConfirmManager(_Manager):
            def __init__(self):
                self.target_confirm_nvsm = _NVSM()

            def _ensure_target_confirm_nvsm_fem(self):
                return self.target_confirm_nvsm, object(), None

            def _match_candidate_with_nvsm_views(
                self, center, indices, stage=None, step=0, metadata=None
            ):
                result = self.target_confirm_nvsm.render_novel_views()
                count = len(result["rendered_images"])
                return {
                    "available": True,
                    "score": 9.0 if self.target_confirm_nvsm.num_views == 8 else 24.0,
                    "match_count": 30,
                    "rendered_view_count": count,
                }

        manager = _ConfirmManager()
        install_minav_strategy_guards(manager)
        result = manager._match_candidate_with_nvsm_views(
            np.zeros(3), np.arange(20), stage="target_confirmation", step=3,
            metadata={"candidate_rank": 0},
        )
        self.assertEqual(manager.target_confirm_nvsm.batch_sizes, [40, 40])
        self.assertEqual(result["rendered_view_count"], 80)
        self.assertEqual(result["coarse_views_reused"], 40)

    def test_precomputed_mask_builds_semantic_features_without_segmenter(self):
        class _Fem:
            segmenter = None
            target_category = "plant"
            target_image = np.zeros((80, 60, 3), dtype=np.uint8)
            target_features_semantic = None

            def _build_target_feature_sets(self):
                if self.target_sampling_mask is not None:
                    self.target_features_semantic = {"ready": True}

        fem = _Fem()
        mask = np.zeros((40, 30), dtype=bool)
        mask[10:35, 8:25] = True
        self.assertTrue(_refresh_fem_foreground_features(fem, mask))
        self.assertEqual(fem.target_semantic_mask.shape, (80, 60))
        self.assertEqual(fem.default_feature_mode, "semantic")
        self.assertIsNotNone(fem.target_features_semantic)

    def test_phase3_force_is_blocked_before_start(self):
        manager = _Manager()
        install_minav_strategy_guards(manager)
        manager.targeted_phase2_force_phase3 = True
        self.assertEqual(manager._desired_strategy_phase(799), 2)
        self.assertFalse(manager._strategy_phase3_active(799))
        manager._run_targeted_update({}, 799)
        self.assertEqual(manager.phase2_no_reachable_retry_limit_seen, 10 ** 9)

    def test_fallback_stop_requires_threshold_evidence(self):
        manager = _Manager()
        install_minav_strategy_guards(manager)
        self.assertFalse(manager._maybe_activate_strategy_phase3_fallback_stop(900, "near"))
        manager.phase1_confirmation_evidence = [{"confirmed": False, "best_score": 21.0, "all_cluster_scores": [{"best_score": 21.0, "support_count": 2}]}]
        self.assertTrue(manager._maybe_activate_strategy_phase3_fallback_stop(900, "near"))

    def test_adaptive_blacklist_radius(self):
        manager = _Manager()
        install_minav_strategy_guards(manager)
        result = manager._record_confirmation_failure_for_blacklist({"confirmed": False}, 10)
        self.assertEqual(result[0], 1.15)
        self.assertEqual(result[1], 3)
        self.assertEqual(result[2], 80)

    def test_hard_rejection_persists_until_reset(self):
        manager = _Manager()
        install_minav_strategy_guards(manager)
        for step in range(5):
            manager._record_confirmation_failure_for_blacklist(
                {"confirmed": False, "anchor": [1.0, 0.0, 2.0]}, step
            )
        self.assertTrue(manager._is_blacklisted([1.1, 0.0, 2.0], 1000))
        manager.reset_episode()
        self.assertFalse(manager._is_blacklisted([1.1, 0.0, 2.0], 1000))

    def test_first_internal_arrival_requires_an_alternate_view(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        manager.update_frame(
            {
                "step": 53,
                "internal_complete": True,
                "minav_fmm_stop": True,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertFalse(manager.navigation_complete)
        self.assertEqual(len(manager._multi_navigation_attempted_goals), 1)
        override = manager.get_goal_override({"internal_complete": True})
        self.assertFalse(override["navigation_complete"])
        self.assertFalse(override["found_goal"])

    def test_second_internal_standoff_accepts_stop_without_evaluator_metrics(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        manager._multi_navigation_attempted_goals = [
            {"world": np.asarray([0.6, 0.0, 0.0], dtype=np.float32)}
        ]
        manager.update_frame(
            {
                "step": 323,
                "internal_complete": False,
                "minav_fmm_stop": True,
                "raw_target_distance": 0.55,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertTrue(manager.navigation_complete)
        override = manager.get_goal_override({"internal_complete": False})
        self.assertTrue(override["navigation_complete"])
        self.assertTrue(override["found_goal"])

    def test_evaluator_metrics_cannot_change_online_stop_decision(self):
        decisions = []
        for habitat_distance, habitat_success in ((0.01, 1.0), (100.0, 0.0)):
            manager = _NavigationManager()
            install_minav_strategy_guards(manager)
            manager.update_frame(
                {
                    "step": 53,
                    "distance_to_goal": habitat_distance,
                    "success": habitat_success,
                    "internal_complete": True,
                    "minav_fmm_stop": True,
                    "raw_target_distance": 0.4,
                    "agent_position": [0.0, 0.0, 0.0],
                }
            )
            decisions.append(
                (manager.navigation_complete, len(manager._multi_navigation_attempted_goals))
            )
        self.assertEqual(decisions[0], decisions[1])

    def test_localized_camera_pose_does_not_replace_object_center(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        camera = np.asarray([4.0, 1.2, -3.0], dtype=np.float32)
        center = np.asarray([5.0, 0.5, -2.0], dtype=np.float32)
        manager._switch_to_navigation_from_localization(
            10,
            "localization",
            camera,
            {
                "target_position": camera,
                "localized_camera_position": camera,
                "target_object_center": center,
                "position_source": "localized_target_camera_pose",
            },
        )
        np.testing.assert_allclose(manager.localized_target_position, center)
        np.testing.assert_allclose(manager.navigation_localizer_suggested_goal, camera)
        np.testing.assert_allclose(manager._multi_localized_camera_viewpoint, camera)

    def test_phase3_does_not_fall_back_to_weak_reachable_identity(self):
        class _Phase3Manager(_Manager):
            class _Occupancy:
                map_resolution = 5.0

            occupancy_adapter = _Occupancy()

            def _select_reachable_navigation_goal(self, center, context, phase3=False):
                del context, phase3
                reachable = float(np.asarray(center)[2]) < 0.0
                return {"selected": reachable}

            def _select_targeted_phase3_evidence_object(self, payload, step):
                del payload, step
                return {"position_3d": np.asarray([0.2, 1.1, -4.4]), "historical_score": 33.68}

        manager = _Phase3Manager()
        manager.phase1_confirmation_evidence = [{
            "center": [0.0, 1.8, 0.87],
            "best_score": 59.997,
            "best_match_count": 113,
            "confirmed": True,
            "all_cluster_scores": [
                {
                    "target_cluster_center": [0.0, 1.8, 0.87],
                    "best_score": 59.997,
                    "best_match_count": 113,
                    "candidate_cluster_size": 14482,
                },
                {
                    "target_cluster_center": [0.2, 1.1, -4.4],
                    "best_score": 33.68,
                    "best_match_count": 66,
                    "candidate_cluster_size": 103,
                },
            ],
        }]
        install_minav_strategy_guards(manager)
        selected = manager._select_targeted_phase3_evidence_object(
            {"agent_position": [0.0, 0.0, 0.0]}, 800
        )
        self.assertIsNone(selected)
        self.assertEqual(
            manager.last_trace,
            "multi_phase3_strongest_identity_waiting_for_reachability",
        )

    def test_phase3_uses_upstream_payload_when_strongest_is_reachable(self):
        class _Phase3Manager(_Manager):
            class _Occupancy:
                map_resolution = 5.0

            occupancy_adapter = _Occupancy()

            def _select_reachable_navigation_goal(self, center, context, phase3=False):
                del center, context, phase3
                return {"selected": True}

            def _select_targeted_phase3_evidence_object(self, payload, step):
                del payload, step
                return {"position_3d": np.asarray([1.0, 0.5, 2.0]), "historical_score": 60.0}

        manager = _Phase3Manager()
        manager.phase1_confirmation_evidence = [{
            "center": [1.0, 0.5, 2.0],
            "best_score": 60.0,
            "best_match_count": 100,
            "confirmed": True,
        }]
        install_minav_strategy_guards(manager)
        selected = manager._select_targeted_phase3_evidence_object(
            {"agent_position": [0.0, 0.0, 0.0]}, 800
        )
        self.assertEqual(selected["historical_score"], 60.0)

    def test_phase3_repeated_identity_beats_one_off_higher_score(self):
        class _Phase3Manager(_Manager):
            class _Occupancy:
                map_resolution = 5.0

            occupancy_adapter = _Occupancy()

            def _select_reachable_navigation_goal(self, center, context, phase3=False):
                del context, phase3
                # The repeated identity is intentionally unreachable.  If the
                # one-off high-score false positive wins, upstream is called.
                return {"selected": float(np.asarray(center)[0]) > 0.0}

            def _select_targeted_phase3_evidence_object(self, payload, step):
                del payload, step
                return {"position_3d": np.asarray([4.0, 0.5, 0.0]), "historical_score": 80.0}

        manager = _Phase3Manager()
        manager.phase1_confirmation_evidence = [{
            "center": [-2.0, 0.5, 0.0],
            "best_score": 35.0,
            "best_match_count": 90,
            "confirmed": True,
            "all_cluster_scores": [
                {
                    "target_cluster_center": [-2.0, 0.5, 0.0],
                    "best_score": 35.0,
                    "best_match_count": 90,
                    "support_count": 3,
                },
                {
                    "target_cluster_center": [4.0, 0.5, 0.0],
                    "best_score": 80.0,
                    "best_match_count": 120,
                    "support_count": 1,
                },
            ],
        }]
        install_minav_strategy_guards(manager)
        selected = manager._select_targeted_phase3_evidence_object(
            {"agent_position": [0.0, 0.0, 0.0]}, 800
        )
        self.assertIsNone(selected)
        np.testing.assert_allclose(manager._multi_phase3_frontier_target, [-2.0, 0.5, 0.0])

    def test_new_alternate_goal_gets_time_to_move(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        manager._multi_navigation_perception_guard_active = True
        manager.update_frame(
            {
                "step": 54,
                "internal_complete": True,
                "minav_fmm_stop": False,
                "execution_distance": 0.8,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertFalse(manager.navigation_complete)
        self.assertEqual(manager._multi_navigation_attempted_goals, [])
        self.assertIsNotNone(manager.navigation_goal_world)

    def test_upstream_completion_without_fmm_arrival_cannot_bypass_guard(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        manager.update_frame(
            {
                "step": 157,
                "internal_complete": True,
                "minav_fmm_stop": False,
                "execution_distance": 0.8,
                "raw_target_distance": 0.2,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertFalse(manager.navigation_complete)
        self.assertEqual(len(manager._multi_navigation_attempted_goals), 1)

    def test_reached_standoff_outside_estimated_radius_replans_immediately(self):
        manager = _NavigationManager()
        install_minav_strategy_guards(manager)
        manager._multi_navigation_attempted_goals = [
            {"world": np.asarray([0.6, 0.0, 0.0], dtype=np.float32)}
        ]
        manager.navigation_goal_world = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        manager.update_frame(
            {
                "step": 700,
                "internal_complete": False,
                "minav_fmm_stop": True,
                "raw_target_distance": 2.2,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertFalse(manager.navigation_complete)
        self.assertIsNone(manager.navigation_goal_world)
        self.assertEqual(len(manager._multi_navigation_attempted_goals), 2)

    def test_anchor_distance_precedes_larger_same_class_cluster(self):
        manager = _Manager()
        install_minav_strategy_guards(manager)
        near = {
            "detection_dist_m": 0.2,
            "size": 100,
            "support_count": 2,
            "confidence": 0.7,
        }
        far_large = {
            "detection_dist_m": 1.5,
            "size": 5000,
            "support_count": 5,
            "confidence": 0.99,
        }
        ranked = manager._filter_and_rank_target_confirm_candidates(
            [far_large, near]
        )
        self.assertIs(ranked[0], near)
        self.assertGreater(
            manager._target_confirm_pass_rank(near, {"score": 21.0}),
            manager._target_confirm_pass_rank(far_large, {"score": 100.0}),
        )

    def test_centered_instance_mask_handles_wrong_goal_category_prediction(self):
        manager = _Manager()
        manager.target_category = "plant"
        centred = np.zeros((100, 100), dtype=bool)
        centred[30:80, 40:65] = True
        background = np.zeros((100, 100), dtype=bool)
        background[5:40, 5:95] = True
        selected = _select_centered_goal_instance_mask(
            manager,
            [
                {"category": "sofa", "binary_mask": background},
                {"category": "chair", "binary_mask": centred},
            ],
            (100, 100, 3),
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["category"], "chair")
        np.testing.assert_array_equal(selected["mask"], centred)

    def test_centered_fallback_is_exposed_to_mr_as_current_task_category(self):
        centred = np.zeros((100, 100), dtype=bool)
        centred[30:80, 40:65] = True

        class _Segmenter:
            def segment_frame(self, image, largest_only=False):
                return (
                    np.zeros(image.shape[:2], dtype=np.int32),
                    [{
                        "category": "sofa",
                        "binary_mask": centred,
                        "pixel_count": int(centred.sum()),
                    }],
                    None,
                )

        class _GoalMaskManager(_Manager):
            target_category = "chair"

            def __init__(self):
                self.target_image = np.zeros((100, 100, 3), dtype=np.uint8)

            def _analyze_goal_image_center_mask(self, segmenter, step):
                return None

            def _analyze_all_goal_image_masks(self, segmenter, step):
                semantic, labels, _ = segmenter.segment_frame(
                    self.target_image, largest_only=False
                )
                return semantic, [{
                    "mask": labels[0]["binary_mask"],
                    "category": labels[0]["category"],
                    "pixel_count": labels[0]["pixel_count"],
                }]

        manager = _GoalMaskManager()
        install_minav_strategy_guards(manager)
        segmenter = _Segmenter()
        selected = manager._analyze_goal_image_center_mask(segmenter, 1)
        _, all_masks = manager._analyze_all_goal_image_masks(segmenter, 1)

        np.testing.assert_array_equal(selected, centred)
        self.assertEqual([item["category"] for item in all_masks], ["sofa", "chair"])
        self.assertEqual(
            all_masks[-1]["source"],
            "multi_goal_centered_instance_task_category_proxy",
        )
        np.testing.assert_array_equal(all_masks[-1]["mask"], centred)

    def test_compact_center_prior_is_not_the_old_broad_box(self):
        mask = _compact_center_prior_mask("chair", (100, 100, 3))
        self.assertIsNotNone(mask)
        self.assertLess(float(mask.mean()), 0.35)
        self.assertTrue(mask[50, 50])


if __name__ == "__main__":
    unittest.main()
