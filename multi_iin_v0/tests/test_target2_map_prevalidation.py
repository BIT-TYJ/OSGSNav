import unittest

import numpy as np
import torch

from multi_iin_v0.runtime.target2_map_prevalidation import (
    _candidate_prior_rank,
    _local_gaussian_clusters,
    _match_with_temporary_nvsm_budget,
    install_target2_map_prevalidation,
)


class _Args:
    multi_variant = "Continuous-Shared"
    multi_target2_map_prevalidation = 1
    multi_target2_map_prevalidation_grace_steps = 2
    multi_target2_map_match_threshold = 40.0
    multi_target2_map_use_gaussian_labels = 1
    multi_target2_map_gaussian_label_confidence = 0.55
    multi_target2_map_gaussian_label_min_observations = 1
    multi_target2_map_max_candidates = 3
    multi_target2_map_clusters_per_component = 2
    multi_target2_map_shortlist_candidates = 2
    multi_target2_map_coarse_views_per_scale = 4
    multi_target2_map_coarse_min_score = 8.0
    multi_target2_map_coarse_save_visualizations = 0
    multi_target2_map_full_views_per_scale = 16
    multi_target2_map_gaussian_radius_m = 1.0
    multi_target2_map_min_gaussians = 20
    multi_target2_map_max_gaussians = 50000


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    @property
    def shape(self):
        return self.value.shape


class _Gaussians:
    def __init__(self):
        rng = np.random.RandomState(1)
        points = rng.normal(scale=0.08, size=(100, 3)).astype(np.float32)
        points[:, 1] += 0.7
        self.get_xyz = _Tensor(points)


class _Mode:
    value = "buffer"


class _Fem:
    target_features_semantic = {"ready": True}

    def __init__(self):
        self.save_visualization_results = True


class _Nvsm:
    def __init__(self):
        self.num_views = 16
        self.save_visualization_results = True


class _Manager:
    def __init__(self):
        self.args = _Args()
        self.slam_mapper = object()
        self.target_category = "sofa"
        self.confirm_match_threshold = 20.0
        self.gaussians = _Gaussians()
        self.target_confirm_fem = _Fem()
        self.target_confirm_nvsm = _Nvsm()
        self.save_stage_visualizations = True
        self.events = []
        self.mode = _Mode()

    def reset_episode(self, info, mapper=None, target_image=None, category=None):
        self.slam_mapper = mapper
        self.target_category = category

    def update_frame(self, payload):
        self.last_update_payload = payload
        return {"mode": self.mode.value}

    def _trace(self, event, **kwargs):
        self.events.append((event, kwargs))

    def _extract_targeted_objects_from_map(self, payload):
        if not payload.get("has_semantic", False):
            return {}
        return {
            1: {
                "obj_id": 1,
                "position_3d": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "area": 100,
            }
        }

    def _get_gaussians_for_confirmation(self):
        return self.gaussians, None

    def _ensure_goal_image_masks(self, payload, step):
        return None

    def _ensure_target_confirm_nvsm_fem(self):
        return self.target_confirm_nvsm, self.target_confirm_fem, None

    def _cluster_target_confirm_indices(self, gaussians, indices):
        xyz = gaussians.get_xyz.numpy()[indices]
        return [
            {
                "indices": indices,
                "global_indices": indices,
                "center": xyz.mean(axis=0),
                "center_3d": xyz.mean(axis=0),
                "size": int(indices.size),
            }
        ]

    @staticmethod
    def _cluster_center(cluster):
        return np.asarray(cluster["center_3d"], dtype=np.float32)

    @staticmethod
    def _cluster_indices(cluster):
        return np.asarray(cluster["global_indices"], dtype=np.int64)

    def _match_candidate_with_nvsm_views(self, *args, **kwargs):
        return {
            "available": True,
            "score": 45.0,
            "match_count": 40,
            "source": "nvsm_test",
            "best_index": 0,
            "best_pose": np.eye(4, dtype=np.float32),
        }

    def _lock_confirmed_cluster(self, indices, **kwargs):
        return int(len(indices))

    def _new_confirm_result(self):
        return {}

    def _record_strategy_evidence(self, **kwargs):
        self.evidence = kwargs

    def _enter_buffer_from_confirmation(self, result, step, payload):
        self.entered_result = result


class Target2MapPrevalidationTest(unittest.TestCase):
    def test_shortlist_prior_prefers_supported_observed_instance(self):
        tiny_pure_cluster = {
            "candidate_source": "trained_semantic_gaussians_near_2d_component",
            "semantic_observation_mean": 1.0,
            "target_semantic_confidence_mean": 0.99,
            "dominant_instance_points": 50,
            "dominant_instance_ratio": 1.0,
            "map_anchor_distance_m": 0.1,
            "size": 50,
        }
        supported_cluster = {
            "candidate_source": "trained_semantic_gaussians_near_2d_component",
            "semantic_observation_mean": 2.0,
            "target_semantic_confidence_mean": 0.9,
            "dominant_instance_points": 1800,
            "dominant_instance_ratio": 0.9,
            "map_anchor_distance_m": 0.4,
            "size": 2000,
        }
        self.assertGreater(
            _candidate_prior_rank(supported_cluster),
            _candidate_prior_rank(tiny_pure_cluster),
        )

    def test_trained_category_gaussians_are_preferred(self):
        manager = _Manager()
        gaussians = manager.gaussians
        gaussians._multi_semantic_evidence = torch.as_tensor(
            np.vstack(
                [
                    np.tile([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], (25, 1)),
                    np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (75, 1)),
                ]
            ),
            dtype=torch.float32,
        )
        gaussians._multi_semantic_observations = torch.ones(
            (100,), dtype=torch.int32
        )
        clusters = _local_gaussian_clusters(
            manager,
            gaussians,
            {"position_3d": [0.0, 0.0, 0.0], "obj_id": 1, "area": 100},
            {"agent_position": [0.0, 0.0, 0.0]},
        )
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["size"], 25)
        self.assertEqual(
            clusters[0]["candidate_source"],
            "trained_semantic_gaussians_near_2d_component",
        )

    def test_only_target2_shared_arms_prevalidation(self):
        manager = _Manager()
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 0}, object(), np.zeros((4, 4, 3)), "chair"
        )
        self.assertFalse(manager._multi_target2_map_prevalidation_pending)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        self.assertTrue(manager._multi_target2_map_prevalidation_pending)

    def test_existing_map_candidate_confirms_before_motion(self):
        manager = _Manager()
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        manager.update_frame(
            {
                "step": 0,
                "has_semantic": True,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertFalse(manager._multi_target2_map_prevalidation_pending)
        self.assertTrue(manager._multi_target2_map_prevalidation_attempted)
        self.assertEqual(
            manager.entered_result["method"],
            "target2_existing_map_coarse_top1_nvsm_fem_prevalidation_v1",
        )
        self.assertIn(
            "target2_map_prevalidation_confirmed",
            [event for event, _kwargs in manager.events],
        )

    def test_existing_candidates_are_evaluated_only_once_when_rejected(self):
        manager = _Manager()
        calls = []

        def low_score(*args, **kwargs):
            calls.append((args, kwargs))
            return {
                "available": True,
                "score": 2.0,
                "match_count": 20,
                "source": "nvsm_test",
                "best_index": 0,
                "best_pose": np.eye(4, dtype=np.float32),
            }

        manager._match_candidate_with_nvsm_views = low_score
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        manager.update_frame(
            {
                "step": 0,
                "has_semantic": True,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertEqual(len(calls), 1)
        self.assertFalse(manager._multi_target2_map_prevalidation_pending)
        self.assertTrue(manager._multi_target2_map_prevalidation_attempted)
        manager.update_frame(
            {
                "step": 1,
                "has_semantic": True,
                "agent_position": [0.0, 0.0, 0.0],
            }
        )
        self.assertEqual(len(calls), 1)

    def test_two_candidates_are_coarse_ranked_and_only_top1_runs_full(self):
        manager = _Manager()

        def two_clusters(gaussians, indices):
            halves = np.array_split(indices, 2)
            records = []
            for part in halves:
                xyz = gaussians.get_xyz.numpy()[part]
                records.append(
                    {
                        "indices": part,
                        "global_indices": part,
                        "center": xyz.mean(axis=0),
                        "center_3d": xyz.mean(axis=0),
                        "size": int(part.size),
                    }
                )
            return records

        manager._cluster_target_confirm_indices = two_clusters
        calls = []

        def ranked_match(center, indices, **kwargs):
            validation_pass = kwargs["metadata"]["validation_pass"]
            calls.append(
                {
                    "pass": validation_pass,
                    "center": np.asarray(center).copy(),
                    "views": manager.target_confirm_nvsm.num_views,
                    "save": manager.save_stage_visualizations,
                }
            )
            if validation_pass == "coarse":
                score = (12.0, 30.0)[sum(c["pass"] == "coarse" for c in calls) - 1]
            else:
                score = 45.0
            return {
                "available": True,
                "score": score,
                "match_count": int(score),
                "source": "nvsm_test",
                "best_index": 0,
                "best_pose": np.eye(4, dtype=np.float32),
                "rendered_view_count": manager.target_confirm_nvsm.num_views * 5,
            }

        manager._match_candidate_with_nvsm_views = ranked_match
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        manager.update_frame(
            {"step": 0, "has_semantic": True, "agent_position": [0.0, 0.0, 0.0]}
        )
        self.assertEqual(
            [call["pass"] for call in calls], ["coarse", "coarse", "full"]
        )
        self.assertEqual([call["views"] for call in calls], [4, 4, 16])
        self.assertEqual([call["save"] for call in calls], [False, False, True])
        np.testing.assert_allclose(calls[2]["center"], calls[1]["center"])
        self.assertTrue(hasattr(manager, "entered_result"))

    def test_coarse_winner_below_full_threshold_falls_back(self):
        manager = _Manager()
        calls = []

        def pass_dependent_match(*args, **kwargs):
            validation_pass = kwargs["metadata"]["validation_pass"]
            calls.append(validation_pass)
            score = 50.0 if validation_pass == "coarse" else 39.0
            return {
                "available": True,
                "score": score,
                "match_count": 50,
                "source": "nvsm_test",
            }

        manager._match_candidate_with_nvsm_views = pass_dependent_match
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        manager.update_frame(
            {"step": 0, "has_semantic": True, "agent_position": [0.0, 0.0, 0.0]}
        )
        self.assertEqual(calls, ["coarse", "full"])
        self.assertFalse(hasattr(manager, "entered_result"))
        failures = [
            kwargs
            for event, kwargs in manager.events
            if event == "target2_map_prevalidation_failed"
        ]
        self.assertEqual(
            failures[-1]["reason"], "coarse_winner_failed_full_match_threshold"
        )

    def test_temporary_nvsm_budget_is_restored_after_exception(self):
        manager = _Manager()

        def explode(*args, **kwargs):
            self.assertEqual(manager.target_confirm_nvsm.num_views, 3)
            self.assertFalse(manager.save_stage_visualizations)
            raise RuntimeError("test failure")

        manager._match_candidate_with_nvsm_views = explode
        with self.assertRaises(RuntimeError):
            _match_with_temporary_nvsm_budget(
                manager,
                np.zeros(3, dtype=np.float32),
                np.arange(20, dtype=np.int64),
                views_per_scale=3,
                save_visualizations=False,
                stage="coarse",
                step=0,
                metadata={},
            )
        self.assertEqual(manager.target_confirm_nvsm.num_views, 16)
        self.assertTrue(manager.target_confirm_nvsm.save_visualization_results)
        self.assertTrue(manager.target_confirm_fem.save_visualization_results)
        self.assertTrue(manager.save_stage_visualizations)

    def test_weak_map_match_falls_back_to_exploration(self):
        manager = _Manager()
        manager._match_candidate_with_nvsm_views = lambda *args, **kwargs: {
            "available": True,
            "score": 24.0,
            "match_count": 90,
            "source": "ambiguous_background",
        }
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "chair"
        )
        manager.update_frame(
            {"step": 2, "has_semantic": True, "agent_position": [0.0, 0.0, 0.0]}
        )
        self.assertFalse(manager._multi_target2_map_prevalidation_pending)
        self.assertFalse(hasattr(manager, "entered_result"))
        self.assertIn(
            "target2_map_prevalidation_failed",
            [event for event, _kwargs in manager.events],
        )

    def test_uniform_features_never_confirm_map_only_candidate(self):
        manager = _Manager()
        manager.target_confirm_fem = type(
            "UniformFem", (), {"target_features_semantic": None}
        )()
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "chair"
        )
        manager.update_frame(
            {"step": 2, "has_semantic": True, "agent_position": [0.0, 0.0, 0.0]}
        )
        self.assertFalse(hasattr(manager, "entered_result"))

    def test_no_map_candidate_falls_back_after_grace_period(self):
        manager = _Manager()
        install_target2_map_prevalidation(manager)
        manager.reset_episode(
            {"multi_target_index": 1}, object(), np.zeros((4, 4, 3)), "sofa"
        )
        manager.update_frame({"step": 0, "has_semantic": False})
        self.assertTrue(manager._multi_target2_map_prevalidation_pending)
        manager.update_frame({"step": 2, "has_semantic": False})
        self.assertFalse(manager._multi_target2_map_prevalidation_pending)
        self.assertTrue(manager._multi_target2_map_prevalidation_attempted)


if __name__ == "__main__":
    unittest.main()
