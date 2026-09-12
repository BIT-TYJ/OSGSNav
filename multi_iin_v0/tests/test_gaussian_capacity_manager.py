import unittest

import numpy as np

from multi_iin_v0.runtime.gaussian_capacity_manager import (
    GaussianCapacityConfig,
    GaussianCapacityManager,
    choose_capacity_prune_mask,
    install_minav_full_snapshot_policy,
)


class GaussianCapacitySelectionTest(unittest.TestCase):
    def test_protects_confirmed_semantic_and_recent_gaussians(self):
        xyz = np.zeros((10, 3), dtype=np.float32)
        xyz[:, 0] = np.linspace(0.000, 0.009, 10)
        opacity = np.linspace(0.01, 0.90, 10, dtype=np.float32)
        semantic = np.zeros((10,), dtype=np.float32)
        semantic[1] = 0.9
        keyframes = np.zeros((10,), dtype=np.int64)
        keyframes[8:] = 100

        prune, stats = choose_capacity_prune_mask(
            xyz,
            opacity,
            target_count=5,
            semantic=semantic,
            reliability=np.full((10,), 0.5, dtype=np.float32),
            observations=np.arange(10),
            keyframe_ids=keyframes,
            confirmed_indices=np.array([0]),
            current_keyframe_id=100,
            recent_keyframe_grace=2,
            voxel_size_m=0.03,
        )

        self.assertEqual(int(prune.sum()), 5)
        self.assertFalse(bool(prune[0]))
        self.assertFalse(bool(prune[1]))
        self.assertFalse(bool(prune[8]))
        self.assertFalse(bool(prune[9]))
        self.assertEqual(stats["protected_confirmed"], 1)
        self.assertEqual(stats["protected_semantic"], 1)
        self.assertEqual(stats["protected_recent"], 2)

    def test_preserves_at_least_one_gaussian_per_voxel(self):
        xyz = np.array(
            [
                [0.000, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.050, 0.0, 0.0],
                [0.051, 0.0, 0.0],
                [0.100, 0.0, 0.0],
                [0.101, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        opacity = np.array([0.9, 0.1, 0.8, 0.1, 0.7, 0.1], dtype=np.float32)
        prune, stats = choose_capacity_prune_mask(
            xyz,
            opacity,
            target_count=3,
            voxel_size_m=0.03,
            max_voxel_size_m=0.03,
            recent_keyframe_grace=0,
        )

        kept_voxels = np.unique(np.floor(xyz[~prune] / 0.03), axis=0)
        self.assertEqual(kept_voxels.shape[0], 3)
        self.assertEqual(stats["effective_target"], 3)
        self.assertEqual(stats["capacity_shortfall"], 0)

    def test_capacity_wins_when_even_coarse_coverage_exceeds_target(self):
        xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
        opacity = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        prune, stats = choose_capacity_prune_mask(
            xyz,
            opacity,
            target_count=4,
            voxel_size_m=0.03,
            max_voxel_size_m=0.10,
        )
        self.assertEqual(int(prune.sum()), 6)
        self.assertEqual(stats["effective_target"], 4)
        self.assertGreater(stats["coverage_anchor_dropped_for_capacity"], 0)

    def test_low_quality_dense_duplicates_are_removed_first(self):
        xyz = np.zeros((8, 3), dtype=np.float32)
        xyz[:, 0] = np.linspace(0.000, 0.007, 8)
        opacity = np.array(
            [0.95, 0.90, 0.85, 0.80, 0.04, 0.03, 0.02, 0.01],
            dtype=np.float32,
        )
        prune, _ = choose_capacity_prune_mask(
            xyz,
            opacity,
            target_count=4,
            voxel_size_m=0.03,
            dense_voxel_threshold=4,
            low_opacity_threshold=0.05,
        )
        self.assertTrue(np.all(prune[4:]))
        self.assertFalse(np.any(prune[:4]))

    def test_config_clamps_watermarks(self):
        class Args:
            multi_gaussian_hard_cap = 100
            multi_gaussian_prune_trigger = 120
            multi_gaussian_prune_target = 150

        config = GaussianCapacityConfig.from_args(Args())
        self.assertEqual(config.hard_cap, 100)
        self.assertEqual(config.prune_trigger, 100)
        self.assertEqual(config.prune_target, 99)


class GaussianSnapshotPolicyTest(unittest.TestCase):
    def test_disables_only_repeated_full_scene_snapshots(self):
        class Manager:
            def _save_confirmed_full_gaussian_snapshot(self, *args, **kwargs):
                return "confirmed.ply"

            def _save_failed_target_confirmation_full_gaussian_snapshot(
                self, *args, **kwargs
            ):
                return "failed.ply"

            def save_target_only(self):
                return "target_only.ply"

        manager = Manager()
        install_minav_full_snapshot_policy(manager, enabled=False)
        self.assertIsNone(manager._save_confirmed_full_gaussian_snapshot())
        self.assertIsNone(
            manager._save_failed_target_confirmation_full_gaussian_snapshot()
        )
        self.assertEqual(manager.save_target_only(), "target_only.ply")


class GaussianCapacityRuntimeTest(unittest.TestCase):
    def test_disabled_manager_does_not_change_mapper_cap(self):
        class FakeSlam:
            def __init__(self):
                self.max_gaussians = 123
                self.config = {"Training": {"max_gaussians": 123}}

        class FakeMapper:
            def __init__(self):
                self.slam = FakeSlam()

        mapper = FakeMapper()
        manager = GaussianCapacityManager(
            GaussianCapacityConfig(enabled=False, hard_cap=999),
            output_dir=None,
        )
        manager.configure_mapper(mapper)
        self.assertEqual(mapper.slam.max_gaussians, 123)

    def test_manager_uses_model_prune_api_and_records_event(self):
        import torch

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.zeros((8, 3), dtype=torch.float32)
                self._xyz[:, 0] = torch.linspace(0.000, 0.007, 8)
                self._opacity_value = torch.tensor(
                    [0.95, 0.90, 0.85, 0.80, 0.04, 0.03, 0.02, 0.01]
                ).reshape(-1, 1)
                self._semantic_target = torch.zeros((8,), dtype=torch.float32)
                self._reliability = torch.full((8,), 0.5, dtype=torch.float32)
                self.n_obs = torch.arange(8, dtype=torch.int32)
                self.unique_kfIDs = torch.zeros((8,), dtype=torch.int32)
                self.optimizer = object()
                self.prune_calls = 0

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_opacity(self):
                return self._opacity_value

            @property
            def get_reliability_1dim(self):
                return self._reliability

            def get_confirmed_target_indices(self):
                return torch.tensor([0], dtype=torch.int64)

            def prune_points(self, mask):
                self.prune_calls += 1
                keep = ~mask.cpu()
                self._xyz = self._xyz[keep]
                self._opacity_value = self._opacity_value[keep]

        class FakeSlam:
            def __init__(self):
                self.gaussians = FakeGaussians()
                self.frame_count = 10
                self.max_gaussians = 8
                self.config = {"Training": {}}

        class FakeMapper:
            def __init__(self):
                self.slam = FakeSlam()

        mapper = FakeMapper()
        manager = GaussianCapacityManager(
            GaussianCapacityConfig(
                hard_cap=8,
                prune_trigger=8,
                prune_target=4,
                recent_keyframe_grace=0,
            ),
            output_dir=None,
        )
        event = manager.maybe_prune(
            mapper,
            env_step=20,
            stage="minav_exploration",
            timing="after_update",
        )
        self.assertIsNotNone(event)
        self.assertEqual(mapper.slam.gaussians.prune_calls, 1)
        self.assertEqual(mapper.slam.gaussians.get_xyz.shape[0], 4)
        self.assertEqual(manager.total_removed, 4)

    def test_growth_guard_disables_topology_above_watermark(self):
        class FakeGaussians:
            class Shape:
                shape = (10, 3)

            get_xyz = Shape()

        class FakeSlam:
            def __init__(self):
                self.gaussians = FakeGaussians()
                self.config = {"Training": {}}
                self.max_gaussians = 10
                self.calls = []

            def _train_mapping(self, *args, **kwargs):
                self.calls.append(kwargs.get("allow_topology_growth"))
                return "ok"

        class FakeMapper:
            def __init__(self):
                self.slam = FakeSlam()

        mapper = FakeMapper()
        manager = GaussianCapacityManager(
            GaussianCapacityConfig(prune_trigger=8, hard_cap=10, prune_target=7),
            output_dir=None,
        )
        manager.configure_mapper(mapper)
        self.assertEqual(mapper.slam._train_mapping(), "ok")
        self.assertEqual(mapper.slam.calls, [False])


if __name__ == "__main__":
    unittest.main()
