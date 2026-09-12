import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from multi_iin_v0.runtime.semantic_instance_filter import (
    InstanceFilterConfig,
    _category_size_compatible,
    _prefilter_indices_by_instance_region,
    attach_instance_filter_to_manager,
    filter_instance_indices,
    save_filter_projection,
)


def _cluster(rng, center, count, sigma=0.025):
    return rng.normal(center, sigma, size=(count, 3)).astype(np.float32)


class SemanticInstanceFilterTest(unittest.TestCase):
    def test_manager_bridge_exposes_exact_confirmed_indices(self):
        class Localizer:
            pass

        class Manager:
            def __init__(self):
                self.confirmed_indices = np.asarray([2, 7, 9])
                self.localizer = Localizer()

            def _ensure_full_localizer(self, payload):
                return self.localizer

        manager = Manager()
        attach_instance_filter_to_manager(manager)
        localizer = manager._ensure_full_localizer({})
        np.testing.assert_array_equal(
            localizer._multi_iin_confirmed_indices_provider(),
            manager.confirmed_indices,
        )

    def test_filter_projection_png_contains_keep_and_remove_colours(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "projection.png"
            kept = np.asarray([[0.0, 0.5, 0.0], [0.2, 0.7, 0.2]])
            removed = np.asarray([[1.0, 0.5, 1.0], [1.2, 0.7, 1.2]])
            save_filter_projection(kept, removed, output, "plant")
            image = np.asarray(Image.open(str(output)).convert("RGB"))
            self.assertTrue(np.any(np.all(image == (25, 180, 75), axis=2)))
            self.assertTrue(np.any(np.all(image == (220, 55, 45), axis=2)))

    def test_category_size_gate_rejects_tiny_bed_satellite(self):
        compatible, debug = _category_size_compatible(
            [0.12, 0.13, 0.06], "bed"
        )
        self.assertFalse(compatible)
        self.assertEqual(debug["reason"], "bbox_size_mismatch")

    def test_category_size_gate_allows_partial_bed_body(self):
        compatible, debug = _category_size_compatible(
            [0.80, 0.35, 1.10], "bed"
        )
        self.assertTrue(compatible)
        self.assertEqual(debug["reason"], "within_bounds")

    def test_soim_region_prefilter_separates_same_category_instances(self):
        class GridMapper:
            @staticmethod
            def world_to_map(world_x, world_z):
                return int(round(world_x)) + 5, int(round(world_z)) + 5

        xyz = np.asarray(
            [
                [0.0, 0.5, 0.0],
                [0.1, 0.6, 0.1],
                [3.0, 0.5, 3.0],
                [3.1, 0.6, 3.1],
            ],
            dtype=np.float32,
        )
        mask = np.zeros((12, 12), dtype=bool)
        mask[5, 5] = True
        selected, stats = _prefilter_indices_by_instance_region(
            xyz,
            np.arange(4),
            {"mask": mask},
            GridMapper(),
            min_points=2,
        )

        self.assertIs(stats["applied"], True)
        np.testing.assert_array_equal(selected, np.asarray([0, 1]))

    def test_removes_small_satellite_components(self):
        rng = np.random.RandomState(7)
        main = _cluster(rng, [0.0, 0.7, 0.0], 1000)
        satellites = np.concatenate(
            [
                _cluster(rng, [2.0, 0.7, 0.0], 30),
                _cluster(rng, [-2.0, 0.7, 0.0], 30),
            ]
        )
        xyz = np.concatenate([main, satellites])
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(voxel_size_m=0.015, cluster_eps_m=0.10),
            anchor_3d=np.asarray([0.1, 0.0, 0.1], dtype=np.float32),
        )

        self.assertIs(stats["applied"], True)
        self.assertEqual(selected.size, main.shape[0])
        self.assertEqual(stats["removed_points"], satellites.shape[0])

    def test_anchor_neighbourhood_beats_much_larger_far_component(self):
        rng = np.random.RandomState(11)
        near = _cluster(rng, [0.0, 0.6, 0.0], 500)
        far = _cluster(rng, [2.0, 0.6, 0.0], 5000)
        xyz = np.concatenate([near, far])
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(
                voxel_size_m=0.015,
                cluster_eps_m=0.10,
                anchor_neighbourhood_slack_m=0.25,
            ),
            anchor_3d=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertIs(stats["applied"], True)
        self.assertEqual(selected.size, near.shape[0])
        self.assertEqual(stats["selection_mode"], "anchor_neighbourhood_then_size")
        self.assertLess(np.linalg.norm(xyz[selected].mean(axis=0)[[0, 2]]), 0.1)

    def test_support_size_wins_inside_anchor_neighbourhood(self):
        rng = np.random.RandomState(13)
        tiny_near = _cluster(rng, [0.0, 0.6, 0.0], 60)
        body = _cluster(rng, [0.20, 0.6, 0.0], 600)
        xyz = np.concatenate([tiny_near, body])
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(
                voxel_size_m=0.015,
                cluster_eps_m=0.08,
                anchor_neighbourhood_slack_m=0.25,
            ),
            anchor_3d=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertIs(stats["applied"], True)
        self.assertEqual(selected.size, body.shape[0])
        body_center_xz = xyz[selected].mean(axis=0)[[0, 2]]
        self.assertLess(np.linalg.norm(body_center_xz - [0.2, 0.0]), 0.1)

    def test_tiny_cluster_on_anchor_is_not_an_object_body(self):
        rng = np.random.RandomState(17)
        anchor_noise = _cluster(rng, [0.0, 0.6, 0.0], 60)
        body = _cluster(rng, [1.5, 0.6, 0.0], 5000)
        xyz = np.concatenate([anchor_noise, body])
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(
                voxel_size_m=0.015,
                cluster_eps_m=0.08,
                min_keep_ratio=0.02,
                anchor_neighbourhood_slack_m=0.25,
            ),
            anchor_3d=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertIs(stats["applied"], True)
        self.assertEqual(selected.size, body.shape[0])
        self.assertFalse(stats["body_candidate_floor_fallback"])

    def test_confirmed_seed_overrides_anchor_and_cluster_size(self):
        rng = np.random.RandomState(19)
        confirmed = _cluster(rng, [0.0, 0.6, 0.0], 60)
        large = _cluster(rng, [1.5, 0.6, 0.0], 5000)
        xyz = np.concatenate([confirmed, large])
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(
                voxel_size_m=0.015,
                cluster_eps_m=0.08,
                min_keep_ratio=0.02,
            ),
            anchor_3d=np.asarray([1.5, 0.0, 0.0], dtype=np.float32),
            seed_indices=np.arange(confirmed.shape[0]),
        )

        self.assertIs(stats["applied"], True)
        self.assertEqual(selected.size, confirmed.shape[0])
        self.assertEqual(
            stats["selection_mode"], "confirmed_seed_overlap_then_size"
        )
        self.assertEqual(stats["selected_seed_overlap_count"], confirmed.shape[0])

    def test_safety_fallback_keeps_small_input(self):
        xyz = np.zeros((20, 3), dtype=np.float32)
        selected, stats = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0]),
            InstanceFilterConfig(min_cluster_points=50),
        )

        self.assertIs(stats["applied"], False)
        self.assertEqual(stats["fallback_reason"], "too_few_candidate_points")
        np.testing.assert_array_equal(selected, np.arange(xyz.shape[0]))


if __name__ == "__main__":
    unittest.main()
