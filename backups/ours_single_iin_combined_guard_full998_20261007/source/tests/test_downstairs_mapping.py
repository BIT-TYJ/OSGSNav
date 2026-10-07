import unittest

import numpy as np

from envs.utils.depth_utils import update_minav_full_grid_from_depth


class DownstairsMappingTest(unittest.TestCase):
    def _new_maps(self):
        shape = (100, 100)
        return (
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.bool_),
            np.zeros(shape, dtype=np.uint16),
        )

    def _update(self, maps, enabled, depth=None):
        obstacle_count, free_count, occupancy, explored, downstairs, evidence = maps
        if depth is None:
            depth = np.zeros((8, 8), dtype=np.float32)
            depth[7, 2:7] = 2.0
        return update_minav_full_grid_from_depth(
            obstacle_count,
            free_count,
            occupancy,
            explored,
            depth,
            agent_position=[0.0, 0.0, 0.0],
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            episode_origin_xy=[0.0, 0.0],
            map_size_cm=500,
            map_resolution=5,
            fov_deg=79,
            camera_height=0.88,
            point_sample_step=1,
            ray_sample_rate=1,
            prevent_downstairs=enabled,
            downstairs_height_threshold_m=0.2,
            downstairs_obstacle_mask=downstairs,
            downstairs_evidence_count=evidence,
            downstairs_min_observations=2,
        )

    def test_only_leading_edge_becomes_persistent_obstacle(self):
        maps = self._new_maps()
        first = self._update(maps, enabled=True)
        stats = self._update(maps, enabled=True)
        _, free_count, occupancy, explored, downstairs, _ = maps

        self.assertGreater(stats["downstairs_raw_points"], 0)
        self.assertGreater(stats["downstairs_points"], 0)
        self.assertLess(
            stats["downstairs_points"], stats["downstairs_raw_points"]
        )
        self.assertEqual(first["downstairs_confirmed_cells"], 0)
        self.assertGreater(stats["downstairs_confirmed_cells"], 0)
        self.assertGreater(np.count_nonzero(downstairs), 0)
        self.assertTrue(np.all(occupancy[downstairs] == 1.0))
        self.assertTrue(np.all(explored[downstairs] == 1.0))

        free_count[downstairs] = 1000
        occupancy[downstairs] = 0.0
        self._update(maps, enabled=True)
        self.assertTrue(np.all(occupancy[downstairs] == 1.0))

    def test_low_surface_is_not_obstacle_when_disabled(self):
        maps = self._new_maps()
        stats = self._update(maps, enabled=False)
        _, _, occupancy, _, downstairs, _ = maps

        self.assertEqual(stats["downstairs_points"], 0)
        self.assertEqual(np.count_nonzero(downstairs), 0)
        self.assertEqual(np.count_nonzero(occupancy), 0)

    def test_flat_ground_is_not_marked_as_downstairs(self):
        maps = self._new_maps()
        height, width = 40, 60
        depth = np.zeros((height, width), dtype=np.float32)
        fov = np.deg2rad(79.0)
        focal = width / (2.0 * np.tan(fov / 2.0))
        center_y = height / 2.0
        for row in range(int(center_y) + 1, height):
            depth[row, :] = 0.88 * focal / (row - center_y)

        self._update(maps, enabled=True, depth=depth)
        stats = self._update(maps, enabled=True, depth=depth)
        _, _, occupancy, _, downstairs, _ = maps

        self.assertEqual(stats["downstairs_raw_points"], 0)
        self.assertEqual(np.count_nonzero(downstairs), 0)
        self.assertEqual(np.count_nonzero(occupancy), 0)

    def test_robot_footprint_clears_stale_downstairs_obstacle(self):
        maps = self._new_maps()
        _, _, occupancy, _, downstairs, evidence = maps
        downstairs[50, 50] = True
        evidence[50, 50] = 10
        occupancy[50, 50] = 1.0

        empty_depth = np.zeros((8, 8), dtype=np.float32)
        empty_depth[4, 4] = 1.0
        self._update(maps, enabled=True, depth=empty_depth)

        self.assertFalse(downstairs[50, 50])
        self.assertEqual(evidence[50, 50], 0)
        self.assertEqual(occupancy[50, 50], 0.0)


if __name__ == "__main__":
    unittest.main()
