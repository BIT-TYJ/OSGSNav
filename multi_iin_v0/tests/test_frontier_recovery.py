import unittest

import numpy as np

from multi_iin_v0.runtime.frontier_recovery import recover_reachable_frontiers


class FrontierRecoveryTest(unittest.TestCase):
    def test_projects_disconnected_representative_inside_start_component(self):
        traversible = np.ones((20, 20), dtype=np.uint8)
        traversible[8, 2:16] = 0
        frontier = np.zeros_like(traversible, dtype=bool)
        frontier[7, 10] = True
        # Representative is on the other side of a remembered obstacle.
        candidates, debug = recover_reachable_frontiers(
            traversible,
            frontier,
            (3, 3),
            [(9, 10)],
            max_projection_px=20,
            min_goal_dist_px=1,
        )
        self.assertTrue(candidates)
        self.assertTrue(candidates[0]["frontier_recovery"])
        self.assertGreaterEqual(debug["output_count"], 1)

    def test_empty_component_is_safe(self):
        traversible = np.zeros((8, 8), dtype=np.uint8)
        frontier = np.ones((8, 8), dtype=bool)
        candidates, debug = recover_reachable_frontiers(
            traversible, frontier, (3, 3), [], min_goal_dist_px=1
        )
        self.assertEqual(candidates, [])
        self.assertEqual(debug["reason"], "empty_start_component")

    def test_fully_explored_component_returns_patrol_goal_when_enabled(self):
        traversible = np.ones((12, 12), dtype=np.uint8)
        frontier = np.zeros_like(traversible, dtype=bool)
        candidates, debug = recover_reachable_frontiers(
            traversible,
            frontier,
            (6, 6),
            [],
            min_goal_dist_px=3,
            allow_patrol_fallback=True,
        )
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["frontier_patrol"])
        self.assertTrue(debug["used_patrol_fallback"])


if __name__ == "__main__":
    unittest.main()
