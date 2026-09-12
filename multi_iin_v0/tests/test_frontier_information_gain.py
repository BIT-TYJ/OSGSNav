import unittest

import numpy as np

from multi_iin_v0.runtime.frontier_information_gain import (
    rank_frontier_candidates,
    resolve_target_local_step,
)


class FrontierInformationGainTest(unittest.TestCase):
    def test_larger_unknown_region_can_beat_shorter_path(self):
        explored = np.ones((60, 60), dtype=np.float32)
        explored[5:16, 5:16] = 0.0
        explored[32:55, 32:55] = 0.0
        candidates = [
            {"point": (10, 10), "fmm_cost": 10.0},
            {"point": (42, 42), "fmm_cost": 80.0},
        ]
        ranked, debug = rank_frontier_candidates(
            candidates, explored, radius_px=9, fmm_cost_weight=0.20
        )
        self.assertEqual(ranked[0]["point"], (42, 42))
        self.assertGreater(
            ranked[0]["unknown_gain_cells"], ranked[1]["unknown_gain_cells"]
        )
        self.assertEqual(debug["candidate_count"], 2)

    def test_fmm_cost_breaks_equal_information_gain(self):
        explored = np.zeros((50, 50), dtype=np.float32)
        candidates = [
            {"point": (15, 15), "fmm_cost": 20.0},
            {"point": (35, 35), "fmm_cost": 70.0},
        ]
        ranked, _ = rank_frontier_candidates(candidates, explored, radius_px=6)
        self.assertEqual(ranked[0]["point"], (15, 15))

    def test_visited_neighbourhood_is_penalized(self):
        explored = np.zeros((50, 50), dtype=np.float32)
        visited = np.zeros_like(explored)
        visited[9:22, 9:22] = 1.0
        candidates = [
            {"point": (15, 15), "fmm_cost": 30.0},
            {"point": (35, 35), "fmm_cost": 30.0},
        ]
        ranked, _ = rank_frontier_candidates(
            candidates,
            explored,
            visited_map=visited,
            radius_px=6,
            visited_ratio_penalty=0.25,
        )
        self.assertEqual(ranked[0]["point"], (35, 35))

    def test_target_clock_resets_independently(self):
        self.assertEqual(resolve_target_local_step({"time": 0}, 363), 0)
        self.assertEqual(resolve_target_local_step({"time": 137}, 500), 137)
        self.assertEqual(resolve_target_local_step({}, 500), 500)


if __name__ == "__main__":
    unittest.main()
