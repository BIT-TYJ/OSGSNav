import unittest

import numpy as np

from gaussian_mapping.mapper.realtime_3dgs import RealtimeSLAM


def _make_slam(target_ids=()):
    slam = RealtimeSLAM.__new__(RealtimeSLAM)
    slam.kf_indices = list(range(1, 21))
    slam.keyframes = {kf_id: object() for kf_id in slam.kf_indices}
    slam.kf_metadata = {}
    for kf_id in slam.kf_indices:
        pixels = 100 + kf_id if kf_id in target_ids else 0
        slam.kf_metadata[kf_id] = {
            "stage": "minav_exploration",
            "has_target": pixels > 0,
            "target_loss_mask_pixels": pixels,
        }
    slam.training_window_size = 8
    slam.target_focus_window_size = 12
    slam.phase1_current_frame_iters = 4
    slam.phase1_recent_window_iters = 4
    slam.phase1_target_replay_iters = 4
    slam.phase1_other_replay_iters = 8
    slam.semantic_guided_optimization_enabled = True
    return slam


class Phase1TrainingScheduleTest(unittest.TestCase):
    def test_uses_disjoint_4_4_4_8_pools(self):
        np.random.seed(7)
        slam = _make_slam(target_ids={2, 5, 15, 19})
        schedule = slam._phase1_training_schedule(20)

        recent = set(range(13, 20))
        historical_targets = {2, 5}
        ordinary_history = set(range(1, 13)) - historical_targets
        self.assertEqual(len(schedule), 20)
        self.assertEqual(schedule.count(20), 4)
        self.assertEqual(sum(kf_id in recent for kf_id in schedule), 4)
        self.assertEqual(
            sum(kf_id in historical_targets for kf_id in schedule), 4
        )
        self.assertEqual(
            sum(kf_id in ordinary_history for kf_id in schedule), 8
        )

    def test_missing_target_pool_falls_back_to_ordinary_history(self):
        np.random.seed(11)
        slam = _make_slam()
        schedule = slam._phase1_training_schedule(20)

        recent = set(range(13, 20))
        ordinary_history = set(range(1, 13))
        self.assertEqual(schedule.count(20), 4)
        self.assertEqual(sum(kf_id in recent for kf_id in schedule), 4)
        self.assertEqual(
            sum(kf_id in ordinary_history for kf_id in schedule), 12
        )

    def test_disabled_uses_original_recent_window_and_no_target_mask(self):
        slam = _make_slam(target_ids={2, 5, 15, 19})
        slam.semantic_guided_optimization_enabled = False
        sentinel_mask = object()
        slam.kf_metadata[19]["target_loss_mask"] = sentinel_mask

        self.assertEqual(slam._mapping_keyframe_pool(), list(range(13, 21)))
        self.assertIsNone(slam._keyframe_target_loss_mask(19))
        self.assertIsNone(
            slam.refresh_training_focus(stage="prelocalization", reason="test")
        )

    def test_enabled_keeps_semantic_mask_available(self):
        slam = _make_slam(target_ids={19})
        sentinel_mask = object()
        slam.kf_metadata[19]["target_loss_mask"] = sentinel_mask

        self.assertEqual(slam._mapping_keyframe_pool(), list(range(1, 21)))
        self.assertIs(slam._keyframe_target_loss_mask(19), sentinel_mask)


if __name__ == "__main__":
    unittest.main()
