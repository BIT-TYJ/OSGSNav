import unittest

from multi_iin_v0.runtime.localization_runtime import FastLocalizationConfig
from multi_iin_v0.runtime.runtime_timing import RuntimeTimingRecorder


class RuntimeTimingTest(unittest.TestCase):
    def test_default_fast_pre_budget_is_384_renders(self):
        self.assertEqual(FastLocalizationConfig().expected_pre_renders, 384)

    def test_ordinary_summary_excludes_long_virtual_view_frame(self):
        recorder = RuntimeTimingRecorder()
        recorder.record_frame(
            {"total_frame_s": 0.8, "gaussmi_s": 0.0,
             "goal_override_localization_s": 0.01}
        )
        recorder.record_frame(
            {"total_frame_s": 180.0, "gaussmi_s": 179.0,
             "goal_override_localization_s": 0.01}
        )
        summary = recorder.summary()
        self.assertEqual(summary["all_frames"]["total_frame_s"]["count"], 2)
        self.assertAlmostEqual(
            summary["all_frames"]["total_frame_s"]["total_s"], 180.8
        )
        self.assertAlmostEqual(
            summary["all_frames"]["total_frame_s"]["mean_s"], 90.4
        )
        self.assertEqual(
            summary["ordinary_frames"]["total_frame_s"]["count"], 1
        )
        self.assertAlmostEqual(
            summary["ordinary_frames"]["total_frame_s"]["median_s"], 0.8
        )


if __name__ == "__main__":
    unittest.main()
