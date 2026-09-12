import unittest

import numpy as np

from multi_iin_v0.runtime import localization_runtime


class _Localizer:
    _multi_iin_confirmed_center = np.asarray([5.0, 0.5, -2.0], dtype=np.float32)


class LocalizationRuntimeViewpointTest(unittest.TestCase):
    def test_center_lock_preserves_localized_camera_navigation_pose(self):
        camera = np.asarray([4.0, 1.2, -3.0], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = camera
        w2c = np.linalg.inv(c2w)
        original = localization_runtime._ORIGINAL_COMPLETE
        localization_runtime._ORIGINAL_COMPLETE = lambda self, *a, **k: {
            "status": "localization_complete",
            "best_pose": w2c,
            "target_position": np.asarray([99.0, 0.0, 99.0]),
        }
        try:
            result = localization_runtime._locked_complete(_Localizer())
        finally:
            localization_runtime._ORIGINAL_COMPLETE = original
        np.testing.assert_allclose(result["target_position"], camera)
        np.testing.assert_allclose(
            result["target_object_center"], _Localizer._multi_iin_confirmed_center
        )
        self.assertEqual(result["position_source"], "localized_target_camera_pose")


if __name__ == "__main__":
    unittest.main()
