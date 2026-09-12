from types import SimpleNamespace

import torch

from multi_iin_v0.runtime.render_matching_acceleration import (
    RenderMatchingAccelerationConfig,
    RenderMatchingAccelerationManager,
    _create_render_only_cameras,
    _match_lightglue_score_only,
    _rgb_u8_to_cv_gray_tensor,
    _split_batched_features,
)


def test_split_batched_features_preserves_batch_dimension():
    features = {
        "keypoints": torch.arange(24).reshape(3, 4, 2),
        "descriptors": torch.arange(60).reshape(3, 4, 5),
        "shared": "value",
    }
    split = _split_batched_features(features, 3)
    assert len(split) == 3
    assert split[1]["keypoints"].shape == (1, 4, 2)
    assert torch.equal(split[1]["keypoints"], features["keypoints"][1:2])
    assert split[2]["shared"] == "value"


def test_config_and_empty_snapshot_are_stable():
    args = SimpleNamespace(
        multi_render_matching_acceleration=1,
        multi_fem_superpoint_batch_size=8,
        multi_fem_lightglue_cuda_streams=3,
        multi_defer_render_cpu_transfer=1,
        multi_pre_score_only_matching=0,
        multi_pre_gpu_direct_matching=0,
    )
    config = RenderMatchingAccelerationConfig.from_args(args)
    manager = RenderMatchingAccelerationManager(config)
    snapshot = manager.snapshot()
    assert config.superpoint_batch_size == 8
    assert config.lightglue_cuda_streams == 3
    assert snapshot["fem"]["calls"] == 0
    assert snapshot["render"]["mean_call_s"] == 0.0
    assert config.pre_score_only is False
    assert config.pre_gpu_direct is False
    assert snapshot["pre_score_only"]["calls"] == 0


def test_validated_acceleration_is_enabled_by_default():
    config = RenderMatchingAccelerationConfig.from_args(SimpleNamespace())
    assert config.enabled is True
    assert config.superpoint_batch_size == 1
    assert config.lightglue_cuda_streams == 1
    assert config.defer_render_cpu_transfer is False
    assert config.pre_score_only is True
    assert config.pre_gpu_direct is True


def test_gpu_gray_conversion_is_pixel_equal_to_opencv():
    import cv2
    import numpy as np

    rng = np.random.default_rng(123)
    rgb = rng.integers(0, 256, (73, 91, 3), dtype=np.uint8)
    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    actual = (
        _rgb_u8_to_cv_gray_tensor(torch.from_numpy(rgb).permute(2, 0, 1))
        .mul(255.0)
        .to(torch.uint8)[0]
        .numpy()
    )
    assert np.array_equal(actual, expected)


def test_render_float_to_direct_gray_matches_previous_cpu_round_trip():
    import cv2
    import numpy as np

    rng = np.random.default_rng(456)
    rendered = rng.random((3, 61, 79), dtype=np.float32)
    previous_rgb = (
        rendered.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    expected = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    direct = _rgb_u8_to_cv_gray_tensor(
        (torch.from_numpy(rendered).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    )
    actual = direct.mul(255.0).to(torch.uint8)[0].numpy()
    assert np.array_equal(actual, expected)


def test_pre_score_only_uses_identical_confidence_sum_without_mutating_cache():
    class _FakeFEM:
        device = torch.device("cpu")
        _last_match_cache = {"keep": True}

        def _get_target_features(self, feature_mode=None):
            return {"keypoints": torch.zeros((1, 12, 2))}, "uniform"

        def _extract_features(self, image):
            return {"keypoints": torch.zeros((1, 12, 2))}

        def matcher(self, data):
            return {
                "matches": [torch.stack((torch.arange(12), torch.arange(12)), dim=1)],
                "scores": [torch.linspace(0.1, 0.9, 12)],
            }

    fem = _FakeFEM()
    expected = float(torch.linspace(0.1, 0.9, 12).sum().item())
    actual = _match_lightglue_score_only(
        fem, torch.zeros((8, 8, 3)).numpy(), feature_mode="uniform"
    )
    assert actual == expected
    assert fem._last_match_cache == {"keep": True}


def test_pre_score_only_preserves_minimum_match_rule():
    class _FakeFEM:
        device = torch.device("cpu")

        def _get_target_features(self, feature_mode=None):
            return {"keypoints": torch.zeros((1, 9, 2))}, "uniform"

        def _extract_features(self, image):
            return {"keypoints": torch.zeros((1, 9, 2))}

        def matcher(self, data):
            return {
                "matches": [torch.stack((torch.arange(9), torch.arange(9)), dim=1)],
                "scores": [torch.ones(9)],
            }

    assert _match_lightglue_score_only(
        _FakeFEM(), torch.zeros((8, 8, 3)).numpy()
    ) == 0.0


def test_render_only_camera_matches_training_camera_matrices():
    import math
    import numpy as np
    from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from gaussian_mapping.utils.camera_utils import Camera

    width, height = 64, 48
    fx = fy = 50.0
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    projection = getProjectionMatrix2(
        0.01, 100.0, width / 2, height / 2, fx, fy, width, height
    ).transpose(0, 1)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray([0.4, -0.2, 1.3], dtype=np.float32)
    transform = torch.from_numpy(pose)
    original = Camera(
        -1,
        torch.zeros((3, height, width)),
        np.zeros((height, width), dtype=np.float32),
        transform,
        projection,
        fx,
        fy,
        width / 2,
        height / 2,
        fovx,
        fovy,
        height,
        width,
        device="cpu",
    )
    original.update_RT(transform[:3, :3], transform[:3, 3])
    slam = SimpleNamespace(
        gaussians=SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        projection_matrix=projection,
        width=width,
        height=height,
        fovx=fovx,
        fovy=fovy,
    )
    compact = _create_render_only_cameras(slam, [pose])[0]
    assert torch.equal(compact.world_view_transform, original.world_view_transform)
    assert torch.equal(compact.full_proj_transform, original.full_proj_transform)
    assert torch.equal(compact.camera_center, original.camera_center)
