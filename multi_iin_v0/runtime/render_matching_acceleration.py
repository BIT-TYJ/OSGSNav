"""Additive Multi-IIN acceleration for NVSM/PRE rendering and FEM extraction.

The original IEVE classes are left unchanged.  The Multi-IIN runner installs
these process-local wrappers before it creates any localizers.  View counts,
image resolution, LightGlue order, scores, and thresholds are preserved.  The
default path uses inference-only cameras and upstream per-image SuperPoint;
optional experimental batch/stream knobs remain disabled by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List

import cv2
import numpy as np


def _create_render_only_cameras(slam: Any, poses: List[np.ndarray]) -> List[Any]:
    """Create the minimal immutable camera state required by ``gs_render``.

    ``RealtimeSLAM.create_camera`` uploads a full zero RGB-D observation and
    allocates trainable pose/exposure parameters for every novel view.  None
    of those tensors are read by inference rendering.  For an 80-view NVSM
    call that means hundreds of MB of avoidable allocation and host-to-device
    traffic.  This path computes exactly the four matrices/fields consumed by
    the renderer and falls back at the caller if a mapper is incompatible.
    """

    import torch
    from types import SimpleNamespace
    from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getWorld2View2

    xyz = slam.gaussians.get_xyz
    device = xyz.device
    projection = getattr(slam, "projection_matrix").to(device=device)
    height = int(getattr(slam, "height", 480))
    width = int(getattr(slam, "width", 640))
    fovx = float(getattr(slam, "fovx"))
    fovy = float(getattr(slam, "fovy"))
    cameras = []
    for index, pose in enumerate(poses):
        transform = torch.as_tensor(
            np.asarray(pose, dtype=np.float32), device=device
        )
        world_view = getWorld2View2(
            transform[:3, :3], transform[:3, 3]
        ).transpose(0, 1)
        full_projection = world_view.unsqueeze(0).bmm(
            projection.unsqueeze(0)
        ).squeeze(0)
        camera_center = torch.linalg.inv(world_view)[3, :3]
        cameras.append(
            SimpleNamespace(
                uid=-(index + 1),
                image_height=height,
                image_width=width,
                FoVx=fovx,
                FoVy=fovy,
                projection_matrix=projection,
                world_view_transform=world_view,
                full_proj_transform=full_projection,
                camera_center=camera_center,
            )
        )
    return cameras


@dataclass(frozen=True)
class RenderMatchingAccelerationConfig:
    # Enabled after the 1.5M-Gaussian/80-view pixel-equivalence speed gate.
    enabled: bool = True
    # Batch-1 is the numerical-equivalence path.  Larger ragged SuperPoint
    # batches remain available for profiling, but do not improve latency on
    # the current L40 workload.
    superpoint_batch_size: int = 1
    lightglue_cuda_streams: int = 1
    defer_render_cpu_transfer: bool = False
    # PRE needs only a scalar score while searching.  Avoid materializing
    # per-view PnP/visualization caches; the winning REF view still uses the
    # unchanged full matcher afterwards.
    pre_score_only: bool = True
    # Keep PRE render tensors on GPU and reproduce OpenCV RGB->GRAY exactly
    # before SuperPoint, avoiding a GPU->CPU->GPU round trip per candidate.
    pre_gpu_direct: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "RenderMatchingAccelerationConfig":
        return cls(
            enabled=bool(int(getattr(args, "multi_render_matching_acceleration", 1))),
            superpoint_batch_size=max(
                1, int(getattr(args, "multi_fem_superpoint_batch_size", 1))
            ),
            lightglue_cuda_streams=max(
                1, int(getattr(args, "multi_fem_lightglue_cuda_streams", 1))
            ),
            defer_render_cpu_transfer=bool(
                int(getattr(args, "multi_defer_render_cpu_transfer", 0))
            ),
            pre_score_only=bool(
                int(getattr(args, "multi_pre_score_only_matching", 1))
            ),
            pre_gpu_direct=bool(
                int(getattr(args, "multi_pre_gpu_direct_matching", 1))
            ),
        )


class RenderMatchingAccelerationManager:
    def __init__(self, config: RenderMatchingAccelerationConfig):
        self.config = config
        self.fem_calls = 0
        self.fem_views = 0
        self.superpoint_batches = 0
        self.superpoint_fallback_views = 0
        self.last_superpoint_fallback_error = None
        self.lightglue_streamed_calls = 0
        self.fem_extract_seconds = 0.0
        self.fem_match_seconds = 0.0
        self.fem_total_seconds = 0.0
        self.render_calls = 0
        self.render_views = 0
        self.render_camera_seconds = 0.0
        self.render_gpu_seconds = 0.0
        self.render_transfer_seconds = 0.0
        self.render_total_seconds = 0.0
        self.render_fast_camera_calls = 0
        self.render_camera_fallback_calls = 0
        self.last_render_camera_fallback_error = None
        self.pre_render_calls = 0
        self.pre_render_fallbacks = 0
        self.pre_render_total_seconds = 0.0
        self.last_pre_render_fallback_error = None
        self.pre_score_only_calls = 0
        self.pre_score_only_fallbacks = 0
        self.pre_score_only_total_seconds = 0.0
        self.last_pre_score_only_fallback_error = None
        self.pre_gpu_direct_calls = 0

    @classmethod
    def from_args(cls, args: Any):
        return cls(RenderMatchingAccelerationConfig.from_args(args))

    @staticmethod
    def _mean(total: float, count: int) -> float:
        return float(total / count) if count else 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "fem": {
                "calls": self.fem_calls,
                "views": self.fem_views,
                "superpoint_batches": self.superpoint_batches,
                "fallback_views": self.superpoint_fallback_views,
                "last_fallback_error": self.last_superpoint_fallback_error,
                "lightglue_cuda_streams": self.config.lightglue_cuda_streams,
                "lightglue_streamed_calls": self.lightglue_streamed_calls,
                "extract_total_s": self.fem_extract_seconds,
                "match_total_s": self.fem_match_seconds,
                "total_s": self.fem_total_seconds,
                "mean_call_s": self._mean(self.fem_total_seconds, self.fem_calls),
            },
            "render": {
                "calls": self.render_calls,
                "views": self.render_views,
                "camera_total_s": self.render_camera_seconds,
                "gpu_total_s": self.render_gpu_seconds,
                "transfer_total_s": self.render_transfer_seconds,
                "total_s": self.render_total_seconds,
                "mean_call_s": self._mean(self.render_total_seconds, self.render_calls),
                "fast_camera_calls": self.render_fast_camera_calls,
                "camera_fallback_calls": self.render_camera_fallback_calls,
                "last_camera_fallback_error": self.last_render_camera_fallback_error,
            },
            "pre_render": {
                "calls": self.pre_render_calls,
                "fallbacks": self.pre_render_fallbacks,
                "total_s": self.pre_render_total_seconds,
                "mean_s": self._mean(
                    self.pre_render_total_seconds, self.pre_render_calls
                ),
                "last_fallback_error": self.last_pre_render_fallback_error,
            },
            "pre_score_only": {
                "enabled": self.config.pre_score_only,
                "calls": self.pre_score_only_calls,
                "fallbacks": self.pre_score_only_fallbacks,
                "total_s": self.pre_score_only_total_seconds,
                "mean_s": self._mean(
                    self.pre_score_only_total_seconds,
                    self.pre_score_only_calls,
                ),
                "last_fallback_error": self.last_pre_score_only_fallback_error,
                "gpu_direct_enabled": self.config.pre_gpu_direct,
                "gpu_direct_calls": self.pre_gpu_direct_calls,
            },
        }


def _split_batched_features(features: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    import torch

    split = [dict() for _ in range(batch_size)]
    for key, value in features.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            for index in range(batch_size):
                split[index][key] = value[index : index + 1]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            for index in range(batch_size):
                split[index][key] = value[index]
        else:
            for index in range(batch_size):
                split[index][key] = value
    return split


def _superpoint_forward_ragged(extractor: Any, image: Any) -> List[Dict[str, Any]]:
    """Run SuperPoint's CNN once for a batch and retain ragged keypoints.

    Upstream SuperPoint stacks the final per-image keypoints and therefore
    raises when views contain different numbers of detections.  NVSM renders
    frequently have textureless/black regions, so that assumption does not
    hold.  This mirrors upstream ``SuperPoint.forward`` but returns one feature
    dictionary per image after the shared batched CNN pass.
    """

    import torch
    from kornia.color import rgb_to_grayscale
    from lightglue.superpoint import simple_nms, sample_descriptors, top_k_keypoints

    if image.shape[1] == 3:
        image = rgb_to_grayscale(image)

    x = extractor.relu(extractor.conv1a(image))
    x = extractor.relu(extractor.conv1b(x))
    x = extractor.pool(x)
    x = extractor.relu(extractor.conv2a(x))
    x = extractor.relu(extractor.conv2b(x))
    x = extractor.pool(x)
    x = extractor.relu(extractor.conv3a(x))
    x = extractor.relu(extractor.conv3b(x))
    x = extractor.pool(x)
    x = extractor.relu(extractor.conv4a(x))
    x = extractor.relu(extractor.conv4b(x))

    detector = extractor.relu(extractor.convPa(x))
    scores = extractor.convPb(detector)
    scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
    batch_size, _, height, width = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(
        batch_size, height, width, 8, 8
    )
    scores = scores.permute(0, 1, 3, 2, 4).reshape(
        batch_size, height * 8, width * 8
    )
    scores = simple_nms(scores, extractor.conf.nms_radius)

    if extractor.conf.remove_borders:
        pad = extractor.conf.remove_borders
        scores[:, :pad] = -1
        scores[:, :, :pad] = -1
        scores[:, -pad:] = -1
        scores[:, :, -pad:] = -1

    best = torch.where(scores > extractor.conf.detection_threshold)
    selected_scores = scores[best]
    keypoints = [
        torch.stack(best[1:3], dim=-1)[best[0] == index]
        for index in range(batch_size)
    ]
    selected_scores = [
        selected_scores[best[0] == index] for index in range(batch_size)
    ]
    if extractor.conf.max_num_keypoints is not None:
        selected = [
            top_k_keypoints(points, point_scores, extractor.conf.max_num_keypoints)
            for points, point_scores in zip(keypoints, selected_scores)
        ]
        keypoints = [item[0] for item in selected]
        selected_scores = [item[1] for item in selected]
    keypoints = [torch.flip(points, [1]).float() for points in keypoints]

    descriptor_head = extractor.relu(extractor.convDa(x))
    dense_descriptors = extractor.convDb(descriptor_head)
    dense_descriptors = torch.nn.functional.normalize(
        dense_descriptors, p=2, dim=1
    )
    descriptors = [
        sample_descriptors(points[None], dense[None], 8)[0]
        .transpose(-1, -2)
        .contiguous()
        for points, dense in zip(keypoints, dense_descriptors)
    ]

    return [
        {
            "keypoints": points[None],
            "keypoint_scores": point_scores[None],
            "descriptors": descriptor[None],
        }
        for points, point_scores, descriptor in zip(
            keypoints, selected_scores, descriptors
        )
    ]


def _extract_superpoint_batch(fem: Any, images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Mirror ``Extractor.extract`` while allowing ragged batch outputs."""

    import torch
    from lightglue.utils import ImagePreprocessor

    grays = []
    shape = None
    for image in images:
        array = np.asarray(image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array
        if gray.ndim != 2:
            raise ValueError("FEM batch contains a non-image")
        if shape is None:
            shape = gray.shape
        if gray.shape != shape:
            raise ValueError("FEM batch images must have equal dimensions")
        grays.append(gray.astype(np.float32) / 255.0)

    batch = torch.from_numpy(np.stack(grays, axis=0)).unsqueeze(1).to(fem.device)
    original_wh = batch.shape[-2:][::-1]
    resized, scales = ImagePreprocessor(
        **dict(fem.extractor.preprocess_conf)
    )(batch)
    features_per_image = _superpoint_forward_ragged(fem.extractor, resized)
    image_size = torch.tensor(
        original_wh, device=batch.device, dtype=batch.dtype
    ).unsqueeze(0)
    # ImagePreprocessor returns one common XY scale for an equal-sized batch.
    scale = scales.reshape(1, 1, 2)
    for features in features_per_image:
        features["image_size"] = image_size.clone()
        features["keypoints"] = (
            (features["keypoints"] + 0.5) / scale - 0.5
        )
    return features_per_image


def _match_lightglue_score_only(
    fem: Any, image: np.ndarray, feature_mode: Any = None
) -> float:
    """Return exactly FEM's LightGlue evidence score without debug caches.

    PRE evaluates hundreds of disposable candidates.  The upstream matcher
    copies matched keypoints and confidences to CPU, walks semantic masks in
    Python, and stores a complete MR cache for every candidate even though PRE
    consumes only ``sum(scores)``.  Keep the same SuperPoint, LightGlue,
    minimum-match rule and floating-point reduction, while omitting those
    unused side effects.  REF subsequently runs the full matcher and produces
    the cache that MR actually consumes.
    """

    import torch

    target_features, _active_mode = fem._get_target_features(feature_mode)
    if target_features is None:
        return 0.0

    with torch.no_grad():
        # PRE's optimized renderer can provide the exact normalized grayscale
        # tensor directly.  Normal numpy inputs retain the original FEM path.
        if torch.is_tensor(image):
            query_features = fem.extractor.extract(image)
        else:
            query_features = fem._extract_features(image)
        matches = fem.matcher(
            {"image0": target_features, "image1": query_features}
        )

    matches_data = matches.get("matches", None)
    if isinstance(matches_data, list) and matches_data:
        matched = matches_data[0]
        num_matches = int(matched.shape[0]) if torch.is_tensor(matched) else 0
    elif torch.is_tensor(matches_data):
        num_matches = int(matches_data.shape[0])
    else:
        num_matches = 0
    if num_matches < 10:
        return 0.0

    scores_data = matches.get("scores", None)
    if isinstance(scores_data, list) and scores_data:
        confidence = scores_data[0]
        if not (torch.is_tensor(confidence) and confidence.numel() > 0):
            confidence = torch.ones(num_matches, device=fem.device)
    elif torch.is_tensor(scores_data) and scores_data.numel() > 0:
        confidence = scores_data
    else:
        confidence = torch.ones(num_matches, device=fem.device)
    return float(confidence.sum().item())


def _rgb_u8_to_cv_gray_tensor(rgb: Any) -> Any:
    """Reproduce ``cv2.COLOR_RGB2GRAY`` exactly for a CHW uint8 tensor."""

    import torch

    if not torch.is_tensor(rgb) or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("expected a CHW RGB tensor")
    rgb_i32 = rgb.to(dtype=torch.int32)
    # OpenCV's 15-bit integer coefficients for 8-bit RGB conversion.
    gray_i32 = (
        rgb_i32[0] * 9798
        + rgb_i32[1] * 19235
        + rgb_i32[2] * 3735
        + 16384
    ) >> 15
    return gray_i32.to(dtype=torch.float32).div_(255.0).unsqueeze(0)


def install_render_matching_acceleration(args: Any) -> RenderMatchingAccelerationManager:
    """Install process-local wrappers and return their shared statistics."""

    manager = RenderMatchingAccelerationManager.from_args(args)
    if not manager.config.enabled:
        return manager

    _install_fem_wrapper(manager)
    _install_render_wrapper(manager)
    _install_pre_render_wrapper(manager)
    _install_pre_score_only_wrapper(manager)
    return manager


def _install_fem_wrapper(manager: RenderMatchingAccelerationManager) -> None:
    import torch
    from ieve_minav.localizer.feature_matcher import FeatureMatcher

    if getattr(FeatureMatcher, "_multi_batch_extraction_installed", False):
        return
    original = FeatureMatcher._batch_match_lightglue

    def accelerated(self, rendered_images, target_features=None, feature_mode=None):
        started = time.perf_counter()
        n = len(rendered_images)
        self._last_batch_match_caches = [None] * n
        if target_features is None:
            target_features, _ = self._get_target_features()
        if target_features is None:
            return [0.0] * n, [0] * n

        valid_indices = [
            index
            for index, item in enumerate(rendered_images)
            if item.get("image") is not None
        ]
        all_features = [None] * n
        extraction_started = time.perf_counter()
        chunk_size = manager.config.superpoint_batch_size
        with torch.no_grad():
            for offset in range(0, len(valid_indices), chunk_size):
                indices = valid_indices[offset : offset + chunk_size]
                images = [rendered_images[index]["image"] for index in indices]
                try:
                    if len(images) == 1:
                        # Use upstream extract verbatim on the default path.
                        # This removes batched-convolution numerical drift.
                        tensor = self._numpy_to_torch_image(images[0])
                        batch_features = [self.extractor.extract(tensor)]
                    else:
                        batch_features = _extract_superpoint_batch(self, images)
                        manager.superpoint_batches += 1
                    for index, features in zip(indices, batch_features):
                        all_features[index] = features
                except Exception as exc:
                    manager.last_superpoint_fallback_error = repr(exc)
                    self.logger.debug(
                        "Batched SuperPoint fallback for %s views: %s", len(indices), exc
                    )
                    manager.superpoint_fallback_views += len(indices)
                    for index, image in zip(indices, images):
                        tensor = self._numpy_to_torch_image(image)
                        all_features[index] = self.extractor.extract(tensor)
        extraction_elapsed = time.perf_counter() - extraction_started

        match_started = time.perf_counter()
        scores = [0.0] * n
        match_counts = [0] * n
        match_caches = [None] * n
        raw_matches = [None] * n
        with torch.no_grad():
            use_streams = (
                torch.cuda.is_available()
                and getattr(self, "device", torch.device("cpu")).type == "cuda"
                and manager.config.lightglue_cuda_streams > 1
                and len(valid_indices) > 1
            )
            if use_streams:
                stream_count = min(
                    manager.config.lightglue_cuda_streams, len(valid_indices)
                )
                streams = getattr(manager, "_lightglue_streams", None)
                if streams is None or len(streams) != stream_count:
                    streams = [
                        torch.cuda.Stream(device=self.device)
                        for _ in range(stream_count)
                    ]
                    manager._lightglue_streams = streams
                producer = torch.cuda.current_stream(device=self.device)
                for stream in streams:
                    stream.wait_stream(producer)
                for order, index in enumerate(valid_indices):
                    features = all_features[index]
                    if features is None:
                        continue
                    with torch.cuda.stream(streams[order % stream_count]):
                        raw_matches[index] = self.matcher(
                            {"image0": target_features, "image1": features}
                        )
                for stream in streams:
                    producer.wait_stream(stream)
                manager.lightglue_streamed_calls += 1
            else:
                for index in valid_indices:
                    features = all_features[index]
                    if features is None:
                        continue
                    raw_matches[index] = self.matcher(
                        {"image0": target_features, "image1": features}
                    )

            for index in valid_indices:
                features = all_features[index]
                matches = raw_matches[index]
                if features is None or matches is None:
                    continue
                try:
                    matches_data = matches.get("matches", None)
                    if isinstance(matches_data, list) and matches_data:
                        matched = matches_data[0]
                        num_matches = int(matched.shape[0]) if torch.is_tensor(matched) else 0
                    elif torch.is_tensor(matches_data):
                        num_matches = int(matches_data.shape[0])
                    else:
                        num_matches = 0
                    match_counts[index] = num_matches
                    if num_matches < 10:
                        continue
                    scores_data = matches.get("scores", None)
                    if isinstance(scores_data, list) and scores_data:
                        confidence = scores_data[0]
                        if not (torch.is_tensor(confidence) and confidence.numel() > 0):
                            confidence = torch.ones(num_matches, device=self.device)
                    elif torch.is_tensor(scores_data) and scores_data.numel() > 0:
                        confidence = scores_data
                    else:
                        confidence = torch.ones(num_matches, device=self.device)
                    match_caches[index] = self._build_lightglue_match_cache(
                        target_features=target_features,
                        query_features=features,
                        matches_data=matches_data,
                        scores=confidence,
                        num_matches=num_matches,
                        feature_mode=feature_mode,
                    )
                    scores[index] = float(confidence.sum().item())
                except Exception:
                    scores[index] = 0.0
                    match_counts[index] = 0
                    match_caches[index] = None

        match_elapsed = time.perf_counter() - match_started
        manager.fem_calls += 1
        manager.fem_views += n
        manager.fem_extract_seconds += extraction_elapsed
        manager.fem_match_seconds += match_elapsed
        manager.fem_total_seconds += time.perf_counter() - started
        self._last_batch_match_caches = match_caches
        return scores, match_counts

    FeatureMatcher._multi_original_batch_match_lightglue = original
    FeatureMatcher._batch_match_lightglue = accelerated
    FeatureMatcher._multi_batch_extraction_installed = True


def _install_render_wrapper(manager: RenderMatchingAccelerationManager) -> None:
    import torch
    from gaussian_mapping.gaussian_splatting.gaussian_renderer import render as gs_render
    from ieve_minav.localizer.novel_view_renderer import NovelViewRenderer

    if getattr(NovelViewRenderer, "_multi_deferred_transfer_installed", False):
        return
    original = NovelViewRenderer._batch_render

    def accelerated(self, poses):
        started = time.perf_counter()
        slam = getattr(self.slam_mapper, "slam", None)
        if (
            slam is None
            or not getattr(self.slam_mapper, "is_running", False)
            or not hasattr(slam, "gaussians")
            or slam.gaussians.get_xyz.shape[0] == 0
        ):
            return [None] * len(poses)

        height = getattr(slam, "height", 480)
        width = getattr(slam, "width", 640)
        camera_started = time.perf_counter()
        try:
            cameras = _create_render_only_cameras(slam, poses)
            manager.render_fast_camera_calls += 1
        except Exception as exc:
            manager.render_camera_fallback_calls += 1
            manager.last_render_camera_fallback_error = repr(exc)
            dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            dummy_depth = np.zeros((height, width), dtype=np.float32)
            cameras = []
            for index, pose in enumerate(poses):
                try:
                    cameras.append(
                        slam.create_camera(
                            dummy_rgb, dummy_depth, pose, uid=-(index + 1)
                        )
                    )
                except Exception:
                    cameras.append(None)
        camera_elapsed = time.perf_counter() - camera_started

        render_started = time.perf_counter()
        tensors = []
        tensor_indices = []
        results = [None] * len(cameras)
        with torch.no_grad():
            for index, camera in enumerate(cameras):
                if camera is None:
                    continue
                try:
                    package = gs_render(
                        camera, slam.gaussians, slam.pipeline_params, slam.background
                    )
                    rendered = package["render"].detach()
                    if manager.config.defer_render_cpu_transfer:
                        tensors.append(rendered)
                        tensor_indices.append(index)
                    else:
                        results[index] = (
                            rendered.cpu().numpy().transpose(1, 2, 0) * 255
                        ).astype(np.uint8)
                except Exception:
                    continue
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=slam.gaussians.get_xyz.device)
        render_elapsed = time.perf_counter() - render_started

        transfer_started = time.perf_counter()
        if manager.config.defer_render_cpu_transfer and tensors:
            batch = torch.stack(tensors, dim=0).cpu().numpy()
            batch = (batch.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            for index, image in zip(tensor_indices, batch):
                results[index] = image
        transfer_elapsed = time.perf_counter() - transfer_started

        manager.render_calls += 1
        manager.render_views += len(poses)
        manager.render_camera_seconds += camera_elapsed
        manager.render_gpu_seconds += render_elapsed
        manager.render_transfer_seconds += transfer_elapsed
        manager.render_total_seconds += time.perf_counter() - started
        return results

    NovelViewRenderer._multi_original_batch_render = original
    NovelViewRenderer._batch_render = accelerated
    NovelViewRenderer._multi_deferred_transfer_installed = True


def _install_pre_render_wrapper(manager: RenderMatchingAccelerationManager) -> None:
    """Use the same inference-only camera for PRE's many one-view renders."""

    import torch
    from gaussian_mapping.gaussian_splatting.gaussian_renderer import render as gs_render
    from ieve_minav.localizer.pose_pre_refiner import PosePreRefiner

    if getattr(PosePreRefiner, "_multi_render_only_camera_installed", False):
        return
    original = PosePreRefiner._render_view

    def accelerated(self, w2c):
        started = time.perf_counter()
        manager.pre_render_calls += 1
        try:
            slam = getattr(self.slam_mapper, "slam", None)
            if (
                slam is None
                or not hasattr(slam, "gaussians")
                or slam.gaussians.get_xyz.shape[0] == 0
            ):
                return None
            camera = _create_render_only_cameras(slam, [w2c])[0]
            package = gs_render(
                camera, slam.gaussians, slam.pipeline_params, slam.background
            )
            if package is None:
                return None
            rendered = package["render"].detach().clamp(0.0, 1.0)
            gpu_direct = (
                manager.config.pre_score_only
                and manager.config.pre_gpu_direct
                and bool(
                    getattr(
                        getattr(self, "fem", None),
                        "_multi_pre_score_only_active",
                        False,
                    )
                )
            )
            if gpu_direct:
                manager.pre_gpu_direct_calls += 1
                rgb_u8 = (rendered * 255.0).to(dtype=torch.uint8)
                return _rgb_u8_to_cv_gray_tensor(rgb_u8)
            image = rendered.cpu().numpy()
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            return (image * 255.0).astype(np.uint8)
        except Exception as exc:
            manager.pre_render_fallbacks += 1
            manager.last_pre_render_fallback_error = repr(exc)
            return original(self, w2c)
        finally:
            manager.pre_render_total_seconds += time.perf_counter() - started

    PosePreRefiner._multi_original_render_view = original
    PosePreRefiner._render_view = accelerated
    PosePreRefiner._multi_render_only_camera_installed = True


def _install_pre_score_only_wrapper(
    manager: RenderMatchingAccelerationManager,
) -> None:
    """Use scalar-only LightGlue evaluation only inside PRE candidate search."""

    from ieve_minav.localizer.feature_matcher import FeatureMatcher
    from ieve_minav.localizer.pose_pre_refiner import PosePreRefiner

    if not getattr(FeatureMatcher, "_multi_pre_score_only_installed", False):
        original_match = FeatureMatcher._match_with_lightglue

        def score_aware_match(self, image, feature_mode=None):
            if not getattr(self, "_multi_pre_score_only_active", False):
                return original_match(self, image, feature_mode=feature_mode)
            started = time.perf_counter()
            manager.pre_score_only_calls += 1
            try:
                return _match_lightglue_score_only(
                    self, image, feature_mode=feature_mode
                )
            except Exception as exc:
                # Preserve upstream error behavior and scoring if an unknown
                # LightGlue version returns an incompatible structure.
                manager.pre_score_only_fallbacks += 1
                manager.last_pre_score_only_fallback_error = repr(exc)
                return original_match(self, image, feature_mode=feature_mode)
            finally:
                manager.pre_score_only_total_seconds += (
                    time.perf_counter() - started
                )

        FeatureMatcher._multi_original_match_with_lightglue = original_match
        FeatureMatcher._match_with_lightglue = score_aware_match
        FeatureMatcher._multi_pre_score_only_installed = True

    if not getattr(PosePreRefiner, "_multi_pre_score_only_installed", False):
        original_refine = PosePreRefiner.refine

        def score_only_refine(self, coarse_pose, coarse_score):
            fem = getattr(self, "fem", None)
            if (
                fem is None
                or not manager.config.pre_score_only
                or not bool(getattr(fem, "lightglue_available", False))
            ):
                return original_refine(self, coarse_pose, coarse_score)
            previous = bool(
                getattr(fem, "_multi_pre_score_only_active", False)
            )
            fem._multi_pre_score_only_active = True
            try:
                return original_refine(self, coarse_pose, coarse_score)
            finally:
                fem._multi_pre_score_only_active = previous

        PosePreRefiner._multi_original_refine = original_refine
        PosePreRefiner.refine = score_only_refine
        PosePreRefiner._multi_pre_score_only_installed = True

    if not getattr(PosePreRefiner, "_multi_pre_full_image_save_installed", False):
        original_save = PosePreRefiner._save_refined_render

        def full_image_save(self, *args, **kwargs):
            # The visualization writer needs RGB numpy images, not the
            # grayscale GPU tensors used only for PRE candidate scoring.
            fem = getattr(self, "fem", None)
            if fem is None:
                return original_save(self, *args, **kwargs)
            previous = bool(
                getattr(fem, "_multi_pre_score_only_active", False)
            )
            fem._multi_pre_score_only_active = False
            try:
                return original_save(self, *args, **kwargs)
            finally:
                fem._multi_pre_score_only_active = previous

        PosePreRefiner._multi_original_save_refined_render = original_save
        PosePreRefiner._save_refined_render = full_image_save
        PosePreRefiner._multi_pre_full_image_save_installed = True
