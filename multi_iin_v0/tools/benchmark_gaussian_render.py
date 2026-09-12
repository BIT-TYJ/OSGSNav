#!/usr/bin/env python3
"""Benchmark exact-output NVSM rendering paths on a saved Gaussian map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _look_at_w2c(camera_position, center):
    forward = np.asarray(center) - np.asarray(camera_position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_position
    return np.linalg.inv(c2w @ np.diag([1.0, -1.0, -1.0, 1.0])).astype(np.float32)


def _timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _quality(reference, candidate):
    differences = []
    exact = []
    for first, second in zip(reference, candidate):
        delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
        differences.append(float(delta.mean()))
        exact.append(bool(np.array_equal(first, second)))
    return {
        "mean_abs_u8": float(np.mean(differences)),
        "max_view_mean_abs_u8": float(np.max(differences)),
        "all_pixel_exact": all(exact),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--center", nargs=3, type=float, required=True)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--streams", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    from gaussian_mapping.gaussian_splatting.gaussian_renderer import render
    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
    from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from gaussian_mapping.utils.camera_utils import Camera
    from multi_iin_v0.runtime.render_matching_acceleration import _create_render_only_cameras

    width, height = 640, 480
    fovx = math.radians(79.0)
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = fx
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    cx, cy = width / 2.0, height / 2.0
    projection = getProjectionMatrix2(
        0.01, 100.0, cx, cy, fx, fy, width, height
    ).transpose(0, 1).cuda()

    gaussians = GaussianModel(1)
    gaussians.load_ply(args.ply)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False)
    center = np.asarray(args.center, dtype=np.float32)
    poses = []
    for angle in np.linspace(0.0, 2.0 * np.pi, args.views, endpoint=False):
        position = center + np.asarray(
            [args.radius * np.cos(angle), 0.3, args.radius * np.sin(angle)],
            dtype=np.float32,
        )
        poses.append(_look_at_w2c(position, center))
    slam = SimpleNamespace(
        gaussians=gaussians,
        projection_matrix=projection,
        width=width,
        height=height,
        fovx=fovx,
        fovy=fovy,
    )

    def original_cameras():
        dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_depth = np.zeros((height, width), dtype=np.float32)
        output = []
        for index, pose in enumerate(poses):
            color = torch.from_numpy(dummy_rgb).float().permute(2, 0, 1).cuda() / 255.0
            transform = torch.from_numpy(pose).float().cuda()
            camera = Camera(
                -(index + 1), color, dummy_depth, transform, projection,
                fx, fy, cx, cy, fovx, fovy, height, width,
            )
            camera.update_RT(transform[:3, :3], transform[:3, 3])
            output.append(camera)
        return output

    def baseline():
        images = []
        cameras = original_cameras()
        with torch.no_grad():
            for camera in cameras:
                image = render(camera, gaussians, pipeline, background)["render"]
                images.append((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        return images

    def deferred():
        cameras = _create_render_only_cameras(slam, poses)
        with torch.no_grad():
            tensors = [
                render(camera, gaussians, pipeline, background)["render"].detach()
                for camera in cameras
            ]
        array = torch.stack(tensors).cpu().numpy().transpose(0, 2, 3, 1)
        return (array * 255).astype(np.uint8)

    streams = [torch.cuda.Stream() for _ in range(max(0, args.streams))]

    def compact_immediate():
        images = []
        cameras = _create_render_only_cameras(slam, poses)
        with torch.no_grad():
            for camera in cameras:
                image = render(camera, gaussians, pipeline, background)["render"]
                images.append((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        return images

    def streamed():
        if not streams:
            raise RuntimeError("streamed benchmark disabled")
        cameras = _create_render_only_cameras(slam, poses)
        tensors = [None] * len(cameras)
        producer = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(producer)
        with torch.no_grad():
            for index, camera in enumerate(cameras):
                with torch.cuda.stream(streams[index % len(streams)]):
                    tensors[index] = render(
                        camera, gaussians, pipeline, background
                    )["render"].detach()
        for stream in streams:
            producer.wait_stream(stream)
        array = torch.stack(tensors).cpu().numpy().transpose(0, 2, 3, 1)
        return (array * 255).astype(np.uint8)

    reference, _ = _timed(baseline)
    _timed(baseline)
    _timed(compact_immediate)
    _timed(deferred)
    if streams:
        _timed(streamed)
    baseline_times, compact_times, deferred_times, streamed_times = [], [], [], []
    compact_candidate = None
    deferred_candidate = None
    streamed_candidate = None
    for _ in range(max(1, args.repeats)):
        _, elapsed = _timed(baseline)
        baseline_times.append(elapsed)
        compact_candidate, elapsed = _timed(compact_immediate)
        compact_times.append(elapsed)
        deferred_candidate, elapsed = _timed(deferred)
        deferred_times.append(elapsed)
        if streams:
            streamed_candidate, elapsed = _timed(streamed)
            streamed_times.append(elapsed)
    base_median = statistics.median(baseline_times)
    compact_median = statistics.median(compact_times)
    deferred_median = statistics.median(deferred_times)
    streamed_median = statistics.median(streamed_times) if streamed_times else None
    print({
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "views": len(poses),
        "baseline_median_s": base_median,
        "compact_camera_immediate_median_s": compact_median,
        "compact_camera_speedup": base_median / compact_median,
        "deferred_median_s": deferred_median,
        "deferred_speedup": base_median / deferred_median,
        "stream_count": len(streams),
        "streamed_median_s": streamed_median,
        "streamed_speedup": (
            base_median / streamed_median if streamed_median is not None else None
        ),
        "baseline_times_s": baseline_times,
        "compact_camera_times_s": compact_times,
        "deferred_times_s": deferred_times,
        "streamed_times_s": streamed_times,
        "deferred_quality": _quality(reference, deferred_candidate),
        "compact_camera_quality": _quality(reference, compact_candidate),
        "streamed_quality": (
            _quality(reference, streamed_candidate)
            if streamed_candidate is not None else None
        ),
    })


if __name__ == "__main__":
    main()
