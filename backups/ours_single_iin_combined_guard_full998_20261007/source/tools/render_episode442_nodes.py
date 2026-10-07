#!/usr/bin/env python3
"""Render pose-aligned keyframes for Habitat episode 442 on CPU."""

from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_gaussian_ply_cpu import render_view


ROOT = Path("/nas_ds925/IEVE/results/habitat_442_20260816_212148_a8BtkwhxdRV")
SCENE = ROOT / "episode_00001_a8BtkwhxdRV.basis.glb"
OUT = Path("/nas_ds925/tyj/IEVE/仿真实验关键节点_Gaussian原位渲染/01_快速成功_无二次探索_Episode442")


def direction_from_y_quaternion(y: float, w: float) -> np.ndarray:
    # Habitat camera forward is local -Z.  All recorded poses here are yaw-only.
    return np.asarray([-2.0 * y * w, 0.0, -(1.0 - 2.0 * y * y)], dtype=np.float32)


NODES = [
    # filename, map snapshot, agent xyz, quaternion y,w
    ("01_任务开始_step000.png", "checkpoint_49/point_cloud.ply",
     (-0.9598471, -1.8091916, 3.1881013), 0.4073555, 0.9132697),
    ("02_探索中间状态_step020.png", "checkpoint_49/point_cloud.ply",
     (-1.6104164, -1.8091916, 1.6344843), -0.0845620, -0.9964182),
    ("03_目标确认前看到语义_step025.png", "checkpoint_49/point_cloud.ply",
     (-1.5094128, -1.8091916, 1.2038432), -0.6118598, -0.7909663),
    ("05_闭环修正结束_step140.png", "checkpoint_149/point_cloud.ply",
     (-4.8719090, -1.8091916, -0.9968424), 0.9098632, -0.4149084),
    ("06_目标导航中_step170.png", "final/point_cloud.ply",
     (-3.8875775, -1.8091916, 1.0300148), 0.2925991, 0.9562352),
    # Last recorded in-situ camera pose immediately before the episode ends.
    ("07_实验结束前最后一帧高斯原位渲染_step176.png", "final/point_cloud.ply",
     (-4.4737670, -1.8091916, 0.4046542), 0.5663029, 0.8241972),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for filename, rel_ply, agent_xyz, y, w in NODES:
        if rel_ply not in cache:
            cache[rel_ply] = PlyData.read(SCENE / rel_ply)["vertex"].data
        camera = np.asarray(agent_xyz, dtype=np.float32)
        camera[1] += 0.88
        rgb, _, _ = render_view(
            cache[rel_ply], camera, direction_from_y_quaternion(y, w),
            width=480, height=640, hfov_deg=79.0,
        )
        cv2.imwrite(str(OUT / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(OUT / filename, flush=True)

    # Localization starts at algorithm step 41.  The saved NVSM candidate at
    # step 44 is badly occluded, so render the same early checkpoint from the
    # actual start-of-localization position, looking at the confirmed cluster.
    data = cache["checkpoint_49/point_cloud.ply"]
    camera = np.asarray((-2.0927145, -1.8091916 + 0.88, 0.8655484), dtype=np.float32)
    look = np.asarray((-0.917, 0.0, -0.399), dtype=np.float32)
    rgb, _, _ = render_view(data, camera, look, width=480, height=640, hfov_deg=79.0)
    path = OUT / "04_定位开始_step041.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(path, flush=True)


if __name__ == "__main__":
    main()
