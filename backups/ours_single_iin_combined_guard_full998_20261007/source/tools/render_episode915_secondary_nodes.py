#!/usr/bin/env python3
"""Render clean secondary-exploration Gaussian keyframes for Episode 915."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_gaussian_ply_cpu import render_view


ROOT = Path("/nas_ds925/IEVE/results/habitat_914_20260708_181234_yr17PDCnDDW")
SCENE = ROOT / "episode_00001_yr17PDCnDDW.basis.glb"
OUT = Path("/nas_ds925/tyj/IEVE/仿真实验关键节点_Gaussian原位渲染/02_二次探索成功_Episode915")


def yaw_direction(y: float, w: float) -> np.ndarray:
    return np.asarray([-2.0 * y * w, 0.0, -(1.0 - 2.0 * y * y)], dtype=np.float32)


def look_direction(camera: np.ndarray, target: tuple[float, float, float]) -> np.ndarray:
    d = np.asarray(target, dtype=np.float32) - camera
    return d / max(float(np.linalg.norm(d)), 1e-6)


# Each candidate is a genuinely different quick-lock cluster recorded by the
# algorithm.  Camera locations come from the adjacent buffer-update trace.  We
# face the confirmed cluster to keep the instance unambiguous in the figure.
NODES = [
    ("01_二次探索开始_algo_step400_env_step400.png", "checkpoint_401/point_cloud.ply",
     (-3.015736, 2.4483883, 0.93599755), (0.27014333, 0.9628202), None),
    ("02_候选实例1进入视野_algo_step429.png", "checkpoint_401/point_cloud.ply",
     (-3.2947686, 2.4483883, -0.8055804), None, (-1.9978186, 3.2289515, -1.4397591)),
    ("03_候选实例2进入视野_algo_step473.png", "checkpoint_451/point_cloud.ply",
     (-1.1373317, 2.4483883, 1.4010993), None, (0.2337430, 3.5358212, 1.5729040)),
    ("04_候选实例3进入视野_algo_step522.png", "checkpoint_501/point_cloud.ply",
     (-0.0497413, 2.4483883, -2.1259034), None, (0.3989370, 3.1596131, -3.0779588)),
    ("05_正确实例进入视野_algo_step565.png", "checkpoint_551/point_cloud.ply",
     (-5.0111175, 2.4483883, -5.3494363), None, (-5.4894447, 3.7458763, -6.5206885)),
    ("06_导航结束成功前最后一帧_env_step565.png", "final/point_cloud.ply",
     (-5.3077025, 2.4483883, -5.6588078), (0.37291652, 0.9278649), None),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for filename, rel_ply, agent_xyz, quat, target in NODES:
        if rel_ply not in cache:
            cache[rel_ply] = PlyData.read(SCENE / rel_ply)["vertex"].data
        camera = np.asarray(agent_xyz, dtype=np.float32)
        camera[1] += 0.88
        direction = yaw_direction(*quat) if quat is not None else look_direction(camera, target)
        rgb, _, _ = render_view(cache[rel_ply], camera, direction,
                                width=480, height=640, hfov_deg=79.0)
        path = OUT / filename
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(path, flush=True)


if __name__ == "__main__":
    main()
