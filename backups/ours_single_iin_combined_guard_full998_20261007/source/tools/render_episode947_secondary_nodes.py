#!/usr/bin/env python3
"""Render pose-aligned secondary-exploration keyframes for Episode 947."""

from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_gaussian_ply_cpu import render_view


ROOT = Path("/nas_ds925/IEVE/results/habitat_946_20260709_034721_ziup5kvtCCR")
SCENE = ROOT / "episode_00001_ziup5kvtCCR.basis.glb"
OUT = Path("/nas_ds925/tyj/IEVE/仿真实验关键节点_Gaussian原位渲染/02_二次探索成功_Episode947")


def direction_from_y_quaternion(y: float, w: float) -> np.ndarray:
    return np.asarray([-2.0 * y * w, 0.0, -(1.0 - 2.0 * y * y)], dtype=np.float32)


# Poses are copied from MINAV_STOP_DEBUG.  Checkpoints are the closest saved
# Gaussian maps at or before each observation, avoiding future-map leakage.
NODES = [
    ("01_二次探索开始_algo_step400_env_step400.png", "checkpoint_401/point_cloud.ply",
     (0.51820165, 0.02122317, 1.2155404), 0.1096268, -0.9939729),
    ("02_候选实例1进入视野_algo_step456_env_step456.png", "checkpoint_451/point_cloud.ply",
     (3.0488384, 0.02122317, 3.9561539), 0.99524766, 0.09737583),
    ("03_候选实例2进入视野_algo_step468_env_step468.png", "final/point_cloud.ply",
     (4.0226197, 0.02122317, 5.715124), 0.7963506, -0.6048354),
    ("04_多个候选实例同时进入视野_algo_step477_env_step477.png", "final/point_cloud.ply",
     (5.4900208, 0.02122317, 6.3377533), 0.98305804, 0.18329442),
    ("05_导航结束成功前最后一帧_algo_step482_env_step482.png", "final/point_cloud.ply",
     (5.4900208, 0.02122317, 6.3377533), 0.46718085, 0.8841618),
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
        path = OUT / filename
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(path, flush=True)


if __name__ == "__main__":
    main()
