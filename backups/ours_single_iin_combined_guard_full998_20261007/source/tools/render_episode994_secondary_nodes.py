#!/usr/bin/env python3
"""Render pose-aligned secondary-exploration keyframes for episode 994."""

from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_gaussian_ply_cpu import render_view


ROOT = Path("/nas_ds925/IEVE/results/habitat_994_20260709_151103_zt1RVoi7PcG")
SCENE = ROOT / "episode_00001_zt1RVoi7PcG.basis.glb"
OUT = Path("/nas_ds925/tyj/IEVE/仿真实验关键节点_Gaussian原位渲染/02_二次探索成功_Episode994")


def direction(y: float, w: float) -> np.ndarray:
    return np.asarray([-2.0 * y * w, 0.0, -(1.0 - 2.0 * y * y)], dtype=np.float32)


NODES = [
    ("02_候选实例1进入视野_step404.png",
     "3dgs_confirmed/step_000404_frame_000405_confirmed_targeted_quick_lock.ply",
     (-2.6316123, -2.8870633, 7.3540993), 0.9559288, -0.2935987),
    ("03_候选实例2进入视野_step462.png",
     "3dgs_confirmed/step_000462_frame_000463_confirmed_targeted_quick_lock.ply",
     (-0.2123508, -2.8870633, 6.1446238), -0.3092810, 0.9509707),
    ("04_候选实例3进入视野_step492.png",
     "3dgs_confirmed/step_000492_frame_000493_confirmed_targeted_quick_lock.ply",
     (-0.6369772, -2.8870633, 4.4732050), -0.6464449, -0.7629607),
    ("05_导航结束成功_step471.png", "final/point_cloud.ply",
     (-1.3063363, -2.8870633, 4.3617710), 0.6464449, 0.7629607),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rel_ply, agent_xyz, y, w in NODES:
        data = PlyData.read(SCENE / rel_ply)["vertex"].data
        camera = np.asarray(agent_xyz, dtype=np.float32)
        camera[1] += 0.88
        rgb, _, _ = render_view(data, camera, direction(y, w), 480, 640, 79.0)
        cv2.imwrite(str(OUT / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(OUT / filename, flush=True)


if __name__ == "__main__":
    main()
