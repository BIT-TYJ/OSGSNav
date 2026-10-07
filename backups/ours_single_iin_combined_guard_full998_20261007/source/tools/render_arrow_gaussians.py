#!/usr/bin/env python3
"""Render a small yaw sweep from a map-picked camera pose."""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def look_at_w2c(camera_position, look_direction):
    camera_position = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(look_direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_position
    habitat_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w @ habitat_to_opencv)


def rotate_y(direction, angle_deg):
    x, _, z = direction
    angle = math.radians(angle_deg)
    return np.asarray([
        math.cos(angle) * x + math.sin(angle) * z,
        0.0,
        -math.sin(angle) * x + math.cos(angle) * z,
    ])


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def normalized_3dgs_ply(source, destination):
    """Strip display-only uchar RGB fields that break legacy float field strides."""
    ply = PlyData.read(source)
    vertices = ply["vertex"].data
    names = [name for name in vertices.dtype.names if name not in {"red", "green", "blue"}]
    dtype = [(name, vertices.dtype.fields[name][0]) for name in names]
    contiguous = np.empty(len(vertices), dtype=dtype)
    for name in names:
        contiguous[name] = vertices[name]
    PlyData([PlyElement.describe(contiguous, "vertex")], text=False).write(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-position", nargs=3, type=float, required=True)
    parser.add_argument("--look-direction", nargs=3, type=float, default=(-1.0, 0.0, 0.0))
    parser.add_argument("--yaw-offsets", nargs="+", type=float, default=(-15, -7.5, 0, 7.5, 15))
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--scaling-modifier", type=float, default=1.0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from gaussian_mapping.gaussian_splatting.gaussian_renderer import render
    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
    from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from gaussian_mapping.utils.camera_utils import Camera

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = args.width, args.height
    fovx = math.radians(args.hfov)
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    projection = getProjectionMatrix2(
        znear=0.01, zfar=100.0, cx=cx, cy=cy, fx=fx, fy=fy, W=width, H=height
    ).transpose(0, 1).cuda()

    model = GaussianModel(1, config={"Dataset": {"sensor_type": "rgbd"}})
    source_ply = Path(args.ply)
    temporary_ply = None
    if PlyData.read(str(source_ply))["vertex"].data.dtype.itemsize % 4:
        temporary_ply = tempfile.NamedTemporaryFile(suffix="_3dgs.ply", delete=False)
        temporary_ply.close()
        normalized_3dgs_ply(str(source_ply), temporary_ply.name)
        model.load_ply(temporary_ply.name)
        os.unlink(temporary_ply.name)
    else:
        model.load_ply(str(source_ply))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False)
    dummy_color = torch.zeros((3, height, width), dtype=torch.float32, device="cuda")
    dummy_depth = np.zeros((height, width), dtype=np.float32)
    camera_position = np.asarray(args.camera_position, dtype=np.float64)
    base_direction = np.asarray(args.look_direction, dtype=np.float64)
    base_direction[1] = 0.0
    base_direction /= np.linalg.norm(base_direction)

    rendered = []
    metadata = []
    for index, yaw in enumerate(args.yaw_offsets):
        direction = rotate_y(base_direction, yaw)
        w2c = look_at_w2c(camera_position, direction)
        pose = torch.as_tensor(w2c, dtype=torch.float32, device="cuda")
        camera = Camera(
            -(index + 1), dummy_color, dummy_depth, pose, projection, fx, fy, cx, cy,
            fovx, fovy, height, width, device="cuda"
        )
        camera.update_RT(pose[:3, :3], pose[:3, 3])
        with torch.no_grad():
            pkg = render(camera, model, pipe, background, scaling_modifier=args.scaling_modifier)
        rgb = pkg["render"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * 255.0).astype(np.uint8)
        suffix = f"{yaw:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        path = out_dir / f"red_arrow_yaw_{suffix}deg.png"
        save_rgb(path, rgb)
        rendered.append((yaw, rgb, path))
        metadata.append({
            "yaw_offset_deg": yaw,
            "camera_position": camera_position.tolist(),
            "look_direction": direction.tolist(),
            "w2c": w2c.tolist(),
            "output": str(path.resolve()),
        })

    thumbs = []
    thumb_w, thumb_h = 360, 480
    for yaw, rgb, _ in rendered:
        thumb = cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (thumb_w, 44), (255, 255, 255), -1)
        cv2.putText(thumb, f"yaw {yaw:+g} deg", (12, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (20, 20, 20), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    save_rgb(out_dir / "red_arrow_yaw_sweep.png", np.hstack(thumbs))

    run_metadata = {
        "source_ply": str(Path(args.ply).resolve()),
        "gaussians": int(model.get_xyz.shape[0]),
        "resolution": [width, height],
        "hfov_deg": args.hfov,
        "vfov_deg": math.degrees(fovy),
        "views": metadata,
    }
    (out_dir / "render_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "gaussians": run_metadata["gaussians"],
                      "output": str(out_dir.resolve())}))


if __name__ == "__main__":
    main()
