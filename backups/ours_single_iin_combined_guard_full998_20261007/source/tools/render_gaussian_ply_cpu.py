#!/usr/bin/env python3
"""CPU fallback renderer for RGB Gaussian PLY files."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from numba import njit
from plyfile import PlyData


def look_at_w2c(camera_position, look_direction):
    camera_position = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(look_direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up_world = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_position
    return np.linalg.inv(c2w)


def rotate_y(direction, angle_deg):
    x, _, z = direction
    angle = math.radians(angle_deg)
    return np.asarray([
        math.cos(angle) * x + math.sin(angle) * z,
        0.0,
        -math.sin(angle) * x + math.cos(angle) * z,
    ])


def quaternion_rotation(quaternions):
    q = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-10)
    w, x, y, z = q.T
    rotation = np.empty((len(q), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - w * z)
    rotation[:, 0, 2] = 2 * (x * z + w * y)
    rotation[:, 1, 0] = 2 * (x * y + w * z)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - w * x)
    rotation[:, 2, 0] = 2 * (x * z - w * y)
    rotation[:, 2, 1] = 2 * (y * z + w * x)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


@njit(cache=True)
def composite_gaussians(image, u, v, conic_a, conic_b, conic_c, radii, opacity, colors):
    height, width, _ = image.shape
    for index in range(len(u)):
        radius = radii[index]
        x0 = max(0, int(math.floor(u[index] - radius)))
        x1 = min(width - 1, int(math.ceil(u[index] + radius)))
        y0 = max(0, int(math.floor(v[index] - radius)))
        y1 = min(height - 1, int(math.ceil(v[index] + radius)))
        for py in range(y0, y1 + 1):
            dy = (py + 0.5) - v[index]
            for px in range(x0, x1 + 1):
                dx = (px + 0.5) - u[index]
                power = conic_a[index] * dx * dx + 2.0 * conic_b[index] * dx * dy + conic_c[index] * dy * dy
                if power > 9.0:
                    continue
                alpha = min(0.995, opacity[index] * math.exp(-0.5 * power))
                if alpha < 1.0 / 255.0:
                    continue
                image[py, px, 0] = alpha * colors[index, 0] + (1.0 - alpha) * image[py, px, 0]
                image[py, px, 1] = alpha * colors[index, 1] + (1.0 - alpha) * image[py, px, 1]
                image[py, px, 2] = alpha * colors[index, 2] + (1.0 - alpha) * image[py, px, 2]


def render_view(data, camera_position, direction, width, height, hfov_deg):
    xyz = np.column_stack([data[name] for name in ("x", "y", "z")]).astype(np.float32)
    scales = np.exp(np.column_stack([data[name] for name in ("scale_0", "scale_1", "scale_2")])).astype(np.float32)
    rotations = quaternion_rotation(
        np.column_stack([data[name] for name in ("rot_0", "rot_1", "rot_2", "rot_3")]).astype(np.float32)
    )
    opacity = (1.0 / (1.0 + np.exp(-data["opacity"]))).astype(np.float32)
    if all(name in data.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack([data[name] for name in ("red", "green", "blue")]).astype(np.float32) / 255.0
    elif all(name in data.dtype.names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        # Native IEVE checkpoints store the DC spherical-harmonic coefficient
        # rather than display RGB fields.  Convert it to the same base colour
        # used by the project renderer; higher-order view dependence is small
        # enough for these diagnostic, pose-aligned snapshots.
        colors = np.clip(
            0.5 + 0.2820947918 * np.column_stack(
                [data[name] for name in ("f_dc_0", "f_dc_1", "f_dc_2")]
            ).astype(np.float32),
            0.0,
            1.0,
        )
    else:
        raise ValueError("PLY has neither RGB nor f_dc colour fields")

    w2c = look_at_w2c(camera_position, direction)
    rotation_camera = w2c[:3, :3].astype(np.float32)
    camera_xyz = xyz @ rotation_camera.T + w2c[:3, 3]
    depth = camera_xyz[:, 2]
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    u = fx * camera_xyz[:, 0] / depth + width / 2.0
    v = fy * camera_xyz[:, 1] / depth + height / 2.0

    margin = 80.0
    visible = ((depth > 0.03) & (depth < 30.0) &
               (u > -margin) & (u < width + margin) &
               (v > -margin) & (v < height + margin))
    ids = np.nonzero(visible)[0]
    xyz_c = camera_xyz[ids]
    depth_c = depth[ids]

    covariance_world = rotations[ids] @ (np.eye(3, dtype=np.float32)[None] * scales[ids, :, None] ** 2) @ np.transpose(rotations[ids], (0, 2, 1))
    covariance_camera = rotation_camera[None] @ covariance_world @ rotation_camera.T[None]
    jacobian = np.zeros((len(ids), 2, 3), dtype=np.float32)
    jacobian[:, 0, 0] = fx / depth_c
    jacobian[:, 0, 2] = -fx * xyz_c[:, 0] / (depth_c * depth_c)
    jacobian[:, 1, 1] = fy / depth_c
    jacobian[:, 1, 2] = -fy * xyz_c[:, 1] / (depth_c * depth_c)
    covariance_2d = jacobian @ covariance_camera @ np.transpose(jacobian, (0, 2, 1))
    covariance_2d[:, 0, 0] += 0.3
    covariance_2d[:, 1, 1] += 0.3
    determinant = covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1] - covariance_2d[:, 0, 1] ** 2
    good = determinant > 1e-8
    ids = ids[good]
    covariance_2d = covariance_2d[good]
    determinant = determinant[good]
    inv_a = covariance_2d[:, 1, 1] / determinant
    inv_b = -covariance_2d[:, 0, 1] / determinant
    inv_c = covariance_2d[:, 0, 0] / determinant
    max_eigenvalue = np.linalg.eigvalsh(covariance_2d)[:, 1]
    radii = np.minimum(80.0, 3.0 * np.sqrt(np.maximum(max_eigenvalue, 0.3))).astype(np.float32)

    order = np.argsort(depth[ids])[::-1]
    ids = ids[order]
    image = np.zeros((height, width, 3), dtype=np.float32)
    composite_gaussians(
        image,
        u[ids].astype(np.float32), v[ids].astype(np.float32),
        inv_a[order].astype(np.float32), inv_b[order].astype(np.float32), inv_c[order].astype(np.float32),
        radii[order], opacity[ids], colors[ids],
    )
    return np.clip(image * 255.0, 0, 255).astype(np.uint8), w2c, len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-position", nargs=3, type=float, required=True)
    parser.add_argument("--look-direction", nargs=3, type=float, default=(-1.0, 0.0, 0.0))
    parser.add_argument("--yaw-offsets", nargs="+", type=float, default=(-15, -7.5, 0, 7.5, 15))
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=79.0)
    args = parser.parse_args()

    data = PlyData.read(args.ply)["vertex"].data
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_position = np.asarray(args.camera_position, dtype=np.float32)
    base_direction = np.asarray(args.look_direction, dtype=np.float32)
    base_direction[1] = 0.0
    base_direction /= np.linalg.norm(base_direction)
    views = []
    thumbs = []
    for yaw in args.yaw_offsets:
        direction = rotate_y(base_direction, yaw)
        rgb, w2c, visible = render_view(data, camera_position, direction, args.width, args.height, args.hfov)
        suffix = f"{yaw:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        path = out_dir / f"red_arrow_yaw_{suffix}deg_cpu.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        thumb = cv2.resize(rgb, (360, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (360, 44), (255, 255, 255), -1)
        cv2.putText(thumb, f"yaw {yaw:+g} deg", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (20, 20, 20), 2, cv2.LINE_AA)
        thumbs.append(thumb)
        views.append({"yaw_offset_deg": yaw, "direction": direction.tolist(), "w2c": w2c.tolist(),
                      "visible_gaussians": visible, "output": str(path.resolve())})

    montage = np.hstack(thumbs)
    cv2.imwrite(str(out_dir / "red_arrow_yaw_sweep_cpu.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    metadata = {"source_ply": str(Path(args.ply).resolve()), "camera_position": camera_position.tolist(),
                "resolution": [args.width, args.height], "hfov_deg": args.hfov, "views": views}
    (out_dir / "render_metadata_cpu.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(out_dir.resolve()), "views": len(views)}))


if __name__ == "__main__":
    main()
