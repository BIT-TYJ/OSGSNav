#!/usr/bin/env python3
"""Render a faithful GauSS-MI selected-view process figure."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch
from plyfile import PlyData, PlyElement

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


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


def normalized_3dgs_ply(source, destination):
    ply = PlyData.read(source)
    vertices = ply["vertex"].data
    names = [
        name
        for name in vertices.dtype.names
        if name not in {"red", "green", "blue"}
    ]
    dtype = [(name, vertices.dtype.fields[name][0]) for name in names]
    contiguous = np.empty(len(vertices), dtype=dtype)
    for name in names:
        contiguous[name] = vertices[name]
    PlyData([PlyElement.describe(contiguous, "vertex")], text=False).write(
        destination
    )


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def heatmap_rgba(values, vmax, mask, alpha=0.9):
    normalized = np.clip(values / max(float(vmax), 1e-12), 0.0, 1.0)
    rgba = plt.get_cmap("magma")(normalized)
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def percentile_max(values, mask, percentile=99.5):
    selected = np.asarray(values)[mask]
    selected = selected[np.isfinite(selected) & (selected > 0)]
    if selected.size == 0:
        return 1.0
    return float(max(np.percentile(selected, percentile), 1e-8))


def directional_information(xyz, reliability, camera_position):
    direction = xyz - camera_position[None]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    abs_direction = np.abs(direction)
    coefficients = np.column_stack(
        [
            np.where(direction[:, 0] >= 0, abs_direction[:, 0], 0.0),
            np.where(direction[:, 2] >= 0, abs_direction[:, 2], 0.0),
            np.where(direction[:, 0] < 0, abs_direction[:, 0], 0.0),
            np.where(direction[:, 2] < 0, abs_direction[:, 2], 0.0),
        ]
    ).astype(np.float32)
    information_gain = -np.log(np.clip(reliability, 0.0001, 0.9999))
    return np.sum(information_gain * coefficients, axis=1), coefficients


def semantic_weights_from_highlighted_ply(
    ply,
    xyz,
    target_weight=2.0,
    nearby_weight=1.5,
    nearby_radius_m=2.0,
):
    """Reproduce IEVE Phase-I semantic weights from persisted target colors."""
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"red", "green", "blue"}.issubset(names):
        raise ValueError("Semantic weighting requires RGB fields in the PLY")
    target_mask = (
        (np.asarray(vertex["red"]) == 255)
        & (np.asarray(vertex["green"]) == 0)
        & (np.asarray(vertex["blue"]) == 255)
    )
    if not np.any(target_mask):
        raise ValueError("No confirmed magenta target Gaussians found in the PLY")

    from sklearn.cluster import DBSCAN

    target_xyz = xyz[target_mask]
    labels = DBSCAN(eps=0.15, min_samples=10).fit(target_xyz).labels_
    cluster_ids = sorted(set(labels.tolist()) - {-1})
    centers = [target_xyz[labels == label].mean(axis=0) for label in cluster_ids]
    if not centers:
        centers = [target_xyz.mean(axis=0)]
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    horizontal_delta = (
        xyz[:, None, :][:, :, [0, 2]]
        - centers[None, :, :][:, :, [0, 2]]
    )
    nearby_mask = (
        np.linalg.norm(horizontal_delta, axis=2).min(axis=1)
        <= float(nearby_radius_m)
    ) & (~target_mask)
    weights = np.ones(xyz.shape[0], dtype=np.float32)
    weights[nearby_mask] = max(1.0, float(nearby_weight))
    weights[target_mask] = max(1.0, float(target_weight))
    return weights, target_mask, {
        "target_count": int(target_mask.sum()),
        "nearby_count": int(nearby_mask.sum()),
        "cluster_count": int(centers.shape[0]),
        "target_weight": float(target_weight),
        "nearby_weight": float(nearby_weight),
        "nearby_radius_m": float(nearby_radius_m),
        "target_centers": centers.tolist(),
        "target_source": "PLY confirmed color RGB=(255,0,255)",
    }


def project_semantic_markers(
    rgb,
    xyz,
    w2c,
    target_mask,
    nearby_mask,
    width,
    height,
    hfov_deg,
    depth_image=None,
    max_nearby=300,
):
    """Mark projected target/nearby semantic Gaussians on the RGB render."""
    import cv2

    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cam_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    depth = cam_xyz[:, 2]
    u = fx * cam_xyz[:, 0] / np.maximum(depth, 1e-8) + width / 2.0
    v = fx * cam_xyz[:, 1] / np.maximum(depth, 1e-8) + height / 2.0
    visible = (
        (depth > 0.03)
        & (depth < 100.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    if depth_image is not None:
        px = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
        py = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
        rendered_depth = np.asarray(depth_image)[py, px]
        visible &= (
            (rendered_depth <= 0.0)
            | (~np.isfinite(rendered_depth))
            | (depth <= rendered_depth + 0.25)
        )
    target_ids = np.flatnonzero(target_mask & visible)
    nearby_ids = np.flatnonzero(nearby_mask & visible)
    if nearby_ids.size > max_nearby:
        # Deterministic spatially spread subset, avoiding a noisy solid fill.
        keep = np.linspace(0, nearby_ids.size - 1, max_nearby).astype(np.int64)
        nearby_ids = nearby_ids[keep]
    marked = np.asarray(np.clip(rgb * 255.0, 0, 255), dtype=np.uint8).copy()
    nearby_layer = marked.copy()
    for idx in nearby_ids:
        cv2.circle(
            nearby_layer,
            (int(round(u[idx])), int(round(v[idx]))),
            1,
            (0, 220, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    marked = cv2.addWeighted(nearby_layer, 0.72, marked, 0.28, 0.0)
    for idx in target_ids:
        cv2.circle(
            marked,
            (int(round(u[idx])), int(round(v[idx]))),
            4,
            (255, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    return marked.astype(np.float32) / 255.0, {
        "projected_target_count": int(target_ids.size),
        "projected_nearby_count": int(nearby_ids.size),
        "visible_target_count": int((target_mask & visible).sum()),
        "visible_nearby_count": int((nearby_mask & visible).sum()),
    }


def add_process_arrow(fig, left_axis, right_axis, label):
    left = left_axis.get_position()
    right = right_axis.get_position()
    start = (left.x1 + 0.004, 0.905)
    end = (right.x0 - 0.004, 0.905)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#555555",
    )
    fig.add_artist(arrow)
    fig.text(
        (start[0] + end[0]) / 2,
        0.918,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )


def build_figure(
    rgb,
    directional_map,
    mutual_info,
    opacity,
    score,
    output_png,
    output_pdf,
):
    valid = opacity > 0.02
    directional_vmax = percentile_max(directional_map, valid)
    mi_vmax = percentile_max(mutual_info, valid)
    grayscale = np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])

    fig = plt.figure(figsize=(16, 5), dpi=240, facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        left=0.025,
        right=0.985,
        bottom=0.22,
        top=0.86,
        wspace=0.10,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    titles = [
        "A  Rendered Novel View",
        "B  Directional Information Gain",
        "C  Pixel-wise Mutual Information",
        "D  GauSS-MI View Score",
    ]
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=12, fontweight="semibold", pad=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    axes[0].imshow(rgb)

    axes[1].imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].imshow(
        heatmap_rgba(directional_map, directional_vmax, valid, alpha=0.88)
    )

    axes[2].imshow(grayscale * 0.42, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(heatmap_rgba(mutual_info, mi_vmax, valid, alpha=0.94))

    axes[3].imshow(rgb)
    axes[3].imshow(heatmap_rgba(mutual_info, mi_vmax, valid, alpha=0.52))
    positive = mutual_info[valid & (mutual_info > 0)]
    if positive.size:
        levels = np.unique(np.percentile(positive, [75.0, 90.0]))
        if levels.size:
            axes[3].contour(
                mutual_info,
                levels=levels,
                colors=["#f7f7f7", "#00d4ff"][: levels.size],
                linewidths=[0.7, 1.0][: levels.size],
                alpha=0.9,
            )
    axes[3].text(
        0.04,
        0.955,
        f"Score = {score:.3f} k-nats",
        transform=axes[3].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 4},
    )

    fig.suptitle(
        "GauSS-MI at the Selected View",
        y=0.985,
        fontsize=16,
        fontweight="semibold",
    )
    add_process_arrow(fig, axes[0], axes[1], "directional weighting")
    add_process_arrow(fig, axes[1], axes[2], r"$\alpha T$ compositing")
    add_process_arrow(fig, axes[2], axes[3], "sum over pixels")

    captions = [
        r"Selected camera $v^*$",
        r"$u_g(v^*)=\sum_d c_{g,d}(v^*)[-\log r_{g,d}]$",
        r"$I(p\mid v^*)=\sum_g u_g(v^*)\,\alpha_g(p)\,T_g(p)$",
        r"$S(v^*)=\lfloor\sum_p I(p\mid v^*)\rfloor/1000$",
    ]
    for axis, caption in zip(axes, captions):
        box = axis.get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            0.095,
            caption,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#222222",
        )

    for axis, vmax, label in (
        (axes[1], directional_vmax, "directional information (nats)"),
        (axes[2], mi_vmax, "mutual information (nats / pixel)"),
    ):
        box = axis.get_position()
        colorbar_axis = fig.add_axes([box.x0 + 0.02, 0.145, box.width - 0.04, 0.014])
        scalar = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax),
            cmap="magma",
        )
        colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label(label, fontsize=7.5, labelpad=1)
        colorbar.outline.set_linewidth(0.5)

    fig.text(
        0.985,
        0.025,
        "Heatmap limits: 99.5th percentile; score uses unclipped values.",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#666666",
    )
    fig.savefig(str(output_png), dpi=240, facecolor="white")
    fig.savefig(str(output_pdf), facecolor="white")
    plt.close(fig)
    return directional_vmax, mi_vmax


def build_simple_figure(
    rgb,
    mutual_info,
    opacity,
    output_png,
    output_pdf,
    semantic_weighted=False,
    marker_rgb=None,
):
    """Build a compact two-panel paper figure with the MI equations below."""
    valid = opacity > 0.02
    mi_vmax = percentile_max(mutual_info, valid)
    normalized_mi = np.clip(mutual_info / max(mi_vmax, 1e-12), 0.0, 1.0)

    fig = plt.figure(figsize=(8.0, 8.2), dpi=260, facecolor="white")
    # The two axes share the exact same boundary at x=0.5: zero inter-image gap.
    rgb_axis = fig.add_axes([0.08, 0.31, 0.42, 0.545])
    mi_axis = fig.add_axes([0.50, 0.31, 0.42, 0.545])

    for axis in (rgb_axis, mi_axis):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    rgb_axis.imshow(marker_rgb if marker_rgb is not None else rgb)
    rgb_axis.set_title("Gaussian Render", fontsize=19, pad=11)

    mi_axis.imshow(np.zeros_like(normalized_mi), cmap="gray", vmin=0, vmax=1)
    mi_axis.imshow(
        heatmap_rgba(normalized_mi, 1.0, valid, alpha=1.0),
        interpolation="nearest",
    )
    density_name = "SWMI" if semantic_weighted else "MI"
    mi_axis.set_title(f"Pixel-wise {density_name} Density", fontsize=19, pad=11)

    box = mi_axis.get_position()
    colorbar_axis = fig.add_axes(
        [box.x0 + 0.035, 0.205, box.width - 0.07, 0.022]
    )
    scalar = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
        cmap="magma",
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=14, length=0, pad=3)
    colorbar.set_label(
        f"normalized {density_name} density", fontsize=14, labelpad=3
    )
    colorbar.outline.set_linewidth(0.5)

    figure_title = (
        "Field of View at the Circled Sampled Pose in Fig. 1"
        if semantic_weighted
        else "GauSS-MI Density for the Selected View"
    )
    fig.suptitle(figure_title, y=0.975, fontsize=20)
    if semantic_weighted and marker_rgb is not None:
        fig.text(
            0.155,
            0.250,
            "●",
            ha="center",
            va="center",
            fontsize=20,
            color="#ff00ff",
        )
        fig.text(
            0.270,
            0.250,
            "target cluster",
            ha="center",
            va="center",
            fontsize=15,
            color="#333333",
        )
        fig.text(
            0.410,
            0.250,
            "●",
            ha="center",
            va="center",
            fontsize=20,
            color="#00dcff",
        )
        fig.text(
            0.490,
            0.250,
            "nearby",
            ha="center",
            va="center",
            fontsize=15,
            color="#333333",
        )
    if semantic_weighted:
        left_equation = (
            r"$I_{\rm SWMI}(p\mid v)=$" "\n"
            r"$\sum_g w_g^{\rm sem}T_g(p)\alpha_g(p)[-\log r_g]$"
        )
        right_equation = (
            r"$I_{\rm SWMI}(v)=$" "\n"
            r"$\sum_{p\in\Omega(v)}I_{\rm SWMI}(p\mid v)$"
        )
    else:
        left_equation = (
            r"$I(p\mid v)=$" "\n"
            r"$\sum_g T_g(p)\,\alpha_g(p)[-\log r_g]$"
        )
        right_equation = (
            r"$I(v)=$" "\n"
            r"$\sum_{p\in\Omega(v)} I(p\mid v)$"
        )
    fig.text(
        0.29,
        0.075,
        left_equation,
        ha="center",
        va="center",
        fontsize=17,
        color="#111111",
        linespacing=1.15,
    )
    fig.text(
        0.71,
        0.075,
        right_equation,
        ha="center",
        va="center",
        fontsize=17,
        color="#111111",
        linespacing=1.15,
    )
    fig.savefig(str(output_png), dpi=260, facecolor="white")
    fig.savefig(str(output_pdf), facecolor="white")
    plt.close(fig)
    return mi_vmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--state-npz")
    parser.add_argument(
        "--uniform-prior",
        action="store_true",
        help="Use the model's uniform reliability prior when no state snapshot exists.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Render a compact RGB + MI density + pixel-sum figure.",
    )
    parser.add_argument(
        "--semantic-weighted",
        action="store_true",
        help="Apply IEVE target and nearby semantic weights before rasterization.",
    )
    parser.add_argument("--target-weight", type=float, default=2.0)
    parser.add_argument("--nearby-weight", type=float, default=1.5)
    parser.add_argument("--nearby-radius-m", type=float, default=2.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--camera-position", nargs=3, type=float, default=(-0.276, -2.007, 2.286)
    )
    parser.add_argument(
        "--look-direction",
        nargs=3,
        type=float,
        default=(-0.9659258263, 0.0, 0.2588190451),
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--output-stem", default="gaussmi_selected_view_process")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from gaussian_mapping.gaussian_splatting.gaussmi_renderer import render_gaussmi
    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
    from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from gaussian_mapping.utils.camera_utils import Camera

    ply = PlyData.read(args.ply)
    ply_xyz = np.column_stack(
        [ply["vertex"].data[name] for name in ("x", "y", "z")]
    ).astype(np.float32)
    if args.state_npz:
        state = np.load(args.state_npz)
        snapshot_xyz = np.ascontiguousarray(state["xyz"], dtype=np.float32)
        reliability = np.ascontiguousarray(state["reliability"], dtype=np.float32)
        if ply_xyz.shape != snapshot_xyz.shape or not np.array_equal(
            ply_xyz, snapshot_xyz
        ):
            raise ValueError(
                "Snapshot reliability is not aligned with the input PLY geometry"
            )
        state_source = str(Path(args.state_npz).resolve())
    elif args.uniform_prior:
        snapshot_xyz = np.ascontiguousarray(ply_xyz, dtype=np.float32)
        reliability = np.full(
            (ply_xyz.shape[0], 4), 0.5, dtype=np.float32
        )
        state_source = "uniform reliability prior r=0.5"
    else:
        parser.error("provide --state-npz or explicitly select --uniform-prior")
    if reliability.shape != (ply_xyz.shape[0], 4):
        raise ValueError(
            f"Expected reliability shape {(ply_xyz.shape[0], 4)}, got {reliability.shape}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    source_ply = Path(args.ply)
    if ply["vertex"].data.dtype.itemsize % 4:
        temporary = tempfile.NamedTemporaryFile(suffix="_3dgs.ply", delete=False)
        temporary_path = temporary.name
        temporary.close()
        normalized_3dgs_ply(str(source_ply), temporary_path)
        load_path = temporary_path
    else:
        load_path = str(source_ply)

    try:
        model = GaussianModel(1, config={"Dataset": {"sensor_type": "rgbd"}})
        model.load_ply(load_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    model._reliability = torch.as_tensor(
        reliability, dtype=torch.float32, device="cuda"
    )
    semantic_weights = np.ones(ply_xyz.shape[0], dtype=np.float32)
    target_mask = np.zeros(ply_xyz.shape[0], dtype=bool)
    semantic_debug = None
    if args.semantic_weighted:
        semantic_weights, target_mask, semantic_debug = (
            semantic_weights_from_highlighted_ply(
                ply,
                ply_xyz,
                target_weight=args.target_weight,
                nearby_weight=args.nearby_weight,
                nearby_radius_m=args.nearby_radius_m,
            )
        )
    information_gain = -np.log(np.clip(reliability, 0.0001, 0.9999))
    information_gain *= semantic_weights[:, None]
    information_gain_tensor = torch.as_tensor(
        information_gain, dtype=torch.float32, device="cuda"
    )

    width, height = args.width, args.height
    fovx = math.radians(args.hfov)
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    projection = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        W=width,
        H=height,
    ).transpose(0, 1).cuda()
    camera_position = np.asarray(args.camera_position, dtype=np.float64)
    look_direction = np.asarray(args.look_direction, dtype=np.float64)
    w2c = look_at_w2c(camera_position, look_direction)
    pose = torch.as_tensor(w2c, dtype=torch.float32, device="cuda")
    camera = Camera(
        -1,
        torch.zeros((3, height, width), dtype=torch.float32, device="cuda"),
        np.zeros((height, width), dtype=np.float32),
        pose,
        projection,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        height,
        width,
        device="cuda",
    )
    camera.update_RT(pose[:3, :3], pose[:3, 3])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_package = render_gaussmi(
            camera,
            model,
            pipe,
            background,
            information_gain=information_gain_tensor,
        )
    if render_package is None:
        raise RuntimeError("GauSS-MI renderer returned no output")

    rgb = (
        render_package["render"]
        .detach()
        .clamp(0, 1)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    mutual_info = render_package["mutual_info"][0].detach().cpu().numpy()
    opacity = render_package["opacity"][0].detach().cpu().numpy()
    rendered_depth = render_package["depth"][0].detach().cpu().numpy()
    directional_map = np.divide(
        mutual_info,
        opacity,
        out=np.zeros_like(mutual_info),
        where=opacity > 1e-6,
    )
    per_gaussian_info, direction_coefficients = directional_information(
        snapshot_xyz, reliability, camera_position.astype(np.float32)
    )
    per_gaussian_info *= semantic_weights
    score = int(float(mutual_info.sum())) / 1000.0
    marker_rgb = None
    marker_debug = None
    if args.semantic_weighted:
        nearby_mask = (
            np.isclose(semantic_weights, float(args.nearby_weight))
            & (~target_mask)
        )
        marker_rgb, marker_debug = project_semantic_markers(
            rgb,
            snapshot_xyz,
            w2c,
            target_mask,
            nearby_mask,
            width,
            height,
            args.hfov,
            depth_image=rendered_depth,
        )

    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    save_rgb(out_dir / "01_rendered_novel_view.png", rgb_u8)
    if marker_rgb is not None:
        marker_rgb_u8 = np.clip(marker_rgb * 255.0, 0, 255).astype(np.uint8)
        save_rgb(out_dir / "02_semantic_marked_render.png", marker_rgb_u8)
    np.savez_compressed(
        str(out_dir / f"{args.output_stem}_data.npz"),
        rgb=rgb.astype(np.float32),
        mutual_info=mutual_info.astype(np.float32),
        opacity=opacity.astype(np.float32),
        directional_information=directional_map.astype(np.float32),
        per_gaussian_directional_information=per_gaussian_info.astype(np.float32),
        direction_coefficients=direction_coefficients,
        reliability=reliability,
        semantic_weights=semantic_weights,
        semantic_target_mask=target_mask,
        marker_rgb=(marker_rgb if marker_rgb is not None else rgb).astype(np.float32),
    )
    output_png = out_dir / f"{args.output_stem}.png"
    output_pdf = out_dir / f"{args.output_stem}.pdf"
    if args.simple:
        mi_vmax = build_simple_figure(
            rgb,
            mutual_info,
            opacity,
            output_png,
            output_pdf,
            semantic_weighted=args.semantic_weighted,
            marker_rgb=marker_rgb,
        )
        directional_vmax = percentile_max(directional_map, opacity > 0.02)
    else:
        directional_vmax, mi_vmax = build_figure(
            rgb,
            directional_map,
            mutual_info,
            opacity,
            score,
            output_png,
            output_pdf,
        )

    delta = np.abs(reliability - float(model.init_reliability))
    metadata = {
        "source_ply": str(source_ply.resolve()),
        "source_state": state_source,
        "gaussians": int(snapshot_xyz.shape[0]),
        "xyz_sha256": hashlib.sha256(snapshot_xyz.tobytes()).hexdigest(),
        "reliability_shape": list(reliability.shape),
        "reliability_range": [float(reliability.min()), float(reliability.max())],
        "updated_reliability_ratio": float(np.any(delta > 0.02, axis=1).mean()),
        "camera_position": camera_position.tolist(),
        "look_direction": (look_direction / np.linalg.norm(look_direction)).tolist(),
        "w2c": w2c.tolist(),
        "resolution": [width, height],
        "hfov_deg": args.hfov,
        "vfov_deg": math.degrees(fovy),
        "visible_gaussians": int(render_package["visibility_filter"].sum().item()),
        "mutual_info_sum": float(mutual_info.sum()),
        "gaussmi_score_k_nats": score,
        "semantic_weighted": bool(args.semantic_weighted),
        "semantic_weighting": semantic_debug,
        "semantic_marker_projection": marker_debug,
        "directional_heatmap_vmax_p99_5": directional_vmax,
        "mutual_info_heatmap_vmax_p99_5": mi_vmax,
        "figure_png": str(output_png.resolve()),
        "figure_pdf": str(output_pdf.resolve()),
    }
    metadata_path = out_dir / f"{args.output_stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", **metadata}))


if __name__ == "__main__":
    main()
