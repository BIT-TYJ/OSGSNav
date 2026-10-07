#!/usr/bin/env python3
"""Create four standalone localization-module method figures.

Pose locations and image evidence come from
showcase_episode_994_zt1RVoi7PcG. Spatial panels use a clean schematic
proxy for the target chair so the localization operations remain legible.
"""

from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import numpy as np


ROOT = Path("/nas_ds925/tyj/IEVE")
EPISODE = Path("/nas_ds925/IEVE/showcase_episode_994_zt1RVoi7PcG")
RUN = Path("/nas_ds925/IEVE/results/habitat_994_20260709_151103_zt1RVoi7PcG")
OUT = ROOT / "detectron2/docs/localization_module_figures"

CLUSTER_PLY = RUN / "NVSM/instance_semantic_20260709_152429.ply"
GLOBAL_PLY = EPISODE / "04_global_gaussian_ply/final_point_cloud_1000000_gaussians.ply"
MR_META = EPISODE / "08_mr_comparison/render_metadata.json"
MATCH_IMAGE = EPISODE / "10_fem_lightglue/fem_lightglue_matches_native.png"

INK = "#1d2939"
MUTED = "#667085"
GRID = "#d9e1ea"
BLUE = "#176b9c"
CYAN = "#00a7c4"
MAGENTA = "#d81b9c"
ORANGE = "#e97917"
GREEN = "#2a9d6f"
PURPLE = "#6f52a2"
SELECTED_RED = "#d92d20"
PRE_SKY_BLUE = "#55b7e8"
PRE_INITIAL_PURPLE = "#582C83"
BG = "#fbfcfe"


def load_ascii_ply_xyz(path):
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            if line.strip() == "end_header":
                break
    return np.loadtxt(path, skiprows=header_lines, usecols=(0, 1, 2), dtype=np.float32)


def load_binary_ply(path):
    """Memory-map geometry and DC colors from a binary little-endian 3DGS PLY."""
    raw = Path(path).read_bytes()
    header_end = raw.index(b"end_header") + len(b"end_header")
    data_offset = header_end
    while raw[data_offset:data_offset + 1] in (b"\n", b"\r"):
        data_offset += 1
    header = raw[:header_end].decode("ascii")
    type_map = {
        "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
        "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    }
    properties = []
    vertex_count = None
    for line in header.splitlines():
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        elif len(fields) >= 3 and fields[0] == "property" and fields[1] in type_map:
            properties.append((fields[2], type_map[fields[1]]))
    if vertex_count is None or not properties:
        raise ValueError(f"Could not parse vertex schema from {path}")
    dtype = np.dtype(properties)
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(vertex_count,))
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32, copy=False)
    if all(name in vertices.dtype.names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        # 3DGS stores the DC spherical-harmonic coefficient; this converts it
        # back to a display RGB color and keeps the scene context recognizable.
        rgb = np.clip(0.5 + 0.2820947918 * np.column_stack(
            (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])
        ), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        rgb = np.full((vertex_count, 3), 0.55, dtype=np.float32)
    return xyz, rgb


def to_local(points, center):
    """Habitat XYZ -> plotting coordinates (X, Z, vertical Y)."""
    p = np.asarray(points, dtype=float)
    c = np.asarray(center, dtype=float)
    return np.column_stack((p[..., 0] - c[0], p[..., 2] - c[2], p[..., 1] - c[1]))


def camera_center(w2c):
    w2c = np.asarray(w2c, dtype=float)
    return -w2c[:3, :3].T @ w2c[:3, 3]


def camera_forward(w2c):
    w2c = np.asarray(w2c, dtype=float)
    direction = w2c[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return direction / np.linalg.norm(direction)


def direction_to_local(direction):
    d = np.asarray(direction, dtype=float)
    local = np.array([d[0], d[2], d[1]])
    return local / max(np.linalg.norm(local), 1e-9)


def load_assets():
    semantic_xyz = load_ascii_ply_xyz(CLUSTER_PLY)
    center = np.median(semantic_xyz, axis=0)
    with MR_META.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"target_center": np.zeros(3, dtype=float)}, center, meta


def style_3d(ax, limits=(-1.8, 1.8, -1.8, 1.8, -0.45, 1.45), labels=True, zoom=1.0):
    ax.set_facecolor(BG)
    ax.view_init(elev=26, azim=-58)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_zlim(limits[4], limits[5])
    ax.set_box_aspect(
        (limits[1] - limits[0], limits[3] - limits[2], 0.78 * (limits[5] - limits[4])),
        zoom=zoom,
    )
    # These are method schematics rather than metric plots: remove the full
    # coordinate frame, ticks, panes, and grid so geometry remains the focus.
    ax.set_axis_off()


def _cuboid_vertices(center, size):
    center = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    x0, x1 = center[0] - half[0], center[0] + half[0]
    y0, y1 = center[1] - half[1], center[1] + half[1]
    z0, z1 = center[2] - half[2], center[2] + half[2]
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def draw_cuboid(ax, center, size, facecolor="#58738a", edgecolor=MAGENTA,
                alpha=0.95, linewidth=1.2):
    v = _cuboid_vertices(center, size)
    faces = [[v[i] for i in face] for face in ((0, 1, 2, 3), (4, 5, 6, 7),
                                                (0, 1, 5, 4), (1, 2, 6, 5),
                                                (2, 3, 7, 6), (3, 0, 4, 7))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=facecolor,
                                          edgecolor=edgecolor, linewidth=linewidth,
                                          alpha=alpha, zorder=5))


def draw_beam(ax, start, end, width=0.085, color="#48677e", edgecolor=MAGENTA,
              alpha=0.95):
    """Draw a rectangular beam between two 3-D points."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return
    axis /= length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    r = width / 2
    corners = np.array([p + su * r * u + sv * r * v for p in (start, end)
                        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    faces = [[corners[i + j] for j in range(4)] for i in (0, 4)]
    faces += [[corners[i], corners[(i + 1) % 4], corners[4 + (i + 1) % 4], corners[4 + i]]
              for i in range(4)]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=edgecolor,
                                          linewidth=0.85, alpha=alpha, zorder=5))


def plot_cluster(ax, cluster=None, n=2200, alpha=0.32, size=5.0, nvsm_style=False,
                 proxy_scale=0.45):
    """Draw a complete low-poly proxy for the target chair."""
    # Keep the proxy compact so orbit poses and refinement arrows remain visible.
    scale = proxy_scale
    def cp(point):
        return np.asarray(point, dtype=float) * scale

    edge = "#282d32" if nvsm_style else MAGENTA
    seat_color = "#50575e" if nvsm_style else "#5f7e93"
    back_color = "#383e45" if nvsm_style else "#496b82"
    leg_color = "#454c53" if nvsm_style else "#48677e"
    support_color = "#343a40" if nvsm_style else "#6d8799"
    seat_depth = 0.92 if nvsm_style else 0.82
    back_depth = 0.09 if nvsm_style else 0.13
    draw_cuboid(ax, cp((0.0, 0.03, 0.42)), np.asarray((1.05, seat_depth, 0.16)) * scale,
                facecolor=seat_color, edgecolor=edge, linewidth=1.35)
    draw_cuboid(ax, cp((0.0, -0.31, 0.91)), np.asarray((1.05, back_depth, 1.10)) * scale,
                facecolor=back_color, edgecolor=edge, linewidth=1.35)
    for x in (-0.40, 0.40):
        draw_beam(ax, cp((x, -0.27, 0.34)), cp((x * 1.18, -0.40, -0.12)),
                  width=0.105 * scale, color=leg_color, edgecolor=edge)
        draw_beam(ax, cp((x, 0.30, 0.34)), cp((x * 1.18, 0.43, -0.12)),
                  width=0.105 * scale, color=leg_color, edgecolor=edge)
        draw_beam(ax, cp((x, -0.33, 0.12)), cp((x, 0.33, 0.12)), width=0.065 * scale,
                  color=support_color, edgecolor=edge, alpha=0.9)
        draw_beam(ax, cp((x, -0.28, 0.39)), cp((x, -0.28, 0.68)), width=0.075 * scale,
                  color=leg_color, edgecolor=edge)
    outline = np.asarray([[-0.53, 0.44, 0.51], [0.53, 0.44, 0.51],
                          [0.53, -0.38, 0.51], [-0.53, -0.38, 0.51],
                          [-0.53, 0.44, 0.51]]) * scale
    ax.plot(outline[:, 0], outline[:, 1], outline[:, 2],
            color=edge, linewidth=1.1, alpha=0.85, zorder=8)
    if not nvsm_style:
        ax.scatter([0], [0.04 * scale], [0.53 * scale], marker="*", s=62, color=MAGENTA,
                   edgecolor="white", linewidth=0.8, zorder=10)


def _shade(color, factor):
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=float)
    if factor >= 1.0:
        return tuple(1.0 - (1.0 - rgb) / factor)
    return tuple(rgb * factor)


def draw_solid_arrow(ax, position, direction, length, color, alpha=1.0, linewidth=2.0):
    """Draw a broad, smoothly shaded 3-D arrow with a cylinder and cone."""
    p = np.asarray(position, dtype=float)
    d = np.asarray(direction, dtype=float)
    d /= max(np.linalg.norm(d), 1e-9)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.88 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(d, u)
    v /= max(np.linalg.norm(v), 1e-9)

    head_fraction = 0.40
    head_base = p + d * length * (1.0 - head_fraction)
    tip = p + d * length
    shaft_radius = length * (0.043 + 0.004 * linewidth)
    head_radius = length * 0.22
    dark = _shade(color, 0.72)
    segments = 18
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    radial = [math.cos(a) * u + math.sin(a) * v for a in angles]

    # Cylindrical shaft overlaps the cone slightly to avoid a visible seam.
    shaft_end = head_base + d * length * 0.025
    shaft_start_ring = [p + shaft_radius * r for r in radial]
    shaft_end_ring = [shaft_end + shaft_radius * r for r in radial]
    shaft_faces = [
        [shaft_start_ring[i], shaft_start_ring[(i + 1) % segments],
         shaft_end_ring[(i + 1) % segments], shaft_end_ring[i]]
        for i in range(segments)
    ]
    side_colors = [_shade(color, 0.76 + 0.44 * (0.5 + 0.5 * math.cos(a - 0.7))) for a in angles]
    ax.add_collection3d(Poly3DCollection(
        shaft_faces,
        facecolors=side_colors,
        edgecolor=dark,
        linewidth=0.12,
        alpha=alpha,
        zorder=34,
    ))

    cone_ring = [head_base + head_radius * r for r in radial]
    cone_faces = [[tip, cone_ring[i], cone_ring[(i + 1) % segments]] for i in range(segments)]
    cone_colors = [_shade(color, 0.74 + 0.56 * (0.5 + 0.5 * math.cos(a - 0.7))) for a in angles]
    ax.add_collection3d(Poly3DCollection(
        cone_faces,
        facecolors=cone_colors,
        edgecolor=dark,
        linewidth=0.12,
        alpha=alpha,
        zorder=35,
    ))


def draw_pose(ax, position, target=(0.0, 0.0, 0.12), color=BLUE, length=0.36,
              alpha=1.0, linewidth=2.0, circle=False, label=None, direction=None):
    p = np.asarray(position, dtype=float)
    if direction is None:
        d = np.asarray(target, dtype=float) - p
    else:
        d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-9)
    draw_solid_arrow(ax, p, d, length, color, alpha=alpha, linewidth=linewidth)
    if circle:
        ax.scatter([p[0]], [p[1]], [p[2]], s=165, facecolors="none", edgecolors=MAGENTA, linewidths=2.0, zorder=13)
    if label:
        ax.text(p[0], p[1], p[2] + 0.13, label, color=color, fontsize=8.5, fontweight="semibold", ha="center")


def draw_box(ax, center, size, color=ORANGE, linewidth=1.15):
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    lo = center - size / 2
    hi = center + size / 2
    corners = np.array([
        [x, y, z]
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                edges.append((i, j))
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color, linestyle=(0, (4, 2.2)), linewidth=linewidth,
            alpha=1.0, antialiased=True, solid_capstyle="round",
            dash_capstyle="round",
        )


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_clean_axes_crop(fig, ax, stem):
    """Export one method panel without any title, caption, or annotation."""
    OUT.mkdir(parents=True, exist_ok=True)
    title_visible = ax.title.get_visible()
    text_visibility = [artist.get_visible() for artist in ax.texts]
    ax.title.set_visible(False)
    for artist in ax.texts:
        artist.set_visible(False)
    fig.canvas.draw()
    position = ax.get_position()
    bbox = Bbox.from_bounds(
        position.x0 * fig.get_figwidth(),
        position.y0 * fig.get_figheight(),
        position.width * fig.get_figwidth(),
        position.height * fig.get_figheight(),
    )
    for suffix, options in (
        ("png", {"dpi": 260}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            facecolor="white",
            bbox_inches=bbox,
            pad_inches=0.0,
            **options,
        )
    ax.title.set_visible(title_visible)
    for artist, visible in zip(ax.texts, text_visibility):
        artist.set_visible(visible)


def draw_chair_topdown(ax):
    """Draw a recognizable chair using strict top-view geometry."""
    edge = "#282d32"
    # Scale about the object origin, then compensate for the asymmetric
    # backrest so the complete chair silhouette is centered on the orbit.
    chair_transform = Affine2D().scale(0.78).translate(0.0, -0.035) + ax.transData
    # Four short diagonal footprints protrude from the seat corners. Their
    # symmetry is a strong top-view cue while remaining physically plausible.
    legs = [
        Polygon([(-0.26, 0.20), (-0.19, 0.25), (-0.34, 0.40), (-0.42, 0.33)],
                closed=True, facecolor="#343a40", edgecolor=edge, linewidth=1.6, zorder=9,
                transform=chair_transform),
        Polygon([(0.19, 0.25), (0.26, 0.20), (0.42, 0.33), (0.34, 0.40)],
                closed=True, facecolor="#343a40", edgecolor=edge, linewidth=1.6, zorder=9,
                transform=chair_transform),
        Polygon([(-0.27, -0.20), (-0.19, -0.25), (-0.34, -0.41), (-0.43, -0.34)],
                closed=True, facecolor="#454c53", edgecolor=edge, linewidth=1.6, zorder=9,
                transform=chair_transform),
        Polygon([(0.19, -0.25), (0.27, -0.20), (0.43, -0.34), (0.34, -0.41)],
                closed=True, facecolor="#454c53", edgecolor=edge, linewidth=1.6, zorder=9,
                transform=chair_transform),
    ]
    for leg in legs:
        ax.add_patch(leg)
    # Two visible posts separate the rear backrest from the seat cushion.
    for x in (-0.22, 0.22):
        ax.add_patch(FancyBboxPatch(
            (x - 0.035, 0.22), 0.07, 0.15,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor="#343a40", edgecolor=edge, linewidth=1.4, zorder=10,
            transform=chair_transform,
        ))
    seat = FancyBboxPatch(
        (-0.31, -0.27), 0.62, 0.54,
        boxstyle="round,pad=0.018,rounding_size=0.07",
        facecolor="#50575e", edgecolor=edge, linewidth=2.2, zorder=12,
        transform=chair_transform,
    )
    back = FancyBboxPatch(
        (-0.38, 0.37), 0.76, 0.09,
        boxstyle="round,pad=0.010,rounding_size=0.035",
        facecolor="#383e45", edgecolor=edge, linewidth=2.2, zorder=13,
        transform=chair_transform,
    )
    ax.add_patch(seat)
    ax.add_patch(back)
    ax.plot([-0.24, 0.24], [0.18, 0.18], color="#858d95", linewidth=1.3,
            alpha=0.9, zorder=14, transform=chair_transform)
    ax.plot([-0.29, 0.29], [0.405, 0.405], color="#6d747b", linewidth=1.0,
            alpha=0.9, zorder=14, transform=chair_transform)
    ax.text(0.0, -0.01, "chair", color="#f4f6f8", fontsize=9.5,
            fontweight="semibold", ha="center", va="center", zorder=15,
            transform=chair_transform)


def nvsm_selected_pose(center_world):
    """Return the framework-consistent NVSM winner as PRE's input pose."""
    radii = np.array([0.90, 1.36, 1.82])
    angle_offsets = np.radians([0.0, 12.0, 24.0])
    coarse_world = np.array([-1.24, -2.20, 4.83])
    coarse_local = to_local(coarse_world[None, :], center_world)[0]

    # In the paper framework, the coarse view carried into fine localization
    # is the upper-left novel view around the chair.  Map its approximate
    # 135-degree pose to the nearest candidate in the displayed six-sample
    # inner orbit (120 degrees), instead of selecting the mirrored view.
    selected_ring = 0
    framework_angle = math.radians(135.0)
    selected_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle_offsets[selected_ring]
    angle_error = np.abs(np.angle(np.exp(1j * (selected_angles - framework_angle))))
    selected_sample = int(np.argmin(angle_error))
    selected_angle = selected_angles[selected_sample]
    selected_pose = np.array([
        radii[selected_ring] * math.cos(selected_angle),
        # The top-down paper view uses image-style vertical orientation.  Its
        # upper half therefore maps to negative local Y, which is the 3-D
        # proxy's backrest side.  This preserves the framework's rear view.
        -radii[selected_ring] * math.sin(selected_angle),
        coarse_local[2],
    ])
    return selected_pose, selected_ring, selected_sample


def make_nvsm(cluster, center_world):
    fig = plt.figure(figsize=(8.2, 6.5), facecolor="white")
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.76])
    ax.set_aspect("equal")
    ax.set_xlim(-2.06, 2.06)
    ax.set_ylim(-2.06, 2.06)
    ax.set_facecolor(BG)
    ax.set_axis_off()

    radii = [0.90, 1.36, 1.82]
    # Matched cool palette with comparable luminance: violet -> blue -> green.
    ring_colors = ["#7658b5", "#347fb6", "#2c9a78"]
    angle_offsets = [0.0, math.radians(12), math.radians(24)]
    # This winner is shared verbatim with PRE Stage 1.
    _, selected_ring, selected_sample = nvsm_selected_pose(center_world)
    for ridx, (radius, color, offset) in enumerate(zip(radii, ring_colors, angle_offsets), start=1):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=color,
                            linewidth=2.6, alpha=1.0, zorder=3 + ridx))
        sample_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + offset
        for sample_idx, angle in enumerate(sample_angles):
            start = np.array([radius * math.cos(angle), radius * math.sin(angle)])
            direction = -start / radius
            end = start + 0.34 * direction
            is_selected = (ridx - 1 == selected_ring and sample_idx == selected_sample)
            arrow_color = SELECTED_RED if is_selected else color
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle="-|>", mutation_scale=19 if is_selected else 16,
                linewidth=2.8 if is_selected else 2.2, color=arrow_color,
                shrinkA=0, shrinkB=0, zorder=20,
            )
            ax.add_patch(arrow)
        tag_angle = math.radians(28)
        ax.text(
            radius * math.cos(tag_angle) + 0.055,
            radius * math.sin(tag_angle) + 0.035,
            rf"$r_{ridx}$", color=color, fontsize=12, fontweight="bold",
            ha="left", va="bottom", zorder=25,
        )

    draw_chair_topdown(ax)

    fig.text(0.5, 0.915, "Equally spaced camera poses at adaptive orbit radii, all facing the target instance", ha="center", fontsize=9.5, color=MUTED)
    legend = [
        Line2D([0], [0], color=ring_colors[i], lw=2.6, label=rf"orbit radius $r_{i+1}$")
        for i in range(3)
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=9)
    save_figure(fig, "NVSM_multiscale_orbit_sampling")


def _segment_distance_2d(segment_a, segment_b):
    """Approximate minimum distance between two short projected segments."""
    a0, a1 = np.asarray(segment_a[0]), np.asarray(segment_a[1])
    b0, b1 = np.asarray(segment_b[0]), np.asarray(segment_b[1])
    ta = np.linspace(0.0, 1.0, 9)[:, None]
    tb = np.linspace(0.0, 1.0, 9)[:, None]
    pa = a0 + ta * (a1 - a0)
    pb = b0 + tb * (b1 - b0)
    return float(np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)))


def _project_segment(ax, start, end):
    x, y, _ = proj3d.proj_transform(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]], ax.get_proj()
    )
    return np.column_stack((x, y))


def _box_edge_segments(center, size):
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    lo, hi = center - size / 2, center + size / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
            if np.sum(corners[i] != corners[j]) == 1]


def pre_candidates(
    ax, center, size, count, selected, arrow_length,
    initial_position=None, initial_direction=None,
):
    """Place equal-size arrows while avoiding projected box edges and overlap."""
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    selected = np.asarray(selected, dtype=float)
    grid = [-0.30, -0.15, 0.0, 0.15, 0.30]
    anchors = np.array([[x, y, z] for x in grid for y in grid for z in grid
                        if not (x == 0.0 and y == 0.0 and z == 0.0)])
    pts = center + anchors * size
    directions = []
    for index, pose in enumerate(pts):
        base = np.array([0.0, 0.0, 0.12]) - pose
        base /= max(np.linalg.norm(base), 1e-9)
        side = np.cross(np.array([0.0, 0.0, 1.0]), base)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        yaw_offset = 0.22 * math.sin(1.7 * index)
        pitch_offset = 0.16 * math.cos(1.3 * index)
        varied = base + yaw_offset * side + pitch_offset * np.array([0.0, 0.0, 1.0])
        directions.append(varied / max(np.linalg.norm(varied), 1e-9))
    selected_direction = np.array([0.0, 0.0, 0.12]) - selected
    selected_direction /= max(np.linalg.norm(selected_direction), 1e-9)
    box_segments_2d = [_project_segment(ax, a, b) for a, b in _box_edge_segments(center, size)]
    red_segment = _project_segment(ax, selected, selected + arrow_length * selected_direction)
    initial_segment = None
    if initial_position is not None and initial_direction is not None:
        initial_position = np.asarray(initial_position, dtype=float)
        initial_segment = _project_segment(
            ax, initial_position, initial_position + arrow_length * initial_direction
        )
    candidates = []
    for pose, direction in zip(pts, directions):
        segment = _project_segment(ax, pose, pose + arrow_length * direction)
        edge_clearance = min(_segment_distance_2d(segment, edge) for edge in box_segments_2d)
        red_clearance = _segment_distance_2d(segment, red_segment)
        initial_clearance = (
            _segment_distance_2d(segment, initial_segment)
            if initial_segment is not None else np.inf
        )
        if edge_clearance >= 0.0020 and red_clearance >= 0.009 and initial_clearance >= 0.006:
            candidates.append((pose, direction, segment, edge_clearance))

    # Select projected segments greedily for maximum visual separation.
    chosen = []
    while candidates and len(chosen) < count - 1:
        if not chosen:
            best = max(range(len(candidates)), key=lambda i: candidates[i][3])
        else:
            best = max(
                range(len(candidates)),
                key=lambda i: min(_segment_distance_2d(candidates[i][2], item[2]) for item in chosen),
            )
        item = candidates.pop(best)
        if chosen and min(_segment_distance_2d(item[2], other[2]) for other in chosen) < 0.0075:
            continue
        chosen.append(item)
    chosen_pts = [item[0] for item in chosen]
    chosen_dirs = [item[1] for item in chosen]
    return np.vstack(chosen_pts + [selected]), np.vstack(chosen_dirs + [selected_direction])


def pre_candidates_fine_stage(
    ax, center, size, selected, arrow_length,
    initial_position=None, initial_direction=None, count=7,
):
    """Screen-space-balanced samples for the unchanged Stage-4 volume."""
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    selected = np.asarray(selected, dtype=float)
    selected_direction = np.array([0.0, 0.0, 0.12]) - selected
    selected_direction /= max(np.linalg.norm(selected_direction), 1e-9)
    red_segment = _project_segment(ax, selected, selected + arrow_length * selected_direction)
    initial_segment = None
    if initial_position is not None and initial_direction is not None:
        initial_position = np.asarray(initial_position, dtype=float)
        initial_segment = _project_segment(
            ax, initial_position, initial_position + arrow_length * initial_direction
        )
    box_segments_2d = [_project_segment(ax, a, b) for a, b in _box_edge_segments(center, size)]

    # Generate many valid 3-D samples, then distribute their *origins* across
    # the final 2-D projection.  Requiring the complete equal-length arrow to
    # remain inside this tiny box biases every pose toward one side, which is
    # precisely the visual clustering this stage must avoid.
    rng = np.random.default_rng(43)
    pool = []
    for _ in range(5000):
        pose = center + rng.uniform(-0.38, 0.38, size=3) * size
        # Fine-stage samples still observe the target chair.  Only a small
        # angular perturbation is used to depict pose sampling; arbitrary
        # outward-facing arrows would misrepresent the search objective.
        base = np.array([0.0, 0.0, 0.12]) - pose
        base /= max(np.linalg.norm(base), 1e-9)
        side = np.cross(np.array([0.0, 0.0, 1.0]), base)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        direction = base + rng.uniform(-0.09, 0.09) * side
        direction += rng.uniform(-0.06, 0.06) * np.array([0.0, 0.0, 1.0])
        direction /= max(np.linalg.norm(direction), 1e-9)
        endpoint = pose + arrow_length * direction
        segment = _project_segment(ax, pose, endpoint)
        origin_2d = segment[0]
        projected_length = float(np.linalg.norm(segment[1] - segment[0]))
        if projected_length < 0.0048:
            continue
        origin_clearance = min(
            _segment_distance_2d((origin_2d, origin_2d), edge)
            for edge in box_segments_2d
        )
        red_clearance = _segment_distance_2d(segment, red_segment)
        initial_clearance = (
            _segment_distance_2d(segment, initial_segment)
            if initial_segment is not None else np.inf
        )
        if origin_clearance >= 0.0020 and red_clearance >= 0.0050 and initial_clearance >= 0.0040:
            pool.append((
                pose, direction, segment, origin_2d, origin_clearance,
                red_clearance, initial_clearance, projected_length,
            ))

    red_origin = red_segment[0]
    initial_origin = initial_segment[0] if initial_segment is not None else red_origin
    chosen = []
    while pool and len(chosen) < count - 1:
        best = max(
            range(len(pool)),
            key=lambda i: min(
                [
                    np.linalg.norm(pool[i][3] - red_origin),
                    np.linalg.norm(pool[i][3] - initial_origin),
                ]
                + [np.linalg.norm(pool[i][3] - other[3]) for other in chosen]
            ),
        )
        item = pool.pop(best)
        if chosen and min(_segment_distance_2d(item[2], other[2]) for other in chosen) < 0.0065:
            continue
        chosen.append(item)

    pts = [item[0] for item in chosen]
    directions = [item[1] for item in chosen]
    return np.vstack(pts + [selected]), np.vstack(directions + [selected_direction])


def pre_schematic_geometry(center_world):
    """Shared PRE geometry so the REF input exactly matches PRE's output."""
    nvsm_pose, _, _ = nvsm_selected_pose(center_world)
    box_sizes = [
        1.25 * np.array([0.38, 0.38, 1.05]),
        1.25 * np.array([1.00, 1.00, 0.62]),
        1.50 * np.array([0.68, 0.68, 0.42]),
        1.60 * np.array([0.30, 0.30, 0.24]),
    ]
    # These small offsets define the established box layout.  Keep them fixed:
    # later display exaggeration must not resize or reposition any volume.
    layout_offsets = [
        np.array([0.04, -0.05, 0.04]),
        np.array([-0.05, 0.05, 0.04]),
        np.array([0.05, -0.04, 0.03]),
        np.array([-0.04, 0.04, 0.02]),
    ]
    box_centers = [nvsm_pose]
    for stage_idx in range(3):
        layout_pose = box_centers[stage_idx] + layout_offsets[stage_idx] * box_sizes[stage_idx]
        box_centers.append(layout_pose)

    # Exaggerate the update within each unchanged volume so PRE's effect is
    # legible at paper scale.  The next stage inherits the preceding red pose
    # exactly, even though the schematic displacement is larger than the run.
    selected = [
        box_centers[0] + np.array([0.18, -0.14, 0.15]) * box_sizes[0],
        box_centers[1] + np.array([-0.18, 0.18, 0.12]) * box_sizes[1],
        box_centers[3] + np.array([0.15, -0.15, 0.10]) * box_sizes[3],
        box_centers[3] + np.array([-0.25, 0.24, 0.18]) * box_sizes[3],
    ]
    initial_poses = [box_centers[0]] + selected[:3]
    return {
        "box_sizes": box_sizes,
        "box_centers": box_centers,
        "selected": selected,
        "initial_poses": initial_poses,
    }


def make_pre(cluster, center_world):
    geometry = pre_schematic_geometry(center_world)
    box_sizes = geometry["box_sizes"]
    box_centers = geometry["box_centers"]
    selected = geometry["selected"]
    initial_poses = geometry["initial_poses"]
    # Fewer, substantially larger pose glyphs survive paper-scale reduction
    # far better than the former dense set of tiny arrows.  Candidate
    # selection below explicitly rejects projected overlaps.
    candidate_counts = [6, 9, 8, 6]
    arrow_length = 0.20
    # Stage 1 receives the coarse pose.  Every later purple initial pose is
    # exactly the preceding panel's red selection, in both position and
    # orientation, matching the progressive-search data flow.
    target = np.array([0.0, 0.0, 0.12])
    selected_directions = [target - pose for pose in selected]
    selected_directions = [direction / np.linalg.norm(direction) for direction in selected_directions]
    coarse_direction = target - initial_poses[0]
    coarse_direction /= np.linalg.norm(coarse_direction)
    initial_directions = [coarse_direction] + selected_directions[:3]
    # One continuous canvas, read as a compact snake-shaped sequence:
    # upper-left -> upper-right -> lower-right -> lower-left.  Axes are
    # intentionally flush and transparent, so the result reads as one method
    # diagram rather than four separately pasted pictures.
    fig = plt.figure(figsize=(9.2, 9.2), facecolor=BG)
    panel_positions = [
        [0.000, 0.505, 0.485, 0.485],
        [0.515, 0.505, 0.485, 0.485],
        [0.515, 0.010, 0.485, 0.485],
        [0.000, 0.010, 0.485, 0.485],
    ]
    axes = []
    for idx in range(4):
        ax = fig.add_axes(panel_positions[idx], projection="3d")
        axes.append(ax)
        # A common, tighter crop preserves the relative box scales across all
        # stages while letting the chair-and-volume pair fill each panel.
        # Stage 2 owns the widest search volume, so give only that panel a
        # little extra projection margin.  Keeping the tighter crop elsewhere
        # preserves the enlarged chair/pose glyphs requested for PRE.
        limits = (-1.10, 0.38, -1.48, 0.38, -0.36, 1.04)
        panel_zoom = 1.25
        if idx == 1:
            limits = (-1.16, 0.44, -1.60, 0.44, -0.38, 1.06)
            panel_zoom = 1.18
        style_3d(ax, limits, labels=False, zoom=panel_zoom)
        ax.patch.set_alpha(0.0)
        # Rotate the complete PRE scene by 180 degrees so the chair back faces
        # the reader and the important initial/selected poses are no longer
        # hidden behind the chair body.
        ax.view_init(elev=24, azim=210)
        # PRE previously used the same small chair proxy in each quarter-size
        # panel.  Enlarge it here so its printed size matches the standalone
        # localization-module figures.
        plot_cluster(
            ax, cluster, n=1050, alpha=0.19, size=3.2,
            nvsm_style=True, proxy_scale=0.70 if idx == 1 else 0.64,
        )
        draw_box(
            ax, box_centers[idx], box_sizes[idx], color="#000000",
            linewidth=1.55,
        )
        if idx == 3:
            candidates, directions = pre_candidates_fine_stage(
                ax, box_centers[idx], box_sizes[idx], selected[idx], arrow_length,
                initial_position=initial_poses[idx],
                initial_direction=initial_directions[idx],
                count=candidate_counts[idx],
            )
        else:
            candidates, directions = pre_candidates(
                ax, box_centers[idx], box_sizes[idx], candidate_counts[idx], selected[idx], arrow_length,
                initial_position=initial_poses[idx],
                initial_direction=initial_directions[idx],
            )
        draw_pose(
            ax,
            initial_poses[idx],
            color=PRE_INITIAL_PURPLE,
            length=arrow_length,
            alpha=1.0,
            linewidth=2.75,
            circle=False,
            direction=initial_directions[idx],
        )
        for cidx, (pose, direction) in enumerate(zip(candidates, directions)):
            is_selected = cidx == len(candidates) - 1
            draw_pose(
                ax,
                pose,
                color=SELECTED_RED if is_selected else PRE_SKY_BLUE,
                length=arrow_length,
                alpha=1.0 if is_selected else 0.82,
                linewidth=3.0 if is_selected else 1.75,
                circle=False,
                direction=direction,
            )

    pure_stems = [
        "PRE_stage_1_height_pitch_pure",
        "PRE_stage_2_local_position_angles_pure",
        "PRE_stage_3_recentered_search_pure",
        "PRE_stage_4_dense_fine_search_pure",
    ]
    for ax, stem in zip(axes, pure_stems):
        save_clean_axes_crop(fig, ax, stem)

    connections = [
        # Stage 1 -> Stage 2
        ((0.474, 0.755), (0.526, 0.755)),
        # Stage 2 -> Stage 3
        ((0.755, 0.535), (0.755, 0.465)),
        # Stage 3 -> Stage 4
        ((0.526, 0.250), (0.474, 0.250)),
    ]
    for start, end in connections:
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=23,
            linewidth=2.8, color=MAGENTA,
        )
        fig.add_artist(arrow)
    save_figure(fig, "PRE_progressive_random_pose_search")


def ref_schematic_geometry(center_world):
    """Shared REF geometry so MR begins at the displayed REF output."""
    # REF begins exactly at the red output of the final PRE panel.  The real
    # update is only a few centimetres and projects almost onto itself in this
    # view, so use a deliberately enlarged translation to expose REF's joint
    # position/orientation correction.  Both cameras observe the same chair.
    pre_pos = pre_schematic_geometry(center_world)["selected"][-1]
    # Keep the refinement local: enough separation to read in print, without
    # implying that differentiable optimization makes a large viewpoint jump.
    ref_pos_display = pre_pos + np.array([-0.22, 0.10, 0.06])
    target = np.array([0.0, 0.0, 0.12])
    pre_dir = target - pre_pos
    pre_dir /= np.linalg.norm(pre_dir)
    ref_dir = target - ref_pos_display
    ref_dir /= np.linalg.norm(ref_dir)
    return {
        "pre_pos": pre_pos,
        "ref_pos": ref_pos_display,
        "pre_dir": pre_dir,
        "ref_dir": ref_dir,
    }


def make_ref(cluster, center_world, meta):
    geometry = ref_schematic_geometry(center_world)
    pre_pos = geometry["pre_pos"]
    ref_pos_display = geometry["ref_pos"]
    pre_dir = geometry["pre_dir"]
    ref_dir = geometry["ref_dir"]
    arrow_length = 0.20

    # Compact single-panel view: both poses are overlaid so the refinement
    # displacement is visible without a redundant before/after split.
    fig = plt.figure(figsize=(6.8, 5.8), facecolor="white")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.78], projection="3d")
    style_3d(ax, (-1.05, 1.15, -0.65, 1.40, -0.18, 1.32), labels=False)
    ax.view_init(elev=24, azim=30)
    plot_cluster(ax, cluster, n=2100, alpha=0.31, size=4.2, nvsm_style=True)

    draw_pose(
        ax, pre_pos, color=PRE_INITIAL_PURPLE, length=arrow_length,
        linewidth=2.4, label=None, direction=pre_dir,
    )
    ax.plot(
        [pre_pos[0], ref_pos_display[0]],
        [pre_pos[1], ref_pos_display[1]],
        [pre_pos[2], ref_pos_display[2]],
        color="#000000", linestyle=(0, (3, 2)), linewidth=1.45, alpha=0.82,
    )
    draw_pose(
        ax, ref_pos_display, color=SELECTED_RED, length=arrow_length,
        linewidth=2.5, circle=False, label=None, direction=ref_dir,
    )
    save_figure(fig, "REF_differentiable_pose_refinement")


def make_mr(cluster, center_world, meta):
    match = mpimg.imread(MATCH_IMAGE)
    ref_geometry = ref_schematic_geometry(center_world)
    ref_pos = ref_geometry["ref_pos"]
    ref_dir = ref_geometry["ref_dir"]
    # Enlarge the PnP correction at paper scale, as in REF, while retaining
    # the same target-facing camera convention.
    mr_pos = ref_pos + np.array([-0.30, 0.18, -0.12])
    target = np.array([0.0, 0.0, 0.12])
    mr_dir = target - mr_pos
    mr_dir /= np.linalg.norm(mr_dir)
    arrow_length = 0.20

    fig = plt.figure(figsize=(13.0, 5.5), facecolor="white")
    img_ax = fig.add_axes([0.025, 0.17, 0.54, 0.66])
    img_ax.imshow(match)
    img_ax.set_axis_off()
    for spine in img_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.0)
    fig.text(0.17, 0.845, "Target image", ha="center", fontsize=10.5, color=INK, fontweight="semibold")
    fig.text(0.435, 0.845, "Rendered view", ha="center", fontsize=10.5, color=INK, fontweight="semibold")
    fig.text(0.295, 0.115, "cached 2D correspondences", ha="center", fontsize=9, color=MUTED)

    ax3 = fig.add_axes([0.66, 0.10, 0.33, 0.76], projection="3d")
    style_3d(ax3, (-1.05, 1.15, -0.65, 1.40, -0.18, 1.32), labels=False)
    ax3.view_init(elev=24, azim=30)
    plot_cluster(ax3, cluster, n=2300, alpha=0.30, size=4.3, nvsm_style=True)
    draw_pose(
        ax3, ref_pos, color=PRE_INITIAL_PURPLE, length=arrow_length,
        linewidth=2.4, label=None, direction=ref_dir,
    )
    ax3.plot(
        [ref_pos[0], mr_pos[0]], [ref_pos[1], mr_pos[1]],
        [ref_pos[2], mr_pos[2]], color="#000000",
        linestyle=(0, (3, 2)), linewidth=1.45, alpha=0.82,
    )
    draw_pose(
        ax3, mr_pos, color=SELECTED_RED, length=arrow_length,
        linewidth=2.5, circle=False, label=None, direction=mr_dir,
    )

    arrow = FancyArrowPatch((0.565, 0.50), (0.655, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=21, linewidth=2.3, color=MAGENTA)
    fig.add_artist(arrow)
    fig.text(0.61, 0.57, "depth back-project\n+ RANSAC-PnP", ha="center", fontsize=8.5, color=MAGENTA, fontweight="semibold")
    save_figure(fig, "MR_correspondence_viewpoint_correction")


def main():
    cluster, center_world, meta = load_assets()
    make_nvsm(cluster, center_world)
    make_pre(cluster, center_world)
    make_ref(cluster, center_world, meta)
    make_mr(cluster, center_world, meta)
    for stem in (
        "NVSM_multiscale_orbit_sampling",
        "PRE_progressive_random_pose_search",
        "REF_differentiable_pose_refinement",
        "MR_correspondence_viewpoint_correction",
    ):
        print(OUT / f"{stem}.png")


if __name__ == "__main__":
    main()
