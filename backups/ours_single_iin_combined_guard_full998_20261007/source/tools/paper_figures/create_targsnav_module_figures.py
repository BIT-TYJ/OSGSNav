#!/usr/bin/env python3
"""Create TarGSNav method and qualitative figures from IEVE visualizations."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "detectron2" / "docs" / "targsnav_figures"
SRC = Path("/nas_ds925/IEVE/showcase_episode_994_zt1RVoi7PcG")
W, H = 3200, 1800

C = {
    "ink": "#18222D",
    "muted": "#66727E",
    "blue": "#245B8A",
    "cyan": "#DFF2F5",
    "green": "#E7F4CF",
    "green2": "#BFE39E",
    "purple": "#EEE8F6",
    "yellow": "#FFF0C8",
    "pink": "#FCE4EE",
    "magenta": "#F04B87",
    "white": "#FFFFFF",
    "gray": "#F5F7F8",
    "orange": "#F2A64A",
    "red": "#E65252",
}

ASSET = {
    "goal": SRC / "02_target_image" / "target_image.png",
    "target_obs": SRC / "01_rgb_depth" / "target_observation_rgb.png",
    "early": SRC / "01_rgb_depth" / "no_chair_after_existing_step_000372" / "rgb.png",
    "recent": SRC / "01_rgb_depth" / "no_chair_within_8_steps_before_step_000398" / "rgb.png",
    "mask": SRC / "03_instance_masks" / "best_per_category" / "best_chair.png",
    "map": SRC / "05_semantic_grid_map" / "final_semantic_grid_map_clean.png",
    "map_full": SRC / "05_semantic_grid_map" / "final_full_visualization.png",
    "views": SRC / "06_nvsm_four_views" / "four_views_montage.png",
    "sbo": SRC / "07_pre_ref_comparison" / "01_pre_before_after.png",
    "gbo": SRC / "07_pre_ref_comparison" / "02_ref_before_after.png",
    "pva": SRC / "08_mr_comparison" / "mr_before_after.png",
    "matches": SRC / "10_fem_lightglue" / "fem_lightglue_matches.png",
    "gaussian": ROOT / "docs" / "gaussian_step39_red_arrow_native" / "red_arrow_yaw_sweep.png",
    "map_mid": ROOT / "docs" / "image_hd.png",
}


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


_DATA_URI_CACHE = {}


def data_uri(path: Path) -> str:
    """Embed a paper-resolution JPEG to keep SVGs portable and renderer-friendly."""
    if path not in _DATA_URI_CACHE:
        with Image.open(path) as source:
            source = source.convert("RGB")
            source.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            source.save(buffer, "JPEG", quality=88, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        _DATA_URI_CACHE[path] = f"data:image/jpeg;base64,{encoded}"
    return _DATA_URI_CACHE[path]


def rect(x, y, w, h, fill=C["white"], stroke=C["ink"], sw=4, rx=24, dash=None, opacity=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"']
    if stroke:
        attrs += [f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if dash:
        attrs += [f'stroke-dasharray="{dash}"']
    if opacity is not None:
        attrs += [f'opacity="{opacity}"']
    return f"<rect {' '.join(attrs)}/>"


def text(x, y, value, size=38, weight="normal", anchor="middle", fill=C["ink"], family=None):
    family = family or "Comic Sans MS, DejaVu Sans, sans-serif"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f'{esc(value)}</text>'
    )


def lines(x, y, values, size=34, leading=1.22, weight="normal", anchor="middle", fill=C["ink"]):
    return "".join(
        text(x, y + i * size * leading, value, size, weight, anchor, fill)
        for i, value in enumerate(values)
    )


def image(parts, path: Path, x, y, w, h, ident, fit="slice", border=True, rx=22):
    parts.append(
        f'<clipPath id="clip_{ident}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>'
    )
    aspect = "xMidYMid slice" if fit == "slice" else "xMidYMid meet"
    parts.append(
        f'<image href="{data_uri(path)}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="{aspect}" clip-path="url(#clip_{ident})"/>'
    )
    if border:
        parts.append(rect(x, y, w, h, "none", C["ink"], 4, rx))


def arrow(x1, y1, x2, y2, color=C["blue"], sw=8, dashed=False):
    dash = ' stroke-dasharray="18 14"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="round" marker-end="url(#arrow)"{dash}/>'
    )


def pill(x, y, w, h, label, fill, size=29, stroke=C["ink"]):
    return rect(x, y, w, h, fill, stroke, 3, h / 2) + text(x + w / 2, y + h * 0.66, label, size, "bold")


def title(parts, name, subtitle):
    parts.append(text(70, 92, name, 58, "bold", "start"))
    parts.append(text(W - 70, 88, "TarGSNav", 32, "bold", "end", C["blue"]))
    parts.append(text(72, 142, subtitle, 26, "normal", "start", C["muted"]))


def begin():
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#245B8A"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]


def save(name, parts):
    parts.append("</svg>")
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(path)


def figure_mapping():
    p = begin()
    title(p, "Semantic-Guided Incremental 3DGS Optimization", "Target-centric replay improves instance reconstruction without increasing the optimization budget.")

    p.append(rect(55, 190, 980, 1390, C["cyan"], C["ink"], 4, 30, "16 12"))
    p.append(text(95, 252, "A  Training View Scheduler", 38, "bold", "start"))
    cards = [
        (ASSET["target_obs"], 105, 305, "Current RGB-D frame", "new observation"),
        (ASSET["recent"], 105, 690, "Random recent keyframe", "sliding-window regularization"),
        (ASSET["mask"], 105, 1075, "Target-history keyframe", "semantic replay + instance mask"),
    ]
    for idx, (path, x, y, head, sub) in enumerate(cards):
        p.append(rect(x, y, 880, 335, C["white"], C["ink"], 3, 22))
        image(p, path, x + 18, y + 18, 390, 299, f"map_card_{idx}", "slice", True, 16)
        p.append(text(x + 438, y + 115, head, 29, "bold", "start"))
        p.append(lines(x + 438, y + 169, [sub], 22, anchor="start", fill=C["muted"]))
        p.append(pill(x + 438, y + 226, 365, 56, ["1×", "random", "m× target"][idx], [C["yellow"], C["purple"], C["pink"]][idx], 24))

    p.append(arrow(1035, 875, 1140, 875))
    p.append(rect(1140, 190, 860, 1390, C["pink"], C["ink"], 4, 30, "16 12"))
    p.append(text(1180, 252, "B  Semantic-Weighted Loss", 38, "bold", "start"))
    image(p, ASSET["mask"], 1205, 320, 730, 650, "mapping_mask", "contain", True, 22)
    p.append(pill(1270, 1005, 600, 64, "target semantic mask  Mₜ", C["white"], 29, C["magenta"]))
    p.append(rect(1210, 1115, 720, 245, C["white"], C["ink"], 3, 24))
    p.append(text(1570, 1192, "L = Lrgb + λd Ldepth", 36, "bold", fill=C["blue"], family="DejaVu Sans, sans-serif"))
    p.append(text(1570, 1255, "+  λsem · Mₜ · Lrgb", 36, "bold", fill=C["magenta"], family="DejaVu Sans, sans-serif"))
    p.append(text(1570, 1320, "larger gradient inside target region", 23, fill=C["muted"]))
    p.append(lines(1570, 1432, ["same number of updates", "more target-region supervision"], 25, 1.35, "bold"))

    p.append(arrow(2000, 875, 2105, 875))
    p.append(rect(2105, 190, 1040, 1390, C["green"], C["ink"], 4, 30, "16 12"))
    p.append(text(2145, 252, "C  Online 3DGS Update", 38, "bold", "start"))
    image(p, ASSET["views"], 2165, 315, 920, 520, "mapping_views", "contain", True, 22)
    p.append(text(2625, 885, "Differentiable rendering", 31, "bold"))
    p.append(arrow(2625, 920, 2625, 1010))
    p.append(rect(2200, 1015, 850, 305, C["white"], C["ink"], 3, 24))
    for i, (cx, cy, rx, ry, col) in enumerate([
        (2320, 1120, 78, 34, "#89C9D0"), (2465, 1175, 92, 42, "#E9B569"),
        (2630, 1095, 66, 30, "#9ACD78"), (2775, 1185, 104, 44, "#C59BD9"),
        (2930, 1105, 74, 34, "#E78388"), (2555, 1250, 80, 33, "#75A7D8"),
    ]):
        p.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{col}" opacity="0.75" stroke="{C["ink"]}" stroke-width="3"/>')
    p.append(text(2625, 1385, "Target Gaussians: sharper · more complete", 29, "bold", fill=C["blue"]))
    p.append(text(2625, 1440, "Global map: regularized by recent-frame replay", 24, fill=C["muted"]))

    p.append(pill(705, 1635, 1790, 80, "past observations  →  target-aware optimization  →  reliable novel-view rendering", C["yellow"], 30, C["blue"]))
    save("Fig2_semantic_guided_mapping", p)


def figure_gaussmi():
    p = begin()
    title(p, "Semantic-Weighted Mutual Information Exploration", "A candidate viewpoint is selected by combining geometric frontiers with target-aware Gaussian information gain.")

    p.append(rect(55, 190, 1000, 1230, C["cyan"], C["ink"], 4, 30, "16 12"))
    p.append(text(95, 252, "A  Candidate Viewpoints", 38, "bold", "start"))
    image(p, ASSET["map_mid"], 115, 310, 880, 880, "mi_map", "contain", True, 22)
    candidates = [(315, 980, C["blue"]), (515, 780, C["orange"]), (750, 920, C["magenta"]), (860, 610, C["green2"])]
    for i, (cx, cy, col) in enumerate(candidates, 1):
        p.append(f'<circle cx="{cx}" cy="{cy}" r="30" fill="{col}" stroke="{C["ink"]}" stroke-width="5"/>')
        p.append(f'<path d="M {cx} {cy} L {cx+130} {cy-75} A 150 150 0 0 1 {cx+145} {cy+35} Z" fill="{col}" opacity="0.25" stroke="{col}" stroke-width="4"/>')
        p.append(text(cx, cy + 11, str(i), 24, "bold", fill=C["white"]))
    p.append(pill(185, 1245, 340, 62, "nearest frontier", C["white"], 25, C["blue"]))
    p.append(pill(550, 1245, 370, 62, "sampled poses  {vᵢ}", C["white"], 25, C["magenta"]))
    p.append(text(555, 1370, "position + heading + field of view", 24, fill=C["muted"]))

    p.append(arrow(1055, 800, 1140, 800))
    p.append(rect(1140, 190, 1010, 1230, C["purple"], C["ink"], 4, 30, "16 12"))
    p.append(text(1180, 252, "B  SWMI Evaluation", 38, "bold", "start"))
    image(p, ASSET["gaussian"], 1205, 310, 880, 620, "mi_render", "contain", True, 22)
    for cx, cy, rx, ry, col, op in [
        (1430, 560, 85, 42, "#F25C54", .52), (1585, 650, 125, 54, "#FFB347", .46),
        (1780, 535, 96, 45, "#F25C54", .56), (1900, 720, 120, 52, "#FFE06A", .42),
    ]:
        p.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{col}" opacity="{op}" stroke="{C["red"]}" stroke-width="4"/>')
    p.append(rect(1210, 980, 870, 250, C["white"], C["ink"], 3, 24))
    p.append(text(1645, 1060, "SWMI(v_i) = sum [ w_sem(g) * MI_g(v_i) ]", 30, "bold", fill=C["blue"], family="DejaVu Sans, sans-serif"))
    p.append(text(1645, 1123, "g in FOV(v_i)", 26, "bold", fill=C["muted"], family="DejaVu Sans, sans-serif"))
    p.append(lines(1645, 1175, ["uncertain Gaussian contribution", "× semantic importance"], 23, 1.25, "bold"))
    p.append(pill(1270, 1272, 330, 58, "low uncertainty", C["green"], 23))
    p.append(pill(1630, 1272, 390, 58, "high uncertainty", C["yellow"], 23, C["red"]))

    p.append(arrow(2150, 800, 2235, 800))
    p.append(rect(2235, 190, 910, 1230, C["green"], C["ink"], 4, 30, "16 12"))
    p.append(text(2275, 252, "C  Strategy Selection", 38, "bold", "start"))
    p.append(rect(2300, 340, 780, 225, C["white"], C["ink"], 3, 24))
    p.append(text(2690, 405, "Compare", 31, "bold"))
    p.append(pill(2360, 450, 310, 60, "frontier score", C["cyan"], 24))
    p.append(text(2700, 490, "vs.", 26, "bold", fill=C["muted"]))
    p.append(pill(2760, 450, 260, 60, "max SWMI", C["yellow"], 24, C["red"]))
    p.append(arrow(2690, 570, 2690, 660))
    p.append(rect(2320, 665, 740, 260, C["white"], C["magenta"], 4, 30))
    p.append(lines(2690, 740, ["if max SWMI > tau", "and frontier SWMI <= tau", "select information-gain pose v*"], 27, 1.40, "bold"))
    p.append(arrow(2690, 930, 2690, 1015))
    p.append(pill(2370, 1020, 640, 76, "FMM navigation to selected pose", C["purple"], 28, C["blue"]))
    p.append(arrow(2690, 1100, 2690, 1175))
    p.append(pill(2370, 1180, 640, 76, "new RGB-D → update 3DGS", C["cyan"], 28, C["blue"]))
    p.append(text(2690, 1340, "≈ 1 SWMI step : 4 frontier steps", 25, "bold", fill=C["muted"]))

    p.append(rect(300, 1490, 2600, 205, C["gray"], C["ink"], 3, 28, "14 10"))
    p.append(pill(365, 1555, 680, 74, "Large-scale frontier + SWMI", C["cyan"], 28))
    p.append(arrow(1060, 1592, 1290, 1592))
    p.append(pill(1300, 1555, 580, 74, "target instance confirmed", C["yellow"], 28))
    p.append(arrow(1895, 1592, 2125, 1592))
    p.append(pill(2135, 1555, 690, 74, "single-instance best-view orbit", C["green"], 28))
    save("Fig3_semantic_gaussmi_exploration", p)


def figure_localization():
    p = begin()
    title(p, "Coarse-to-Fine Localization with Closed-Loop Correction", "Multi-view rendering, feature matching, pose refinement and PnP feedback resolve free-view ambiguity.")

    p.append(rect(55, 190, 3090, 500, C["cyan"], C["ink"], 4, 30, "16 12"))
    p.append(text(95, 252, "A  Instance-Aware Coarse Localization", 38, "bold", "start"))
    image(p, ASSET["goal"], 105, 315, 390, 300, "loc_goal", "slice", True, 20)
    p.append(text(300, 655, "Image goal", 25, "bold"))
    p.append(arrow(510, 465, 620, 465))
    image(p, ASSET["views"], 625, 315, 900, 300, "loc_views", "contain", True, 20)
    p.append(text(1075, 655, "multi-scale orbit rendering", 25, "bold"))
    p.append(arrow(1540, 465, 1650, 465))
    image(p, ASSET["matches"], 1655, 315, 820, 300, "loc_match", "slice", True, 20)
    p.append(text(2065, 655, "semantic-weighted LightGlue", 25, "bold"))
    p.append(arrow(2490, 465, 2600, 465))
    p.append(rect(2605, 315, 470, 300, C["white"], C["magenta"], 4, 24))
    p.append(lines(2840, 405, ["best rendered pose", "score > tau_coarse", "→ coarse pose T0"], 28, 1.45, "bold"))

    panels = [
        (55, 760, 940, C["yellow"], "B  Sampling-based Optimization (SBO)", ASSET["sbo"], "large correction: height + pitch + local sampling"),
        (1130, 760, 940, C["purple"], "C  Gradient-based Optimization (GBO)", ASSET["gbo"], "small differentiable refinement: RGB / depth / opacity"),
        (2205, 760, 940, C["green"], "D  PnP Viewpoint Alignment (PVA)", ASSET["pva"], "feedback when matches are strong but target evidence is weak"),
    ]
    for idx, (x, y, w, fill, head, path, cap) in enumerate(panels):
        p.append(rect(x, y, w, 760, fill, C["ink"], 4, 30, "16 12"))
        p.append(text(x + 40, y + 62, head, 31, "bold", "start"))
        image(p, path, x + 45, y + 105, w - 90, 470, f"loc_panel_{idx}", "contain", True, 20)
        p.append(lines(x + w / 2, y + 635, cap.split(": "), 23, 1.30, "bold"))

    p.append(arrow(995, 1140, 1125, 1140))
    p.append(arrow(2070, 1140, 2200, 1140))
    p.append(rect(330, 1590, 2540, 125, C["gray"], C["ink"], 3, 28, "14 10"))
    p.append(pill(390, 1620, 500, 65, "semantic matches sufficient", C["green"], 25))
    p.append(arrow(900, 1652, 1130, 1652))
    p.append(pill(1140, 1620, 460, 65, "final target pose", C["yellow"], 25))
    p.append(text(1670, 1663, "otherwise", 24, "bold", fill=C["muted"]))
    p.append(arrow(1770, 1652, 1980, 1652, dashed=True))
    p.append(pill(1990, 1620, 810, 65, "PnP pose → move → rebuild → SBO/GBO", C["pink"], 25, C["magenta"]))
    save("Fig4_coarse_to_fine_localization", p)


def figure_second_round():
    p = begin()
    title(p, "Accuracy-Oriented Second-Round Exploration", "When global map growth stalls, exploration switches from unknown-space coverage to targeted instance completion.")

    p.append(rect(55, 190, 3090, 1320, C["gray"], C["ink"], 4, 30, "16 12"))
    stages = [
        (95, 290, 470, 285, C["cyan"], "Phase-I exploration", ["frontier + SWMI", "global coverage"]),
        (650, 290, 470, 285, C["yellow"], "Trigger", ["map growth < 1%", "target not localized"]),
        (1205, 290, 470, 285, C["purple"], "Instance mining", ["semantic connected", "components"]),
        (1760, 290, 470, 285, C["pink"], "Candidate ranking", ["evidence + FEM +", "support − path cost"]),
        (2315, 290, 470, 285, C["green"], "Local completion", ["target GaussMI", "best-view observation"]),
    ]
    for i, (x, y, w, h, fill, head, body) in enumerate(stages):
        p.append(rect(x, y, w, h, fill, C["ink"], 3, 25))
        p.append(text(x + w / 2, y + 80, head, 29, "bold"))
        p.append(lines(x + w / 2, y + 145, body, 24, 1.35, "bold", fill=C["muted"]))
        if i < len(stages) - 1:
            p.append(arrow(x + w + 5, y + h / 2, stages[i + 1][0] - 10, y + h / 2))
    p.append(arrow(2785, 432, 2960, 432))
    p.append(rect(2960, 290, 130, 285, C["yellow"], C["ink"], 3, 25))
    p.append(lines(3025, 382, ["re-", "localize"], 25, 1.35, "bold"))

    image(p, ASSET["map"], 120, 700, 1020, 670, "round_map", "contain", True, 24)
    p.append(text(630, 1420, "candidate instances on the semantic grid map", 27, "bold"))
    for i, (cx, cy, col) in enumerate([(420, 980, C["orange"]), (710, 1080, C["magenta"]), (910, 870, C["blue"])], 1):
        p.append(f'<circle cx="{cx}" cy="{cy}" r="42" fill="none" stroke="{col}" stroke-width="12"/>')
        p.append(text(cx, cy + 11, str(i), 26, "bold", fill=col))

    p.append(rect(1270, 700, 820, 670, C["white"], C["ink"], 3, 24))
    p.append(text(1680, 765, "Targeted candidate queue", 30, "bold"))
    rows = [
        ("#1", "match  ↑", "support  ↑", "path  short", C["green"]),
        ("#2", "match  mid", "support  ↑", "path  mid", C["yellow"]),
        ("#3", "visited", "failed before", "skip", C["pink"]),
    ]
    for i, (rank, a, b, c, fill) in enumerate(rows):
        y = 825 + i * 145
        p.append(rect(1325, y, 710, 110, fill, C["ink"], 2, 18))
        p.append(text(1370, y + 69, rank, 27, "bold", "start"))
        p.append(text(1485, y + 69, a, 23, "bold", "start"))
        p.append(text(1685, y + 69, b, 23, "bold", "start"))
        p.append(text(1900, y + 69, c, 23, "bold", "start"))
    p.append(pill(1360, 1260, 640, 66, "select first reachable unvisited instance", C["purple"], 24))

    image(p, ASSET["views"], 2220, 700, 800, 420, "round_views", "contain", True, 24)
    p.append(pill(2290, 1160, 660, 70, "max target-semantic SWMI view", C["green"], 25, C["blue"]))
    p.append(arrow(2620, 1235, 2620, 1300))
    p.append(pill(2300, 1305, 640, 70, "update 3DGS → relocalize", C["yellow"], 25, C["blue"]))

    p.append(arrow(850, 1485, 850, 1585, dashed=True))
    p.append(pill(425, 1600, 850, 72, "if explored area grows again → return to Phase I", C["cyan"], 25, C["blue"]))
    p.append(pill(1790, 1600, 980, 72, "candidate rejected → visit the next ranked instance", C["pink"], 25, C["magenta"]))
    save("Fig5_second_round_exploration", p)


def figure_qualitative():
    p = begin()
    title(p, "Qualitative Navigation Process — Episode 994", "A successful free-view chair search using online mapping, active exploration and coarse-to-fine localization.")
    p.append(rect(55, 190, 3090, 1440, C["gray"], C["ink"], 4, 30, "16 12"))
    cards = [
        ("1", "Image Goal", ASSET["goal"], "free-view target"),
        ("2", "Initial Observation", ASSET["early"], "target not visible"),
        ("3", "Exploration & Mapping", ASSET["map_full"], "online semantic map"),
        ("4", "Novel-View Rendering", ASSET["views"], "candidate target views"),
        ("5", "Feature Matching", ASSET["matches"], "semantic LightGlue"),
        ("6", "Pose Refinement", ASSET["gbo"], "SBO + GBO"),
        ("7", "Final Observation", ASSET["target_obs"], "navigation succeeds"),
    ]
    x0, y0, cw, ch, gap = 95, 300, 405, 1020, 30
    for i, (number, head, path, cap) in enumerate(cards):
        x = x0 + i * (cw + gap)
        fill = [C["cyan"], C["cyan"], C["green"], C["purple"], C["yellow"], C["purple"], C["green"]][i]
        p.append(rect(x, y0, cw, ch, fill, C["ink"], 3, 24))
        p.append(f'<circle cx="{x+45}" cy="{y0+48}" r="31" fill="{C["blue"]}"/>')
        p.append(text(x + 45, y0 + 59, number, 25, "bold", fill=C["white"]))
        p.append(text(x + cw / 2 + 22, y0 + 60, head, 25, "bold"))
        image(p, path, x + 20, y0 + 100, cw - 40, 760, f"qual_{i}", "slice" if i not in (2, 3, 5) else "contain", True, 18)
        p.append(text(x + cw / 2, y0 + 920, cap, 23, "bold"))
        if i < len(cards) - 1:
            p.append(arrow(x + cw + 3, y0 + 520, x + cw + gap - 3, y0 + 520, sw=6))
    p.append(pill(560, 1410, 2080, 86, "goal image  →  active online 3DGS  →  free-view match  →  final navigation", C["white"], 30, C["blue"]))
    p.append(text(1600, 1580, "Visualization source: showcase_episode_994_zt1RVoi7PcG", 24, fill=C["muted"]))
    save("Fig6_simulation_qualitative", p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in ASSET.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing visualization assets:\n" + "\n".join(missing))
    figure_mapping()
    figure_gaussmi()
    figure_localization()
    figure_second_round()
    figure_qualitative()


if __name__ == "__main__":
    main()
