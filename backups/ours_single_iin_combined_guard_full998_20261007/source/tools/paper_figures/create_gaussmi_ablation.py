#!/usr/bin/env python3
"""Create a minimal same-pose GaussMI ablation figure for Episode 994."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "detectron2" / "docs" / "targsnav_figures"
ABLATION = OUT / "gaussmi_ablation"
W, H = 3200, 1800

FONT_BOLD = Path("/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS_Bold.ttf")
INK = (24, 34, 45)
WHITE = (255, 255, 255)

WITHOUT_GAUSSMI = ABLATION / "no_gaussmi" / "nvsm_four_views" / "view_270deg.png"
WITH_GAUSSMI = ABLATION / "with_gaussmi" / "nvsm_four_views" / "view_270deg.png"


def load_render(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        return source.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def save_svg_wrapper(image: Image.Image, path: Path) -> None:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f'  <image href="data:image/png;base64,{encoded}" width="{W}" height="{H}"/>\n'
        "</svg>\n",
        encoding="utf-8",
    )


def save_metadata(path: Path) -> None:
    without_meta = json.loads(
        (ABLATION / "no_gaussmi" / "render_metadata.json").read_text(encoding="utf-8")
    )
    with_meta = json.loads(
        (ABLATION / "with_gaussmi" / "render_metadata.json").read_text(encoding="utf-8")
    )
    without_view = next(
        item for item in without_meta["orbit"] if item["relative_angle_deg"] == 270
    )
    with_view = next(
        item for item in with_meta["orbit"] if item["relative_angle_deg"] == 270
    )
    metadata = {
        "figure": "Fig9_gaussmi_ablation",
        "episode": 994,
        "scene": "zt1RVoi7PcG",
        "target_category": "chair",
        "layout": "two unmodified final-global-PLY renders side by side",
        "without_gaussmi": {
            "label": "Frontier-only exploration (w/o GaussMI)",
            "phase1_gaussmi_enabled": False,
            "run_log": "/nas_ds925/IEVE/results/models/ablation100_no_gaussmi_seed20260815_s0of1_ep0994/train.log",
            "final_global_ply": without_meta["source_ply"],
            "render": str(WITHOUT_GAUSSMI),
        },
        "with_gaussmi": {
            "label": "Ours: GaussMI-guided exploration",
            "phase1_gaussmi_enabled": True,
            "secondary_exploration_enabled": False,
            "scope_note": "GaussMI-enabled run; not presented as a full-system result.",
            "run_log": "/nas_ds925/IEVE/results/models/ablation100_no_secondary_exploration_seed20260815_s6of7_ep0994/train.log",
            "final_global_ply": with_meta["source_ply"],
            "render": str(WITH_GAUSSMI),
        },
        "render_control": {
            "same_pose_verified": without_view["w2c"] == with_view["w2c"],
            "same_camera_position_verified": without_view["camera_position"]
            == with_view["camera_position"],
            "relative_angle_deg": 270,
            "world_angle_deg": without_view["world_angle_deg"],
            "camera_position": without_view["camera_position"],
            "w2c": without_view["w2c"],
            "resolution": without_meta["resolution"],
            "hfov_deg": 79.0,
        },
        "image_processing": "uniform resize, two labels and thin borders only; render pixels otherwise unchanged",
        "overlay": False,
        "target_crop": False,
        "quantitative_metrics": None,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype(str(FONT_BOLD), 48)

    image_size = (1490, 1490)
    left_xy = (70, 205)
    right_xy = (1640, 205)
    left = load_render(WITHOUT_GAUSSMI, image_size)
    right = load_render(WITH_GAUSSMI, image_size)
    canvas.paste(left, left_xy)
    canvas.paste(right, right_xy)
    draw.rectangle(
        (*left_xy, left_xy[0] + image_size[0], left_xy[1] + image_size[1]),
        outline=INK,
        width=5,
    )
    draw.rectangle(
        (*right_xy, right_xy[0] + image_size[0], right_xy[1] + image_size[1]),
        outline=INK,
        width=5,
    )
    draw.text(
        (left_xy[0] + image_size[0] // 2, 108),
        "(a) Frontier-only exploration (w/o GaussMI)",
        font=label_font,
        fill=INK,
        anchor="mm",
    )
    draw.text(
        (right_xy[0] + image_size[0] // 2, 108),
        "(b) Ours: GaussMI-guided exploration",
        font=label_font,
        fill=INK,
        anchor="mm",
    )

    stem = OUT / "Fig9_gaussmi_ablation"
    canvas.save(stem.with_suffix(".png"), format="PNG", optimize=True)
    canvas.save(stem.with_suffix(".pdf"), format="PDF", resolution=240.0)
    save_svg_wrapper(canvas, stem.with_suffix(".svg"))
    save_metadata(ABLATION / "comparison_metadata.json")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
