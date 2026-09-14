#!/usr/bin/env python3
"""Finalize a secondary-exploration on/off paired figure."""

import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_run(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"Final InstanceImageNav succ/spl/dtg:\s*([0-9.]+)/([0-9.]+)/([0-9.]+)", text
    )
    if not matches:
        raise RuntimeError(f"run has not finished: {path}")
    success, spl, dtg = map(float, matches[-1])
    steps = [int(value) for value in re.findall(r'"step":\s*(\d+)', text)]
    return {"success": success, "spl": spl, "dtg_m": dtg, "last_algorithm_step": max(steps)}


def final_grid(run_root):
    habitat_dirs = sorted((run_root / "results").glob("habitat_*"))
    if not habitat_dirs:
        raise FileNotFoundError(f"no habitat result under {run_root}")
    root = habitat_dirs[-1]
    grids = sorted((root / "2d_grid").glob("*.png"))
    if not grids:
        raise FileNotFoundError(f"no 2d_grid images under {root}")
    return root, grids[-1]


def fit(path, size):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main():
    output = Path(sys.argv[1]).resolve()
    run_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else output
    variants = [("Enabled", run_root / "enabled"),
                ("Disabled", run_root / "disabled")]
    if not variants[0][1].exists():
        variants = [("Enabled", run_root / "paired_current_enabled"),
                    ("Disabled", run_root / "paired_current_disabled")]
    records = []
    for label, root in variants:
        metrics = parse_run(root / "run.log")
        result_root, grid = final_grid(root)
        metrics.update({"variant": label.lower(), "result_root": str(result_root),
                        "final_grid": str(grid)})
        records.append(metrics)

    canvas = Image.new("RGB", (1880, 850), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        x = 20 + index * 930
        title = (f"Secondary exploration {record['variant']} | "
                 f"SR {record['success']:.0f} | SPL {record['spl']:.3f} | "
                 f"DTG {record['dtg_m']:.2f} m")
        draw.text((x + 450, 38), title, fill=(20, 20, 20), font=font(27, True), anchor="mm")
        canvas.paste(fit(Path(record["final_grid"]), (900, 700)), (x, 76))
    draw.rounded_rectangle((20, 784, 1860, 838), radius=12, fill=(239, 246, 255),
                           outline=(53, 112, 190), width=2)
    draw.text((940, 811), "Same episode, seed, model and configuration; only the secondary-exploration switch differs",
              fill=(22, 66, 120), font=font(24, True), anchor="mm")
    figure = output / "Figure_secondary_exploration_strict_pair.png"
    canvas.save(figure, quality=95)

    for record in records:
        name = record["variant"]
        shutil.copy2(record["final_grid"], output / f"final_grid_{name}.png")
    (output / "paired_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
