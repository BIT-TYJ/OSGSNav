#!/usr/bin/env python3
"""Create semantic- and instance-colored PLY views of labelled Gaussians."""

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


CLASS_NAMES = ("chair", "sofa", "plant", "bed", "toilet", "tv_monitor")
SEMANTIC_COLORS = np.asarray(
    [
        (220, 20, 60), (255, 140, 0), (34, 139, 34),
        (65, 105, 225), (0, 206, 209), (186, 85, 211),
    ], dtype=np.uint8,
)
UNLABELLED_COLOR = np.asarray((128, 128, 128), dtype=np.uint8)
UNKNOWN_INSTANCE_COLOR = np.asarray((80, 80, 80), dtype=np.uint8)


def instance_color(instance_id: int) -> np.ndarray:
    hue = (int(instance_id) * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return np.asarray([round(channel * 255) for channel in rgb], dtype=np.uint8)


def write_colored(source, vertices, colors, path):
    output = vertices.copy()
    for channel, name in enumerate(("red", "green", "blue")):
        output[name] = colors[:, channel]
    PlyData(
        [PlyElement.describe(output, "vertex")], text=False,
        comments=list(source.comments), obj_info=list(source.obj_info),
    ).write(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    source = PlyData.read(str(args.input_ply))
    vertices = np.asarray(source["vertex"].data)
    names = set(vertices.dtype.names or ())
    required = {"red", "green", "blue", "semantic_class", "instance_id"}
    missing = sorted(required - names)
    if missing:
        raise ValueError("input PLY lacks properties: " + ", ".join(missing))

    semantic = np.asarray(vertices["semantic_class"], dtype=np.int32)
    instances = np.asarray(vertices["instance_id"], dtype=np.int64)
    labelled = (semantic >= 0) & (semantic < len(CLASS_NAMES))

    semantic_colors = np.broadcast_to(UNLABELLED_COLOR, (len(vertices), 3)).copy()
    semantic_colors[labelled] = SEMANTIC_COLORS[semantic[labelled]]

    instance_colors = np.broadcast_to(UNLABELLED_COLOR, (len(vertices), 3)).copy()
    known_instance = labelled & (instances >= 0)
    instance_colors[labelled & (instances < 0)] = UNKNOWN_INSTANCE_COLOR
    instance_ids = sorted(int(value) for value in np.unique(instances[known_instance]))
    instance_palette = {value: instance_color(value) for value in instance_ids}
    for value, color in instance_palette.items():
        instance_colors[known_instance & (instances == value)] = color

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_path = args.output_dir / "gaussians_semantic_colored.ply"
    instance_path = args.output_dir / "gaussians_instance_colored.ply"
    write_colored(source, vertices, semantic_colors, semantic_path)
    write_colored(source, vertices, instance_colors, instance_path)

    legend = {
        "source": str(args.input_ply.resolve()),
        "total_gaussians": int(len(vertices)),
        "semantic_labelled_gaussians": int(np.count_nonzero(labelled)),
        "unlabelled_gaussians": int(np.count_nonzero(~labelled)),
        "semantic_colors_rgb": {
            name: SEMANTIC_COLORS[index].tolist()
            for index, name in enumerate(CLASS_NAMES)
        },
        "unlabelled_color_rgb": UNLABELLED_COLOR.tolist(),
        "semantic_without_instance_color_rgb": UNKNOWN_INSTANCE_COLOR.tolist(),
        "semantic_class_counts": {
            name: int(np.count_nonzero(semantic == class_id))
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "instance_count": len(instance_ids),
        "instance_colors_rgb": {
            str(value): instance_palette[value].tolist() for value in instance_ids
        },
        "outputs": {"semantic": semantic_path.name, "instance": instance_path.name},
    }
    (args.output_dir / "color_legend.json").write_text(
        json.dumps(legend, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: legend[key] for key in (
        "total_gaussians", "semantic_labelled_gaussians",
        "unlabelled_gaussians", "instance_count", "semantic_class_counts"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
