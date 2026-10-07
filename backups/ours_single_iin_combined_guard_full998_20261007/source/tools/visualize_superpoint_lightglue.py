#!/usr/bin/env python3
"""Match the two halves of one image with SuperPoint + LightGlue.

The output shows every sampled SuperPoint keypoint in gray. Matched points and
their connecting line share a Turbo heat-map color derived from the native
LightGlue confidence (blue=low, red=high).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTGLUE_ROOT = REPO_ROOT / "LightGlue"
if str(LIGHTGLUE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTGLUE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one image vertically, match its halves, and visualize matches."
    )
    parser.add_argument("input", type=Path, help="Input image containing left/right views")
    parser.add_argument("output", type=Path, help="Output match visualization (.png/.jpg)")
    parser.add_argument(
        "--split-x",
        type=int,
        default=None,
        help="Vertical split coordinate (default: image_width // 2)",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=1024,
        help="Maximum SuperPoint samples per half (default: 1024)",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=0.1,
        help="Minimum LightGlue match confidence (default: 0.1)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable match statistics",
    )
    return parser.parse_args()


def image_to_tensor(image_bgr: np.ndarray, device: "torch.device") -> "torch.Tensor":
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
    return tensor.float().div_(255.0).to(device)


def unbatch(value):
    """Remove LightGlue's leading single-image batch dimension recursively."""
    import torch

    if isinstance(value, dict):
        return {key: unbatch(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else value
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def confidence_color_bgr(score: float) -> tuple[int, int, int]:
    """Map an absolute confidence in [0, 1] to OpenCV's Turbo heat map."""
    level = np.array([[round(255 * float(np.clip(score, 0.0, 1.0)))]], dtype=np.uint8)
    color = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)[0, 0]
    return tuple(int(channel) for channel in color)


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> None:
    """Draw all SuperPoint samples unobtrusively underneath the matches."""
    for x, y in keypoints:
        point = (int(round(x)) + x_offset, int(round(y)) + y_offset)
        cv2.circle(image, point, 1, (220, 220, 220), 1, cv2.LINE_AA)


def render_visualization(
    left: np.ndarray,
    right: np.ndarray,
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    matched0: np.ndarray,
    matched1: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    header_h = 50
    # Reassemble the two halves at their original boundary without a divider.
    gap = 0
    height = max(left.shape[0], right.shape[0])
    left_w, right_w = left.shape[1], right.shape[1]
    canvas = np.full((header_h + height, left_w + gap + right_w, 3), 22, dtype=np.uint8)
    canvas[header_h : header_h + left.shape[0], :left_w] = left
    canvas[
        header_h : header_h + right.shape[0],
        left_w + gap : left_w + gap + right_w,
    ] = right

    draw_keypoints(canvas, keypoints0, 0, header_h)
    draw_keypoints(canvas, keypoints1, left_w + gap, header_h)

    # Low-confidence lines are drawn first so high-confidence evidence remains visible.
    for index in np.argsort(scores):
        score = float(scores[index])
        color = confidence_color_bgr(score)
        p0 = (
            int(round(matched0[index, 0])),
            int(round(matched0[index, 1])) + header_h,
        )
        p1 = (
            int(round(matched1[index, 0])) + left_w + gap,
            int(round(matched1[index, 1])) + header_h,
        )
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"SuperPoint samples: {len(keypoints0)} / {len(keypoints1)}   "
        f"LightGlue matches: {len(scores)}",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "confidence",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    bar_x0, bar_x1 = 92, min(322, canvas.shape[1] - 72)
    if bar_x1 > bar_x0:
        gradient = np.linspace(0, 255, bar_x1 - bar_x0, dtype=np.uint8)[None, :]
        gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        canvas[28:42, bar_x0:bar_x1] = gradient
        cv2.putText(canvas, "0", (bar_x0 - 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1", (bar_x1 + 4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and the bundled LightGlue package are required. "
            "Run this script in the project's IEVE environment."
        ) from exc

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read input image: {args.input}")
    split_x = image.shape[1] // 2 if args.split_x is None else args.split_x
    if not 0 < split_x < image.shape[1]:
        raise SystemExit(f"--split-x must be between 1 and {image.shape[1] - 1}")
    left = image[:, :split_x].copy()
    right = image[:, split_x:].copy()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        device = torch.device(args.device)

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(
        features="superpoint", filter_threshold=args.filter_threshold
    ).eval().to(device)

    with torch.inference_mode():
        features0 = extractor.extract(image_to_tensor(left, device), resize=None)
        features1 = extractor.extract(image_to_tensor(right, device), resize=None)
        matches01 = matcher({"image0": features0, "image1": features1})

    features0 = unbatch(features0)
    features1 = unbatch(features1)
    matches01 = unbatch(matches01)
    matches = matches01["matches"].detach().cpu().numpy().astype(np.int64)
    scores = matches01["scores"].detach().cpu().numpy().astype(np.float32)
    keypoints0 = features0["keypoints"].detach().cpu().numpy().astype(np.float32)
    keypoints1 = features1["keypoints"].detach().cpu().numpy().astype(np.float32)
    matched0 = keypoints0[matches[:, 0]]
    matched1 = keypoints1[matches[:, 1]]

    output = render_visualization(
        left, right, keypoints0, keypoints1, matched0, matched1, scores
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise SystemExit(f"Could not write output image: {args.output}")

    stats = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "device": str(device),
        "split_x": int(split_x),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "sample_count_left": int(len(keypoints0)),
        "sample_count_right": int(len(keypoints1)),
        "match_count": int(len(matches)),
        "confidence_min": float(scores.min()) if len(scores) else None,
        "confidence_mean": float(scores.mean()) if len(scores) else None,
        "confidence_max": float(scores.max()) if len(scores) else None,
    }
    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
