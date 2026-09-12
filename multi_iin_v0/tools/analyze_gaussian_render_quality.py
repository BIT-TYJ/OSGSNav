#!/usr/bin/env python3
"""Read-only diagnostics for online 3DGS debug renders.

The tool pairs ``gt_XXXXX.png`` and ``render_XXXXX.png`` files, reports image
quality by episode and trajectory phase, and highlights pixels that are dark in
the render despite being visibly illuminated in the observation.  It never
uses Habitat semantic observations, goal coordinates, or success measures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from skimage.metrics import structural_similarity


FRAME_RE = re.compile(r"^(?:gt|render)_(\d+)\.png$")
EPISODE_RE = re.compile(r"episode_(\d+)")
TARGET_RE = re.compile(r"target_\d+__(.+?)__(?:obj\d+__)?(?:success|failure)$")


def compute_frame_metrics(gt_bgr: np.ndarray, render_bgr: np.ndarray) -> Dict[str, float]:
    """Compute full-frame quality and conservative black-artifact indicators."""
    if gt_bgr is None or render_bgr is None:
        raise ValueError("GT and render images must both be readable")
    if render_bgr.shape[:2] != gt_bgr.shape[:2]:
        render_bgr = cv2.resize(
            render_bgr,
            (gt_bgr.shape[1], gt_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    gt = gt_bgr.astype(np.float32)
    rendered = render_bgr.astype(np.float32)
    mse = float(np.mean((gt - rendered) ** 2))
    psnr = float("inf") if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    # SSIM is diagnostic rather than a benchmark score.  Bounding its input
    # keeps a 1000-step episode analysis to seconds instead of minutes while
    # preserving large reconstruction artifacts.
    ssim_gt = gt_bgr
    ssim_render = render_bgr
    max_side = max(gt_bgr.shape[:2])
    if max_side > 256:
        scale = 256.0 / float(max_side)
        size = (
            max(8, int(round(gt_bgr.shape[1] * scale))),
            max(8, int(round(gt_bgr.shape[0] * scale))),
        )
        ssim_gt = cv2.resize(gt_bgr, size, interpolation=cv2.INTER_AREA)
        ssim_render = cv2.resize(render_bgr, size, interpolation=cv2.INTER_AREA)
    ssim = float(
        structural_similarity(
            ssim_gt,
            ssim_render,
            channel_axis=2,
            data_range=255,
        )
    )

    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    render_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    gt_visible = gt_gray >= 32
    render_black = render_gray <= 8
    excess_black = render_black & gt_visible
    strong_darkening = (gt_gray >= 48) & (
        render_gray.astype(np.float32) < 0.35 * gt_gray.astype(np.float32)
    )
    abs_error = np.abs(gt - rendered)

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mae": float(abs_error.mean()),
        "gt_black_fraction": float(np.mean(gt_gray <= 8)),
        "render_black_fraction": float(np.mean(render_black)),
        "excess_black_fraction": float(np.mean(excess_black)),
        "strong_darkening_fraction": float(np.mean(strong_darkening)),
    }


def phase_for_index(index: int, total: int) -> str:
    if total <= 1:
        return "late"
    third = min(2, (3 * index) // total)
    return ("early", "middle", "late")[third]


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    keys = (
        "psnr_db",
        "ssim",
        "mae",
        "gt_black_fraction",
        "render_black_fraction",
        "excess_black_fraction",
        "strong_darkening_fraction",
    )
    summary: Dict[str, float] = {"frames": int(len(rows))}
    for key in keys:
        summary[f"mean_{key}"] = _finite_mean(float(row[key]) for row in rows)
    return summary


def _episode_number(path: Path) -> int:
    match = EPISODE_RE.search(path.parent.name)
    return int(match.group(1)) if match else -1


def _target_labels(experiment_root: Path) -> Dict[int, List[str]]:
    labels: Dict[int, List[str]] = {}
    compact = experiment_root / "compact"
    if not compact.exists():
        return labels
    for episode_dir in compact.rglob("episode_*"):
        if not episode_dir.is_dir():
            continue
        match = EPISODE_RE.match(episode_dir.name)
        if not match:
            continue
        found = []
        for target_dir in sorted(episode_dir.glob("target_*")):
            target_match = TARGET_RE.match(target_dir.name)
            if target_match:
                found.append(target_match.group(1))
        if found:
            labels[int(match.group(1))] = found
    return labels


def discover_debug_dirs(root: Path) -> List[Path]:
    if root.name == "debug_renders":
        return [root]
    return sorted(path for path in root.rglob("debug_renders") if path.is_dir())


def analyze_debug_dir(debug_dir: Path, labels: Sequence[str]) -> List[Dict[str, object]]:
    frame_ids = sorted(
        int(match.group(1))
        for path in debug_dir.glob("gt_*.png")
        if (match := FRAME_RE.match(path.name))
        and (debug_dir / f"render_{int(match.group(1)):05d}.png").exists()
    )
    rows: List[Dict[str, object]] = []
    episode = _episode_number(debug_dir)
    for index, frame_id in enumerate(frame_ids):
        gt_path = debug_dir / f"gt_{frame_id:05d}.png"
        render_path = debug_dir / f"render_{frame_id:05d}.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        rendered = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        metrics = compute_frame_metrics(gt, rendered)
        rows.append(
            {
                "episode": episode,
                "targets": "+".join(labels),
                "phase": phase_for_index(index, len(frame_ids)),
                "frame": frame_id,
                "debug_dir": str(debug_dir),
                **metrics,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_worst_contact_sheet(path: Path, rows: Sequence[Dict[str, object]], limit: int) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["excess_black_fraction"]),
            -float(row["psnr_db"]),
        ),
        reverse=True,
    )[: max(1, int(limit))]
    tiles = []
    for row in ranked:
        debug_dir = Path(str(row["debug_dir"]))
        frame = int(row["frame"])
        gt = cv2.imread(str(debug_dir / f"gt_{frame:05d}.png"), cv2.IMREAD_COLOR)
        rendered = cv2.imread(str(debug_dir / f"render_{frame:05d}.png"), cv2.IMREAD_COLOR)
        if gt is None or rendered is None:
            continue
        gt = cv2.resize(gt, (256, 256))
        rendered = cv2.resize(rendered, (256, 256))
        tile = np.hstack((gt, rendered))
        cv2.putText(
            tile,
            f"ep{int(row['episode']):02d} f{frame} PSNR {float(row['psnr_db']):.1f} black+ {100.0 * float(row['excess_black_fraction']):.1f}%",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(path), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Experiment root or debug_renders directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-frames", type=int, default=12)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _target_labels(input_root)
    rows: List[Dict[str, object]] = []
    for debug_dir in discover_debug_dirs(input_root):
        episode = _episode_number(debug_dir)
        rows.extend(analyze_debug_dir(debug_dir, labels.get(episode, [])))
    if not rows:
        raise SystemExit(f"No paired gt/render debug frames found under {input_root}")

    episodes = {}
    for episode in sorted({int(row["episode"]) for row in rows}):
        episode_rows = [row for row in rows if int(row["episode"]) == episode]
        episodes[str(episode)] = {
            "targets": episode_rows[0]["targets"],
            "overall": summarize_rows(episode_rows),
            "phases": {
                phase: summarize_rows([row for row in episode_rows if row["phase"] == phase])
                for phase in ("early", "middle", "late")
            },
        }
    report = {
        "input_root": str(input_root),
        "metric_note": (
            "excess_black_fraction counts render luminance <=8 where GT luminance >=32; "
            "metrics are observation/render comparisons and use no task ground truth"
        ),
        "overall": summarize_rows(rows),
        "episodes": episodes,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "frames.csv", rows)
    _make_worst_contact_sheet(output_dir / "worst_black_frames.png", rows, args.worst_frames)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
