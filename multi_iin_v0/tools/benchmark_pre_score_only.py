#!/usr/bin/env python3
"""Benchmark full-cache and score-only FEM on saved real render images."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ieve_minav.localizer.feature_matcher import FeatureMatcher
from multi_iin_v0.runtime.render_matching_acceleration import (
    _match_lightglue_score_only,
    _rgb_u8_to_cv_gray_tensor,
)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read {}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _timed(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--views", type=int, default=80)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    paths = sorted(Path(args.render_dir).glob("render_*.png"))[: args.views]
    if not paths:
        raise RuntimeError("No render_*.png inputs found")
    images = [_read_rgb(path) for path in paths]
    fem = FeatureMatcher(
        target_image=_read_rgb(Path(args.target_image)),
        segmenter=None,
        save_visualization_results=False,
    )

    def full():
        return [fem._match_with_lightglue(image) for image in images]

    def score_only():
        return [_match_lightglue_score_only(fem, image) for image in images]

    gpu_gray_images = [
        _rgb_u8_to_cv_gray_tensor(
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .to(device=fem.device)
        )
        for image in images
    ]

    def gpu_direct():
        return [_match_lightglue_score_only(fem, image) for image in gpu_gray_images]

    for _ in range(max(0, args.warmup)):
        _timed(full)
        _timed(score_only)
        _timed(gpu_direct)

    full_runs, score_runs = [], []
    gpu_runs = []
    full_times, score_times, gpu_times = [], [], []
    for _ in range(max(1, args.repeats)):
        values, elapsed = _timed(full)
        full_runs.append(values)
        full_times.append(elapsed)
        values, elapsed = _timed(score_only)
        score_runs.append(values)
        score_times.append(elapsed)
        values, elapsed = _timed(gpu_direct)
        gpu_runs.append(values)
        gpu_times.append(elapsed)

    reference = np.asarray(full_runs[0], dtype=np.float64)
    max_full_repeat_diff = max(
        (float(np.max(np.abs(reference - np.asarray(run)))) for run in full_runs[1:]),
        default=0.0,
    )
    max_score_only_diff = max(
        float(np.max(np.abs(reference - np.asarray(run)))) for run in score_runs
    )
    max_gpu_direct_diff = max(
        float(np.max(np.abs(reference - np.asarray(run)))) for run in gpu_runs
    )
    full_median = statistics.median(full_times)
    score_median = statistics.median(score_times)
    gpu_median = statistics.median(gpu_times)
    print({
        "views": len(images),
        "repeats": args.repeats,
        "full_cache_median_s": full_median,
        "score_only_median_s": score_median,
        "score_only_speedup": full_median / score_median,
        "gpu_direct_median_s": gpu_median,
        "gpu_direct_speedup": full_median / gpu_median,
        "full_cache_times_s": full_times,
        "score_only_times_s": score_times,
        "gpu_direct_times_s": gpu_times,
        "full_repeat_max_score_diff": max_full_repeat_diff,
        "score_only_max_score_diff": max_score_only_diff,
        "gpu_direct_max_score_diff": max_gpu_direct_diff,
        "best_index_equal": all(
            int(np.argmax(run)) == int(np.argmax(reference)) for run in score_runs
        ),
        "gpu_direct_best_index_equal": all(
            int(np.argmax(run)) == int(np.argmax(reference)) for run in gpu_runs
        ),
    })


if __name__ == "__main__":
    main()
