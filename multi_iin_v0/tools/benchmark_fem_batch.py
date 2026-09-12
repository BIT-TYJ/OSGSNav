#!/usr/bin/env python3
"""Compare original and Multi-IIN batched FEM on saved real render images."""

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
    install_render_matching_acceleration,
)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
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
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--match-streams", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    paths = sorted(Path(args.render_dir).glob("render_*.png"))[: args.views]
    if not paths:
        raise RuntimeError("No render_*.png inputs found")
    rendered = [
        {"image": _read_rgb(path), "index": index, "pose": None}
        for index, path in enumerate(paths)
    ]
    fem = FeatureMatcher(
        target_image=_read_rgb(Path(args.target_image)),
        segmenter=None,
        save_visualization_results=False,
    )
    target_features, mode = fem._get_target_features("uniform")

    original_method = FeatureMatcher._batch_match_lightglue
    acceleration = install_render_matching_acceleration(
        argparse.Namespace(
            multi_render_matching_acceleration=1,
            multi_fem_superpoint_batch_size=args.batch_size,
            multi_fem_lightglue_cuda_streams=args.match_streams,
            multi_defer_render_cpu_transfer=0,
        )
    )
    original_call = lambda: original_method(
        fem, rendered, target_features=target_features, feature_mode=mode
    )
    accelerated_call = lambda: fem._batch_match_lightglue(
        rendered, target_features=target_features, feature_mode=mode
    )
    for _ in range(max(0, args.warmup)):
        _timed(original_call)
        _timed(accelerated_call)

    original_runs = []
    accelerated_runs = []
    original_times = []
    accelerated_times = []
    for _ in range(max(1, args.repeats)):
        result, elapsed = _timed(original_call)
        original_runs.append(result)
        original_times.append(elapsed)
        result, elapsed = _timed(accelerated_call)
        accelerated_runs.append(result)
        accelerated_times.append(elapsed)

    reference = original_runs[0]

    def signature(result):
        scores, counts = result
        best = int(np.argmax(scores)) if scores else -1
        return {
            "best_index": best,
            "success": bool(scores and scores[best] >= fem.match_threshold),
            "best_match_count": int(counts[best]) if best >= 0 else 0,
        }

    def max_diff(result):
        return float(
            np.max(np.abs(np.asarray(reference[0]) - np.asarray(result[0])))
        )

    ref_signature = signature(reference)
    baseline_diffs = [max_diff(result) for result in original_runs[1:]]
    accelerated_diffs = [max_diff(result) for result in accelerated_runs]
    baseline_signature_equal = [
        signature(result) == ref_signature for result in original_runs[1:]
    ]
    accelerated_signature_equal = [
        signature(result) == ref_signature for result in accelerated_runs
    ]
    original_s = statistics.median(original_times)
    accelerated_s = statistics.median(accelerated_times)
    print(
        {
            "views": len(rendered),
            "batch_size": args.batch_size,
            "match_streams": args.match_streams,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "original_median_s": original_s,
            "accelerated_median_s": accelerated_s,
            "median_speedup": original_s / accelerated_s,
            "original_times_s": original_times,
            "accelerated_times_s": accelerated_times,
            "reference_signature": ref_signature,
            "baseline_max_score_diff": max(baseline_diffs, default=0.0),
            "accelerated_max_score_diff": max(accelerated_diffs, default=0.0),
            "baseline_signatures_equal": all(baseline_signature_equal),
            "accelerated_signatures_equal": all(accelerated_signature_equal),
            "acceleration": acceleration.snapshot()["fem"],
        }
    )


if __name__ == "__main__":
    main()
