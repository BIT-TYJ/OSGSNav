#!/usr/bin/env python3
"""Score and visualize Goal-vs-PRE/REF/MR using the runtime FEM matcher."""

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np


IEVE_ROOT = Path("/code3/tyj/IEVE")
sys.path[:0] = [str(IEVE_ROOT), str(IEVE_ROOT / "LightGlue")]
from ieve_minav.localizer.feature_matcher import FeatureMatcher  # noqa: E402


def load(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = load(source / "01_目标图像.png")
    matcher = FeatureMatcher(target, match_threshold=40.0, results_dir=None,
                             segmenter=None, target_category=None,
                             save_visualization_results=False,
                             default_feature_mode="uniform")
    rows = []
    for stage, path in [
        ("PRE", source / "02_PRE后位姿渲染图.png"),
        ("REF", source / "03_REF后位姿渲染图.png"),
        ("MR", source / "04_MR后位姿渲染图.png"),
    ]:
        image = load(path)
        score = float(matcher._compute_match_score(image, feature_mode="uniform"))
        cache = copy.deepcopy(matcher._last_match_cache or {})
        count = int(cache.get("num_matches", 0))
        matcher._save_lightglue_match_visualization(output, stage.lower(), image, cache)
        rows.append({"stage": stage, "score_sum": score, "match_count": count})
    payload = {"target": str(source / "01_目标图像.png"), "threshold": 40.0,
               "feature_mode": "uniform", "stages": rows}
    (output / "match_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
