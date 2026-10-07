#!/usr/bin/env python3
"""Create clear, semantically annotated Gaussian views for Episode 915."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/nas_ds925/IEVE/results/habitat_914_20260708_181234_yr17PDCnDDW/NVSM")
OUT = Path("/nas_ds925/tyj/IEVE/仿真实验关键节点_Gaussian原位渲染/02_二次探索成功_Episode915")


ITEMS = [
    {
        "src": "best_match_step_000445_score_0060.33_nvsm_s4_v31_1783506801.png",
        "dst": "02_候选实例1_目标语义区域_algo_step429_nvsm_step445.png",
        "kind": "ellipse",
        "region": ((238, 332), (20, 39), 0),
        "label_xy": (24, 42),
        "anchor": (238, 304),
    },
    {
        "src": "best_match_step_000538_score_0063.32_nvsm_s4_v31_1783507096.png",
        "dst": "03_候选实例2_目标语义区域_algo_step522_nvsm_step538.png",
        "kind": "ellipse",
        "region": ((239, 332), (21, 41), 0),
        "label_xy": (24, 42),
        "anchor": (239, 303),
    },
    {
        "src": "best_match_step_000581_score_0082.01_nvsm_s0_v29_1783507308.png",
        "dst": "04_正确实例_目标语义区域_algo_step565_nvsm_step581.png",
        "kind": "polygon",
        "region": np.asarray([(181, 207), (241, 208), (269, 278), (261, 405),
                              (238, 481), (197, 454), (177, 368), (166, 278)], np.int32),
        "label_xy": (24, 42),
        "anchor": (202, 246),
    },
]


def annotate(item: dict) -> None:
    image = cv2.imread(str(ROOT / item["src"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(ROOT / item["src"])

    overlay = image.copy()
    color = (82, 214, 72)  # BGR: semantic green
    if item["kind"] == "ellipse":
        center, axes, angle = item["region"]
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(image, center, axes, angle, 0, 360, color, 4, cv2.LINE_AA)
    else:
        polygon = item["region"]
        cv2.fillPoly(overlay, [polygon], color, cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, color, 4, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.24, image, 0.76, 0.0)

    label = "Target semantic region"
    x, y = item["label_xy"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.62, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(image, (x - 9, y - th - 10), (x + tw + 9, y + baseline + 8),
                  (24, 118, 39), -1, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.arrowedLine(image, (x + tw, y - th // 2), item["anchor"], color, 3,
                    cv2.LINE_AA, tipLength=0.08)

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / item["dst"]), image)
    print(OUT / item["dst"])


def main() -> None:
    for item in ITEMS:
        annotate(item)


if __name__ == "__main__":
    main()
