#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ieve_minav.ieve_segmenter import IEVEMinAVSegmenter


def build_args():
    parser = argparse.ArgumentParser(
        description="Run IEVE's Mask R-CNN semantic segmenter on one image."
    )
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument(
        "--categories",
        default="bed,chair",
        help="comma-separated categories to keep",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="output directory; default: <image_dir>/ieve_maskrcnn_vis",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "pretrained_models" / "model_final_f10217.pkl"),
        help="local Mask R-CNN weights; defaults to the cached detectron2 COCO model copied into pretrained_models",
    )
    parser.add_argument(
        "--sem-gpu-id",
        type=int,
        default=0,
        help="-2 for CPU, otherwise CUDA device id used by IEVE segmenter",
    )
    parser.add_argument(
        "--crop",
        default="",
        help="optional crop as x,y,w,h before segmentation",
    )
    parser.add_argument(
        "--crop-observation-panel",
        action="store_true",
        help="crop the middle observation panel from IEVE 2d_grid visualizations",
    )
    return parser.parse_args()


def load_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_crop(crop_text):
    if not crop_text:
        return None
    values = [int(float(item.strip())) for item in crop_text.split(",")]
    if len(values) != 4:
        raise ValueError("--crop must be x,y,w,h")
    return tuple(values)


def observation_panel_crop(image):
    height, width = image.shape[:2]
    if width < 900 or height < 400:
        return (0, 0, width, height)
    x0 = int(round(width * 0.3735))
    x1 = int(round(width * 0.6375))
    y0 = int(round(height * 0.077))
    y1 = int(round(height * 0.818))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    return (x0, y0, x1 - x0, y1 - y0)


def apply_crop(image, crop):
    if crop is None:
        return image, (0, 0, image.shape[1], image.shape[0])
    x, y, w, h = crop
    height, width = image.shape[:2]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return image[y : y + h, x : x + w].copy(), (x, y, w, h)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main():
    args = build_args()
    image_path = Path(args.image)
    out_dir = Path(args.output_dir) if args.output_dir else image_path.parent / "ieve_maskrcnn_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_full = load_image(image_path)
    crop = parse_crop(args.crop)
    if args.crop_observation_panel:
        crop = observation_panel_crop(image_full)
    image_rgb, crop_box = apply_crop(image_full, crop)

    segmenter_args = SimpleNamespace(
        sem_pred_prob_thr=float(args.threshold),
        sem_gpu_id=int(args.sem_gpu_id),
        visualize=0,
        quiet_log=1,
        maskrcnn_weights=str(args.weights),
    )
    segmenter = IEVEMinAVSegmenter(segmenter_args)
    categories = [item.strip().lower() for item in args.categories.split(",") if item.strip()]
    semantic_mask, labels, visualized = segmenter.segment_frame(
        image_rgb,
        largest_only=False,
        categories=categories,
    )

    detections = []
    for idx, label in enumerate(labels):
        mask = np.asarray(label.get("binary_mask"), dtype=np.uint8)
        mask_path = out_dir / "{}_mask_{:02d}.png".format(
            str(label.get("category", "unknown")).replace(" ", "_"),
            idx,
        )
        cv2.imwrite(str(mask_path), mask * 255)
        detections.append(
            {
                "category": label.get("category"),
                "confidence": float(label.get("confidence", 0.0) or 0.0),
                "bbox_xyxy": [int(v) for v in label.get("bbox", [])],
                "pixel_count": int(label.get("pixel_count", int(mask.sum())) or 0),
                "source": label.get("source", "ieve_maskrcnn"),
                "mask_path": str(mask_path),
            }
        )

    crop_path = out_dir / "{}_crop.png".format(image_path.stem)
    overlay_path = out_dir / "{}_ieve_maskrcnn_overlay.png".format(image_path.stem)
    mask_path = out_dir / "{}_semantic_mask.png".format(image_path.stem)
    json_path = out_dir / "{}_ieve_maskrcnn_results.json".format(image_path.stem)

    cv2.imwrite(str(crop_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), np.asarray(semantic_mask, dtype=np.uint8))
    with open(json_path, "w") as f:
        json.dump(
            json_safe(
                {
                    "image": str(image_path),
                    "crop_xywh": list(crop_box),
                    "categories": categories,
                    "threshold": float(args.threshold),
                    "detections": detections,
                    "overlay_path": str(overlay_path),
                    "semantic_mask_path": str(mask_path),
                    "crop_path": str(crop_path),
                }
            ),
            f,
            indent=2,
        )

    print("saved_overlay={}".format(overlay_path))
    print("saved_results={}".format(json_path))
    print("detections={}".format(len(detections)))
    for det in detections:
        print(
            "{category} score={confidence:.3f} pixels={pixel_count} bbox={bbox_xyxy}".format(
                **det
            )
        )


if __name__ == "__main__":
    main()
