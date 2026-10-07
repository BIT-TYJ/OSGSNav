#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
GSA_ROOT = REPO_ROOT / "3rdparty" / "minav" / "third_party" / "Grounded-Segment-Anything"
GROUNDING_DINO_ROOT = GSA_ROOT / "GroundingDINO"
SAM_ROOT = GSA_ROOT / "segment_anything"
MODEL_ROOT = REPO_ROOT / "3rdparty" / "minav" / "models"

sys.path.insert(0, str(GSA_ROOT))
sys.path.insert(0, str(GROUNDING_DINO_ROOT))
sys.path.insert(0, str(SAM_ROOT))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_model_registry


COLORS = {
    "bed": (120, 90, 255),
    "chair": (255, 150, 40),
    "couch": (70, 180, 255),
    "sofa": (70, 180, 255),
    "potted plant": (70, 220, 120),
    "plant": (70, 220, 120),
    "toilet": (220, 120, 220),
    "tv": (80, 220, 220),
}


def build_args():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM on one image and save mask visualization."
    )
    parser.add_argument("--image", required=True, help="input RGB/BGR image path")
    parser.add_argument(
        "--categories",
        default="bed,chair",
        help="comma-separated categories; each category is prompted separately",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="output directory; default: <image_dir>/grounded_sam_vis",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda/cpu",
    )
    parser.add_argument(
        "--grounding-config",
        default=str(
            GROUNDING_DINO_ROOT
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        ),
    )
    parser.add_argument(
        "--grounding-ckpt",
        default=str(MODEL_ROOT / "groundingdino_swint_ogc.pth"),
    )
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument(
        "--sam-ckpt",
        default=str(MODEL_ROOT / "sam_vit_b_01ec64.pth"),
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
    parser.add_argument("--alpha", type=float, default=0.38)
    return parser.parse_args()


def load_grounding_model(config_path, ckpt_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


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
    # IEVE 2d_grid layout: left target, middle observation, right map.
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


def transform_image_for_dino(image_rgb):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def get_grounding_output(model, image_tensor, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    with torch.no_grad():
        outputs = model(image_tensor[None].to(device), captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    keep = logits.max(dim=1)[0] > float(box_threshold)
    logits = logits[keep]
    boxes = boxes[keep]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    scores = []
    for logit in logits:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        score = float(logit.max().item())
        phrases.append(phrase)
        scores.append(score)
    return boxes, phrases, scores


def xywh_center_to_xyxy(boxes, width, height):
    scaled = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    scaled[:, :2] -= scaled[:, 2:] / 2
    scaled[:, 2:] += scaled[:, :2]
    scaled[:, 0::2].clamp_(0, width)
    scaled[:, 1::2].clamp_(0, height)
    return scaled


def draw_label(image, text, x, y, color):
    x = int(max(0, min(image.shape[1] - 1, x)))
    y = int(max(16, min(image.shape[0] - 1, y)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - th - 6), (min(image.shape[1] - 1, x + tw + 6), y + 4), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    args = build_args()
    image_path = Path(args.image)
    out_dir = Path(args.output_dir) if args.output_dir else image_path.parent / "grounded_sam_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_full = load_image(image_path)
    crop = parse_crop(args.crop)
    if args.crop_observation_panel:
        crop = observation_panel_crop(image_full)
    image_rgb, crop_box = apply_crop(image_full, crop)
    height, width = image_rgb.shape[:2]

    grounding_model = load_grounding_model(args.grounding_config, args.grounding_ckpt, args.device)
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)

    image_tensor = transform_image_for_dino(image_rgb)
    sam_predictor.set_image(image_rgb)

    overlay = image_rgb.copy()
    combined_mask = np.zeros((height, width), dtype=np.int32)
    detections = []
    categories = [item.strip().lower() for item in args.categories.split(",") if item.strip()]

    for cat_idx, category in enumerate(categories, start=1):
        boxes, phrases, scores = get_grounding_output(
            grounding_model,
            image_tensor,
            category,
            args.box_threshold,
            args.text_threshold,
            args.device,
        )
        if boxes.numel() == 0:
            continue
        boxes_xyxy = xywh_center_to_xyxy(boxes, width, height)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_rgb.shape[:2]
        ).to(args.device)
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        color = COLORS.get(category, (60 + cat_idx * 53 % 180, 180, 90 + cat_idx * 37 % 160))
        color_arr = np.array(color, dtype=np.uint8)
        for det_idx, (mask, box, phrase, score) in enumerate(
            zip(masks, boxes_xyxy, phrases, scores)
        ):
            mask_np = mask[0].detach().cpu().numpy().astype(bool)
            if int(mask_np.sum()) == 0:
                continue
            combined_mask[mask_np] = cat_idx
            overlay[mask_np] = (
                overlay[mask_np].astype(np.float32) * (1.0 - args.alpha)
                + color_arr.astype(np.float32) * args.alpha
            ).astype(np.uint8)
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
            label = "{} {:.2f}".format(category, float(score))
            draw_label(overlay, label, x0, max(16, y0 - 4), color)
            mask_path = out_dir / "{}_mask_{:02d}.png".format(category.replace(" ", "_"), det_idx)
            cv2.imwrite(str(mask_path), (mask_np.astype(np.uint8) * 255))
            detections.append(
                {
                    "category": category,
                    "phrase": phrase,
                    "score": float(score),
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "pixel_count": int(mask_np.sum()),
                    "mask_path": str(mask_path),
                }
            )

    crop_path = out_dir / "{}_crop.png".format(image_path.stem)
    overlay_path = out_dir / "{}_grounded_sam_overlay.png".format(image_path.stem)
    mask_path = out_dir / "{}_semantic_mask.png".format(image_path.stem)
    json_path = out_dir / "{}_grounded_sam_results.json".format(image_path.stem)

    cv2.imwrite(str(crop_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), combined_mask.astype(np.uint8))
    with open(json_path, "w") as f:
        json.dump(
            {
                "image": str(image_path),
                "crop_xywh": list(crop_box),
                "categories": categories,
                "box_threshold": float(args.box_threshold),
                "text_threshold": float(args.text_threshold),
                "detections": detections,
                "overlay_path": str(overlay_path),
                "semantic_mask_path": str(mask_path),
                "crop_path": str(crop_path),
            },
            f,
            indent=2,
        )
    print("saved_overlay={}".format(overlay_path))
    print("saved_results={}".format(json_path))
    print("detections={}".format(len(detections)))
    for det in detections:
        print(
            "{category} score={score:.3f} pixels={pixel_count} bbox={bbox_xyxy}".format(
                **det
            )
        )


if __name__ == "__main__":
    main()
