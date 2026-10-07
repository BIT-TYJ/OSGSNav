from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np


class IEVEMinAVSegmenter:
    """MinAV segmenter API backed by IEVE's Mask R-CNN predictor."""

    indoor_categories = {
        "chair": ["chair"],
        "plant": ["plant", "potted plant", "houseplant"],
        "sofa": ["sofa", "couch"],
        "toilet": ["toilet"],
        "bed": ["bed"],
        "tv_monitor": ["tv", "monitor", "television", "screen", "tv_monitor"],
    }

    _ieve_to_minav_category = {
        "chair": "chair",
        "couch": "sofa",
        "sofa": "sofa",
        "potted plant": "plant",
        "plant": "plant",
        "bed": "bed",
        "toilet": "toilet",
        "tv": "tv_monitor",
        "tv_monitor": "tv_monitor",
        "tv monitor": "tv_monitor",
        "monitor": "tv_monitor",
        "television": "tv_monitor",
        "screen": "tv_monitor",
    }

    category_colors = {
        "chair": (255, 80, 80),
        "plant": (80, 200, 120),
        "sofa": (80, 140, 255),
        "toilet": (220, 220, 80),
        "bed": (220, 120, 220),
        "tv_monitor": (80, 220, 220),
    }

    def __init__(self, args: Any):
        from agents.utils.semantic_prediction import SemanticPredMaskRCNN

        self.args = self._build_args(args)
        self.predictor = SemanticPredMaskRCNN(self.args)

    @staticmethod
    def _build_args(args: Any) -> Any:
        sem_gpu_id = int(getattr(args, "sem_gpu_id", -2))
        if sem_gpu_id == -1:
            sem_gpu_id = int(getattr(args, "gpu_id", 0))
        try:
            import torch

            if not torch.cuda.is_available():
                sem_gpu_id = -2
        except Exception:
            sem_gpu_id = -2
        return SimpleNamespace(
            sem_pred_prob_thr=float(getattr(args, "sem_pred_prob_thr", 0.5)),
            sem_gpu_id=sem_gpu_id,
            visualize=int(getattr(args, "visualize", 0)),
            quiet_log=int(getattr(args, "quiet_log", 1)),
            maskrcnn_weights=getattr(args, "maskrcnn_weights", None),
        )

    @classmethod
    def normalize_category(cls, category_name: Any) -> str:
        key = str(category_name or "").strip().lower().replace("-", " ")
        return cls._ieve_to_minav_category.get(key, key)

    @classmethod
    def category_matches(cls, detected: Any, target: Any) -> bool:
        detected_key = cls.normalize_category(detected)
        target_key = cls.normalize_category(target)
        if not detected_key or not target_key:
            return False
        if detected_key == target_key:
            return True
        aliases = cls.indoor_categories.get(target_key, [])
        return detected_key in {cls.normalize_category(item) for item in aliases}

    def get_category_id(self, category_name: Any) -> int:
        return self.category_id_from_name(category_name)

    @classmethod
    def category_id_from_name(cls, category_name: Any) -> int:
        key = cls.normalize_category(category_name)
        keys = list(cls.indoor_categories.keys())
        if key in keys:
            return keys.index(key) + 1
        return 0

    def segment_frame(
        self,
        rgb_image: np.ndarray,
        largest_only: bool = False,
        categories: Optional[List[str]] = None,
    ):
        if rgb_image is None:
            return None, [], rgb_image

        image = self._as_uint8_rgb(rgb_image)
        height, width = image.shape[:2]
        _, _, bbox = self.predictor.get_prediction(image)
        instances = list(getattr(self.predictor, "last_instances", []) or [])
        labels = self._instances_to_labels(instances, height, width, categories)

        if not labels and bbox:
            labels = self._boxes_to_labels(bbox, height, width, categories)

        if largest_only and labels:
            labels = [max(labels, key=lambda item: int(item.get("pixel_count", 0)))]

        semantic_mask = np.zeros((height, width), dtype=np.int32)
        for label in labels:
            cat_id = self.get_category_id(label.get("category", ""))
            if cat_id <= 0:
                continue
            mask = np.asarray(label.get("binary_mask"), dtype=bool)
            if mask.shape == (height, width):
                semantic_mask[mask] = cat_id

        visualized = self.visualize_semantic_segmentation(image, semantic_mask, labels)
        return semantic_mask, labels, visualized

    def _instances_to_labels(
        self,
        instances: List[Dict[str, Any]],
        height: int,
        width: int,
        categories: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for instance in instances:
            category = self.normalize_category(instance.get("category_name"))
            if not self._category_allowed(category, categories):
                continue
            mask = np.asarray(instance.get("binary_mask"), dtype=bool)
            if mask.shape != (height, width):
                continue
            pixel_count = int(mask.sum())
            if pixel_count <= 0:
                continue
            bbox = self._clean_bbox(instance.get("bbox"), width, height)
            if bbox is None:
                bbox = self._bbox_from_mask(mask)
            labels.append(
                {
                    "category": category,
                    "bbox": bbox,
                    "confidence": float(instance.get("confidence", 0.0) or 0.0),
                    "pixel_count": pixel_count,
                    "binary_mask": mask,
                    "source": "ieve_maskrcnn",
                }
            )
        return labels

    def _boxes_to_labels(
        self,
        boxes: List[Any],
        height: int,
        width: int,
        categories: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        names = [
            "chair",
            "couch",
            "potted plant",
            "bed",
            "toilet",
            "tv",
            "dining-table",
            "oven",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "cup",
            "bottle",
        ]
        for box_item in boxes:
            if not isinstance(box_item, (list, tuple)) or len(box_item) < 3:
                continue
            try:
                category = self.normalize_category(names[int(box_item[0])])
            except Exception:
                continue
            if not self._category_allowed(category, categories):
                continue
            bbox = self._clean_bbox(box_item[2], width, height)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            pixel_count = int(mask.sum())
            if pixel_count <= 0:
                continue
            labels.append(
                {
                    "category": category,
                    "bbox": bbox,
                    "confidence": float(box_item[1] or 0.0),
                    "pixel_count": pixel_count,
                    "binary_mask": mask,
                    "source": "ieve_maskrcnn_bbox",
                }
            )
        return labels

    def _category_allowed(
        self, category: str, categories: Optional[List[str]]
    ) -> bool:
        if self.get_category_id(category) <= 0:
            return False
        if not categories:
            return True
        return any(self.category_matches(category, item) for item in categories)

    @staticmethod
    def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("rgb_image must be HxWx3")
        arr = arr[:, :, :3]
        if arr.dtype == np.uint8:
            return arr.copy()
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _clean_bbox(bbox: Any, width: int, height: int) -> Optional[List[int]]:
        if bbox is None:
            return None
        arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            return None
        x0 = int(np.clip(np.floor(arr[0]), 0, width - 1))
        y0 = int(np.clip(np.floor(arr[1]), 0, height - 1))
        x1 = int(np.clip(np.ceil(arr[2]), x0 + 1, width))
        y1 = int(np.clip(np.ceil(arr[3]), y0 + 1, height))
        return [x0, y0, x1, y1]

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> List[int]:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 1, 1]
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def visualize_semantic_segmentation(
        self,
        rgb_image: np.ndarray,
        semantic_mask: np.ndarray,
        semantic_labels: List[Dict[str, Any]],
    ) -> np.ndarray:
        visualized = rgb_image.copy()
        try:
            import cv2
        except Exception:
            return visualized

        for cat_id in np.unique(semantic_mask):
            if cat_id == 0:
                continue
            keys = list(self.indoor_categories.keys())
            if 1 <= int(cat_id) <= len(keys):
                name = keys[int(cat_id) - 1]
                color = self.category_colors.get(name, (255, 255, 255))
                mask = semantic_mask == int(cat_id)
                overlay = np.zeros_like(visualized)
                overlay[mask] = color
                visualized = cv2.addWeighted(visualized, 0.7, overlay, 0.3, 0)

        for label in semantic_labels:
            bbox = label.get("bbox")
            if not bbox:
                continue
            color = self.category_colors.get(label.get("category"), (255, 255, 255))
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(visualized, (x0, y0), (x1, y1), color, 2)
            text = "{} {:.2f}".format(
                label.get("category", "?"),
                float(label.get("confidence", 0.0) or 0.0),
            )
            cv2.putText(
                visualized,
                text,
                (x0, max(12, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return visualized


def minav_semantic_class_from_category(category_name: Any) -> int:
    return IEVEMinAVSegmenter.category_id_from_name(category_name)
