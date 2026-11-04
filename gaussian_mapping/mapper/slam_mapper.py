import logging
import traceback
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from gaussian_mapping.utils.metrics import compute_psnr, compute_ssim


logger = logging.getLogger(__name__)


class SLAMMapper:
    def __init__(self, config, output_dir, camera_params, enable_uaemi: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.camera_params = camera_params
        self.enable_uaemi = enable_uaemi
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slam = None
        self.is_running = False
        self.frame_count = 0

    def start_slam(self):
        try:
            from gaussian_mapping.mapper.realtime_3dgs import RealtimeSLAM

            self.slam = RealtimeSLAM(
                config=self.config,
                camera_params=self.camera_params,
                output_dir=self.output_dir,
                use_gui=False,
                enable_uaemi=self.enable_uaemi,
            )
            self.is_running = True
            logger.info("3DGS SLAM initialized: %s", self.output_dir)
        except Exception as exc:
            logger.error("3DGS SLAM init failed: %s", exc)
            traceback.print_exc()
            self.slam = None
            self.is_running = False

    def incremental_train_3dgs(self, obs):
        if self.slam is None:
            return False
        self.slam.add_frame(
            obs["rgb"],
            obs["depth"],
            obs["position"],
            obs["rotation"],
            target_mask=obs.get("target_mask"),
            training_context=obs.get("training_context"),
        )
        self.frame_count += 1
        return True

    def refresh_training_focus(
        self,
        object_id: Optional[int] = None,
        reason: str = "",
        extra_iters: Optional[int] = None,
        preferred_kf_ids: Optional[list] = None,
        stage: Optional[str] = None,
    ):
        if self.slam is None or not hasattr(self.slam, "refresh_training_focus"):
            return None
        return self.slam.refresh_training_focus(
            object_id=object_id,
            reason=reason,
            extra_iters=extra_iters,
            preferred_kf_ids=preferred_kf_ids,
            stage=stage,
        )

    def get_reconstruction_quality(self) -> Dict[str, float]:
        if self.slam is None:
            return {"num_gaussians": 0, "avg_reliability": 0.0, "reliable_percent": 0.0}
        return self.slam.get_reconstruction_quality()

    def _gaussians(self):
        slam = getattr(self, "slam", None)
        if slam is None:
            return None
        return getattr(slam, "gaussians", None)

    def set_confirmed_cluster_mask(self, indices) -> int:
        gaussians = self._gaussians()
        if gaussians is None or not hasattr(gaussians, "set_confirmed_target_indices"):
            return 0
        return int(gaussians.set_confirmed_target_indices(indices))

    def set_target_semantic_growth_policy(self, target_category: Optional[str] = None, **kwargs) -> bool:
        gaussians = self._gaussians()
        if gaussians is None or not hasattr(gaussians, "set_target_semantic_growth_policy"):
            return False
        gaussians.set_target_semantic_growth_policy(
            target_category=target_category,
            **kwargs,
        )
        return True

    def get_target_semantic_gaussians(self, threshold: float = 0.4):
        gaussians = self._gaussians()
        if gaussians is None:
            return np.array([], dtype=np.int64)
        confirmed_fallback = None
        if getattr(gaussians, "_confirmed_cluster_locked", False) and hasattr(
            gaussians, "get_confirmed_target_indices"
        ):
            confirmed = gaussians.get_confirmed_target_indices()
            if confirmed is not None:
                if torch.is_tensor(confirmed):
                    confirmed = confirmed.detach().cpu().numpy()
                confirmed = np.asarray(confirmed, dtype=np.int64).reshape(-1)
                if confirmed.size > 0:
                    confirmed_fallback = confirmed
        if not hasattr(gaussians, "get_semantic_target"):
            if confirmed_fallback is not None:
                return confirmed_fallback
            return np.array([], dtype=np.int64)
        semantic = gaussians.get_semantic_target
        if callable(semantic):
            semantic = semantic()
        if semantic is None:
            if confirmed_fallback is not None:
                return confirmed_fallback
            return np.array([], dtype=np.int64)
        if torch.is_tensor(semantic):
            if semantic.numel() == 0:
                if confirmed_fallback is not None:
                    return confirmed_fallback
                return np.array([], dtype=np.int64)
            indices = (
                torch.where(semantic > float(threshold))[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            if indices.size > 0:
                return indices
            if confirmed_fallback is not None:
                return confirmed_fallback
            return np.array([], dtype=np.int64)
        semantic = np.asarray(semantic)
        if semantic.size == 0:
            if confirmed_fallback is not None:
                return confirmed_fallback
            return np.array([], dtype=np.int64)
        indices = np.where(semantic > float(threshold))[0].astype(np.int64)
        if indices.size > 0:
            return indices
        if confirmed_fallback is not None:
            return confirmed_fallback
        return np.array([], dtype=np.int64)

    def get_target_semantic_stats(self, threshold: float = 0.4) -> Dict[str, object]:
        gaussians = self._gaussians()
        if gaussians is None:
            return {"num_target": 0, "centroid": None}
        indices = self.get_target_semantic_gaussians(threshold)
        centroid = None
        if indices.size > 0 and hasattr(gaussians, "get_xyz"):
            try:
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                valid = indices[(indices >= 0) & (indices < xyz.shape[0])]
                if valid.size > 0:
                    centroid = xyz[valid].mean(axis=0).astype(np.float32)
                    indices = valid
            except Exception:
                centroid = None
        bbox_min = None
        bbox_max = None
        if indices.size > 0 and hasattr(gaussians, "get_xyz"):
            try:
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                valid = indices[(indices >= 0) & (indices < xyz.shape[0])]
                if valid.size > 0:
                    target_xyz = xyz[valid]
                    bbox_min = target_xyz.min(axis=0).astype(np.float32)
                    bbox_max = target_xyz.max(axis=0).astype(np.float32)
                    indices = valid
            except Exception:
                bbox_min = None
                bbox_max = None
        return {
            "count": int(indices.size),
            "num_target": int(indices.size),
            "center": centroid,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": None if bbox_min is None or bbox_max is None else bbox_max - bbox_min,
            "indices": indices,
        }

    def get_target_gaussian_centroid(self, threshold: float = 0.4):
        gaussians = self._gaussians()
        if gaussians is None:
            return None
        if hasattr(gaussians, "get_target_gaussian_centroid"):
            return gaussians.get_target_gaussian_centroid(threshold)
        return self.get_target_semantic_stats(threshold).get("centroid")

    def set_target_semantic_region(
        self,
        region_2d: dict,
        map_resolution: float = 0.05,
        dilate_pixels: int = 3,
        max_height: float = None,
    ) -> int:
        del map_resolution, dilate_pixels, max_height
        gaussians = self._gaussians()
        if gaussians is None or not hasattr(gaussians, "set_semantic_target"):
            return 0
        indices = None
        if isinstance(region_2d, dict):
            for key in ("indices", "gaussian_indices", "target_indices"):
                if region_2d.get(key) is not None:
                    indices = region_2d.get(key)
                    break
        if indices is None:
            indices = self.get_target_semantic_gaussians()
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if getattr(gaussians, "_confirmed_cluster_locked", False) and hasattr(
            gaussians, "get_confirmed_target_indices"
        ):
            confirmed = gaussians.get_confirmed_target_indices()
            if confirmed is not None:
                if torch.is_tensor(confirmed):
                    confirmed = confirmed.detach().cpu().numpy()
                confirmed = np.asarray(confirmed, dtype=np.int64).reshape(-1)
                if confirmed.size > 0:
                    indices = np.unique(np.concatenate([indices, confirmed]))
        if indices.size == 0:
            return 0
        try:
            if hasattr(gaussians, "reset_semantic_target_values"):
                gaussians.reset_semantic_target_values()
            else:
                gaussians.reset_semantic_target()
            slam = getattr(self, "slam", None)
            if slam is not None:
                mask = np.zeros((gaussians.get_xyz.shape[0],), dtype=bool)
                valid = indices[(indices >= 0) & (indices < mask.shape[0])]
                if valid.size == 0:
                    return 0
                gaussians.set_semantic_target(valid, value=1.0)
                mask[valid] = True
                try:
                    slam._target_semantic_mask = torch.from_numpy(mask).to(
                        gaussians.get_xyz.device
                    )
                except Exception:
                    slam._target_semantic_mask = mask
            else:
                valid = indices[
                    (indices >= 0) & (indices < gaussians.get_xyz.shape[0])
                ]
                if valid.size == 0:
                    return 0
                gaussians.set_semantic_target(valid, value=1.0)
            return int(valid.size)
        except Exception as exc:
            logger.debug("set_target_semantic_region failed: %s", exc)
            return 0

    def visualize_target_semantic_gaussians(self, output_path: str):
        empty = {"count": 0, "center": None, "bbox_min": None, "bbox_max": None}
        gaussians = self._gaussians()
        if gaussians is None or not hasattr(gaussians, "get_xyz"):
            return empty
        indices = self.get_target_semantic_gaussians()
        if indices.size == 0:
            return empty
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            valid = indices[(indices >= 0) & (indices < xyz.shape[0])]
            if valid.size == 0:
                return empty
            target_xyz = xyz[valid]
            center = target_xyz.mean(axis=0).astype(np.float32)
            bbox_min = target_xyz.min(axis=0).astype(np.float32)
            bbox_max = target_xyz.max(axis=0).astype(np.float32)
            with open(output_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {target_xyz.shape[0]}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for p in target_xyz:
                    f.write(f"{p[0]} {p[1]} {p[2]} 255 0 0\n")
            return {
                "count": int(target_xyz.shape[0]),
                "center": center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        except Exception as exc:
            logger.debug("visualize_target_semantic_gaussians failed: %s", exc)
            return empty

    def get_instance_cluster_centers(
        self,
        threshold=0.4,
        eps=0.15,
        min_samples=10,
        min_cluster_size=50,
    ):
        empty = {
            "centers_3d": [],
            "cluster_sizes": [],
            "num_clusters": 0,
            "overall_center_3d": None,
        }
        gaussians = self._gaussians()
        if gaussians is None:
            return empty
        cached_centers = getattr(getattr(self, "slam", None), "_cached_instance_centers", None)
        if cached_centers is not None and len(cached_centers) > 0:
            cached_sizes = getattr(
                getattr(self, "slam", None),
                "_cached_instance_sizes",
                [0] * len(cached_centers),
            )
            pairs = sorted(zip(cached_sizes, cached_centers), key=lambda item: item[0], reverse=True)
            centers = [np.asarray(center, dtype=np.float32) for _, center in pairs]
            sizes = [int(size) for size, _ in pairs]
            return {
                "centers_3d": centers,
                "cluster_sizes": sizes,
                "num_clusters": len(centers),
                "overall_center_3d": np.mean(centers, axis=0) if centers else None,
            }
        if not hasattr(gaussians, "cluster_target_gaussians"):
            stats = self.get_target_semantic_stats(threshold)
            center = stats.get("center")
            if center is None:
                center = stats.get("centroid")
            if center is None:
                return empty
            return {
                "centers_3d": [np.asarray(center, dtype=np.float32)],
                "cluster_sizes": [int(stats.get("count", 0))],
                "num_clusters": 1,
                "overall_center_3d": np.asarray(center, dtype=np.float32),
            }
        try:
            result = gaussians.cluster_target_gaussians(
                threshold=threshold, eps=eps, min_samples=min_samples
            )
        except Exception as exc:
            logger.debug("get_instance_cluster_centers failed: %s", exc)
            return empty
        if result.get("num_clusters", 0) <= 0:
            return empty
        centers = []
        sizes = []
        for center, size in zip(result.get("cluster_centers", []), result.get("cluster_sizes", [])):
            if int(size) >= int(min_cluster_size):
                centers.append(np.asarray(center, dtype=np.float32))
                sizes.append(int(size))
        if not centers:
            return empty
        pairs = sorted(zip(sizes, centers), key=lambda item: item[0], reverse=True)
        sizes = [size for size, _ in pairs]
        centers = [center for _, center in pairs]
        target_xyz = np.asarray(result.get("target_xyz", []), dtype=np.float32)
        return {
            "centers_3d": centers,
            "cluster_sizes": sizes,
            "num_clusters": len(centers),
            "overall_center_3d": (
                target_xyz.mean(axis=0) if target_xyz.size > 0 else np.mean(centers, axis=0)
            ),
        }

    def get_next_best_view_gaussmi_semantic(
        self,
        current_position,
        current_rotation,
        target_region_2d: dict = None,
        occupancy_map=None,
        explored_map=None,
        map_resolution=0.05,
        batch_size=32,
        search_radius_m: float = 1.5,
        num_clusters: int = 8,
        num_yaw_samples: int = 8,
        use_multiview_fusion: bool = True,
        hint_target_world_pos=None,
    ):
        slam = getattr(self, "slam", None)
        if slam is not None and hasattr(slam, "get_next_best_view_gaussmi_semantic"):
            return slam.get_next_best_view_gaussmi_semantic(
                current_position,
                current_rotation,
                target_region_2d,
                occupancy_map,
                explored_map,
                map_resolution,
                batch_size,
                search_radius_m=search_radius_m,
                num_clusters=num_clusters,
                num_yaw_samples=num_yaw_samples,
                use_multiview_fusion=use_multiview_fusion,
                hint_target_world_pos=hint_target_world_pos,
            )
        logger.warning("GauSS-MI semantic NBV unavailable: SLAM backend has no implementation")
        return None

    def score_phase1_gaussmi_yaws(
        self,
        current_position,
        current_rotation,
        yaw_offsets_deg,
        **kwargs,
    ):
        slam = getattr(self, "slam", None)
        if slam is None or not hasattr(slam, "score_phase1_gaussmi_yaws"):
            return {
                "ready": False,
                "reason": "slam_backend_unavailable",
                "candidates": [],
            }
        return slam.score_phase1_gaussmi_yaws(
            current_position,
            current_rotation,
            yaw_offsets_deg,
            **kwargs,
        )

    @staticmethod
    def _to_uint8_rgb(image) -> Optional[np.ndarray]:
        if image is None:
            return None
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[-1] < 3:
            return None
        image = image[..., :3]
        if image.dtype != np.uint8:
            finite = image[np.isfinite(image)]
            max_value = float(finite.max()) if finite.size else 0.0
            if max_value <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @classmethod
    def _extract_rendered_rgb(cls, render_result) -> Optional[np.ndarray]:
        if isinstance(render_result, dict):
            for key in ("rendered", "image", "rgb", "render"):
                if key in render_result:
                    return cls._to_uint8_rgb(render_result[key])
            return None
        return cls._to_uint8_rgb(render_result)

    def render_and_compute_psnr(self, gt_rgb: np.ndarray, frame_idx: int, result_dir: Path) -> Optional[Dict]:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        gt_image = self._to_uint8_rgb(gt_rgb)
        if gt_image is None:
            logger.warning("3DGS debug render skipped: invalid GT image at frame %s", frame_idx)
            return None
        cv2.imwrite(str(result_dir / f"gt_{frame_idx:05d}.png"), cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR))

        if self.slam is None or not getattr(self.slam, "kf_indices", None):
            logger.warning("3DGS debug render skipped: no keyframes at frame %s", frame_idx)
            return None

        latest_kf_idx = self.slam.kf_indices[-1]
        try:
            render_result = self.slam.render_frame(latest_kf_idx)
        except Exception as exc:
            logger.exception("3DGS debug render failed at frame %s: %s", frame_idx, exc)
            return None

        rendered_rgb = self._extract_rendered_rgb(render_result)
        if rendered_rgb is None:
            logger.warning("3DGS debug render skipped: cannot read rendered RGB at frame %s", frame_idx)
            return None

        if rendered_rgb.shape[:2] != gt_image.shape[:2]:
            rendered_rgb = cv2.resize(rendered_rgb, (gt_image.shape[1], gt_image.shape[0]))

        psnr_value = compute_psnr(gt_image, rendered_rgb)
        try:
            ssim_value = compute_ssim(gt_image, rendered_rgb)
        except Exception:
            ssim_value = 0.0

        cv2.imwrite(str(result_dir / f"render_{frame_idx:05d}.png"), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))

        comparison = np.hstack([gt_image, rendered_rgb])
        comparison_with_text = comparison.copy()
        psnr_str = f"{psnr_value:.2f}" if psnr_value != float("inf") else "inf"
        cv2.putText(
            comparison_with_text,
            f"PSNR: {psnr_str} dB | SSIM: {ssim_value:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            comparison_with_text,
            "RGB",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            comparison_with_text,
            "Rendered",
            (gt_image.shape[1] + 10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            comparison_with_text,
            f"Frame: {frame_idx} | KF: {latest_kf_idx}",
            (10, comparison_with_text.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imwrite(
            str(result_dir / f"comparison_{frame_idx:05d}.png"),
            cv2.cvtColor(comparison_with_text, cv2.COLOR_RGB2BGR),
        )

        logger.info(
            "3DGS debug render frame %s: PSNR=%s dB, SSIM=%.4f, saved to %s",
            frame_idx,
            psnr_str,
            ssim_value,
            result_dir,
        )
        return {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "kf_idx": latest_kf_idx,
            "rendered": rendered_rgb,
            "gt": gt_image,
        }

    def save_slam_results(self, save_checkpoint: bool = True):
        if self.slam is None:
            return None
        name = f"checkpoint_{self.frame_count}" if save_checkpoint else "final"
        return self.slam.save_checkpoint(name)

    def save_full_ply_frame(
        self,
        env_step: Optional[int] = None,
        stage: str = "online",
        output_subdir: str = "3dgs_frames",
        highlight_confirmed: bool = True,
        confirmed_color=(255, 0, 255),
    ):
        if self.slam is None:
            return None
        if hasattr(self.slam, "save_frame_ply"):
            return self.slam.save_frame_ply(
                env_step=env_step,
                stage=stage,
                output_subdir=output_subdir,
                highlight_confirmed=highlight_confirmed,
                confirmed_color=confirmed_color,
            )
        return None

    def stop_slam(self):
        self.is_running = False
