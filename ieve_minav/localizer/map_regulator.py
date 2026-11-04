"""
MapRegulator (MR) - 闭环定位模块

新方案（替换原模糊检测 + GaussMI）：
1. 对 REF 渲染图和目标图像做 LightGlue 匹配
2. 对目标图像做语义分割，检查语义物体掩码内匹配点数
3. DBSCAN 聚类匹配点，选取最大聚类
4. 取置信度最高的 50 个匹配点做 RANSAC PnP，得到下一次探索位姿
5. 返回位姿供智能体在 buffer 模式中前进

触发条件：
- REF 后匹配度 >= 40 → 进入闭环定位
"""

import logging
import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.cluster import DBSCAN

from ieve_minav.localizer.colors import Colors


class MapRegulator:
    """
    闭环定位模块（Map Regulator, MR）

    职责：
    - 对 REF 渲染图与目标图像做 LightGlue 特征匹配
    - 语义分割过滤
    - DBSCAN 聚类 + 最大聚类筛选
    - RANSAC PnP 求解下一探索位姿
    """

    def __init__(self,
                 slam_mapper,
                 segmenter=None,
                 target_category: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):

        self.logger = logging.getLogger(__name__)
        self.slam_mapper = slam_mapper
        self.segmenter = segmenter
        self.target_category = target_category
        self.results_dir = results_dir

        default_config = {
            # A semantic candidate is usable when it has either enough
            # matched points or enough aggregate LightGlue confidence.
            'min_semantic_points': 8,
            'target_mask_score_threshold': 4.0,
            'max_pnp_points': 50,
            'cluster_eps': 20.0,
            'cluster_min_samples': 5,
            'ransac_reproj_threshold': 8.0,
            'keypoint_budget': 2048,
        }

        self.config = {**default_config, **(config or {})}

        self.lightglue_available = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_lightglue()

        self.logger.info("MapRegulator (closed-loop) initialized")

    # ===============================================================
    # 初始化
    # ===============================================================

    def _init_lightglue(self):
        try:
            from lightglue import LightGlue, SuperPoint

            budget = self.config['keypoint_budget']

            self.extractor = SuperPoint(
                max_num_keypoints=budget
            ).eval().to(self.device)

            self.lg_matcher = LightGlue(
                features='superpoint'
            ).eval().to(self.device)

            self.lightglue_available = True

            # print(f"{Colors.GREEN}[MR] LightGlue initialized (budget={budget}){Colors.END}")

        except ImportError as e:

            self.logger.warning(f"LightGlue not available for MR: {e}")
            print(f"{Colors.RED}[MR] LightGlue NOT available: {e}{Colors.END}")

    # ===============================================================
    # 公开 API
    # ===============================================================

    # ===============================================================
    # 渲染参考视图
    # ===============================================================

    def render_from_pose(self, w2c_pose: np.ndarray) -> Dict[str, Any]:
        """
        从指定位姿渲染 RGB + Depth 图像

        Args:
            w2c_pose: 世界坐标系到相机坐标系的变换矩阵(4x4)

        Returns:
            {
                'rgb': np.ndarray (H,W,3) uint8,
                'depth': np.ndarray (H,W) float32
            }
            or {'rgb': None, 'depth': None} on failure
        """
        try:
            # 从slam_mapper 获取3DGS渲染器
            slam = self.slam_mapper.slam
            cp = self.slam_mapper.camera_params
            height = cp.get('height', 480)
            width = cp.get('width', 640)

            # 创建虚拟的RGB-D输入（仅用于初始化相机）
            dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            dummy_depth = np.zeros((height, width), dtype=np.float32)

            # 创建相机并渲染
            cam = slam.create_camera(dummy_rgb, dummy_depth, w2c_pose, uid=999998)
            result = slam.render_frame(cam)

            if result is None:
                self.logger.warning("render_frame returned None")
                return {'rgb': None, 'depth': None}

            # 处理RGB
            rendered = result.get('rendered')
            if isinstance(rendered, torch.Tensor):
                rendered = rendered.detach().cpu().numpy()
            if rendered is not None and rendered.ndim == 3 and rendered.shape[0] == 3:
                rendered = rendered.transpose(1, 2, 0)
            if rendered is not None and rendered.dtype != np.uint8:
                rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)

            # 处理 Depth. RealtimeSLAM.render_frame() stores it under
            # result["render_pkg"]["depth"], while some callers may expose it
            # directly as result["depth"].
            depth = result.get('depth')
            if depth is None:
                render_pkg = result.get('render_pkg')
                if isinstance(render_pkg, dict):
                    depth = render_pkg.get('depth')
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().numpy()
            if depth is not None and depth.ndim == 3:
                depth = depth[0]  # (1,H,W) → (H,W)

            return {'rgb': rendered, 'depth': depth}

        except Exception as e:
            self.logger.warning(f"render_from_pose failed: {e}", exc_info=True)
            return {'rgb': None, 'depth': None}

    def run_closed_loop(self,
                        ref_rendered_image: np.ndarray,
                        ref_depth: np.ndarray,
                        ref_pose_w2c: np.ndarray,
                        target_image: np.ndarray,
                        camera_params: Dict[str, Any],
                        cached_match: Optional[Dict[str, Any]] = None,
                        goal_all_masks: list = None) -> Dict[str, Any]:

        if not self.lightglue_available:

            msg = "LightGlue not available, cannot run closed-loop"
            print(f"{Colors.RED}[MR] {msg}{Colors.END}")

            return {
                'success': False,
                'new_pose': None,
                'message': msg
            }

        # ----------------------------------------------------------
        # Step 1 LightGlue 匹配（优先复用 FEM 缓存结果）
        # ----------------------------------------------------------

        if cached_match is not None and cached_match.get('kpts_ref') is not None and cached_match.get('kpts_query') is not None:
            # ★★★ 复用 FEM 匹配结果，跳过重复匹配 ★★★
            # 注意：FEM 里 image0=target, image1=query(REF渲染图)
            # MR 约定：kpts_ref = REF渲染图上的点, kpts_target = 目标图上的点
            kpts_ref = cached_match['kpts_query']   # REF渲染图上的匹配点
            kpts_target = cached_match['kpts_ref']  # 目标图上的匹配点
            scores = cached_match['scores']
            num_matches = cached_match.get('num_matches', len(scores))
            print(f"{Colors.CYAN}[MR] 复用FEM匹配结果: {num_matches} 个匹配点（跳过重复匹配）{Colors.END}")
        else:
            match_result = self._lightglue_match(
                ref_rendered_image,
                target_image
            )

            if not match_result['success']:
                return {
                    'success': False,
                    'new_pose': None,
                    'message': match_result['message']
                }

            kpts_ref = match_result['kpts_ref']
            kpts_target = match_result['kpts_target']
            scores = match_result['scores']
            num_matches = len(scores)

        kpts_ref = np.asarray(kpts_ref, dtype=np.float32)
        kpts_target = np.asarray(kpts_target, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n_common = min(len(kpts_ref), len(kpts_target), len(scores))
        kpts_ref = kpts_ref[:n_common]
        kpts_target = kpts_target[:n_common]
        scores = scores[:n_common]

        # ----------------------------------------------------------
        # Step 2 语义过滤候选（仅目标类别单物体 → 目标类别并集）
        # ----------------------------------------------------------

        pnp_candidates = []

        def add_pnp_candidate(name: str, source_type: str, indices=None, mask_info=None):
            if indices is None:
                idx_arr = np.arange(len(scores), dtype=np.int64)
            else:
                idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
                if idx_arr.size > 0:
                    idx_arr = np.unique(idx_arr)

            if idx_arr.size == 0:
                return

            pnp_candidates.append({
                'name': name,
                'source_type': source_type,
                'indices': idx_arr,
                'kpts_ref': kpts_ref[idx_arr],
                'kpts_target': kpts_target[idx_arr],
                'scores': scores[idx_arr],
                'mask_info': mask_info or {'type': source_type, 'n_pts': int(idx_arr.size)},
            })

        if not self.target_category:
            message = "MR target category is unavailable; refusing unconstrained/full-image PnP"
            print(f"{Colors.YELLOW}[MR] {message}{Colors.END}")
            return {'success': False, 'new_pose': None, 'message': message}

        if goal_all_masks:
            target_mask_pairs = [
                (idx, obj) for idx, obj in enumerate(goal_all_masks)
                if self._category_matches(obj.get('category', ''), self.target_category)
            ]
            print(f"{Colors.CYAN}[MR] goal_all_masks: "
                  f"{len(goal_all_masks)} 个语义实例（目标类别 "
                  f"'{self.target_category}'={len(target_mask_pairs)} 个），"
                  f"仅目标类别参与PnP候选{Colors.END}")

            if not target_mask_pairs:
                message = (f"MR target mask missing for category '{self.target_category}'; "
                           "refusing all-semantic/full-image PnP")
                print(f"{Colors.YELLOW}[MR] {message}{Colors.END}")
                return {'success': False, 'new_pose': None, 'message': message}

            H_t, W_t = target_image.shape[:2]

            # 统计目标类别物体掩码内的匹配点数和加权评分（Σconfidence）
            per_obj_indices = []
            for _, obj in target_mask_pairs:
                bmask = obj['mask']  # (H,W) bool
                if bmask.shape[0] != H_t or bmask.shape[1] != W_t:
                    # 尺寸不匹配，跳过
                    per_obj_indices.append([])
                    continue
                in_mask = []
                for idx, (x, y) in enumerate(kpts_target):
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= iy < H_t and 0 <= ix < W_t and bmask[iy, ix]:
                        in_mask.append(idx)
                per_obj_indices.append(in_mask)

            min_semantic_points = int(self.config.get('min_semantic_points', 8))
            score_threshold = float(self.config.get('target_mask_score_threshold', 4.0))
            obj_match_counts = [len(idxs) for idxs in per_obj_indices]
            obj_match_scores = [float(scores[idxs].sum()) if len(idxs) > 0 else 0.0 for idxs in per_obj_indices]
            print(f"{Colors.CYAN}[MR] 目标类别掩码内匹配评分（Σconf）: "
                  f"{[(target_mask_pairs[i][0], target_mask_pairs[i][1].get('category', ''), f'{s:.2f}', obj_match_counts[i]) for i, s in enumerate(obj_match_scores)]}{Colors.END}")

            # 第一级：单实例满足“分数或点数”任一门槛时优先尝试。
            eligible_objects = [
                i for i, (count, score) in enumerate(zip(obj_match_counts, obj_match_scores))
                if count >= min_semantic_points or score >= score_threshold
            ]
            if eligible_objects:
                best_local_idx = max(
                    eligible_objects,
                    key=lambda i: (obj_match_scores[i], obj_match_counts[i]),
                )
                best_obj_idx, best_obj = target_mask_pairs[best_local_idx]
                idxs = per_obj_indices[best_local_idx]
                selected_mask_info = {'type': 'single_target_obj', 'obj_idx': best_obj_idx,
                                      'category': best_obj.get('category', ''),
                                      'n_pts': len(idxs),
                                      'score': obj_match_scores[best_local_idx]}
                add_pnp_candidate(
                    name=f"single_target_obj:{best_obj.get('category', '')}#{best_obj_idx}",
                    source_type='single_target_obj',
                    indices=idxs,
                    mask_info=selected_mask_info,
                )
                print(f"{Colors.GREEN}[MR] 目标物体 '{best_obj.get('category', '')}' "
                      f"内匹配点数={obj_match_counts[best_local_idx]}、"
                      f"评分={obj_match_scores[best_local_idx]:.2f}，"
                      f"满足点数>={min_semantic_points}或评分>={score_threshold:.1f}，"
                      f"优先尝试单目标物体PnP{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[MR] 没有单个目标物体满足点数>={min_semantic_points}"
                      f"或评分>={score_threshold:.1f}，尝试目标类别 mask 并集{Colors.END}")

            # 第二级：目标类别 mask 内匹配点合并，用于均匀采样下的稳健 fallback。
            non_empty = [np.asarray(idxs, dtype=np.int64) for idxs in per_obj_indices if len(idxs) > 0]
            if non_empty:
                union_indices = np.unique(np.concatenate(non_empty))
                union_score = float(scores[union_indices].sum()) if len(union_indices) > 0 else 0.0
                if len(union_indices) >= min_semantic_points or union_score >= score_threshold:
                    add_pnp_candidate(
                        name='all_target_objects',
                        source_type='all_target_objects',
                        indices=union_indices,
                        mask_info={
                            'type': 'all_target_objects',
                            'obj_idx': -1,
                            'category': self.target_category,
                            'n_pts': int(len(union_indices)),
                            'score': union_score,
                        },
                    )
                    print(f"{Colors.CYAN}[MR] 目标类别 mask 并集内匹配点: "
                          f"{len(union_indices)} pts, score={union_score:.2f}，"
                          f"满足点数>={min_semantic_points}或评分>={score_threshold:.1f}{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}[MR] 目标类别 mask 并集仅有 {len(union_indices)} pts, "
                          f"score={union_score:.2f}，未达到点数/评分门槛，MR闭环失败{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[MR] 目标类别 mask 内无匹配点，MR闭环失败{Colors.END}")

        else:
            message = "MR goal_all_masks is empty; refusing full-image PnP"
            print(f"{Colors.YELLOW}[MR] {message}{Colors.END}")
            return {'success': False, 'new_pose': None, 'message': message}

        if not pnp_candidates:
            message = (f"MR has no target-category PnP candidates for "
                       f"'{self.target_category}'; refusing unconstrained PnP")
            print(f"{Colors.YELLOW}[MR] {message}{Colors.END}")
            return {'success': False, 'new_pose': None, 'message': message}

        # ----------------------------------------------------------
        # Step 3/4：按候选顺序尝试 top-K → 2D/3D → RANSAC PnP
        # ----------------------------------------------------------

        max_pts = self.config['max_pnp_points']
        failure_reasons = []

        for candidate in pnp_candidates:
            sel_kpts_ref = candidate['kpts_ref']
            sel_kpts_target = candidate['kpts_target']
            sel_scores = candidate['scores']
            source_name = candidate['name']

            if len(sel_scores) < 4:
                reason = f"{source_name}: too few matches before PnP ({len(sel_scores)})"
                failure_reasons.append(reason)
                print(f"{Colors.YELLOW}[MR] {reason}，尝试下一候选{Colors.END}")
                continue

            if len(sel_scores) > max_pts:
                top_idx = np.argsort(sel_scores)[::-1][:max_pts]
                sel_kpts_ref = sel_kpts_ref[top_idx]
                sel_kpts_target = sel_kpts_target[top_idx]
                sel_scores = sel_scores[top_idx]

            pts_3d = self._backproject_to_3d(
                sel_kpts_ref,
                ref_depth,
                ref_pose_w2c,
                camera_params
            )

            if pts_3d is None:
                reason = f"{source_name}: back-projection failed (depth invalid)"
                failure_reasons.append(reason)
                print(f"{Colors.YELLOW}[MR] {reason}，尝试下一候选{Colors.END}")
                continue

            valid = np.isfinite(pts_3d).all(axis=1) & (pts_3d[:, 2] != 0)

            pts_3d_valid = pts_3d[valid]
            sel_kpts_target_valid = sel_kpts_target[valid]
            sel_scores_valid = sel_scores[valid]

            if len(pts_3d_valid) < 4:
                reason = f"{source_name}: too few valid 3D points after depth filter ({len(pts_3d_valid)})"
                failure_reasons.append(reason)
                print(f"{Colors.YELLOW}[MR] {reason}，尝试下一候选{Colors.END}")
                continue

            pnp_result = self._ransac_pnp(
                pts_3d_valid,
                sel_kpts_target_valid,
                camera_params
            )

            if not pnp_result['success']:
                reason = f"{source_name}: {pnp_result['message']}"
                failure_reasons.append(reason)
                print(f"{Colors.YELLOW}[MR] {reason}，尝试下一候选{Colors.END}")
                continue

            new_pose_w2c = pnp_result['pose_w2c']
            c2w = np.linalg.inv(new_pose_w2c)
            new_position = c2w[:3, 3]
            new_forward = -c2w[:3, 2]
            new_yaw_deg = np.rad2deg(
                np.arctan2(new_forward[0], -new_forward[2])
            )

            print(f"[MR] 闭环成功 source={source_name} "
                  f"pos=[{new_position[0]:.2f},{new_position[1]:.2f},{new_position[2]:.2f}] "
                  f"yaw={new_yaw_deg:.1f}° inliers={pnp_result['num_inliers']} "
                  f"selected_matches={len(sel_scores_valid)} total_matches={num_matches}")

            return {
                'success': True,
                'new_pose': new_pose_w2c,
                'new_position': new_position,
                'new_yaw': new_yaw_deg,
                'num_inliers': pnp_result['num_inliers'],
                'num_matches': num_matches,
                'selected_matches': int(len(sel_scores_valid)),
                'match_source': source_name,
                '_selected_mask_info': candidate.get('mask_info'),
                'message': f'Closed-loop PnP success ({pnp_result["num_inliers"]} inliers, source={source_name})'
            }

        msg = "All MR PnP candidates failed: " + " | ".join(failure_reasons[-4:])
        print(f"{Colors.RED}[MR] {msg}{Colors.END}")
        return {
            'success': False,
            'new_pose': None,
            'message': msg
        }

    # ===============================================================
    # LightGlue
    # ===============================================================

    def _lightglue_match(self, image0, image1):

        try:

            feats0 = self._extract_features(image0)
            feats1 = self._extract_features(image1)

            with torch.no_grad():
                matches = self.lg_matcher({'image0': feats0, 'image1': feats1})

            m = matches['matches'][0]
            s = matches['scores'][0]

            kpts0_all = feats0['keypoints'][0]
            kpts1_all = feats1['keypoints'][0]

            idx0 = m[:, 0].long()
            idx1 = m[:, 1].long()

            kpts0 = kpts0_all[idx0].cpu().numpy()
            kpts1 = kpts1_all[idx1].cpu().numpy()

            scores_np = s.cpu().numpy().astype(np.float64)

            return {
                'success': True,
                'kpts_ref': kpts0,
                'kpts_target': kpts1,
                'scores': scores_np
            }

        except Exception as e:

            self.logger.error(f"LightGlue match error: {e}", exc_info=True)

            return {
                'success': False,
                'message': str(e)
            }

    def _extract_features(self, image):

        tensor = self._numpy_to_torch(image)

        with torch.no_grad():
            feats = self.extractor.extract(tensor)

        return feats

    def _numpy_to_torch(self, image):

        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0

        tensor = torch.from_numpy(gray)[None, None].to(self.device)

        return tensor

    # ===============================================================
    # 语义分割
    # ===============================================================

    def _segment_target(self, target_image):

        try:

            mask, labels, _ = self.segmenter.segment_frame(
                target_image,
                largest_only=True
            )

            if mask is None or len(labels) == 0:
                return None

            cat_id = self.segmenter.get_category_id(
                self.target_category.lower()
            )

            if cat_id == 0:
                return (mask > 0)

            return (mask == cat_id)

        except Exception as e:

            self.logger.warning(f"Segmentation failed: {e}")

            return None

    # ===============================================================
    # 掩码过滤
    # ===============================================================

    def _get_mask_filter(self, kpts, mask):

        H, W = mask.shape
        result = np.zeros(len(kpts), dtype=bool)

        for i, (x, y) in enumerate(kpts):

            ix, iy = int(round(x)), int(round(y))

            if 0 <= iy < H and 0 <= ix < W and mask[iy, ix]:
                result[i] = True

        return result

    # ===============================================================
    # DBSCAN
    # ===============================================================

    def _cluster_and_select(self, kpts_ref, kpts_target, scores):

        if len(kpts_target) < self.config['cluster_min_samples']:

            return {
                'success': False,
                'message': f'Too few points for clustering: {len(kpts_target)}'
            }

        db = DBSCAN(
            eps=self.config['cluster_eps'],
            min_samples=self.config['cluster_min_samples']
        ).fit(kpts_target)

        labels = db.labels_

        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) == 0:

            print(f"{Colors.YELLOW}[MR] DBSCAN: 未发现有效聚类{Colors.END}")

            return {
                'success': True,
                'kpts_ref': np.empty((0, 2)),
                'kpts_target': np.empty((0, 2)),
                'scores': np.empty((0,)),
                'labels': labels
            }

        cluster_sizes = {
            lbl: np.sum(labels == lbl)
            for lbl in unique_labels
        }

        best_label = max(cluster_sizes, key=cluster_sizes.get)

        mask = (labels == best_label)

        return {
            'success': True,
            'kpts_ref': kpts_ref[mask],
            'kpts_target': kpts_target[mask],
            'scores': scores[mask],
            'labels': labels
        }

    # ===============================================================
    # 3D反投影
    # ===============================================================

    def _backproject_to_3d(self,
                           kpts_2d,
                           depth,
                           w2c,
                           camera_params):

        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']

        H, W = depth.shape

        c2w = np.linalg.inv(w2c)

        R = c2w[:3, :3]
        t = c2w[:3, 3]

        pts_world = []

        for x, y in kpts_2d:

            col = int(round(x))
            row = int(round(y))

            col = np.clip(col, 0, W - 1)
            row = np.clip(row, 0, H - 1)

            d = float(depth[row, col])

            if d <= 0 or not np.isfinite(d):

                pts_world.append([np.nan, np.nan, np.nan])
                continue

            z_c = d
            x_c = (col - cx) * z_c / fx
            y_c = (row - cy) * z_c / fy

            pt_cam = np.array([x_c, y_c, z_c])

            pt_world = R @ pt_cam + t

            pts_world.append(pt_world)

        return np.array(pts_world)

    # ===============================================================
    # PnP
    # ===============================================================

    def _ransac_pnp(self, pts_3d, pts_2d, camera_params):

        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist = np.zeros(4)

        obj = pts_3d.reshape(-1, 1, 3).astype(np.float64)
        img = pts_2d.reshape(-1, 1, 2).astype(np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            dist,
            iterationsCount=1000,
            reprojectionError=self.config['ransac_reproj_threshold'],
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None:

            return {
                'success': False,
                'pose_w2c': None,
                'num_inliers': 0,
                'message': 'solvePnPRansac failed'
            }

        R, _ = cv2.Rodrigues(rvec)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.flatten()

        return {
            'success': True,
            'pose_w2c': w2c,
            'num_inliers': len(inliers)
        }

    # ===============================================================
    # 工具
    # ===============================================================

    def _category_matches(self, detected: str, target: str) -> bool:
        """与IIN目标确认保持一致的轻量类别匹配。"""
        detected_lower = str(detected).lower().strip()
        target_lower = str(target).lower().strip()

        if not detected_lower or not target_lower:
            return False
        if detected_lower == target_lower:
            return True
        if target_lower in detected_lower or detected_lower in target_lower:
            return True

        aliases = {
            'chair': ['chair', 'stool', 'seat', 'armchair'],
            'couch': ['couch', 'sofa'],
            'sofa': ['sofa', 'couch'],
            'bed': ['bed'],
            'table': ['table', 'desk', 'dining table', 'coffee table'],
            'plant': ['plant', 'potted plant', 'houseplant'],
            'toilet': ['toilet'],
            'tv': ['tv', 'television', 'monitor', 'screen', 'tv_monitor'],
            'tv_monitor': ['tv', 'television', 'monitor', 'screen', 'tv_monitor'],
        }

        for canonical, alias_list in aliases.items():
            if target_lower == canonical or target_lower in alias_list:
                return detected_lower == canonical or detected_lower in alias_list
        return False

    def reset(self):
        pass
