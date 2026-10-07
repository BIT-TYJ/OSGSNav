import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import cv2


class FeatureMatcher:
    """特征匹配模块 - 支持均匀采样和语义密度采样两套目标图特征。"""
    
    def __init__(self,
                 target_image: np.ndarray,
                 match_threshold: float = 40.0,
                 results_dir: Optional[str] = None,
                 segmenter=None,
                 target_category: Optional[str] = None,
                 semantic_density_ratio: float = 6.0,
                 save_visualization_results: bool = True,
                 default_feature_mode: str = "uniform"):
        """
        初始化特征匹配器
        
        Args:
            target_image: 目标图像 (RGB格式)
            match_threshold: 绝对匹配证据阈值，建议范围 30-50
            results_dir: 结果保存目录
            segmenter: SemanticSegmenter实例（可选），用于语义分割
            target_category: 目标类别名称（如 'bed', 'chair'），配合segmenter使用
            semantic_density_ratio: 目标语义区域内关键点密度与背景的比值（默认3.0）
                例如 3.0 表示目标区域的关键点「留存率」是背景的 3 倍
            default_feature_mode: 默认匹配特征模式，'uniform' 或 'semantic'
        """
        self.logger = logging.getLogger(__name__)
        self.target_image = target_image
        self.match_threshold = match_threshold
        self.results_dir = Path(results_dir) if results_dir else None
        self.save_visualization_results = bool(save_visualization_results)
        
        # ★★★ 语义密度采样参数 ★★★
        self.segmenter = segmenter
        self.target_category = target_category
        self.semantic_density_ratio = semantic_density_ratio
        self.default_feature_mode = self._normalize_feature_mode(default_feature_mode)
        self.target_feature_mode = self.default_feature_mode
        self.target_semantic_mask = None  # 目标图像的语义掩码 (H, W), bool
        self.target_sampling_mask = None  # 仅用于关键点重要性采样的膨胀掩码
        self.keypoint_budget = 2048       # 最终保留的关键点数
        self.oversample_factor = 2        # 初始提取时的过采样倍数
        self.target_features_uniform = None
        self.target_features_semantic = None
        self.target_features = None        # 兼容旧调用，指向当前默认特征
        
        if self.results_dir is not None and self.save_visualization_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # ★★★ 先进行语义分割（在特征提取之前，以便对目标关键点做密度采样）★★★
        self._init_semantic_mask()
        
        # 初始化特征提取和匹配器（内部会利用语义掩码做密度采样）
        self._init_matcher()
        
        self.logger.info(
            f"FEM initialized with absolute match threshold={match_threshold:.1f} "
            f"(default_features={self.default_feature_mode}, "
            f"semantic_density_ratio={semantic_density_ratio})"
        )
        sem_status = "ENABLED" if self.target_semantic_mask is not None else "DISABLED"
        # print(f"[FEM] Initialized with semantic-density sampling ({sem_status}), "
              # f"density_ratio={semantic_density_ratio}, threshold: {match_threshold:.1f}")
    
    # ==================================================================
    #  语义分割相关方法
    # ==================================================================
    
    def _init_semantic_mask(self, precomputed_mask=None):
        """
        对目标图像进行语义分割，提取目标物体的掩码。
        掩码用于在关键点选择时控制目标区域 / 背景区域的采样密度。
        
        Args:
            precomputed_mask: 可选，外部已计算好的 semantic_mask (H,W) int array，
                              有则直接用，跳过内部 segment_frame 调用。
        """
        if self.segmenter is None or self.target_category is None:
            self.target_semantic_mask = None
            self.target_sampling_mask = None
            return
        
        try:
            # ★★★ 优先使用外部预计算的 semantic_mask，跳过重复分割 ★★★
            if precomputed_mask is not None:
                semantic_mask = precomputed_mask
                semantic_labels = [{'category': self.target_category}]  # 占位，只需非空
                print(f"[FEM] Using precomputed semantic mask from analyze_all_goal_image")
            else:
                semantic_mask, semantic_labels, _ = self.segmenter.segment_frame(
                    self.target_image, largest_only=False
                )
            
            if semantic_mask is None or len(semantic_labels) == 0:
                print(f"[FEM] Warning: No objects detected in target image, semantic density disabled")
                self.target_semantic_mask = None
                self.target_sampling_mask = None
                return
            
            target_cat_id = self.segmenter.get_category_id(self.target_category.lower())
            print(f"[FEM-DEBUG] target_category='{self.target_category}' cat_id={target_cat_id} "
                  f"unique_in_mask={np.unique(semantic_mask).tolist()} "
                  f"px_per_label={ {int(l): int(np.sum(semantic_mask==l)) for l in np.unique(semantic_mask) if l>0} }")
            
            if target_cat_id == 0:
                # print(f"[FEM] Category '{self.target_category}' not in known categories, "
                      # f"using all detected objects as target region")
                self.target_semantic_mask = (semantic_mask > 0)
            else:
                self.target_semantic_mask = (semantic_mask == target_cat_id)
            
            target_pixels = np.sum(self.target_semantic_mask)
            total_pixels = self.target_semantic_mask.size
            coverage = target_pixels / total_pixels * 100
            
            if target_pixels == 0:
                # print(f"[FEM] Target category '{self.target_category}' not found in segmentation, "
                      # f"using all detected regions")
                self.target_semantic_mask = (semantic_mask > 0)
                target_pixels = np.sum(self.target_semantic_mask)
                coverage = target_pixels / total_pixels * 100

            if target_pixels > 0:
                # Sampling uses a one-pixel semantic ring so near-boundary texture can still participate.
                dilated_mask = cv2.dilate(
                    self.target_semantic_mask.astype(np.uint8),
                    np.ones((3, 3), dtype=np.uint8),
                    iterations=1,
                ) > 0
                dilated_pixels = int(np.sum(dilated_mask))
                self.target_sampling_mask = dilated_mask
                print(
                    f"[FEM] Sampling mask dilated: base_pixels={int(target_pixels)}, "
                    f"dilated_pixels={dilated_pixels}, added={dilated_pixels - int(target_pixels)}"
                )
            else:
                self.target_sampling_mask = self.target_semantic_mask.copy()
            
            # print(f"[FEM] Semantic mask created: target='{self.target_category}', "
                  # f"target_pixels={target_pixels}, coverage={coverage:.1f}%")
            # print(f"[FEM] Density sampling: target_region={self.semantic_density_ratio}x denser "
                  # f"than background")

            # ★★★ 逐实例标记：用连通域区分同类目标的不同实例 ★★★
            # 过滤掉微小连通域（SAM掩码噪声/孔洞），只保留面积 >= 500px 的实例
            from scipy.ndimage import label as _ndlabel
            _raw_labels, _raw_n = _ndlabel(self.target_semantic_mask)
            _min_instance_pixels = 500
            _kept = 0
            self._target_instance_labels = np.zeros_like(_raw_labels)
            for _i in range(1, _raw_n + 1):
                if np.sum(_raw_labels == _i) >= _min_instance_pixels:
                    _kept += 1
                    self._target_instance_labels[_raw_labels == _i] = _kept
            self._n_target_instances = _kept
            print(f"[FEM] Target instances: {_kept} kept (from {_raw_n} components, min={_min_instance_pixels}px)")
            for _i in range(1, _kept + 1):
                print(f"[FEM]   Inst{_i}: {int(np.sum(self._target_instance_labels == _i))}px")
            
            # 保存语义掩码可视化（调试用）
            if self.results_dir is not None and self.save_visualization_results:
                self._save_semantic_mask_debug(semantic_mask_raw=semantic_mask)
                
        except Exception as e:
            self.logger.error(f"Semantic segmentation of target image failed: {e}", exc_info=True)
            print(f"[FEM] Warning: Semantic segmentation failed ({e}), falling back to uniform sampling")
            self.target_semantic_mask = None
            self.target_sampling_mask = None
    
    def _save_semantic_mask_debug(self, semantic_mask_raw=None):
        """保存语义掩码可视化用于调试，显示所有分割区域和逐实例像素数"""
        try:
            import cv2
            if self.results_dir is None or not self.save_visualization_results:
                return

            vis = self.target_image.copy()  # (H,W,3) RGB
            H, W = vis.shape[:2]

            inst_colors = [
                (0, 255, 0), (0, 128, 255), (255, 0, 128),
                (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (128, 0, 255), (0, 255, 128),
            ]

            overlay = vis.copy()

            # ---- 画整体语义掩码（浅绿半透明底） ----
            if self.target_semantic_mask is not None:
                overlay[self.target_semantic_mask] = [180, 255, 180]

            # ---- 画逐实例颜色 ----
            inst_info_lines = []  # 右上角文字
            if self._n_target_instances > 0 and self._target_instance_labels is not None:
                for lbl in range(1, self._n_target_instances + 1):
                    color = inst_colors[(lbl - 1) % len(inst_colors)]
                    mask_lbl = (self._target_instance_labels == lbl)
                    overlay[mask_lbl] = color
                    px = int(np.sum(mask_lbl))
                    inst_info_lines.append((f"Inst{lbl}: {px}px", color))
            else:
                inst_info_lines.append(("No instances (all filtered)", (128, 128, 128)))

            # ---- 如果有原始 semantic_mask_raw（segment_frame 的原始输出），画所有标签 ----
            if semantic_mask_raw is not None:
                unique_labels = [l for l in np.unique(semantic_mask_raw) if l > 0]
                raw_info_lines = []
                raw_colors = [
                    (200, 80, 80), (80, 200, 80), (80, 80, 200),
                    (200, 200, 80), (200, 80, 200), (80, 200, 200),
                ]
                for i, lbl in enumerate(unique_labels):
                    px = int(np.sum(semantic_mask_raw == lbl))
                    raw_info_lines.append((f"Label{lbl}: {px}px", raw_colors[i % len(raw_colors)]))
            else:
                raw_info_lines = []

            vis = cv2.addWeighted(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), 0.55,
                                  cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), 0.45, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX

            # ---- 右上角：逐实例信息 ----
            header_lines = [
                (f"Category: {self.target_category}", (255, 255, 255)),
                (f"sem_mask_px: {int(np.sum(self.target_semantic_mask)) if self.target_semantic_mask is not None else 0}", (200, 255, 200)),
                (f"n_inst: {self._n_target_instances} (min=500px)", (200, 255, 200)),
            ] + inst_info_lines + raw_info_lines

            for li, (text, color) in enumerate(header_lines):
                y_pos = 22 + li * 22
                bgr = (color[2], color[1], color[0])
                cv2.putText(vis, text, (W - 320, y_pos), font, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, text, (W - 320, y_pos), font, 0.52, bgr, 1, cv2.LINE_AA)

            save_path = str(self.results_dir / 'semantic.png')
            cv2.imwrite(save_path, vis)
            print(f"[FEM] Semantic segmentation visualization saved: {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save semantic mask debug image: {e}", exc_info=True)
    
    # ==================================================================
    #  语义密度采样核心方法
    # ==================================================================
    
    def _semantic_keypoint_selection(self, features, budget: int = 2048):
        """
        ★★★ 核心方法：基于语义掩码的密度采样关键点选择 ★★★
        
        从过采样提取的关键点中，按照密度比对目标区域和背景区域分别筛选，
        使目标区域的关键点密度为背景的 semantic_density_ratio 倍。
        总关键点数保持在 budget 左右。
        
        Args:
            features: SuperPoint 提取的特征字典，包含 keypoints, descriptors, keypoint_scores 等
            budget: 最终保留的关键点数
            
        Returns:
            筛选后的特征字典（与输入结构相同）
        """
        import torch
        
        sampling_mask = self.target_sampling_mask
        if sampling_mask is None:
            return features
        
        # -------- 解析关键点 --------
        kpts = features['keypoints']  # (1, N, 2) 或 (N, 2)
        has_batch = kpts.dim() == 3
        kpts_2d = kpts[0] if has_batch else kpts  # (N, 2)
        N = kpts_2d.shape[0]
        
        if N <= budget:
            # 已在预算内，无需筛选
            return features
        
        # -------- 将关键点分为目标区域 / 背景 --------
        H, W = sampling_mask.shape
        kpts_np = kpts_2d.cpu().numpy()
        
        semantic_indices = []
        background_indices = []
        for i in range(N):
            x, y = kpts_np[i]
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < H and 0 <= ix < W and sampling_mask[iy, ix]:
                semantic_indices.append(i)
            else:
                background_indices.append(i)
        
        N_s = len(semantic_indices)
        N_b = len(background_indices)
        
        # -------- 计算各组保留数量 --------
        # 目标区域留存率 = ratio × 背景留存率
        # keep_s + keep_b = budget
        # keep_s / N_s = ratio × (keep_b / N_b)  (当两组都有富余时)
        ratio = self.semantic_density_ratio
        
        if N_s == 0:
            keep_s = 0
            keep_b = min(N_b, budget)
        elif N_b == 0:
            keep_s = min(N_s, budget)
            keep_b = 0
        else:
            # 理想分配
            keep_s = int(round(budget * ratio * N_s / (ratio * N_s + N_b)))
            keep_s = min(keep_s, N_s)
            keep_b = budget - keep_s
            keep_b = min(keep_b, N_b)
            
            # 如果某组不够，把余额分给另一组
            remaining = budget - keep_s - keep_b
            if remaining > 0:
                extra_s = min(remaining, N_s - keep_s)
                keep_s += extra_s
                remaining -= extra_s
                if remaining > 0:
                    keep_b += min(remaining, N_b - keep_b)
        
        # -------- 按 keypoint_score 排序，各组取 Top-K --------
        scores_tensor = features.get('keypoint_scores', None)
        if scores_tensor is not None:
            scores_1d = scores_tensor[0] if has_batch else scores_tensor  # (N,)
        else:
            scores_1d = None
        
        def topk_indices(group_indices, k):
            """从 group_indices 中选出得分最高的 k 个"""
            if k <= 0:
                return []
            if k >= len(group_indices):
                return list(group_indices)
            if scores_1d is not None:
                idx_tensor = torch.tensor(group_indices, dtype=torch.long, device=scores_1d.device)
                group_scores = scores_1d[idx_tensor]
                _, top_idx = group_scores.topk(k)
                return [group_indices[i] for i in top_idx.cpu().tolist()]
            else:
                return list(group_indices[:k])
        
        selected_sem = topk_indices(semantic_indices, keep_s)
        selected_bg  = topk_indices(background_indices, keep_b)
        
        # -------- 合并并按原始顺序排序 --------
        selected = sorted(selected_sem + selected_bg)
        idx = torch.tensor(selected, dtype=torch.long, device=kpts.device)
        
        # -------- 过滤所有特征字段 --------
        filtered = {}
        for key, val in features.items():
            if torch.is_tensor(val):
                if has_batch and val.dim() >= 2 and val.shape[1] == N:
                    filtered[key] = val[:, idx]
                elif not has_batch and val.dim() >= 1 and val.shape[0] == N:
                    filtered[key] = val[idx]
                else:
                    filtered[key] = val
            else:
                filtered[key] = val
        
        actual_sem = len(selected_sem)
        actual_bg = len(selected_bg)
        total = actual_sem + actual_bg
        sem_pct = actual_sem / total * 100 if total > 0 else 0
        # print(f"[FEM] Semantic keypoint selection: {N} -> {total} "
              # f"(target={actual_sem} [{sem_pct:.0f}%], bg={actual_bg}, "
              # f"density_ratio={ratio}x)")
        
        return filtered

    def _normalize_feature_mode(self, feature_mode: Optional[str] = None) -> str:
        """Normalize target-feature mode names."""
        mode = (feature_mode or "uniform")
        mode = str(mode).strip().lower().replace("-", "_")
        if mode in ("semantic", "semantic_density", "semantic_dense"):
            return "semantic"
        return "uniform"

    def _get_target_features(self, feature_mode: Optional[str] = None):
        """Return the requested target feature set, falling back to uniform."""
        requested = self._normalize_feature_mode(
            feature_mode or getattr(self, "default_feature_mode", "uniform")
        )
        if requested == "semantic" and self.target_features_semantic is not None:
            return self.target_features_semantic, "semantic"
        if self.target_features_uniform is not None:
            return self.target_features_uniform, "uniform"
        if self.target_features is not None:
            return self.target_features, getattr(self, "target_feature_mode", "uniform")
        return None, "none"

    def _build_target_feature_sets(self):
        """Build both target-image feature sets; default matching still uses uniform."""
        if not getattr(self, "lightglue_available", False):
            return

        try:
            from lightglue import SuperPoint
            import torch

            # Uniform branch: exactly the standard SuperPoint budget.
            self.target_features_uniform = self._extract_features(self.target_image)

            # Semantic branch: oversample first, then keep points with higher target-region density.
            self.target_features_semantic = None
            if self.target_sampling_mask is not None:
                oversample_budget = self.keypoint_budget * self.oversample_factor
                target_extractor = SuperPoint(max_num_keypoints=oversample_budget).eval().to(self.device)
                target_tensor = self._numpy_to_torch_image(self.target_image)
                with torch.no_grad():
                    raw_features = target_extractor.extract(target_tensor)

                self.target_features_semantic = self._semantic_keypoint_selection(
                    raw_features, budget=self.keypoint_budget
                )

                del target_extractor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.target_features, self.target_feature_mode = self._get_target_features(
                self.default_feature_mode
            )
            self.logger.info(
                f"Target feature sets ready: default={self.target_feature_mode}, "
                f"uniform={self.target_features_uniform is not None}, "
                f"semantic={self.target_features_semantic is not None}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to build target feature sets: {e}", exc_info=True)
            self.target_features_semantic = None
            if self.target_features_uniform is None:
                self.target_features_uniform = self._extract_features(self.target_image)
            self.target_features = self.target_features_uniform
            self.target_feature_mode = "uniform"

    def refresh_target_features(self,
                                precomputed_mask=None,
                                default_feature_mode: Optional[str] = None):
        """
        Refresh target semantic masks and both target feature branches for a new goal image.
        """
        if default_feature_mode is not None:
            self.default_feature_mode = self._normalize_feature_mode(default_feature_mode)
        self._init_semantic_mask(precomputed_mask=precomputed_mask)
        self._build_target_feature_sets()
        return self.target_features
    
    # ==================================================================
    #  特征提取和匹配器初始化
    # ==================================================================
    
    def _init_matcher(self):
        """初始化LightGlue匹配器，并构建两套目标图特征。"""
        try:
            from lightglue import LightGlue, SuperPoint
            import torch
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 主提取器（用于 query 图像，标准 budget）
            self.extractor = SuperPoint(max_num_keypoints=self.keypoint_budget).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
            self.lightglue_available = True
            # ★★★ 初始化匹配缓存，防止首次调用前 getattr 返回错误默认值 ★★★
            self._last_semantic_matches = 0
            self._last_per_instance_semantic_matches = []
            self._last_matched_kpts_on_target = None
            self._last_match_cache = None
            self._build_target_feature_sets()
            
            self.logger.info(f"LightGlue matcher initialized on {self.device}")
            
        except ImportError as e:
            self.logger.warning(f"LightGlue not available: {e}, using ORB as fallback")
            self.lightglue_available = False
            
            self.extractor = cv2.ORB_create(nfeatures=2000)
            self.matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
            gray_target = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)
            self.target_kp, self.target_desc = self.extractor.detectAndCompute(
                gray_target, None
            )
            self.logger.info(f"ORB matcher initialized (fallback), keypoints: {len(self.target_kp)}")
    
    def match_rendered_views(self,
                             rendered_images: List[Dict[str, Any]],
                             feature_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        批量匹配多个渲染视图 — 先批量提取特征，再逐个匹配
        
        Args:
            rendered_images: 渲染图像列表，每个包含 'image', 'pose', 'index' 等
            
        Returns:
            匹配结果字典
        """
        if not rendered_images:
            return {
                'match_success': False,
                'best_score': 0.0,
                'best_match_count': 0,
                'best_index': -1,
                'best_match_count_index': -1,
                'best_pose': None,
                'all_scores': [],
                'all_match_counts': [],
            }
        
        active_mode = "orb"
        if self.lightglue_available:
            target_features, active_mode = self._get_target_features(feature_mode)
            scores, match_counts = self._batch_match_lightglue(
                rendered_images,
                target_features=target_features,
                feature_mode=active_mode,
            )
        else:
            scores = []
            match_counts = []
            for i, img_data in enumerate(rendered_images):
                image = img_data.get('image')
                if image is None:
                    scores.append(0.0)
                    match_counts.append(0)
                    continue
                score = self._compute_match_score(image, feature_mode=feature_mode)
                scores.append(score)
                match_cache = getattr(self, "_last_match_cache", None) or {}
                match_counts.append(int(match_cache.get("num_matches", 0)))
        
        # 找到最佳匹配
        best_index = int(np.argmax(scores))
        best_score = scores[best_index]
        best_view_match_count = int(match_counts[best_index]) if best_index >= 0 else 0
        best_match_count_index = int(np.argmax(match_counts)) if match_counts else -1
        best_match_count = int(match_counts[best_match_count_index]) if best_match_count_index >= 0 else 0
        match_success = best_score >= self.match_threshold

        # Top-5 views by score for concise output
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"V{idx}={s:.1f}" for idx, s in ranked if s > 0)
        status = "\u2713 SUCCESS" if match_success else "\u2717 FAILED"
        max_match_str = (
            f", max_matches={best_match_count}"
            if best_match_count != best_view_match_count else ""
        )
        print(f"[FEM] {len(rendered_images)} views | mode={active_mode} "
              f"best=V{best_index}({best_score:.2f}) "
              f"best_matches={best_view_match_count}{max_match_str} thresh={self.match_threshold:.0f} "
              f"\u2192 {status}  top5:[{top_str}]")
        
        best_view_data = rendered_images[best_index] if 0 <= best_index < len(rendered_images) else {}
        best_image = best_view_data.get('image') if isinstance(best_view_data, dict) else None
        batch_match_caches = getattr(self, "_last_batch_match_caches", None) or []
        best_match_cache = (
            batch_match_caches[best_index]
            if 0 <= best_index < len(batch_match_caches)
            else None
        )
        if best_match_cache is not None:
            self._last_match_cache = best_match_cache

        result = {
            'match_success': match_success,
            'best_score': best_score,
            'best_view_match_count': best_view_match_count,
            'best_match_count': best_match_count,
            'best_index': best_index,
            'best_match_count_index': best_match_count_index,
            'best_pose': rendered_images[best_index].get('pose') if match_success else None,
            'best_image': best_image,
            'best_view_data': best_view_data,
            'best_match_cache': best_match_cache,
            'all_scores': scores,
            'all_match_counts': match_counts,
            'feature_mode': active_mode,
        }
        
        self.logger.info(
            f"Matched {len(rendered_images)} views: "
            f"best_idx={best_index}, evidence={best_score:.2f}, "
            f"best_matches={best_view_match_count}, max_matches={best_match_count}, "
            f"success={match_success}"
        )
        
        return result
    
    def _batch_match_lightglue(self,
                               rendered_images: List[Dict[str, Any]],
                               target_features=None,
                               feature_mode: Optional[str] = None):
        """
        批量 LightGlue 匹配：先批量提取 SuperPoint 特征，再逐个匹配。
        
        SuperPoint 特征提取支持 batch，LightGlue 匹配是 pairwise 的。
        """
        import torch
        
        n = len(rendered_images)
        self._last_batch_match_caches = [None] * n
        if target_features is None:
            target_features, _ = self._get_target_features()
        if target_features is None:
            return [0.0] * n, [0] * n
        
        # Phase 1: 批量预处理图像 -> 统一 tensor
        valid_indices = []
        gray_tensors = []
        for i, img_data in enumerate(rendered_images):
            image = img_data.get('image')
            if image is None:
                continue
            valid_indices.append(i)
            gray_tensors.append(self._numpy_to_torch_image(image))
        
        if not gray_tensors:
            return [0.0] * n, [0] * n
        
        # Phase 2: 批量提取 SuperPoint 特征（逐个但在 no_grad 下紧密循环）
        all_features = [None] * n
        with torch.no_grad():
            for idx, tensor in zip(valid_indices, gray_tensors):
                all_features[idx] = self.extractor.extract(tensor)
        
        # Phase 3: 批量匹配（每个 query 与 target 匹配）
        scores = [0.0] * n
        match_counts = [0] * n
        match_caches = [None] * n
        with torch.no_grad():
            for idx in valid_indices:
                feats = all_features[idx]
                if feats is None:
                    continue
                try:
                    matches = self.matcher({
                        'image0': target_features,
                        'image1': feats
                    })
                    
                    matches_data = matches.get('matches', None)
                    if isinstance(matches_data, list) and len(matches_data) > 0:
                        m = matches_data[0]
                        num_matches = int(m.shape[0]) if torch.is_tensor(m) else 0
                    elif torch.is_tensor(matches_data):
                        num_matches = int(matches_data.shape[0])
                    else:
                        num_matches = 0

                    match_counts[idx] = int(num_matches)
                    
                    if num_matches < 10:
                        continue
                    
                    scores_data = matches.get('scores', None)
                    if isinstance(scores_data, list) and len(scores_data) > 0:
                        s = scores_data[0]
                        if not (torch.is_tensor(s) and s.numel() > 0):
                            s = torch.ones(num_matches, device=self.device)
                    elif torch.is_tensor(scores_data) and scores_data.numel() > 0:
                        s = scores_data
                    else:
                        s = torch.ones(num_matches, device=self.device)

                    match_caches[idx] = self._build_lightglue_match_cache(
                        target_features=target_features,
                        query_features=feats,
                        matches_data=matches_data,
                        scores=s,
                        num_matches=num_matches,
                        feature_mode=feature_mode,
                    )
                    
                    scores[idx] = float(s.sum().item())
                except Exception:
                    scores[idx] = 0.0
                    match_counts[idx] = 0
                    match_caches[idx] = None
        
        self._last_batch_match_caches = match_caches
        return scores, match_counts

    def _build_lightglue_match_cache(
        self,
        target_features,
        query_features,
        matches_data,
        scores,
        num_matches: int,
        feature_mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract matched target/query keypoints and per-match scores."""
        if num_matches <= 0:
            return None
        try:
            import torch

            if isinstance(matches_data, list) and len(matches_data) > 0:
                matches_tensor = matches_data[0]
            else:
                matches_tensor = matches_data
            if not torch.is_tensor(matches_tensor) or matches_tensor.numel() == 0:
                return None

            kpts0 = target_features.get('keypoints', None)
            if isinstance(kpts0, list) and len(kpts0) > 0:
                kpts0 = kpts0[0]
            elif torch.is_tensor(kpts0) and kpts0.dim() == 3:
                kpts0 = kpts0[0]

            kpts1 = query_features.get('keypoints', None)
            if isinstance(kpts1, list) and len(kpts1) > 0:
                kpts1 = kpts1[0]
            elif torch.is_tensor(kpts1) and kpts1.dim() == 3:
                kpts1 = kpts1[0]

            if not torch.is_tensor(kpts0) or not torch.is_tensor(kpts1):
                return None

            idx0 = matches_tensor[:, 0].clamp(0, kpts0.shape[0] - 1).long()
            idx1 = matches_tensor[:, 1].clamp(0, kpts1.shape[0] - 1).long()
            score_array = (
                scores.detach().cpu().numpy().astype(np.float64)
                if torch.is_tensor(scores)
                else np.asarray(scores, dtype=np.float64)
            )
            return {
                'kpts_ref': kpts0[idx0].detach().cpu().numpy(),
                'kpts_query': kpts1[idx1].detach().cpu().numpy(),
                'scores': score_array,
                'num_matches': int(num_matches),
                'query_features': query_features,
                'feature_mode': feature_mode,
            }
        except Exception:
            return None

    def _numpy_to_torch_image(self, image: np.ndarray) -> 'torch.Tensor':
        """
        将numpy图像转换为LightGlue期望的torch tensor格式
        
        Args:
            image: numpy array, shape (H, W, 3), dtype uint8, RGB格式
            
        Returns:
            torch tensor, shape (1, H, W), dtype float32, 灰度图, 归一化到[0, 1]
        """
        import torch
        
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_float = gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(gray_float).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _extract_features(self, image: np.ndarray):
        """提取图像特征（LightGlue版本）"""
        if self.lightglue_available:
            import torch
            
            image_tensor = self._numpy_to_torch_image(image)
            
            with torch.no_grad():
                features = self.extractor.extract(image_tensor)
            
            return features
        else:
            return None
    
    # ==================================================================
    #  匹配计分（核心：密度采样 → 直接用匹配置信度之和）
    # ==================================================================
    
    def _compute_match_score(self,
                             query_image: np.ndarray,
                             feature_mode: Optional[str] = None) -> float:
        """
        计算匹配证据分数。
        
        目标图像关键点可选择均匀采样或语义密度采样。当前默认使用均匀采样。
        
        匹配证据 = Σ(confidence_i)
        
        Args:
            query_image: 查询图像 (RGB格式)
            
        Returns:
            绝对匹配证据分数
        """
        if self.lightglue_available:
            return self._match_with_lightglue(query_image, feature_mode=feature_mode)
        else:
            return self._match_with_orb(query_image)
    
    def _match_with_lightglue(self,
                              image: np.ndarray,
                              feature_mode: Optional[str] = None) -> float:
        """
        LightGlue 匹配，匹配证据为 Σ(confidence)。
        """
        try:
            import torch
            target_features, active_mode = self._get_target_features(feature_mode)
            if target_features is None:
                return 0.0

            with torch.no_grad():
                query_features = self._extract_features(image)
                matches = self.matcher({
                    'image0': target_features,
                    'image1': query_features
                })

            # -------- 获取匹配点索引 --------
            matches_data = matches.get('matches', None)

            if isinstance(matches_data, list) and len(matches_data) > 0:
                m = matches_data[0]
                num_matches = int(m.shape[0]) if torch.is_tensor(m) else 0
            elif torch.is_tensor(matches_data):
                m = matches_data
                num_matches = int(m.shape[0])
            else:
                num_matches = 0

            # ★★★ 修改：预检测阈值，匹配点数少于10视为无效 ★★★
            if num_matches < 10:
                return 0.0

            # -------- 获取置信度 --------
            scores_data = matches.get('scores', None)

            if isinstance(scores_data, list) and len(scores_data) > 0:
                s = scores_data[0]
                if not (torch.is_tensor(s) and s.numel() > 0):
                    s = torch.ones(num_matches, device=self.device)
            elif torch.is_tensor(scores_data) and scores_data.numel() > 0:
                s = scores_data
            else:
                s = torch.ones(num_matches, device=self.device)

            # -------- 匹配证据 = Σ(confidence) --------
            avg_confidence = float(s.mean().item())
            absolute_evidence = float(s.sum().item())

            # -------- 可选：统计匹配点的语义分布（仅用于日志） --------
            n_semantic = 0
            n_background = 0
            # ★★★ Bug fix: 在 if 块外初始化，防止 NameError ★★★
            matched_np = None
            match_indices_0 = None
            kpts0 = None
            H, W = 0, 0

            # ★★★ 无论是否有语义掩码，都计算 matched_np（供 _last_match_cache 使用）★★★
            kpts0 = target_features.get('keypoints', None)
            if isinstance(kpts0, list) and len(kpts0) > 0:
                kpts0 = kpts0[0]
            elif torch.is_tensor(kpts0) and kpts0.dim() == 3:
                kpts0 = kpts0[0]
            if isinstance(matches_data, list) and len(matches_data) > 0:
                match_indices_0 = matches_data[0][:, 0]
            else:
                match_indices_0 = matches_data[:, 0]
            if kpts0 is not None and torch.is_tensor(match_indices_0) and match_indices_0.numel() > 0:
                match_indices_0 = match_indices_0.clamp(0, kpts0.shape[0] - 1)
                matched_kpts0 = kpts0[match_indices_0.long()]
                matched_np = matched_kpts0.cpu().numpy()

            if self.target_semantic_mask is not None:
                H, W = self.target_semantic_mask.shape
                if matched_np is not None:
                    for i in range(num_matches):
                        x, y = matched_np[i]
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= iy < H and 0 <= ix < W and self.target_semantic_mask[iy, ix]:
                            n_semantic += 1
                        else:
                            n_background += 1

            # ★★★ 保存语义匹配数（总数 + 逐实例），供 REF 后决策使用 ★★★
            self._last_semantic_matches = n_semantic
            # 缓存匹配点在目标图像上的坐标（供可视化使用）
            self._last_matched_kpts_on_target = matched_np.copy() if matched_np is not None else None
            # 逐实例统计
            _inst_labels = getattr(self, '_target_instance_labels', None)
            _n_inst = getattr(self, '_n_target_instances', 0)
            if _inst_labels is not None and _n_inst > 0 and kpts0 is not None and torch.is_tensor(match_indices_0):
                per_inst = [0] * _n_inst
                for i in range(num_matches):
                    x, y = matched_np[i]
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= iy < H and 0 <= ix < W:
                        lbl = _inst_labels[iy, ix]
                        if lbl > 0:
                            per_inst[lbl - 1] += 1
                self._last_per_instance_semantic_matches = per_inst
            else:
                self._last_per_instance_semantic_matches = [n_semantic] if n_semantic > 0 else []

            # ★★★ 缓存完整匹配结果供 MR 复用，避免重复匹配 ★★★
            self._last_match_cache = {
                'kpts_ref': matched_np.copy() if matched_np is not None else None,   # 目标图上的匹配点（image0=target）
                'kpts_query': None,   # query图（REF渲染图）上的匹配点，见下方补充
                'scores': s.cpu().numpy().astype(np.float64) if torch.is_tensor(s) else np.array(s),
                'num_matches': num_matches,
                'query_features': query_features,  # REF渲染图的特征（含keypoints）
                'feature_mode': active_mode,
            }
            # 补充：取 query 图（image1）上的匹配点坐标
            try:
                kpts1 = query_features.get('keypoints', None)
                if isinstance(kpts1, list) and len(kpts1) > 0:
                    kpts1 = kpts1[0]
                elif torch.is_tensor(kpts1) and kpts1.dim() == 3:
                    kpts1 = kpts1[0]
                if isinstance(matches_data, list) and len(matches_data) > 0:
                    match_indices_1 = matches_data[0][:, 1]
                else:
                    match_indices_1 = matches_data[:, 1]
                if kpts1 is not None and torch.is_tensor(match_indices_1):
                    match_indices_1 = match_indices_1.clamp(0, kpts1.shape[0] - 1)
                    self._last_match_cache['kpts_query'] = kpts1[match_indices_1.long()].cpu().numpy()
            except Exception:
                pass

            return absolute_evidence

        except Exception as e:
            self.logger.error(f"LightGlue matching error: {e}", exc_info=True)
            return 0.0

    
    def save_closed_loop_decision_visualization(self, save_path: str,
                                                  per_instance_matches: list,
                                                  sem_threshold: int = 20,
                                                  need_closed_loop: bool = True,
                                                  match_score: float = 0.0,
                                                  target_image_override = None):
        if not self.save_visualization_results:
            return
        """
        保存闭环决策可视化图片：目标图像上同时叠加语义分割实例 + 匹配点，
        右上角文字标注统计信息和判断结果。

        Args:
            save_path: 保存路径
            per_instance_matches: 逐实例匹配点数列表
            sem_threshold: 每个实例需要的最小匹配点数（默认20）
            need_closed_loop: 是否需要闭环
            match_score: REF 匹配证据总分
        """
        try:
            import cv2
            import numpy as np

            vis = target_image_override.copy() if target_image_override is not None else self.target_image.copy()
            H, W = vis.shape[:2]

            # --- 1. 叠加逐实例语义分割掩码（不同实例不同颜色） ---
            _inst_labels = getattr(self, '_target_instance_labels', None)
            _n_inst = getattr(self, '_n_target_instances', 0)
            instance_colors = [
                (255, 80, 80), (80, 255, 80), (80, 80, 255),
                (255, 255, 80), (255, 80, 255), (80, 255, 255),
                (255, 160, 80), (160, 80, 255), (80, 160, 255),
                (160, 255, 80), (255, 80, 160), (80, 255, 160),
            ]

            overlay = np.zeros_like(vis)
            if _inst_labels is not None and _n_inst > 0:
                for inst_id in range(1, _n_inst + 1):
                    color = instance_colors[(inst_id - 1) % len(instance_colors)]
                    mask = _inst_labels == inst_id
                    overlay[mask] = color
            elif self.target_semantic_mask is not None:
                overlay[self.target_semantic_mask] = (80, 255, 80)

            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

            # --- 2. 绘制匹配点 ---
            matched_kpts = getattr(self, '_last_matched_kpts_on_target', None)
            if matched_kpts is not None and _inst_labels is not None:
                for i in range(len(matched_kpts)):
                    x, y = matched_kpts[i]
                    ix, iy = int(round(x)), int(round(y))
                    # 判断该点属于哪个实例
                    if 0 <= iy < H and 0 <= ix < W:
                        lbl = _inst_labels[iy, ix]
                        if lbl > 0:
                            color = instance_colors[(lbl - 1) % len(instance_colors)]
                        else:
                            color = (200, 200, 200)  # 背景匹配点灰色
                    else:
                        color = (200, 200, 200)
                    cv2.circle(vis, (ix, iy), 3, color, -1)
                    cv2.circle(vis, (ix, iy), 3, (255, 255, 255), 1)
            elif matched_kpts is not None:
                for i in range(len(matched_kpts)):
                    x, y = matched_kpts[i]
                    ix, iy = int(round(x)), int(round(y))
                    cv2.circle(vis, (ix, iy), 3, (0, 255, 0), -1)
                    cv2.circle(vis, (ix, iy), 3, (255, 255, 255), 1)

            # --- 3. 绘制每个实例的轮廓 ---
            if _inst_labels is not None and _n_inst > 0:
                for inst_id in range(1, _n_inst + 1):
                    mask_uint8 = (_inst_labels == inst_id).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    color = instance_colors[(inst_id - 1) % len(instance_colors)]
                    cv2.drawContours(vis, contours, -1, color, 2)

            # --- 4. 右上角文字 ---
            n_total_instances = len(per_instance_matches)
            n_covered = sum(1 for c in per_instance_matches if c >= sem_threshold)
            decision_str = "SKIP closed loop" if not need_closed_loop else "NEED closed loop"
            decision_color = (0, 255, 0) if not need_closed_loop else (0, 0, 255)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            line_height = 24
            margin = 8

            lines = [
                (f"Score: {match_score:.1f}", (255, 255, 255)),
                (f"Instances: {n_total_instances}", (255, 255, 255)),
                (f"Covered(>={sem_threshold}): {n_covered}/{n_total_instances}", (255, 255, 255)),
            ]
            # 逐实例详情
            for idx, cnt in enumerate(per_instance_matches):
                tag = "Y" if cnt >= sem_threshold else "N"
                c = (0, 255, 0) if cnt >= sem_threshold else (0, 0, 255)
                lines.append((f"  Inst {idx}: {cnt} pts [{tag}]", c))
            lines.append((decision_str, decision_color))

            # 计算文字区域宽度（估算最宽行）
            max_text_w = 0
            for text, _ in lines:
                (tw, _th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                if tw > max_text_w:
                    max_text_w = tw
            box_w = max_text_w + margin * 2
            box_h = len(lines) * line_height + margin * 2

            # 半透明背景框
            x1 = W - box_w - margin
            y1 = margin
            sub = vis[y1:y1 + box_h, x1:x1 + box_w].copy()
            cv2.rectangle(vis, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
            vis[y1:y1 + box_h, x1:x1 + box_w] = cv2.addWeighted(
                vis[y1:y1 + box_h, x1:x1 + box_w], 0.7, sub, 0.3, 0)

            for i, (text, color) in enumerate(lines):
                ty = y1 + margin + (i + 1) * line_height
                cv2.putText(vis, text, (x1 + margin, ty), font, font_scale, color, thickness)

            # --- 5. 保存 ---
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_bgr)
            print(f"[FEM] Saved closed-loop decision visualization -> {save_path}")

        except Exception as e:
            self.logger.warning(f"save_closed_loop_decision_visualization failed: {e}")
            import traceback; traceback.print_exc()

    def _match_with_orb(self, image: np.ndarray) -> float:
        """
        使用ORB进行匹配（备选方案）
        
        ORB 无法方便地做密度采样，直接使用匹配点数作为证据。
        """
        try:
            if image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            kp, desc = self.extractor.detectAndCompute(gray, None)
            
            if desc is None or len(kp) == 0:
                self.logger.debug("ORB: No keypoints detected")
                return 0.0
            
            matches = self.matcher_orb.knnMatch(self.target_desc, desc, k=2)
            
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) == 0:
                return 0.0
            
            score = float(len(good_matches))
            
            return score
            
        except Exception as e:
            self.logger.error(f"ORB matching error: {e}", exc_info=True)
            return 0.0
    
    def save_best_match(
        self,
        image: np.ndarray,
        index: int,
        score: Optional[float] = None,
        match_count: Optional[int] = None,
        source: Optional[str] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        match_cache: Optional[Dict[str, Any]] = None,
        save_subdir: str = "NVSM",
    ) -> Optional[str]:
        """保存最佳匹配图像"""
        if not self.save_visualization_results or self.results_dir is None:
            return None
        try:
            safe_subdir = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in str(save_subdir or "NVSM")
            )
            save_dir = self.results_dir / safe_subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            source_name = str(source or f"view{index}")
            source_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in source_name)
            score_value = float(score or 0.0)
            step_part = f"step_{int(step):06d}_" if step is not None else ""
            filename_base = (
                f"best_match_{step_part}score_{score_value:07.2f}_{source_name}_{timestamp}"
            )
            filename = save_dir / f"{filename_base}.png"
            
            if image.ndim == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            cv2.imwrite(str(filename), image_bgr)
            match_viz_path = self._save_lightglue_match_visualization(
                save_dir=save_dir,
                filename_base=filename_base,
                query_image=image,
                match_cache=match_cache,
            )
            meta = {
                "stage": "NVSM",
                "image_path": str(filename),
                "match_visualization_path": match_viz_path,
                "step": int(step) if step is not None else None,
                "best_view_index": int(index),
                "score": score_value,
                "match_count": int(match_count or 0),
                "source": source_name,
            }
            if metadata:
                meta.update(metadata)
            with open(save_dir / f"{filename_base}.json", "w") as f:
                json.dump(_json_safe(meta), f, indent=2)
            self.logger.info(f"Saved best match to {filename}")
            return str(filename)
            
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}", exc_info=True)
            return None

    def _save_lightglue_match_visualization(
        self,
        save_dir: Path,
        filename_base: str,
        query_image: np.ndarray,
        match_cache: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Save target/render match overlay colored by per-match LightGlue score."""
        try:
            cache = match_cache or getattr(self, "_last_match_cache", None)
            if not isinstance(cache, dict):
                return None
            kpts_ref = cache.get('kpts_ref')
            kpts_query = cache.get('kpts_query')
            scores = cache.get('scores')
            if kpts_ref is None or kpts_query is None:
                return None
            kpts_ref = np.asarray(kpts_ref, dtype=np.float32)
            kpts_query = np.asarray(kpts_query, dtype=np.float32)
            scores = np.asarray(scores if scores is not None else [], dtype=np.float32)
            n = min(len(kpts_ref), len(kpts_query), len(scores))
            if n <= 0:
                return None
            kpts_ref = kpts_ref[:n]
            kpts_query = kpts_query[:n]
            scores = scores[:n]

            target = self._ensure_rgb_uint8(self.target_image)
            query = self._ensure_rgb_uint8(query_image)
            if target is None or query is None:
                return None

            target_resized, ref_scale = self._resize_for_match_panel(target)
            query_resized, query_scale = self._resize_for_match_panel(query)
            if target_resized.shape[0] != query_resized.shape[0]:
                h = min(target_resized.shape[0], query_resized.shape[0])
                target_resized = cv2.resize(
                    target_resized,
                    (int(round(target_resized.shape[1] * h / target_resized.shape[0])), h),
                    interpolation=cv2.INTER_AREA,
                )
                query_resized = cv2.resize(
                    query_resized,
                    (int(round(query_resized.shape[1] * h / query_resized.shape[0])), h),
                    interpolation=cv2.INTER_AREA,
                )
                ref_scale = target_resized.shape[1] / max(1.0, float(target.shape[1]))
                query_scale = query_resized.shape[1] / max(1.0, float(query.shape[1]))

            gap = 16
            h = max(target_resized.shape[0], query_resized.shape[0])
            w0 = target_resized.shape[1]
            w1 = query_resized.shape[1]
            canvas = np.zeros((h, w0 + gap + w1, 3), dtype=np.uint8)
            canvas[:target_resized.shape[0], :w0] = target_resized
            canvas[:query_resized.shape[0], w0 + gap:w0 + gap + w1] = query_resized

            score_min = float(np.min(scores))
            score_max = float(np.max(scores))
            denom = max(1e-6, score_max - score_min)
            order = np.argsort(scores)
            for idx in order:
                conf = float((scores[idx] - score_min) / denom)
                color = self._score_to_rgb(conf)
                p0 = (
                    int(round(kpts_ref[idx, 0] * ref_scale)),
                    int(round(kpts_ref[idx, 1] * ref_scale)),
                )
                p1 = (
                    int(round(w0 + gap + kpts_query[idx, 0] * query_scale)),
                    int(round(kpts_query[idx, 1] * query_scale)),
                )
                cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
                radius = 2 + int(round(2 * conf))
                cv2.circle(canvas, p0, radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, p1, radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, p0, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, p1, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

            label = f"LightGlue matches: {n}  score_sum: {float(np.sum(scores)):.2f}"
            cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 560), 28), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            out_path = save_dir / f"{filename_base}_matches.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            return str(out_path)
        except Exception as exc:
            self.logger.debug("Failed to save LightGlue match visualization: %s", exc)
            return None

    @staticmethod
    def _ensure_rgb_uint8(image: np.ndarray) -> Optional[np.ndarray]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        return arr.copy()

    @staticmethod
    def _resize_for_match_panel(image: np.ndarray, max_height: int = 360):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image, 1.0
        scale = min(1.0, float(max_height) / float(h))
        if scale >= 0.999:
            return image.copy(), 1.0
        resized = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _score_to_rgb(value: float):
        value = float(np.clip(value, 0.0, 1.0))
        color_bgr = cv2.applyColorMap(
            np.array([[int(round(value * 255))]], dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )[0, 0]
        return tuple(int(c) for c in color_bgr[::-1])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
