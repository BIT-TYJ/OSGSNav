"""
PosePreRefiner (PRE) - 前置位姿优化模块

在第一次FEM粗定位之后、REF细定位之前运行。
通过多阶段采样+LightGlue匹配来修正粗定位位姿的误差。

四阶段流程：
  (1) 大幅修正 Z轴 + Pitch：在粗定位位姿上下各4.0m内采样80个点，
      每个点采样12个pitch角度，共960张渲染图。
  (2) 第一次小幅修正全部自由度（含pitch/yaw）：
      在上步最优点周围1×1×1m立方体内采样100个点，
      每点3种pitch微扰×3种yaw微扰 = 9种，共900张渲染图。
  (3) 第二次小幅修正全部自由度（含pitch/yaw）：
      在上步最优点周围1×1×1m立方体内再次采样100个点，
      每点3种pitch微扰×3种yaw微扰 = 9种，共900张渲染图。
  (4) 精细确定输出：
      在上步最优点周围0.1×0.1×0.1m立方体内采样10个点，
      每点5种pitch微扰×5种yaw微扰 = 25种，共250张渲染图。
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List


class PosePreRefiner:
    """前置位姿优化模块 (PRE)"""

    def __init__(self, slam_mapper, feature_matcher, results_dir: Optional[str] = None,
                 save_visualization_results: bool = True):
        """
        Args:
            slam_mapper: SLAMMapper实例，提供高斯地图和渲染器
            feature_matcher: FeatureMatcher实例（已初始化目标图像），用于匹配打分
            results_dir: 可选，保存调试图像的路径
        """
        self.logger = logging.getLogger(__name__)
        self.slam_mapper = slam_mapper
        self.fem = feature_matcher
        self.results_dir = results_dir
        self.save_visualization_results = bool(save_visualization_results)

        self.logger.info("PosePreRefiner (PRE) initialized")

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------
    def refine(self, coarse_pose: np.ndarray, coarse_score: float) -> Dict[str, Any]:
        """
        对粗定位位姿进行前置优化。

        Args:
            coarse_pose: FEM得到的粗定位w2c位姿 (4×4)
            coarse_score: FEM得到的粗定位匹配证据分数

        Returns:
            {
                'success': bool,
                'refined_pose': np.ndarray (4×4, w2c),
                'refined_score': float,
                'coarse_score': float,
                'stage_info': list,   # 每阶段信息
                'total_renders': int, # 总渲染张数
                'message': str,
            }
        """
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"[PRE] ★ PosePreRefiner Started ★")
        print(f"[PRE] Coarse score: {coarse_score:.2f}")
        print(f"{'='*70}")

        # 提取粗定位的 c2w 信息
        c2w = np.linalg.inv(coarse_pose)
        pos = c2w[:3, 3].copy()
        # 从 c2w 提取 yaw / pitch / roll
        yaw, pitch, roll = self._c2w_to_euler(c2w)

        print(f"[PRE] Coarse position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"[PRE] Coarse euler (deg): yaw={np.rad2deg(yaw):.1f}, pitch={np.rad2deg(pitch):.1f}, roll={np.rad2deg(roll):.1f}")

        best_pose = coarse_pose.copy()
        best_score = coarse_score
        best_pos = pos.copy()
        best_yaw = yaw
        best_pitch = pitch
        best_roll = roll

        stage_info = []
        total_renders = 0

        # ===================== 阶段1: 大幅修正 Z + Pitch =====================
        print(f"\n{'-'*60}")
        print(f"[PRE] Stage 1: Z-axis + Pitch correction (±4.0m, 80 z-samples × 12 pitch-samples = 960 renders)")
        print(f"{'-'*60}")

        stage1_results, n1 = self._stage1_z_pitch(pos, yaw, pitch, roll)
        total_renders += n1

        if stage1_results and stage1_results[0]['score'] > best_score:
            best_score = stage1_results[0]['score']
            best_pose = stage1_results[0]['pose']
            c2w_best = np.linalg.inv(best_pose)
            best_pos = c2w_best[:3, 3].copy()
            best_yaw, best_pitch, best_roll = self._c2w_to_euler(c2w_best)
            print(f"[PRE] Stage 1 improved: score {coarse_score:.2f} → {best_score:.2f}")
            print(f"[PRE]   New pos: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")
        else:
            print(f"[PRE] Stage 1: no improvement, keeping coarse pose")

        stage_info.append({
            'stage': 1, 'renders': n1, 'best_score': best_score,
            'best_pos': best_pos.tolist()
        })

        # ===================== 阶段2: 小幅修正全自由度 (1m cube) =====================
        print(f"\n{'-'*60}")
        print(f"[PRE] Stage 2: All-DOF correction (1×1×1m cube, 100 pts × 9 angle perturbations = 900 renders)")
        print(f"{'-'*60}")

        stage2_results, n2 = self._stage2_cube_perturb(
            best_pos, best_yaw, best_pitch, best_roll,
            cube_half=0.5, n_samples=99,
            angle_offsets_deg=[-10.0, 0.0, 10.0]
        )
        total_renders += n2

        if stage2_results and stage2_results[0]['score'] > best_score:
            best_score = stage2_results[0]['score']
            best_pose = stage2_results[0]['pose']
            c2w_best = np.linalg.inv(best_pose)
            best_pos = c2w_best[:3, 3].copy()
            best_yaw, best_pitch, best_roll = self._c2w_to_euler(c2w_best)
            print(f"[PRE] Stage 2 improved: → {best_score:.2f}")
            print(f"[PRE]   New pos: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")
        else:
            print(f"[PRE] Stage 2: no improvement")

        stage_info.append({
            'stage': 2, 'renders': n2, 'best_score': best_score,
            'best_pos': best_pos.tolist()
        })

        # ===================== 阶段2b: 第二次小幅修正全自由度 (1m cube) =====================
        print(f"\n{'-'*60}")
        print(f"[PRE] Stage 2b: All-DOF correction round 2 (1×1×1m cube, 100 pts × 9 angle perturbations = 900 renders)")
        print(f"{'-'*60}")

        stage2b_results, n2b = self._stage2_cube_perturb(
            best_pos, best_yaw, best_pitch, best_roll,
            cube_half=0.5, n_samples=99,
            angle_offsets_deg=[-10.0, 0.0, 10.0]
        )
        total_renders += n2b

        if stage2b_results and stage2b_results[0]['score'] > best_score:
            best_score = stage2b_results[0]['score']
            best_pose = stage2b_results[0]['pose']
            c2w_best = np.linalg.inv(best_pose)
            best_pos = c2w_best[:3, 3].copy()
            best_yaw, best_pitch, best_roll = self._c2w_to_euler(c2w_best)
            print(f"[PRE] Stage 2b improved: → {best_score:.2f}")
            print(f"[PRE]   New pos: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")
        else:
            print(f"[PRE] Stage 2b: no improvement")

        stage_info.append({
            'stage': '2b', 'renders': n2b, 'best_score': best_score,
            'best_pos': best_pos.tolist()
        })

        # ===================== 阶段3: 精细修正 (0.1m cube) =====================
        print(f"\n{'-'*60}")
        print(f"[PRE] Stage 3: Fine correction (0.1×0.1×0.1m cube, 10 pts × 25 angle perturbations = 250 renders)")
        print(f"{'-'*60}")

        stage3_results, n3 = self._stage3_fine(
            best_pos, best_yaw, best_pitch, best_roll,
            cube_half=0.05, n_samples=9,
            angle_offsets_deg=[-5.0, -2.5, 0.0, 2.5, 5.0]
        )
        total_renders += n3

        if stage3_results and stage3_results[0]['score'] > best_score:
            best_score = stage3_results[0]['score']
            best_pose = stage3_results[0]['pose']
            c2w_best = np.linalg.inv(best_pose)
            best_pos = c2w_best[:3, 3].copy()
            best_yaw, best_pitch, best_roll = self._c2w_to_euler(c2w_best)
            print(f"[PRE] Stage 3 improved: → {best_score:.2f}")
            print(f"[PRE]   Final pos: [{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]")
        else:
            print(f"[PRE] Stage 3: no improvement")

        stage_info.append({
            'stage': 3, 'renders': n3, 'best_score': best_score,
            'best_pos': best_pos.tolist()
        })

        elapsed = time.time() - t0
        improved = best_score > coarse_score

        c2w_final = np.linalg.inv(best_pose)
        final_pos = c2w_final[:3, 3]
        coarse_c2w = np.linalg.inv(coarse_pose)
        coarse_pos = coarse_c2w[:3, 3]
        delta = np.linalg.norm(final_pos - coarse_pos)

        print(f"\n{'='*70}")
        print(f"[PRE] ★ PosePreRefiner Finished ★")
        print(f"[PRE] Score: {coarse_score:.2f} → {best_score:.2f} ({'improved' if improved else 'no change'})")
        print(f"[PRE] Position shift: {delta:.3f}m")
        print(f"[PRE] Total renders: {total_renders}, Time: {elapsed:.1f}s")
        print(f"{'='*70}\n")

        # ★★★ 新增：保存PRE优化后位姿的渲染图 ★★★
        self._save_refined_render(best_pose, coarse_pose, best_score, coarse_score)

        return {
            'success': True,
            'refined_pose': best_pose,
            'refined_score': best_score,
            'coarse_score': coarse_score,
            'improved': improved,
            'stage_info': stage_info,
            'total_renders': total_renders,
            'elapsed': elapsed,
            'message': f"PRE complete: {coarse_score:.2f}→{best_score:.2f}, Δd={delta:.3f}m, {total_renders} renders in {elapsed:.1f}s"
        }

    # ------------------------------------------------------------------
    #  阶段1: Z轴 + Pitch 修正
    # ------------------------------------------------------------------
    def _stage1_z_pitch(self, pos: np.ndarray, yaw: float, pitch: float, roll: float):
        """
        大幅修正Z轴和Pitch：
        - x, yaw, roll 保持不变
        - z_camera（即 habitat 的 Y 轴）：在 pos[1] ± 4.0m 内均匀采样80个点
        - 比原 z 高的点采样俯视 pitch 0~-60°（12个角度）
        - 比原 z 低的点采样仰视 pitch 0~+60°（12个角度）
        """
        original_y = pos[1]  # habitat Y = 垂直轴
        z_offsets = np.linspace(-4.0, 4.0, 82)  # 82个点，去除0
        z_offsets = z_offsets[z_offsets != 0.0]
        if len(z_offsets) > 80:
            z_offsets = z_offsets[:80]

        candidates = []
        n_renders = 0

        for z_off in z_offsets:
            new_y = original_y + z_off

            # 根据偏移方向选择 pitch 采样范围
            if z_off > 0:
                # 比原位置高 → 俯视
                pitch_angles = np.linspace(0, np.deg2rad(-60), 12, endpoint=True)
            else:
                # 比原位置低 → 仰视
                pitch_angles = np.linspace(0, np.deg2rad(60), 12, endpoint=True)

            for p in pitch_angles:
                new_pos = np.array([pos[0], new_y, pos[2]])
                new_pitch = pitch + p

                w2c = self._euler_pos_to_w2c(new_pos, yaw, new_pitch, roll)
                image = self._render_view(w2c)
                n_renders += 1

                if image is None:
                    continue

                score = self.fem._compute_match_score(image, feature_mode="uniform")
                candidates.append({'pose': w2c, 'score': score, 'pos': new_pos.copy()})

                if n_renders % 100 == 0:
                    print(f"[PRE S1] Rendered {n_renders}/600 ...")

        # 按分数降序排列
        candidates.sort(key=lambda x: x['score'], reverse=True)
        print(f"[PRE S1] Total rendered: {n_renders}, candidates with valid render: {len(candidates)}")
        if candidates:
            top3 = [f"{c['score']:.2f}" for c in candidates[:3]]
            print(f"[PRE S1] Top-3 scores: {top3}")

        return candidates, n_renders

    # ------------------------------------------------------------------
    #  阶段2: 1×1×1m 立方体 + pitch/yaw 微扰
    # ------------------------------------------------------------------
    def _stage2_cube_perturb(self, center_pos, yaw, pitch, roll,
                             cube_half=0.5, n_samples=99,
                             angle_offsets_deg=None):
        """
        在 center_pos 周围 1×1×1m 立方体内均匀随机采样 n_samples 个点，
        加上 center_pos 本身共 n_samples+1 个点。
        每个点对 pitch 和 yaw 各做 angle_offsets_deg 微扰（不含 roll）。
        """
        if angle_offsets_deg is None:
            angle_offsets_deg = [-10.0, 0.0, 10.0]

        rng = np.random.default_rng(42)
        # 均匀采样 n_samples 个点
        offsets = rng.uniform(-cube_half, cube_half, size=(n_samples, 3))
        # 加入中心点
        all_positions = np.vstack([center_pos.reshape(1, 3), center_pos + offsets])

        angle_offsets_rad = [np.deg2rad(a) for a in angle_offsets_deg]
        n_angle_combos = len(angle_offsets_rad) ** 2  # pitch × yaw

        candidates = []
        n_renders = 0
        expected_total = len(all_positions) * n_angle_combos

        for pt in all_positions:
            for dp_rad in angle_offsets_rad:
                for dy_rad in angle_offsets_rad:
                    new_pitch = pitch + dp_rad
                    new_yaw = yaw + dy_rad

                    w2c = self._euler_pos_to_w2c(pt, new_yaw, new_pitch, roll)
                    image = self._render_view(w2c)
                    n_renders += 1

                    if image is None:
                        continue

                    score = self.fem._compute_match_score(image, feature_mode="uniform")
                    candidates.append({'pose': w2c, 'score': score, 'pos': pt.copy()})

                    if n_renders % 200 == 0:
                        print(f"[PRE S2] Rendered {n_renders}/{expected_total} ...")

        candidates.sort(key=lambda x: x['score'], reverse=True)
        print(f"[PRE S2] Total rendered: {n_renders}, valid: {len(candidates)}")
        if candidates:
            top3 = [f"{c['score']:.2f}" for c in candidates[:3]]
            print(f"[PRE S2] Top-3 scores: {top3}")

        return candidates, n_renders

    # ------------------------------------------------------------------
    #  阶段3: 0.1×0.1×0.1m 立方体 精细优化
    # ------------------------------------------------------------------
    def _stage3_fine(self, center_pos, yaw, pitch, roll,
                     cube_half=0.05, n_samples=9,
                     angle_offsets_deg=None):
        """
        在 center_pos 周围 0.1×0.1×0.1m 内采样 n_samples 个点 + center，
        每点做 5×5 = 25 种 pitch/yaw 微扰。
        """
        if angle_offsets_deg is None:
            angle_offsets_deg = [-5.0, -2.5, 0.0, 2.5, 5.0]

        rng = np.random.default_rng(123)
        offsets = rng.uniform(-cube_half, cube_half, size=(n_samples, 3))
        all_positions = np.vstack([center_pos.reshape(1, 3), center_pos + offsets])

        angle_offsets_rad = [np.deg2rad(a) for a in angle_offsets_deg]
        n_angle_combos = len(angle_offsets_rad) ** 2

        candidates = []
        n_renders = 0
        expected_total = len(all_positions) * n_angle_combos

        for pt in all_positions:
            for dp_rad in angle_offsets_rad:
                for dy_rad in angle_offsets_rad:
                    new_pitch = pitch + dp_rad
                    new_yaw = yaw + dy_rad

                    w2c = self._euler_pos_to_w2c(pt, new_yaw, new_pitch, roll)
                    image = self._render_view(w2c)
                    n_renders += 1

                    if image is None:
                        continue

                    score = self.fem._compute_match_score(image, feature_mode="uniform")
                    candidates.append({'pose': w2c, 'score': score, 'pos': pt.copy()})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        print(f"[PRE S3] Total rendered: {n_renders}, valid: {len(candidates)}")
        if candidates:
            top3 = [f"{c['score']:.2f}" for c in candidates[:3]]
            print(f"[PRE S3] Top-3 scores: {top3}")

        return candidates, n_renders

    # ------------------------------------------------------------------
    #  保存PRE优化后的渲染图
    # ------------------------------------------------------------------
    def _save_refined_render(self, refined_pose: np.ndarray, coarse_pose: np.ndarray,
                             refined_score: float, coarse_score: float):
        """
        ★★★ 新增：渲染并保存PRE优化前后的位姿渲染图 ★★★
        保存到 results/habitat_imagenav_日期_时间/PRE/ 下
        """
        if not self.save_visualization_results:
            return
        import os
        import cv2
        from datetime import datetime

        try:
            # 确定保存目录
            # ★★★ 修复：不再使用 sorted(subdirs)[-1] 回退，防止多进程保存错乱 ★★★
            if self.results_dir is not None:
                save_dir = os.path.join(str(self.results_dir), "PRE")
            else:
                # 使用进程ID创建隔离目录，避免多进程冲突
                pid = os.getpid()
                save_dir = os.path.join("output", f"pid_{pid}", "PRE")
                self.logger.warning(
                    f"results_dir not set, falling back to process-isolated dir: {save_dir}. "
                    f"Call update_results_dir() to fix."
                )

            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 渲染优化后的位姿
            refined_image = self._render_view(refined_pose)
            # 渲染优化前的位姿（用于对比）
            coarse_image = self._render_view(coarse_pose)

            if refined_image is not None:
                # 保存优化后渲染图
                refined_bgr = cv2.cvtColor(refined_image, cv2.COLOR_RGB2BGR) if refined_image.shape[2] == 3 else refined_image
                c2w = np.linalg.inv(refined_pose)
                pos = c2w[:3, 3]
                refined_path = os.path.join(save_dir, f"pre_refined_{timestamp}.png")
                cv2.imwrite(refined_path, refined_bgr)

                # 如果有优化前的图，拼接对比图
                if coarse_image is not None:
                    coarse_bgr = cv2.cvtColor(coarse_image, cv2.COLOR_RGB2BGR) if coarse_image.shape[2] == 3 else coarse_image
                    c2w_c = np.linalg.inv(coarse_pose)
                    pos_c = c2w_c[:3, 3]

                    h, w = refined_bgr.shape[:2]
                    # 添加标注文字
                    coarse_labeled = coarse_bgr.copy()
                    refined_labeled = refined_bgr.copy()
                    cv2.putText(coarse_labeled, f"BEFORE PRE (score={coarse_score:.1f})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(coarse_labeled, f"pos=[{pos_c[0]:.2f},{pos_c[1]:.2f},{pos_c[2]:.2f}]", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(refined_labeled, f"AFTER PRE (score={refined_score:.1f})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(refined_labeled, f"pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    comparison = np.hstack([coarse_labeled, refined_labeled])
                    comp_path = os.path.join(save_dir, f"pre_comparison_{timestamp}.png")
                    cv2.imwrite(comp_path, comparison)

                print(f"[PRE] Saved refined render to: {save_dir}")
            else:
                print(f"[PRE] Warning: Could not render refined pose for saving")

        except Exception as e:
            print(f"[PRE] Warning: Failed to save refined render: {e}")

    # ------------------------------------------------------------------
    #  辅助方法：渲染
    # ------------------------------------------------------------------
    def _render_view(self, w2c: np.ndarray) -> Optional[np.ndarray]:
        """
        从 w2c 位姿渲染一帧（复用 NVSM 的底层渲染逻辑）。

        Returns:
            RGB numpy (H, W, 3) uint8，渲染失败返回 None
        """
        try:
            import torch

            slam = getattr(self.slam_mapper, 'slam', None)
            if slam is None:
                return None
            if not hasattr(slam, 'create_camera') or not hasattr(slam, 'render_frame'):
                return None

            height = getattr(slam, 'height', 480)
            width = getattr(slam, 'width', 640)
            dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            dummy_depth = np.zeros((height, width), dtype=np.float32)

            virtual_cam = slam.create_camera(dummy_rgb, dummy_depth, w2c, uid=-9999)
            render_result = slam.render_frame(virtual_cam)

            if render_result is not None:
                if isinstance(render_result, dict) and 'rendered' in render_result:
                    return render_result['rendered']
                elif isinstance(render_result, np.ndarray):
                    return render_result

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  辅助方法：坐标 / 位姿转换
    # ------------------------------------------------------------------
    @staticmethod
    def _c2w_to_euler(c2w: np.ndarray):
        """
        从 c2w_slam (4×4) 提取 yaw, pitch, roll (弧度)。

        输入是 c2w_slam = c2w_habitat @ flip_mat，其中 flip_mat = diag(1,-1,-1,1)。
        因此列的含义为：
          col0 = right_habitat（不变）
          col1 = -up_habitat （Y列被 flip 取反）
          col2 = +forward_habitat（Z列被 flip 取反，-forward 变 +forward）
          col3 = position （不变）
        """
        # ★★★ 修复：c2w 实际是 c2w_slam = c2w_habitat @ flip_mat ★★★
        # flip_mat = diag(1,-1,-1,1) 使得：
        #   col0 = right_habitat (不变)
        #   col1 = -up_habitat   (Y列取反)
        #   col2 = +forward_habitat (Z列取反后 -forward 变成 +forward)
        # 所以：forward = col2 (不需要取反)，up = -col1
        forward = c2w[:3, 2]    # c2w_slam Z列 = forward（直接使用）
        up_camera = -c2w[:3, 1]  # c2w_slam Y列 = -up，取反得到真正的 up

        # yaw: 在 XZ 平面上 forward 的朝向角
        yaw = np.arctan2(forward[0], -forward[2])

        # pitch: forward 在 Y 方向上的仰角
        horiz = np.sqrt(forward[0]**2 + forward[2]**2)
        pitch = np.arctan2(forward[1], horiz)

        # roll: 简化估计 — up_camera 在世界 Y 方向上的偏差
        # 当 roll=0 时，up_camera 应与世界 up (0,1,0) 在归一化后对齐
        right = c2w[:3, 0]
        roll = np.arctan2(-right[1], up_camera[1])

        return yaw, pitch, roll

    @staticmethod
    def _euler_pos_to_w2c(pos: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        从 (position, yaw, pitch, roll) 构造 w2c 矩阵。

        步骤：
          1. 构建 c2w：相机位于 pos，朝向由 yaw/pitch/roll 决定
          2. 施加 flip_mat
          3. 取逆得 w2c
        """
        # 相机前方: forward 方向
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        # forward 向量 (在 habitat 世界中)
        forward = np.array([
            sy * cp,
            sp,
            -cy * cp
        ])
        forward /= (np.linalg.norm(forward) + 1e-12)

        # 世界 up
        world_up = np.array([0.0, 1.0, 0.0])

        # right = forward × up
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn

        # 真正的 up = right × forward
        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-12)

        # 应用 roll: 绕 forward 旋转 right 和 up
        if abs(roll) > 1e-6:
            right_new = cr * right + sr * up
            up_new = -sr * right + cr * up
            right = right_new
            up = up_new

        # 构建 c2w（habitat 约定）
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # 注意：c2w 的Z列 = -forward
        c2w[:3, 3] = pos

        # flip_mat: habitat → SLAM 坐标
        flip_mat = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float64)

        c2w_correct = c2w @ flip_mat
        w2c = np.linalg.inv(c2w_correct).astype(np.float32)
        return w2c

    # ------------------------------------------------------------------
    #  重置
    # ------------------------------------------------------------------
    def reset(self):
        """重置模块状态（当前无持久状态需要重置）"""
        pass
