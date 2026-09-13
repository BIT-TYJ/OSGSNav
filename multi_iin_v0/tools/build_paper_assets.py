#!/usr/bin/env python3
"""Assemble reproducible paper-ready qualitative assets from saved runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "论文实验素材补充_20260913"
LATEST = ROOT / "multi_iin_v0/results/experiments/full100_continuous_shared_latest_20260911-差一些/ep087_gpu7"
EP087 = next(LATEST.glob("compact/**/episode_00001__teesavr23of__miin_v0_val_000087"))
T1 = next(EP087.glob("target_01*"))
T2 = next(EP087.glob("target_02*"))
COLORED = ROOT / "multi_iin_v0/results/semantic_instance_visualizations/ep087_target2_shared_final"
POSE_CHAIN = ROOT / "任意视角匹配案例_5组/论文定位方法位姿链_episode_0994"
SECONDARY = ROOT / "仿真实验关键节点_Gaussian原位渲染/02_二次探索成功_Episode915"
RAW915 = Path("/nas_ds925/IEVE/results/habitat_914_20260708_181234_yr17PDCnDDW")


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size=(520, 520), bg="white") -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def montage(items, output: Path, cols: int, panel=(520, 520), note: str | None = None):
    title_h, gap, note_h = 62, 18, (76 if note else 0)
    rows = (len(items) + cols - 1) // cols
    width = cols * panel[0] + (cols + 1) * gap
    height = rows * (panel[1] + title_h) + (rows + 1) * gap + note_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(items):
        row, col = divmod(index, cols)
        x = gap + col * (panel[0] + gap)
        y = gap + row * (panel[1] + title_h)
        canvas.paste(fit_image(Path(path), panel), (x, y + title_h))
        draw.text((x + panel[0] / 2, y + title_h / 2), label,
                  fill=(20, 20, 20), font=font(25, True), anchor="mm")
    if note:
        draw.rounded_rectangle((gap, height - note_h + 8, width - gap, height - 10),
                               radius=14, fill=(239, 246, 255), outline=(53, 112, 190), width=2)
        draw.text((width / 2, height - note_h / 2 + 4), note, fill=(22, 66, 120),
                  font=font(23, True), anchor="mm")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def render_gaussian_overview(output_dir: Path):
    paths = [
        T2 / "03_gaussian_map/semantic_instance_gaussians.ply",
        COLORED / "gaussians_semantic_colored.ply",
        COLORED / "gaussians_instance_colored.ply",
    ]
    clouds = []
    rng = np.random.default_rng(87)
    for path in paths:
        vertices = np.asarray(PlyData.read(str(path))["vertex"].data)
        xyz = np.column_stack([vertices["x"], vertices["z"], vertices["y"]])
        rgb = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]) / 255.0
        semantic = np.asarray(vertices["semantic_class"], dtype=np.int32)
        labelled = semantic >= 0
        background = np.flatnonzero(~labelled)
        chosen_bg = rng.choice(background, min(60000, len(background)), replace=False)
        chosen = np.concatenate([chosen_bg, np.flatnonzero(labelled)])
        clouds.append((xyz[chosen], rgb[chosen], labelled[chosen]))

    all_xyz = clouds[0][0]
    low, high = np.percentile(all_xyz, [1, 99], axis=0)
    fig = plt.figure(figsize=(16.2, 5.5), dpi=180, facecolor="white")
    titles = ["RGB Gaussian map", "Semantic attributes", "Instance attributes"]
    for i, ((xyz, rgb, labelled), title) in enumerate(zip(clouds, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        order = np.argsort(labelled)
        xyz, rgb, labelled = xyz[order], rgb[order], labelled[order]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=np.where(labelled, 0.8, 0.18),
                   alpha=0.82, linewidths=0, rasterized=True)
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
        ax.set_title(title, fontsize=15, weight="bold", pad=8)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("height")
        ax.view_init(elev=33, azim=-58)
        ax.set_box_aspect((high - low) / max(high - low))
        ax.grid(False)
        if i == 2:
            labels = ["chair", "sofa", "plant", "bed", "toilet", "tv"]
            colors = [(220, 20, 60), (255, 140, 0), (34, 139, 34),
                      (65, 105, 225), (0, 206, 209), (186, 85, 211)]
            handles = [Patch(color=np.asarray(c) / 255.0, label=n)
                       for n, c in zip(labels, colors)]
            ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9,
                      ncol=2, borderpad=0.4, handlelength=1.0)
    fig.suptitle("Persistent semantic-instance 3D Gaussian map (Episode 087, after target 2)",
                 fontsize=18, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "Figure_semantic_instance_gaussian_overview.png", bbox_inches="tight")
    plt.close(fig)


def build_semantic_instance():
    dest = OUT / "01_语义实例Gaussian地图_ep087"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("gaussians_semantic_colored.ply", "gaussians_instance_colored.ply", "color_legend.json"):
        copy(COLORED / name, dest / name)
    copy(T2 / "03_gaussian_map/semantic_instance_gaussians.ply", dest / "gaussians_rgb_with_attributes.ply")
    render_gaussian_overview(dest)


def build_multitarget_reuse():
    dest = OUT / "02_target1到target2共享地图_ep087"
    items = [
        (T1 / "01_goal/target_image.png", "Target 1: chair"),
        (T1 / "02_map/final_2d_map.png", "After target 1 (65 steps)"),
        (T1 / "05_final_match/target_render_matches.png", "Target 1 verified"),
        (T2 / "01_goal/target_image.png", "Target 2: plant"),
        (T2 / "02_map/final_2d_map.png", "Shared map continues (56 steps)"),
        (T2 / "05_final_match/target_render_matches.png", "Target 2 verified"),
    ]
    montage(items, dest / "Figure_target1_target2_shared_memory.png", cols=3,
            note="Same mapper: 385,659 Gaussians | boundary pose delta: 0.000 m | target 2 prevalidated at step 0")
    for path, label in items:
        copy(path, dest / (label.lower().replace(" ", "_").replace(":", "") + ".png"))
    evidence = {
        "episode": "miin_v0_val_000087", "scene": "TEEsavR23oF", "ordered_success": 1,
        "target1": {"category": "chair", "steps": 65, "sr": 1, "spl": 0.8120719397337866},
        "target2": {"category": "plant", "steps": 56, "sr": 1, "spl": 0.72502390902011},
        "boundary": {"pose_delta_m": 0.0, "mapper_object_id_before_after": 139651900314288,
                     "gaussians_before_after": 385659},
        "target2_prevalidation": {"step": 0, "coarse_score": 45.25745391845703,
                                  "full_score": 59.10098648071289, "matches": 141,
                                  "threshold": 40.0, "dominant_instance_ratio": 0.8364661654135338},
        "source_result": str(LATEST / "full_result.json"),
    }
    (dest / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def build_localization_chain():
    dest = OUT / "03_粗到细定位与匹配链_ep0994"
    items = [
        (POSE_CHAIN / "01_目标图像.png", "Goal image"),
        (POSE_CHAIN / "02_PRE后位姿渲染图.png", "PRE: pose search"),
        (POSE_CHAIN / "03_REF后位姿渲染图.png", "REF: local refinement"),
        (POSE_CHAIN / "04_MR后位姿渲染图.png", "MR: closed-loop rendering"),
    ]
    montage(items, dest / "Figure_coarse_to_fine_localization_chain.png", cols=4, panel=(430, 430),
            note="Gaussian rendering converts viewpoint variation into a target-like view for feature matching")
    for index, (path, _) in enumerate(items, 1):
        copy(path, dest / f"{index:02d}_{path.name}")
    score_path = dest / "matching/match_scores.json"
    if score_path.exists():
        scores = {row["stage"]: row for row in json.loads(score_path.read_text())["stages"]}
        match_items = []
        for stage in ("PRE", "REF", "MR"):
            row = scores[stage]
            match_items.append((dest / f"matching/{stage.lower()}_matches.png",
                                f"{stage}: score {row['score_sum']:.1f}, {row['match_count']} matches"))
        montage(match_items, dest / "Figure_localization_match_evidence.png", cols=3,
                panel=(540, 430), note="All rendered poses provide strong correspondence evidence (threshold: 40)")


def build_secondary_and_instances():
    secondary_dest = OUT / "04_二次探索严格配对_Episode915"
    instance_dest = OUT / "05_同类多实例辨别_Episode915"
    secondary_dest.mkdir(parents=True, exist_ok=True)
    instance_dest.mkdir(parents=True, exist_ok=True)

    success_items = [
        (SECONDARY / "00_目标图像RGB.png", "Goal: potted plant"),
        (SECONDARY / "01_二次探索开始_algo_step400_env_step400.png", "Secondary exploration starts"),
        (SECONDARY / "04_正确实例_目标语义区域_algo_step565_nvsm_step581.png", "Correct instance recovered"),
        (SECONDARY / "06_导航结束成功前最后一帧_env_step565.png", "Success: 0.0718 m"),
    ]
    montage(success_items, secondary_dest / "Figure_secondary_exploration_enabled.png", cols=4,
            panel=(420, 500), note="Enabled run: secondary exploration at step 400 -> success at environment step 566")

    match_paths = [
        RAW915 / "NVSM/best_match_step_000445_score_0060.33_nvsm_s4_v31_1783506801_matches.png",
        RAW915 / "NVSM/best_match_step_000538_score_0063.32_nvsm_s4_v31_1783507096_matches.png",
        RAW915 / "NVSM/best_match_step_000581_score_0082.01_nvsm_s0_v29_1783507308_matches.png",
    ]
    instance_items = [
        (SECONDARY / "00_目标图像RGB.png", "Goal image"),
        (SECONDARY / "02_候选实例1_目标语义区域_algo_step429_nvsm_step445.png", "Candidate A | score 60.33"),
        (SECONDARY / "03_候选实例2_目标语义区域_algo_step522_nvsm_step538.png", "Candidate B | score 63.32"),
        (SECONDARY / "04_正确实例_目标语义区域_algo_step565_nvsm_step581.png", "Selected instance | score 82.01"),
    ]
    montage(instance_items, instance_dest / "Figure_same_class_instance_disambiguation.png", cols=4,
            panel=(420, 500), note="Semantic class proposes candidates; instance-level rendered matching selects the target identity")
    montage(list(zip(match_paths, ["Candidate A matches", "Candidate B matches", "Selected instance matches"])),
            instance_dest / "Figure_instance_match_evidence.png", cols=3, panel=(540, 500))
    for path in match_paths:
        copy(path, instance_dest / path.name)
    for path, _ in instance_items:
        copy(path, instance_dest / path.name)
    evidence = {
        "episode": 915, "scene": "yr17PDCnDDW", "category": "potted plant",
        "secondary_exploration_start_step": 400, "success_env_step": 566,
        "final_gt_distance_m": 0.0718,
        "candidate_match_scores": [{"algorithm_step": 429, "nvsm_step": 445, "score": 60.33},
                                   {"algorithm_step": 522, "nvsm_step": 538, "score": 63.32},
                                   {"algorithm_step": 565, "nvsm_step": 581, "score": 82.01}],
        "source_result": str(RAW915),
        "paired_disabled_run": "pending: launched separately into this directory",
    }
    (secondary_dest / "enabled_run_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (instance_dest / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def write_readme():
    text = """# 论文实验素材补充（2026-09-13）

1. `01_语义实例Gaussian地图_ep087`：同一持久 Gaussian 地图的 RGB、语义和实例着色视图，含可交互 PLY。
2. `02_target1到target2共享地图_ep087`：严格记录 target1→target2 位姿与 mapper 不重置，并展示 target2 在 step 0 利用共享地图预验证。
3. `03_粗到细定位与匹配链_ep0994`：Goal→PRE→REF→MR 的粗到细视角定位链。
4. `04_二次探索严格配对_Episode915`：开启版本已整理；关闭版本单独补跑后加入同集对照。
5. `05_同类多实例辨别_Episode915`：三个同类 plant 候选及其渲染匹配分数，最终正确实例得分最高。

所有合成图均保留独立原图和 JSON 证据，论文排版时可重新组合。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_semantic_instance()
    build_multitarget_reuse()
    build_localization_chain()
    build_secondary_and_instances()
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()
