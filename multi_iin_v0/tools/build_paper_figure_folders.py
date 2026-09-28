#!/usr/bin/env python3
"""Build a self-contained, evidence-backed folder tree for paper figures.

The script only reads completed experiments.  It copies selected lightweight
assets, exports the underlying metrics as CSV/JSON, and renders quantitative
plots.  Large experiment dumps and PLY files are deliberately not duplicated.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "论文实验图素材_整理_20260928"
OLD = ROOT / "论文实验素材补充_20260913"
ARBITRARY = ROOT / "任意视角匹配案例_5组"
EXP = ROOT / "multi_iin_v0/results/experiments"
OURS = EXP / "full100_compact_correctness_20260908-最好的-改代码后复现不出"
IEVE = EXP / "ieve_multi_truth_target_merged100_20260911"
UNIGOAL = EXP / "unigoal_multi_full100_20260911"
IEVE_VISUAL = EXP / "paper_compare_ieve_visual_20260928"
MANIFEST = ROOT / "multi_iin_v0/manifests/miin_v0_val_k2.json"

METHODS = ["IEVE-MULTI", "UniGoal-MULTI", "Ours"]
COLORS = {"IEVE-MULTI": "#4C78A8", "UniGoal-MULTI": "#F58518", "Ours": "#54A24B"}


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def copy_light(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".json", ".md"}:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def episode_number(value: str) -> int:
    return int(str(value).rsplit("_", 1)[-1])


def collect_records():
    manifest = load(MANIFEST)["episodes"]
    categories = {
        (i, int(goal["order"])): goal["object_category"]
        for i, episode in enumerate(manifest)
        for goal in episode["goals"]
    }
    records = []

    for path in OURS.glob("ep*_gpu*/full_result.json"):
        episode = load(path)["episodes"][0]
        ep = episode_number(episode["multi_episode_id"])
        for goal in episode["goals"]:
            records.append({
                "method": "Ours", "episode": ep,
                "target": int(goal["target_index"]) + 1,
                "category": goal.get("goal_name", categories[(ep, int(goal["target_index"]))]),
                "success": int(goal["success"]), "spl": float(goal["spl"]),
                "steps": int(goal.get("steps", 0)),
            })

    for path in IEVE.glob("results/*/full_result.json"):
        data = load(path)
        episode = (data.get("episodes") or [data])[0]
        ep = episode_number(episode.get("multi_episode_id", episode.get("episode_id")))
        for goal in episode["goals"]:
            records.append({
                "method": "IEVE-MULTI", "episode": ep,
                "target": int(goal["target_index"]) + 1,
                "category": goal.get("goal_name", categories[(ep, int(goal["target_index"]))]),
                "success": int(goal["success"]), "spl": float(goal["spl"]),
                "steps": int(goal.get("steps", 0)),
            })

    for path in UNIGOAL.glob("ep*_gpu*/log/total.json"):
        for goal in load(path).get("targets", []):
            ep = episode_number(goal["episode_id"])
            target = int(goal["target_index"])
            records.append({
                "method": "UniGoal-MULTI", "episode": ep, "target": target + 1,
                "category": categories[(ep, target)],
                "success": int(goal["success"]), "spl": float(goal["spl"]),
                "steps": "",
            })
    return sorted(records, key=lambda x: (x["method"], x["episode"], x["target"]))


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_quantitative(records):
    raw_dir = OUT / "08_三方法按类别SR_SPL"
    write_csv(raw_dir / "all_target_records.csv", records,
              ["method", "episode", "target", "category", "success", "spl", "steps"])

    categories = ["chair", "sofa", "plant", "bed", "toilet", "tv_monitor"]
    present = sorted({r["category"] for r in records})
    categories = [c for c in categories if c in present] + [c for c in present if c not in categories]
    rows = []
    for category in categories:
        for method in METHODS:
            subset = [r for r in records if r["method"] == method and r["category"] == category]
            rows.append({"category": category, "method": method, "n": len(subset),
                         "sr": np.mean([r["success"] for r in subset]),
                         "spl": np.mean([r["spl"] for r in subset])})
    write_csv(raw_dir / "category_metrics.csv", rows, ["category", "method", "n", "sr", "spl"])

    x = np.arange(len(categories)); width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), sharex=True)
    for j, method in enumerate(METHODS):
        vals = {r["category"]: r for r in rows if r["method"] == method}
        axes[0].bar(x + (j - 1) * width, [vals[c]["sr"] for c in categories], width,
                    label=method, color=COLORS[method])
        axes[1].bar(x + (j - 1) * width, [vals[c]["spl"] for c in categories], width,
                    label=method, color=COLORS[method])
    for ax, title, ylabel in zip(axes, ["Success rate by category", "SPL by category"], ["SR", "SPL"]):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_monitor", "") for c in categories], rotation=25, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.22))
    fig.tight_layout()
    save_figure(fig, raw_dir, "Figure_category_sr_spl_comparison")

    pos_dir = OUT / "09_三方法target1_target2与ordered指标"
    position_rows = []
    for method in METHODS:
        per_target = {t: [r for r in records if r["method"] == method and r["target"] == t] for t in (1, 2)}
        by_ep = defaultdict(dict)
        for r in records:
            if r["method"] == method:
                by_ep[r["episode"]][r["target"]] = r
        ordered = [int(v.get(1, {}).get("success", 0) and v.get(2, {}).get("success", 0)) for v in by_ep.values()]
        position_rows.append({
            "method": method,
            "target1_sr": np.mean([r["success"] for r in per_target[1]]),
            "target1_spl": np.mean([r["spl"] for r in per_target[1]]),
            "target2_sr": np.mean([r["success"] for r in per_target[2]]),
            "target2_spl": np.mean([r["spl"] for r in per_target[2]]),
            "ordered_sr": np.mean(ordered),
        })
    write_csv(pos_dir / "target_position_metrics.csv", position_rows,
              ["method", "target1_sr", "target1_spl", "target2_sr", "target2_spl", "ordered_sr"])
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5))
    x = np.arange(len(METHODS))
    axes[0].bar(x - .22, [r["target1_sr"] for r in position_rows], .22, label="Target 1 SR")
    axes[0].bar(x, [r["target2_sr"] for r in position_rows], .22, label="Target 2 SR")
    axes[0].bar(x + .22, [r["ordered_sr"] for r in position_rows], .22, label="Ordered SR")
    axes[1].bar(x - .14, [r["target1_spl"] for r in position_rows], .28, label="Target 1 SPL")
    axes[1].bar(x + .14, [r["target2_spl"] for r in position_rows], .28, label="Target 2 SPL")
    for ax, title in zip(axes, ["Sequential success", "Sequential efficiency"]):
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=15, ha="right")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, pos_dir, "Figure_target_position_metrics")

    matrix_dir = OUT / "10_三方法100集逐集成败矩阵"
    by = defaultdict(dict)
    for r in records:
        by[(r["method"], r["episode"])][r["target"]] = r["success"]
    matrix = np.array([[sum(by[(m, ep)].values()) for ep in range(100)] for m in METHODS])
    matrix_rows = [{"episode": ep, **{m: int(matrix[i, ep]) for i, m in enumerate(METHODS)}} for ep in range(100)]
    write_csv(matrix_dir / "episode_success_count.csv", matrix_rows, ["episode", *METHODS])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig, ax = plt.subplots(figsize=(12.5, 2.2))
    cmap = ListedColormap(["#D9D9D9", "#F2CF5B", "#4CAF50"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N))
    ax.set_yticks(range(3)); ax.set_yticklabels(METHODS); ax.set_xlabel("Episode index")
    ax.set_xticks(np.arange(0, 100, 5)); ax.set_title("Number of successful targets per two-target episode")
    fig.tight_layout()
    save_figure(fig, matrix_dir, "Figure_episode_outcome_matrix")

    difficulty_dir = OUT / "12_按目标间距离分层的多目标结果"
    manifest = load(MANIFEST)["episodes"]
    bins = ["near", "medium", "far"]
    difficulty_rows = []
    for method in METHODS:
        method_records = [r for r in records if r["method"] == method]
        by_ep = defaultdict(dict)
        for record in method_records:
            by_ep[record["episode"]][record["target"]] = record
        for distance_bin in bins:
            episodes = [i for i, episode in enumerate(manifest)
                        if episode["difficulty"]["target_pair_distance_bin"] == distance_bin]
            ordered = [int(by_ep[e][1]["success"] and by_ep[e][2]["success"]) for e in episodes]
            target2 = [by_ep[e][2]["success"] for e in episodes]
            target2_spl = [by_ep[e][2]["spl"] for e in episodes]
            difficulty_rows.append({"method": method, "distance_bin": distance_bin, "n": len(episodes),
                                    "ordered_sr": np.mean(ordered), "target2_sr": np.mean(target2),
                                    "target2_spl": np.mean(target2_spl)})
    write_csv(difficulty_dir / "distance_bin_metrics.csv", difficulty_rows,
              ["method", "distance_bin", "n", "ordered_sr", "target2_sr", "target2_spl"])
    x = np.arange(len(bins)); width = .25
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5))
    for j, method in enumerate(METHODS):
        vals = {r["distance_bin"]: r for r in difficulty_rows if r["method"] == method}
        axes[0].bar(x + (j - 1) * width, [vals[b]["ordered_sr"] for b in bins], width,
                    label=method, color=COLORS[method])
        axes[1].bar(x + (j - 1) * width, [vals[b]["target2_spl"] for b in bins], width,
                    label=method, color=COLORS[method])
    for ax, title, ylabel in zip(axes, ["Ordered SR vs. target-pair distance", "Target 2 SPL vs. target-pair distance"],
                                 ["Ordered SR", "Target 2 SPL"]):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
        ax.set_xticks(x); ax.set_xticklabels(bins)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, difficulty_dir, "Figure_distance_stratified_results")


def copy_case(ep: int, directory: Path, records) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ours_result = next(OURS.glob(f"ep{ep:03d}_gpu*/full_result.json"))
    ours_episode = load(ours_result)["episodes"][0]
    shutil.copy2(ours_result, directory / "ours_full_result.json")
    compact = next(OURS.glob(f"ep{ep:03d}_gpu*/compact/**/episode_*miin_v0_val_{ep:06d}"))
    for target in sorted(compact.glob("target_*")):
        idx = target.name.split("__", 1)[0]
        for rel, suffix in [("02_map/final_2d_map.png", "map"),
                            ("03_gaussian_map/final_gaussian_rgb_comparison.png", "rgb_and_gaussian_render")]:
            src = target / rel
            if src.exists():
                shutil.copy2(src, directory / f"ours_{idx}_{suffix}.png")
                if suffix == "rgb_and_gaussian_render":
                    image = cv2.imread(str(src))
                    midpoint = image.shape[1] // 2
                    cv2.imwrite(str(directory / f"ours_{idx}_final_observation.png"), image[:, :midpoint])
                    cv2.imwrite(str(directory / f"ours_{idx}_gaussian_render.png"), image[:, midpoint:])
        summary = target / "target_summary.json"
        if summary.exists():
            shutil.copy2(summary, directory / f"ours_{idx}_summary.json")

    uni_video = next(UNIGOAL.glob(f"ep*_gpu*/visualization/eps_miin_v0_val_{ep:06d}.mp4"))
    (directory / "unigoal_video_source.txt").write_text(str(uni_video.resolve()) + "\n")
    capture = cv2.VideoCapture(str(uni_video))
    frames, goal_diffs = [], []
    first_goal = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        goal_crop = frame[45:230, 10:260].astype(np.float32)
        if first_goal is None:
            first_goal = goal_crop
        goal_diffs.append(float(np.abs(goal_crop - first_goal).mean()))
    capture.release()
    switch = next((i for i in range(max(0, len(frames) - 10))
                   if min(goal_diffs[i:i + 10]) > 8.0), len(frames) // 2)
    chosen = {
        "target_01_start": 0,
        "target_01_terminal": max(0, switch - 1),
        "target_02_start": min(switch, len(frames) - 1),
        "target_02_terminal": len(frames) - 1,
    }
    for label, frame_index in chosen.items():
        frame = frames[frame_index]
        cv2.imwrite(str(directory / f"unigoal_{label}_frame{frame_index:04d}.png"), frame)
        cv2.imwrite(str(directory / f"unigoal_{label}_observation.png"), frame[45:540, 270:645])
        cv2.imwrite(str(directory / f"unigoal_{label}_map.png"), frame[45:540, 660:1160])
    # The goal panel is shared by all methods because the task input is identical.
    cv2.imwrite(str(directory / "goal_target_01.png"), frames[0][45:255, 15:250])
    cv2.imwrite(str(directory / "goal_target_02.png"), frames[chosen["target_02_start"]][45:255, 15:250])
    with (directory / "unigoal_keyframes.json").open("w") as handle:
        json.dump({"video": str(uni_video.resolve()), "frame_count": len(frames),
                   "detected_target_switch_frame": switch, "selected_frames": chosen}, handle, indent=2)
    case = [r for r in records if r["episode"] == ep]
    with (directory / "case_metrics.json").open("w") as handle:
        json.dump({"episode": ep, "ours_episode": ours_episode["multi_episode_id"], "records": case}, handle, indent=2)

    # A visualization rerun is kept separate from the fixed 100-episode table
    # because the baseline contains stochastic perception/planning components.
    rerun = next(IEVE_VISUAL.glob(f"ep{ep:03d}_gpu*/full_result.json"), None)
    if rerun is not None:
        shutil.copy2(rerun, directory / "ieve_visual_rerun_full_result.json")
        visual_root = rerun.parent / "dump" / rerun.parent.name / "episodes" / "thread_0"
        visuals = list(visual_root.glob("eps_*/*-Vis-*.png"))
        def frame_number(path: Path) -> int:
            return int(path.stem.rsplit("-", 1)[-1])
        if visuals:
            start = min(visuals, key=frame_number)
            target2_steps = int(load(rerun)["episodes"][0]["goals"][1]["steps"])
            terminal_index = max(0, target2_steps - 1)
            terminal = next((path for path in visuals if frame_number(path) == terminal_index),
                            max(visuals, key=frame_number))
            for label, src in [("start", start), ("terminal", terminal)]:
                image = cv2.imread(str(src))
                cv2.imwrite(str(directory / f"ieve_target_02_{label}_full.png"), image)
                cv2.imwrite(str(directory / f"ieve_target_02_{label}_goal.png"), image[50:530, 15:495])
                cv2.imwrite(str(directory / f"ieve_target_02_{label}_observation.png"), image[50:530, 510:870])
                cv2.imwrite(str(directory / f"ieve_target_02_{label}_map.png"), image[50:530, 885:1365])
            with (directory / "ieve_visual_keyframes.json").open("w") as handle:
                json.dump({"source": str(visual_root), "note": "target1 frames were overwritten after the in-place goal switch; these files document target2",
                           "start": str(start), "terminal": str(terminal)}, handle, indent=2)
            build_case_composite(directory, case, load(rerun))


def build_case_composite(directory: Path, fixed_records, ieve_rerun) -> None:
    target2 = {r["method"]: r for r in fixed_records if r["target"] == 2}
    rerun_goal = ieve_rerun["episodes"][0]["goals"][1]
    labels = {
        "IEVE-MULTI": f"SR {int(rerun_goal['success'])} | SPL {rerun_goal['spl']:.3f} | {rerun_goal['steps']} steps",
        "UniGoal-MULTI": f"SR {target2['UniGoal-MULTI']['success']} | SPL {target2['UniGoal-MULTI']['spl']:.3f}",
        "Ours": f"SR {target2['Ours']['success']} | SPL {target2['Ours']['spl']:.3f} | {target2['Ours']['steps']} steps",
    }
    observations = [
        directory / "ieve_target_02_terminal_observation.png",
        directory / "unigoal_target_02_terminal_observation.png",
        directory / "ours_target_02_final_observation.png",
    ]
    maps = [
        directory / "ieve_target_02_terminal_map.png",
        directory / "unigoal_target_02_terminal_map.png",
        directory / "ours_target_02_map.png",
    ]
    fig = plt.figure(figsize=(11.6, 5.7))
    grid = fig.add_gridspec(2, 4, width_ratios=[.72, 1, 1, 1], wspace=.06, hspace=.08)
    goal_ax = fig.add_subplot(grid[:, 0])
    goal_ax.imshow(cv2.cvtColor(cv2.imread(str(directory / "goal_target_02.png")), cv2.COLOR_BGR2RGB))
    goal_ax.set_title("Target 2 goal", fontsize=11); goal_ax.axis("off")
    for col, method in enumerate(METHODS, start=1):
        obs_ax = fig.add_subplot(grid[0, col])
        obs_ax.imshow(cv2.cvtColor(cv2.imread(str(observations[col - 1])), cv2.COLOR_BGR2RGB))
        obs_ax.set_title(f"{method}\n{labels[method]}", fontsize=9); obs_ax.axis("off")
        map_ax = fig.add_subplot(grid[1, col])
        map_ax.imshow(cv2.cvtColor(cv2.imread(str(maps[col - 1])), cv2.COLOR_BGR2RGB))
        map_ax.set_title("Terminal map", fontsize=9); map_ax.axis("off")
    save_figure(fig, directory, "Figure_three_method_target2_comparison")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_light(OLD / "01_语义实例Gaussian地图_ep087", OUT / "01_语义实例Gaussian地图")
    copy_light(ARBITRARY, OUT / "02_大视角差定位与任意视角匹配")
    copy_light(OLD / "04_二次探索严格正向配对_ep243", OUT / "03_二次探索严格配对")
    copy_light(OLD / "05_同类多实例辨别_Episode915", OUT / "04_同类干扰实例排除")
    copy_light(OLD / "02_target1到target2共享地图_ep087", OUT / "05_多目标共享Gaussian地图")
    records = collect_records()
    build_quantitative(records)
    copy_case(20, OUT / "06_三方法同任务对比_ep020_主案例", records)
    copy_case(23, OUT / "07_三方法同任务对比_ep023_补充案例", records)
    copy_case(0, OUT / "11_IEVE随机性说明_ep000_不作主证据", records)
    print(OUT)


if __name__ == "__main__":
    main()
