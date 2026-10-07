#!/usr/bin/env python3
"""Create a qualitative 11-pose SWMI comparison from the sampling sketch."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out_dir = Path("/nas_ds925/tyj/IEVE/detectron2/docs/gaussmi_arrow_view")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Relative SWMI estimates read from the sketch geometry. Pose 5 is the
    # circled/highest sample; pose 6 is the center next-waypoint sample.
    values = np.array([0.66, 0.79, 0.93, 1.08, 1.42, 1.05, 0.96, 0.87, 0.77, 0.70, 0.62])
    poses = np.arange(1, 12)
    circled_idx = 4
    center_idx = 5
    threshold = 1.2 * values[center_idx]

    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=260)
    colors = ["#667085"] * len(values)
    colors[circled_idx] = "#d81b9c"
    colors[center_idx] = "#16a6c9"
    bars = ax.barh(
        poses,
        values,
        height=0.68,
        color=colors,
        edgecolor="#263238",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_xlabel("Relative SWMI / mutual information", fontsize=24, labelpad=14)
    ax.set_ylabel("Sampled pose", fontsize=24, labelpad=16)
    ax.set_yticks(poses)
    ax.set_yticklabels([f"Pose {p}" for p in poses], fontsize=18)
    ax.set_xlim(0.0, 2.08)
    ax.set_xticks(np.arange(0.0, 2.01, 0.2))
    ax.tick_params(axis="x", labelsize=18)
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    ax.axvline(
        threshold,
        color="#e07a16",
        linestyle="--",
        linewidth=1.8,
        zorder=4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.98])
    # The plotting area is shifted right by the large Pose labels. Recenter the
    # long x-axis title against the full canvas so its right margin stays ample.
    ax.xaxis.set_label_coords(0.42, -0.115)
    png = out_dir / "swmi_11_pose_bar_comparison.png"
    pdf = out_dir / "swmi_11_pose_bar_comparison.pdf"
    fig.savefig(png, facecolor="white", dpi=260)
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
