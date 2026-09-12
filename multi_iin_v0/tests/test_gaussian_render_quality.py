import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analyze_gaussian_render_quality.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_gaussian_render_quality", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identical_images_have_perfect_quality_and_no_excess_black():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    metrics = MODULE.compute_frame_metrics(image, image)
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == 1.0
    assert metrics["excess_black_fraction"] == 0.0


def test_black_render_over_visible_gt_is_detected():
    gt = np.full((32, 32, 3), 128, dtype=np.uint8)
    rendered = np.zeros_like(gt)
    metrics = MODULE.compute_frame_metrics(gt, rendered)
    assert metrics["render_black_fraction"] == 1.0
    assert metrics["excess_black_fraction"] == 1.0
    assert metrics["strong_darkening_fraction"] == 1.0


def test_phase_assignment_is_balanced():
    assert [MODULE.phase_for_index(index, 6) for index in range(6)] == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]
