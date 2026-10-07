import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> float:
    gt = gt_rgb.astype(np.float32)
    pred = pred_rgb.astype(np.float32)
    mse = np.mean((gt - pred) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> float:
    return float(
        structural_similarity(
            gt_rgb,
            pred_rgb,
            channel_axis=2,
            data_range=255,
        )
    )

