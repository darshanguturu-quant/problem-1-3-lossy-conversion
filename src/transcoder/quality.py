"""
Objective and Perceptual Video Quality Evaluation (Task 3)
Calculates frame-level PSNR, SSIM, and text/edge preservation ratios between
original master media and compressed derivative streams.
"""

from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class QualityReport:
    """Quantitative quality measurement between original and transcoded video."""
    psnr_db: float
    ssim: float
    edge_preservation_ratio: float
    frames_evaluated: int
    meets_constraints: bool
    psnr_threshold_db: float = 32.0
    ssim_threshold: float = 0.90
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse <= 1e-10:
        return 100.0  # Identical images
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Converts to grayscale and evaluates luminance, contrast, and structure.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Mean and variance
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def calculate_edge_preservation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Measures high-frequency edge and text retention using Laplacian variance ratio.
    Critical for OCR readability and slide text retention in meetings/lectures.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    lap1 = cv2.Laplacian(g1, cv2.CV_64F)
    lap2 = cv2.Laplacian(g2, cv2.CV_64F)

    var1 = float(np.var(lap1))
    var2 = float(np.var(lap2))

    if var1 <= 1e-6:
        return 1.0
    return float(np.clip(var2 / var1, 0.0, 2.0))


def measure_video_quality(
    orig_path: str,
    transcoded_path: str,
    num_eval_frames: int = 5,
    psnr_threshold_db: float = 32.0,
    ssim_threshold: float = 0.90
) -> QualityReport:
    """
    Sample matching frames from original and transcoded video and compute objective metrics.
    """
    if not os.path.exists(orig_path) or not os.path.exists(transcoded_path):
        return QualityReport(
            psnr_db=0.0,
            ssim=0.0,
            edge_preservation_ratio=0.0,
            frames_evaluated=0,
            meets_constraints=False,
            error_message="One or both video paths do not exist"
        )

    cap_orig = cv2.VideoCapture(orig_path)
    cap_trans = cv2.VideoCapture(transcoded_path)

    if not cap_orig.isOpened() or not cap_trans.isOpened():
        if cap_orig.isOpened():
            cap_orig.release()
        if cap_trans.isOpened():
            cap_trans.release()
        return QualityReport(
            psnr_db=0.0,
            ssim=0.0,
            edge_preservation_ratio=0.0,
            frames_evaluated=0,
            meets_constraints=False,
            error_message="Could not open video streams for comparison"
        )

    fc_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fc_trans = int(cap_trans.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = min(fc_orig, fc_trans)

    if total_frames <= 0:
        total_frames = 10  # fallback sequential read

    step = max(1, total_frames // (num_eval_frames + 1))
    target_indices = [step * (i + 1) for i in range(num_eval_frames)]

    psnr_list, ssim_list, edge_list = [], [], []

    for idx in target_indices:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_trans.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret1, f1 = cap_orig.read()
        ret2, f2 = cap_trans.read()

        if ret1 and ret2 and f1 is not None and f2 is not None:
            psnr_list.append(calculate_psnr(f1, f2))
            ssim_list.append(calculate_ssim(f1, f2))
            edge_list.append(calculate_edge_preservation(f1, f2))

    cap_orig.release()
    cap_trans.release()

    if not psnr_list:
        return QualityReport(
            psnr_db=0.0,
            ssim=0.0,
            edge_preservation_ratio=0.0,
            frames_evaluated=0,
            meets_constraints=False,
            error_message="Could not sample valid matching frames from videos"
        )

    mean_psnr = round(float(np.mean(psnr_list)), 2)
    mean_ssim = round(float(np.mean(ssim_list)), 4)
    mean_edge = round(float(np.mean(edge_list)), 3)

    meets_constraints = (mean_psnr >= psnr_threshold_db and mean_ssim >= ssim_threshold)

    return QualityReport(
        psnr_db=mean_psnr,
        ssim=mean_ssim,
        edge_preservation_ratio=mean_edge,
        frames_evaluated=len(psnr_list),
        meets_constraints=meets_constraints,
        psnr_threshold_db=psnr_threshold_db,
        ssim_threshold=ssim_threshold
    )
