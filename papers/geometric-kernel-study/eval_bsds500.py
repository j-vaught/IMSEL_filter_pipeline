"""
Evaluate all three oriented edge filters on 10 BSDS500 test images.
Computes ODS F-score (optimal dataset scale) and saves comparison figures.
"""

import os
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.ndimage import convolve, maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aniso_edge_demo import anisotropic_edge_detect as detect_original
from elliptical_edge_demo import anisotropic_edge_detect as detect_elliptical
from rectangular_edge_demo import anisotropic_edge_detect as detect_rectangular

# Paths
DATASET = ("/home/jvaught/edge-detection-filter-critique/datasets/"
           "BSDS500/BSDS500")
IMG_DIR = os.path.join(DATASET, "images", "test")
GT_DIR = os.path.join(DATASET, "groundTruth", "test")
OUT_DIR = "/home/jvaught/testing/BSDS500"
os.makedirs(OUT_DIR, exist_ok=True)

# Select 10 test images
IMAGE_IDS = ["100007", "100039", "100099", "10081", "101027",
             "101084", "102062", "103006", "103029", "103078"]


def load_gt_boundaries(image_id):
    """Load and average ground truth boundaries from all annotators."""
    mat = sio.loadmat(os.path.join(GT_DIR, f"{image_id}.mat"))
    gt_all = mat["groundTruth"]
    n_annotators = gt_all.shape[1]
    boundaries = []
    for a in range(n_annotators):
        b = gt_all[0, a]["Boundaries"][0, 0].astype(np.float64)
        boundaries.append(b)
    # Average across annotators — soft ground truth
    return np.mean(boundaries, axis=0)


def nms_thin(magnitude, angle):
    """Simple non-maximum suppression for thinner edges."""
    h, w = magnitude.shape
    out = np.zeros_like(magnitude)
    angle_deg = np.degrees(angle) % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle_deg[i, j]
            m = magnitude[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]
            if m >= n1 and m >= n2:
                out[i, j] = m
    return out


def compute_f_scores(pred, gt, thresholds, tolerance=2):
    """Compute precision, recall, F-score at each threshold.

    Uses a distance tolerance (default 2 pixels) for matching,
    consistent with BSDS evaluation protocol.
    """
    precisions = []
    recalls = []
    f_scores = []

    gt_binary = (gt >= 0.5).astype(np.float64)
    # Dilate GT for tolerance matching
    gt_dilated = maximum_filter(gt_binary, size=2 * tolerance + 1)

    for t in thresholds:
        pred_binary = (pred >= t).astype(np.float64)
        # Dilate prediction for recall tolerance
        pred_dilated = maximum_filter(pred_binary, size=2 * tolerance + 1)

        tp_p = np.sum(pred_binary * gt_dilated)
        tp_r = np.sum(gt_binary * pred_dilated)
        n_pred = np.sum(pred_binary)
        n_gt = np.sum(gt_binary)

        p = tp_p / (n_pred + 1e-12)
        r = tp_r / (n_gt + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

    return np.array(precisions), np.array(recalls), np.array(f_scores)


def run_detector(name, detect_fn, gray, kwargs):
    """Run a detector and return normalised magnitude and angle."""
    mag, ang, _ = detect_fn(gray, **kwargs)
    mag_norm = mag / (mag.max() + 1e-12)
    return mag_norm, ang


def main():
    thresholds = np.linspace(0.01, 0.99, 50)

    detectors = {
        "Original": (detect_original,
                     dict(n_orientations=36, length=15, sigma=2.0)),
        "Elliptical": (detect_elliptical,
                       dict(n_orientations=36, sigma_along=2.0,
                            sigma_across=1.2, length=15)),
        "Rectangular": (detect_rectangular,
                        dict(n_orientations=36, half_along=6.0,
                             half_across=3.6, sigma_along=2.0,
                             sigma_across=1.2, length=15)),
    }

    # Accumulate per-image best F-scores for ODS
    all_f_curves = {name: [] for name in detectors}
    all_best_f = {name: [] for name in detectors}

    for idx, img_id in enumerate(IMAGE_IDS):
        print(f"[{idx+1}/10] Processing {img_id} ...")

        # Load image and GT
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt = load_gt_boundaries(img_id)

        results = {}
        for name, (fn, kwargs) in detectors.items():
            mag, ang = run_detector(name, fn, gray, kwargs)
            _, _, f_scores = compute_f_scores(mag, gt, thresholds)
            best_idx = np.argmax(f_scores)
            best_f = f_scores[best_idx]
            all_f_curves[name].append(f_scores)
            all_best_f[name].append(best_f)
            results[name] = (mag, best_f, thresholds[best_idx])

        # --- Save comparison figure for this image --------------------------
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # Row 1: input, GT, blank (scores)
        axes[0, 0].imshow(np.asarray(img))
        axes[0, 0].set_axis_off()
        axes[0, 0].set_title("Input", fontsize=11)

        axes[0, 1].imshow(gt, cmap="gray")
        axes[0, 1].set_axis_off()
        axes[0, 1].set_title("Ground Truth", fontsize=11)

        # Scores summary in third panel
        axes[0, 2].set_axis_off()
        score_text = "ODS F-scores:\n\n"
        for name in detectors:
            score_text += f"{name}: {results[name][1]:.3f}\n"
        axes[0, 2].text(0.1, 0.5, score_text, fontsize=14,
                        verticalalignment="center",
                        fontfamily="monospace",
                        transform=axes[0, 2].transAxes)

        # Row 2: three detector outputs
        for col, name in enumerate(detectors):
            mag, best_f, best_t = results[name]
            axes[1, col].imshow(mag, cmap="gray")
            axes[1, col].set_axis_off()
            axes[1, col].set_title(f"{name}  (F={best_f:.3f})", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{img_id}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # --- ODS: optimal threshold across the dataset --------------------------
    print("\n" + "=" * 60)
    print("ODS F-scores (optimal dataset-scale threshold)")
    print("=" * 60)

    for name in detectors:
        # Stack all F-score curves and average across images
        f_matrix = np.array(all_f_curves[name])  # (10, n_thresholds)
        mean_f_curve = f_matrix.mean(axis=0)
        ods_idx = np.argmax(mean_f_curve)
        ods_f = mean_f_curve[ods_idx]
        ods_t = thresholds[ods_idx]
        per_image = all_best_f[name]
        print(f"  {name:15s}:  ODS={ods_f:.4f}  "
              f"(threshold={ods_t:.3f})  "
              f"per-image mean={np.mean(per_image):.4f}")

    print(f"\nSaved 10 comparison images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
