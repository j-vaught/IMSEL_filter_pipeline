"""
Noise robustness experiment: SNR vs ODS F-score for all three filter designs.

Adds Gaussian noise at varying SNR levels to 10 BSDS500 test images,
runs all three oriented edge filters, and plots ODS as a function of SNR.
"""

import os
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.ndimage import maximum_filter
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

IMAGE_IDS = ["100007", "100039", "100099", "10081", "101027",
             "101084", "102062", "103006", "103029", "103078"]

SNR_DB_LEVELS = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20]


def load_gt_boundaries(image_id):
    mat = sio.loadmat(os.path.join(GT_DIR, f"{image_id}.mat"))
    gt_all = mat["groundTruth"]
    n_annotators = gt_all.shape[1]
    boundaries = []
    for a in range(n_annotators):
        b = gt_all[0, a]["Boundaries"][0, 0].astype(np.float64)
        boundaries.append(b)
    return np.mean(boundaries, axis=0)


def add_gaussian_noise(image, snr_db):
    """Add Gaussian noise to achieve target SNR in dB.

    SNR_dB = 10 * log10(signal_power / noise_power)
    signal_power = mean(image^2)
    noise_std = sqrt(signal_power / 10^(SNR_dB/10))
    """
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noisy = image + np.random.randn(*image.shape) * noise_std
    return np.clip(noisy, 0.0, 1.0)


def compute_f_scores(pred, gt, thresholds, tolerance=2):
    gt_binary = (gt >= 0.5).astype(np.float64)
    gt_dilated = maximum_filter(gt_binary, size=2 * tolerance + 1)

    precisions = []
    recalls = []
    f_scores = []

    for t in thresholds:
        pred_binary = (pred >= t).astype(np.float64)
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

    return np.array(f_scores)


def run_detector(detect_fn, gray, kwargs):
    mag, ang, _ = detect_fn(gray, **kwargs)
    mag_norm = mag / (mag.max() + 1e-12)
    return mag_norm


def main():
    np.random.seed(42)
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

    # Also run clean (no noise) as a reference point
    all_snr_levels = [None] + SNR_DB_LEVELS  # None = clean

    # Results: {detector_name: [ods_at_each_snr]}
    ods_results = {name: [] for name in detectors}

    # Load all images and GTs once
    images = []
    gts = []
    for img_id in IMAGE_IDS:
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt = load_gt_boundaries(img_id)
        images.append(gray)
        gts.append(gt)

    for snr_idx, snr_db in enumerate(all_snr_levels):
        label = "clean" if snr_db is None else f"{snr_db} dB"
        print(f"[{snr_idx+1}/{len(all_snr_levels)}] SNR = {label}")

        for name, (fn, kwargs) in detectors.items():
            f_curves = []
            for gray, gt in zip(images, gts):
                if snr_db is not None:
                    noisy = add_gaussian_noise(gray, snr_db)
                else:
                    noisy = gray

                mag = run_detector(fn, noisy, kwargs)
                f_scores = compute_f_scores(mag, gt, thresholds)
                f_curves.append(f_scores)

            # ODS: best threshold across all images
            f_matrix = np.array(f_curves)
            mean_f = f_matrix.mean(axis=0)
            ods = mean_f.max()
            ods_results[name].append(ods)

    # x-axis: clean + SNR levels
    x_labels = ["Clean"] + [f"{s}" for s in SNR_DB_LEVELS]
    x_positions = list(range(len(all_snr_levels)))

    # --- Print results table ---
    print("\n" + "=" * 70)
    print(f"{'SNR':>8s}", end="")
    for name in detectors:
        print(f"  {name:>14s}", end="")
    print()
    print("-" * 70)
    for i, snr in enumerate(all_snr_levels):
        label = "Clean" if snr is None else f"{snr} dB"
        print(f"{label:>8s}", end="")
        for name in detectors:
            print(f"  {ods_results[name][i]:>14.4f}", end="")
        print()
    print("=" * 70)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Original": "#73000A", "Elliptical": "#466A9F",
              "Rectangular": "#1F414D"}
    markers = {"Original": "s", "Elliptical": "o", "Rectangular": "D"}

    for name in detectors:
        ax.plot(x_positions, ods_results[name],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=7, label=name, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("ODS F-score", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snr_vs_ods.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "snr_vs_ods.pdf"),
                bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {OUT_DIR}/snr_vs_ods.png / .pdf")

    # --- Also save a sample noisy image comparison at SNR=15 dB ---
    sample_gray = images[0]
    sample_gt = gts[0]
    noisy_15 = add_gaussian_noise(sample_gray, 15)
    noisy_5 = add_gaussian_noise(sample_gray, 5)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for row, (noisy, snr_label) in enumerate([(noisy_15, "15 dB"),
                                               (noisy_5, "5 dB")]):
        axes[row, 0].imshow(noisy, cmap="gray")
        axes[row, 0].set_axis_off()
        axes[row, 0].set_title(f"Noisy input (SNR={snr_label})", fontsize=10)

        axes[row, 1].imshow(sample_gt, cmap="gray")
        axes[row, 1].set_axis_off()
        axes[row, 1].set_title("Ground Truth", fontsize=10)

        for col, (name, (fn, kwargs)) in enumerate(detectors.items(), start=2):
            mag = run_detector(fn, noisy, kwargs)
            axes[row, col].imshow(mag, cmap="gray")
            axes[row, col].set_axis_off()
            f_scores = compute_f_scores(mag, sample_gt, thresholds)
            best_f = f_scores.max()
            axes[row, col].set_title(f"{name} (F={best_f:.3f})",
                                     fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "noisy_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/noisy_comparison.png")


if __name__ == "__main__":
    main()
