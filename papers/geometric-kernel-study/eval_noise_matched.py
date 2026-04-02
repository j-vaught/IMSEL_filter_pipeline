"""
Evaluate elliptical and rectangular filters matched to optimal high-noise
ASF configs from the noise ablation study.

Tests at SNR=0.3, 0.5, 1.0 with Gaussian noise on BSDS500.
"""

import os
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = ("/home/jvaught/edge-detection-filter-critique/datasets/"
           "BSDS500/BSDS500")
IMG_DIR = os.path.join(DATASET, "images", "test")
GT_DIR = os.path.join(DATASET, "groundTruth", "test")
OUT_DIR = "/home/jvaught/testing/BSDS500"
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_IDS = ["100007", "100039", "100099", "10081", "101027",
             "101084", "102062", "103006", "103029", "103078"]


def build_elliptical_kernel(theta, sigma_along, sigma_across, length):
    half = length // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    ct, st = np.cos(theta), np.sin(theta)
    u = x * ct + y * st
    v = -x * st + y * ct
    ellipse_arg = (u**2 / sigma_along**2) + (v**2 / sigma_across**2)
    mask = ellipse_arg <= 9.0
    gauss = np.exp(-0.5 * ellipse_arg) * mask
    kernel = -v * gauss
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-12
    return kernel


def build_rectangular_kernel(theta, half_along, half_across,
                             sigma_along, sigma_across, length):
    half = length // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    ct, st = np.cos(theta), np.sin(theta)
    u = x * ct + y * st
    v = -x * st + y * ct
    mask = (np.abs(u) <= half_along) & (np.abs(v) <= half_across)
    gauss = np.exp(-0.5 * (u**2 / sigma_along**2 + v**2 / sigma_across**2))
    gauss *= mask
    kernel = -v * gauss
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-12
    return kernel


def _convolve(image, kernel):
    if kernel.shape[0] > 15:
        ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
        return fftconvolve(padded, kernel, mode="valid")
    else:
        from scipy.ndimage import convolve
        return convolve(image, kernel, mode="reflect")


def detect(gray, build_fn, build_kwargs, n_orientations):
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.zeros((n_orientations, *gray.shape), dtype=np.float64)
    for k, theta in enumerate(thetas):
        kernel = build_fn(theta, **build_kwargs)
        responses[k] = _convolve(gray, kernel)
    abs_resp = np.abs(responses)
    k_star = np.argmax(abs_resp, axis=0)
    magnitude = np.take_along_axis(abs_resp, k_star[None], axis=0)[0]
    return magnitude / (magnitude.max() + 1e-12)


def load_gt(image_id):
    mat = sio.loadmat(os.path.join(GT_DIR, f"{image_id}.mat"))
    gt_all = mat["groundTruth"]
    boundaries = [gt_all[0, a]["Boundaries"][0, 0].astype(np.float64)
                  for a in range(gt_all.shape[1])]
    return np.mean(boundaries, axis=0)


def add_gaussian_noise(image, snr_db):
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noisy = image + np.random.randn(*image.shape) * noise_std
    return np.clip(noisy, 0.0, 1.0)


def compute_ods_f(pred, gt, thresholds, tolerance=2):
    gt_binary = (gt >= 0.5).astype(np.float64)
    gt_dilated = maximum_filter(gt_binary, size=2 * tolerance + 1)
    best_f = 0.0
    for t in thresholds:
        pred_b = (pred >= t).astype(np.float64)
        pred_d = maximum_filter(pred_b, size=2 * tolerance + 1)
        tp_p = np.sum(pred_b * gt_dilated)
        tp_r = np.sum(gt_binary * pred_d)
        p = tp_p / (np.sum(pred_b) + 1e-12)
        r = tp_r / (np.sum(gt_binary) + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        if f > best_f:
            best_f = f
    return best_f


# Configs matched to noise ablation optimal ASF settings
NOISE_CONFIGS = {
    # --- SNR=0.3: LF Np=150, Ns=36, m=14 extent 43x15 ---
    "SNR 0.3": {
        "snr_db": 0.3,
        "configs": {
            "Elliptical (LF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=7.17, sigma_across=2.50, length=43),
                "ns": 36,
            },
            "Rectangular (LF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=21.5, half_across=7.5,
                               sigma_along=7.17, sigma_across=2.50, length=43),
                "ns": 36,
            },
            "Elliptical (WVF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 36,
            },
            "Rectangular (WVF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=11.5, half_across=11.5,
                               sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 36,
            },
        },
    },
    # --- SNR=0.5: LF Np=150, Ns=36, m=14 extent 43x15 ---
    "SNR 0.5": {
        "snr_db": 0.5,
        "configs": {
            "Elliptical (LF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=7.17, sigma_across=2.50, length=43),
                "ns": 36,
            },
            "Rectangular (LF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=21.5, half_across=7.5,
                               sigma_along=7.17, sigma_across=2.50, length=43),
                "ns": 36,
            },
            "Elliptical (WVF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 72,
            },
            "Rectangular (WVF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=11.5, half_across=11.5,
                               sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 72,
            },
        },
    },
    # --- SNR=1.0: LF Np=100, Ns=36, m=10 extent 31x11 ---
    "SNR 1.0": {
        "snr_db": 1.0,
        "configs": {
            "Elliptical (LF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=5.17, sigma_across=1.83, length=31),
                "ns": 36,
            },
            "Rectangular (LF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=15.5, half_across=5.5,
                               sigma_along=5.17, sigma_across=1.83, length=31),
                "ns": 36,
            },
            "Elliptical (WVF-matched)": {
                "build_fn": build_elliptical_kernel,
                "kwargs": dict(sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 72,
            },
            "Rectangular (WVF-matched)": {
                "build_fn": build_rectangular_kernel,
                "kwargs": dict(half_along=11.5, half_across=11.5,
                               sigma_along=3.83, sigma_across=3.83, length=23),
                "ns": 72,
            },
        },
    },
}


def main():
    np.random.seed(42)
    thresholds = np.linspace(0.01, 0.99, 50)

    images, gts = [], []
    for img_id in IMAGE_IDS:
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt = load_gt(img_id)
        images.append(gray)
        gts.append(gt)

    all_results = {}

    for snr_label, snr_cfg in NOISE_CONFIGS.items():
        snr_db = snr_cfg["snr_db"]
        print(f"\n{'='*60}")
        print(f"  {snr_label} (Gaussian noise)")
        print(f"{'='*60}")

        all_results[snr_label] = {}

        for name, cfg in snr_cfg["configs"].items():
            f_scores = []
            for gray, gt in zip(images, gts):
                noisy = add_gaussian_noise(gray, snr_db)
                mag = detect(noisy, cfg["build_fn"], cfg["kwargs"], cfg["ns"])
                f = compute_ods_f(mag, gt, thresholds)
                f_scores.append(f)
            ods = np.mean(f_scores)
            all_results[snr_label][name] = ods
            print(f"  {name:30s}  ODS={ods:.4f}  (Ns={cfg['ns']})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {
        "Elliptical (LF-matched)": "#466A9F",
        "Rectangular (LF-matched)": "#1F414D",
        "Elliptical (WVF-matched)": "#CC2E40",
        "Rectangular (WVF-matched)": "#73000A",
    }

    for idx, (snr_label, res) in enumerate(all_results.items()):
        ax = axes[idx]
        names = list(res.keys())
        vals = [res[n] for n in names]
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color=[colors[n] for n in names], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names],
                           fontsize=7, rotation=0, ha="center")
        ax.set_ylabel("ODS F-score", fontsize=10)
        ax.set_title(snr_label, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "noise_matched_ods.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "noise_matched_ods.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"\nSaved: noise_matched_ods.png/pdf")


if __name__ == "__main__":
    main()
