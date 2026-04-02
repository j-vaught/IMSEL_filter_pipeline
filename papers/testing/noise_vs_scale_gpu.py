"""
GPU-accelerated noise robustness vs kernel scale experiment.

Uses PyTorch conv2d on GPU for all filter bank convolutions.
Sweeps kernel size (scale factor) and SNR simultaneously for all three
filter designs.
"""

import os
import time
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.ndimage import maximum_filter
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Kernel builders (return numpy, converted to torch tensors in batch) ──────

def _build_original_kernel(theta, sigma=2.0, length=15):
    half = length // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    ct, st = np.cos(theta), np.sin(theta)
    u = x * ct + y * st
    v = -x * st + y * ct
    g_along = np.exp(-u**2 / (2 * sigma**2))
    g_across = -v * np.exp(-v**2 / (2 * (sigma * 0.6)**2))
    kernel = g_along * g_across
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-12
    return kernel


def _build_elliptical_kernel(theta, sigma_along=2.0, sigma_across=1.2,
                             length=15):
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


def _build_rectangular_kernel(theta, half_along=6.0, half_across=3.6,
                              sigma_along=2.0, sigma_across=1.2, length=15):
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


def build_kernel_bank(build_fn, build_kwargs, n_orientations=36):
    """Build all orientation kernels and return as a single torch tensor.

    Returns shape (n_orientations, 1, H, W) for use with conv2d groups.
    """
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    kernels = []
    for theta in thetas:
        k = build_fn(theta, **build_kwargs)
        kernels.append(k)
    # Stack into (N_s, H, W) then add channel dim -> (N_s, 1, H, W)
    bank = np.stack(kernels, axis=0)
    return torch.from_numpy(bank).float().unsqueeze(1).to(DEVICE)


def detect_gpu(gray_tensor, kernel_bank):
    """Run oriented filter bank on GPU.

    Args:
        gray_tensor: (1, 1, H, W) float32 tensor on GPU
        kernel_bank: (N_s, 1, H, W) float32 tensor on GPU

    Returns:
        magnitude: (H, W) numpy array, normalised to [0, 1]
    """
    n_s = kernel_bank.shape[0]
    pad_h = kernel_bank.shape[2] // 2
    pad_w = kernel_bank.shape[3] // 2

    # Pad input with reflect
    padded = F.pad(gray_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect")

    # Replicate input for grouped convolution: (1, N_s, H_pad, W_pad)
    padded_rep = padded.expand(-1, n_s, -1, -1)

    # Grouped conv: each kernel applied to its own channel copy
    responses = F.conv2d(padded_rep, kernel_bank, groups=n_s)  # (1, N_s, H, W)

    abs_resp = responses.abs()
    magnitude = abs_resp.max(dim=1).values.squeeze()  # (H, W)

    mag_np = magnitude.cpu().numpy()
    mag_np /= mag_np.max() + 1e-12
    return mag_np


# ── Evaluation helpers ───────────────────────────────────────────────────────

def add_gaussian_noise(image, snr_db):
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noisy = image + np.random.randn(*image.shape) * noise_std
    return np.clip(noisy, 0.0, 1.0)


def compute_ods(pred, gt, thresholds, tolerance=2):
    gt_binary = (gt >= 0.5).astype(np.float64)
    gt_dilated = maximum_filter(gt_binary, size=2 * tolerance + 1)
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
        f_scores.append(f)
    return np.array(f_scores)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    DATASET = ("/home/jvaught/edge-detection-filter-critique/datasets/"
               "BSDS500/BSDS500")
    IMG_DIR = os.path.join(DATASET, "images", "test")
    GT_DIR = os.path.join(DATASET, "groundTruth", "test")
    OUT_DIR = "/home/jvaught/testing/BSDS500"
    os.makedirs(OUT_DIR, exist_ok=True)

    IMAGE_IDS = ["100007", "100039", "100099", "10081", "101027",
                 "101084", "102062", "103006", "103029", "103078"]

    SNR_LEVELS = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20]
    SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    thresholds = np.linspace(0.01, 0.99, 50)
    np.random.seed(42)

    # Load images and GTs
    images_np, gts = [], []
    for img_id in IMAGE_IDS:
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt_mat = sio.loadmat(os.path.join(GT_DIR, f"{img_id}.mat"))
        gt_all = gt_mat["groundTruth"]
        boundaries = [gt_all[0, a]["Boundaries"][0, 0].astype(np.float64)
                      for a in range(gt_all.shape[1])]
        images_np.append(gray)
        gts.append(np.mean(boundaries, axis=0))

    all_snr = [None] + SNR_LEVELS
    results = {}

    total_combos = len(SCALES) * len(all_snr) * 3
    combo_count = 0
    t_start = time.perf_counter()

    for scale in SCALES:
        sigma = 2.0 * scale
        sigma_a = 2.0 * scale
        sigma_c = 1.2 * scale
        half_a = 3.0 * sigma_a
        half_c = 3.0 * sigma_c
        length = 2 * int(np.ceil(3 * sigma_a)) + 1
        if length % 2 == 0:
            length += 1

        # Precompute kernel banks on GPU (one-time per scale)
        detector_configs = {
            "Original": (_build_original_kernel,
                         dict(sigma=sigma, length=length)),
            "Elliptical": (_build_elliptical_kernel,
                           dict(sigma_along=sigma_a, sigma_across=sigma_c,
                                length=length)),
            "Rectangular": (_build_rectangular_kernel,
                            dict(half_along=half_a, half_across=half_c,
                                 sigma_along=sigma_a, sigma_across=sigma_c,
                                 length=length)),
        }

        kernel_banks = {}
        for name, (build_fn, bkw) in detector_configs.items():
            kernel_banks[name] = build_kernel_bank(build_fn, bkw)

        results[scale] = {name: [] for name in detector_configs}

        print(f"\n--- Scale {scale}x  (sigma={sigma:.1f}, "
              f"length={length}, "
              f"kernel_bank={kernel_banks['Original'].shape}) ---")

        for snr_db in all_snr:
            label = "clean" if snr_db is None else f"{snr_db} dB"

            for name in detector_configs:
                combo_count += 1
                f_curves = []
                for gray, gt in zip(images_np, gts):
                    noisy = gray if snr_db is None else \
                        add_gaussian_noise(gray, snr_db)

                    # To GPU
                    noisy_t = torch.from_numpy(noisy).float() \
                        .unsqueeze(0).unsqueeze(0).to(DEVICE)

                    mag = detect_gpu(noisy_t, kernel_banks[name])
                    f_scores = compute_ods(mag, gt, thresholds)
                    f_curves.append(f_scores)

                f_matrix = np.array(f_curves)
                ods = f_matrix.mean(axis=0).max()
                results[scale][name].append(ods)

            elapsed = time.perf_counter() - t_start
            rate = combo_count / elapsed
            remaining = (total_combos - combo_count) / rate
            print(f"  SNR={label:>6s}  "
                  f"Orig={results[scale]['Original'][-1]:.4f}  "
                  f"Ell={results[scale]['Elliptical'][-1]:.4f}  "
                  f"Rect={results[scale]['Rectangular'][-1]:.4f}  "
                  f"[{combo_count}/{total_combos}, "
                  f"~{remaining:.0f}s left]")

        # Free GPU memory between scales
        del kernel_banks
        torch.cuda.empty_cache()

    # ── Plot: one subplot per scale ──────────────────────────────────────────
    n_scales = len(SCALES)
    cols = 3
    rows = int(np.ceil(n_scales / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False)

    colors = {"Original": "#73000A", "Elliptical": "#466A9F",
              "Rectangular": "#1F414D"}
    markers = {"Original": "s", "Elliptical": "o", "Rectangular": "D"}

    x_labels = ["Clean"] + [f"{s}" for s in SNR_LEVELS]
    x_pos = list(range(len(all_snr)))

    for idx, scale in enumerate(SCALES):
        ax = axes[idx // cols, idx % cols]
        for name in ["Original", "Elliptical", "Rectangular"]:
            ax.plot(x_pos, results[scale][name],
                    color=colors[name], marker=markers[name],
                    linewidth=2, markersize=5, label=name, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_ylabel("ODS F-score", fontsize=9)
        ax.set_xlabel("SNR (dB)", fontsize=9)
        sigma_val = 2.0 * scale
        length_val = 2 * int(np.ceil(3 * sigma_val)) + 1
        if length_val % 2 == 0:
            length_val += 1
        ax.set_title(f"Scale {scale}x  "
                     f"($\\sigma$={sigma_val:.1f}, "
                     f"kernel={length_val}x{length_val})",
                     fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(frameon=False, fontsize=8)

    for idx in range(n_scales, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "snr_vs_ods_by_scale.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "snr_vs_ods_by_scale.pdf"),
                bbox_inches="tight")
    plt.close()

    # ── Summary plot: ODS at SNR=2 dB vs scale ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    snr_index_2db = all_snr.index(2)
    snr_index_clean = 0

    for name in ["Original", "Elliptical", "Rectangular"]:
        ods_at_2db = [results[s][name][snr_index_2db] for s in SCALES]
        ax.plot(SCALES, ods_at_2db, color=colors[name],
                marker=markers[name], linewidth=2, markersize=7,
                label=f"{name} (SNR=2 dB)", zorder=3)

    for name in ["Original", "Elliptical", "Rectangular"]:
        ods_clean = [results[s][name][snr_index_clean] for s in SCALES]
        ax.plot(SCALES, ods_clean, color=colors[name],
                marker=markers[name], linewidth=2, markersize=7,
                linestyle="--", alpha=0.5,
                label=f"{name} (clean)", zorder=2)

    ax.set_xlabel("Scale factor", fontsize=11)
    ax.set_ylabel("ODS F-score", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ods_vs_scale.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "ods_vs_scale.pdf"),
                bbox_inches="tight")
    plt.close()

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")
    print(f"Saved: snr_vs_ods_by_scale.png/pdf, ods_vs_scale.png/pdf")


if __name__ == "__main__":
    main()
