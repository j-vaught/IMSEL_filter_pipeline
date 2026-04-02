"""
Generate edge map visualizations for the noise-matched filters at
different SNR levels. Shows input, GT, and all 4 filter outputs.
"""

import os
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.signal import fftconvolve
from scipy.ndimage import convolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = ("/home/jvaught/edge-detection-filter-critique/datasets/"
           "BSDS500/BSDS500")
IMG_DIR = os.path.join(DATASET, "images", "test")
GT_DIR = os.path.join(DATASET, "groundTruth", "test")
OUT_DIR = "/home/jvaught/testing/BSDS500"
os.makedirs(OUT_DIR, exist_ok=True)

# Use 3 images for visual comparison
IMAGE_IDS = ["100007", "101027", "103006"]
SNR_LEVELS = [0.3, 0.5, 1.0, 2.0, 5.0]


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


def _conv(image, kernel):
    if kernel.shape[0] > 15:
        ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
        return fftconvolve(padded, kernel, mode="valid")
    else:
        return convolve(image, kernel, mode="reflect")


def detect(gray, build_fn, build_kwargs, n_orientations):
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.zeros((n_orientations, *gray.shape), dtype=np.float64)
    for k, theta in enumerate(thetas):
        kernel = build_fn(theta, **build_kwargs)
        responses[k] = _conv(gray, kernel)
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


# Filter configs
FILTERS = {
    "Ell. LF-match": {
        "build_fn": build_elliptical_kernel,
        "kwargs": dict(sigma_along=7.17, sigma_across=2.50, length=43),
        "ns": 36,
    },
    "Rect. LF-match": {
        "build_fn": build_rectangular_kernel,
        "kwargs": dict(half_along=21.5, half_across=7.5,
                       sigma_along=7.17, sigma_across=2.50, length=43),
        "ns": 36,
    },
    "Ell. WVF-match": {
        "build_fn": build_elliptical_kernel,
        "kwargs": dict(sigma_along=3.83, sigma_across=3.83, length=23),
        "ns": 36,
    },
    "Rect. WVF-match": {
        "build_fn": build_rectangular_kernel,
        "kwargs": dict(half_along=11.5, half_across=11.5,
                       sigma_along=3.83, sigma_across=3.83, length=23),
        "ns": 36,
    },
}


def main():
    np.random.seed(42)

    for img_id in IMAGE_IDS:
        print(f"\nProcessing {img_id}...")
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt = load_gt(img_id)

        n_snr = len(SNR_LEVELS)
        n_filters = len(FILTERS)
        # Rows: SNR levels, Cols: noisy input | GT | 4 filters
        fig, axes = plt.subplots(n_snr, n_filters + 2,
                                 figsize=(3.2 * (n_filters + 2), 3 * n_snr))

        for row, snr_db in enumerate(SNR_LEVELS):
            noisy = add_gaussian_noise(gray, snr_db)

            axes[row, 0].imshow(noisy, cmap="gray")
            axes[row, 0].set_axis_off()
            if row == 0:
                axes[row, 0].set_title("Noisy input", fontsize=9)
            axes[row, 0].set_ylabel(f"SNR={snr_db}", fontsize=10,
                                     rotation=90, labelpad=10)

            axes[row, 1].imshow(gt, cmap="gray")
            axes[row, 1].set_axis_off()
            if row == 0:
                axes[row, 1].set_title("Ground Truth", fontsize=9)

            for col, (name, cfg) in enumerate(FILTERS.items(), start=2):
                mag = detect(noisy, cfg["build_fn"], cfg["kwargs"], cfg["ns"])
                axes[row, col].imshow(mag, cmap="gray")
                axes[row, col].set_axis_off()
                if row == 0:
                    axes[row, col].set_title(name, fontsize=9)

            print(f"  SNR={snr_db} done")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"noise_edgemaps_{img_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
