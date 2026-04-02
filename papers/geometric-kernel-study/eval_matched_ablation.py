"""
Evaluate elliptical and rectangular filters matched to optimal ASF configs
from the ablation study. Runs on BSDS500 test set (10 images).

Optimal ASF configurations from ablation:
  WVF best:  Np=25,  Ns=4,  d=2         (all datasets)
  WVF BSDS:  Np=100, Ns=4,  d=2         (BSDS500 specifically)
  LF UDED:   Np=25,  Ns=36, m=3, d=4
  LF BIPED:  Np=100, Ns=18, m=1, d=4
  LF BSDS:   Np=250, Ns=18, m=2, d=4
"""

import os
import time
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.ndimage import convolve, maximum_filter
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


# ── Kernel builders ──────────────────────────────────────────────────────────

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


def detect(gray, build_fn, build_kwargs, n_orientations):
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.zeros((n_orientations, *gray.shape), dtype=np.float64)
    for k, theta in enumerate(thetas):
        kernel = build_fn(theta, **build_kwargs)
        responses[k] = convolve(gray, kernel, mode="reflect")
    abs_resp = np.abs(responses)
    k_star = np.argmax(abs_resp, axis=0)
    magnitude = np.take_along_axis(abs_resp, k_star[None], axis=0)[0]
    return magnitude / (magnitude.max() + 1e-12)


# ── Evaluation ───────────────────────────────────────────────────────────────

def load_gt(image_id):
    mat = sio.loadmat(os.path.join(GT_DIR, f"{image_id}.mat"))
    gt_all = mat["groundTruth"]
    boundaries = [gt_all[0, a]["Boundaries"][0, 0].astype(np.float64)
                  for a in range(gt_all.shape[1])]
    return np.mean(boundaries, axis=0)


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


# ── Matched configurations ───────────────────────────────────────────────────
# Derived from ASF ablation optimal stencil extents

CONFIGS = {
    # --- Matched to WVF: Np=25, Ns=4, d=2 (best overall) ---
    "WVF-25 Elliptical (Ns=4)": {
        "build_fn": build_elliptical_kernel,
        "build_kwargs": dict(sigma_along=1.00, sigma_across=0.83, length=7),
        "n_orientations": 4,
    },
    "WVF-25 Rectangular (Ns=4)": {
        "build_fn": build_rectangular_kernel,
        "build_kwargs": dict(half_along=3.0, half_across=2.5,
                             sigma_along=1.00, sigma_across=0.83, length=7),
        "n_orientations": 4,
    },
    # --- Matched to WVF: Np=100, Ns=4, d=2 (best on BSDS500) ---
    "WVF-100 Elliptical (Ns=4)": {
        "build_fn": build_elliptical_kernel,
        "build_kwargs": dict(sigma_along=1.83, sigma_across=1.83, length=11),
        "n_orientations": 4,
    },
    "WVF-100 Rectangular (Ns=4)": {
        "build_fn": build_rectangular_kernel,
        "build_kwargs": dict(half_along=5.5, half_across=5.5,
                             sigma_along=1.83, sigma_across=1.83, length=11),
        "n_orientations": 4,
    },
    # --- Matched to LF: Np=25, Ns=36, m=3, d=4 (best LF on UDED) ---
    "LF-25 Elliptical (Ns=36)": {
        "build_fn": build_elliptical_kernel,
        "build_kwargs": dict(sigma_along=2.00, sigma_across=0.83, length=13),
        "n_orientations": 36,
    },
    "LF-25 Rectangular (Ns=36)": {
        "build_fn": build_rectangular_kernel,
        "build_kwargs": dict(half_along=6.0, half_across=2.5,
                             sigma_along=2.00, sigma_across=0.83, length=13),
        "n_orientations": 36,
    },
    # --- Matched to LF: Np=100, Ns=18, m=1, d=4 (best LF on BIPED) ---
    "LF-100 Elliptical (Ns=18)": {
        "build_fn": build_elliptical_kernel,
        "build_kwargs": dict(sigma_along=2.17, sigma_across=1.83, length=13),
        "n_orientations": 18,
    },
    "LF-100 Rectangular (Ns=18)": {
        "build_fn": build_rectangular_kernel,
        "build_kwargs": dict(half_along=6.5, half_across=5.5,
                             sigma_along=2.17, sigma_across=1.83, length=13),
        "n_orientations": 18,
    },
    # --- Matched to LF: Np=250, Ns=18, m=2, d=4 (best LF on BSDS500) ---
    "LF-250 Elliptical (Ns=18)": {
        "build_fn": build_elliptical_kernel,
        "build_kwargs": dict(sigma_along=3.67, sigma_across=3.00, length=23),
        "n_orientations": 18,
    },
    "LF-250 Rectangular (Ns=18)": {
        "build_fn": build_rectangular_kernel,
        "build_kwargs": dict(half_along=11.0, half_across=9.0,
                             sigma_along=3.67, sigma_across=3.00, length=23),
        "n_orientations": 18,
    },
}


def main():
    thresholds = np.linspace(0.01, 0.99, 50)

    # Load data
    images, gts = [], []
    for img_id in IMAGE_IDS:
        img = Image.open(os.path.join(IMG_DIR, f"{img_id}.jpg"))
        gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
        gt = load_gt(img_id)
        images.append(gray)
        gts.append(gt)

    # Run all configs
    results = {}
    t_start = time.perf_counter()

    for name, cfg in CONFIGS.items():
        f_scores = []
        t0 = time.perf_counter()
        for gray, gt in zip(images, gts):
            mag = detect(gray, cfg["build_fn"], cfg["build_kwargs"],
                         cfg["n_orientations"])
            f = compute_ods_f(mag, gt, thresholds)
            f_scores.append(f)
        elapsed = time.perf_counter() - t0
        ods = np.mean(f_scores)
        results[name] = {"ods": ods, "per_image": f_scores, "time": elapsed}
        print(f"  {name:35s}  ODS={ods:.4f}  ({elapsed:.1f}s)")

    total = time.perf_counter() - t_start
    print(f"\nTotal: {total:.1f}s")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Configuration':35s}  {'ODS':>7s}  {'Ns':>4s}  {'Time':>6s}")
    print("-" * 70)

    # Group by ASF config
    groups = [
        ("WVF Np=25 (best overall)", ["WVF-25 Elliptical (Ns=4)",
                                       "WVF-25 Rectangular (Ns=4)"]),
        ("WVF Np=100 (best BSDS500)", ["WVF-100 Elliptical (Ns=4)",
                                        "WVF-100 Rectangular (Ns=4)"]),
        ("LF Np=25 m=3 (best UDED)", ["LF-25 Elliptical (Ns=36)",
                                        "LF-25 Rectangular (Ns=36)"]),
        ("LF Np=100 m=1 (best BIPED)", ["LF-100 Elliptical (Ns=18)",
                                          "LF-100 Rectangular (Ns=18)"]),
        ("LF Np=250 m=2 (best BSDS)", ["LF-250 Elliptical (Ns=18)",
                                         "LF-250 Rectangular (Ns=18)"]),
    ]

    for group_name, members in groups:
        print(f"\n  {group_name}:")
        for name in members:
            r = results[name]
            ns = CONFIGS[name]["n_orientations"]
            print(f"    {name:33s}  {r['ods']:.4f}  {ns:>4d}  "
                  f"{r['time']:.1f}s")

    # ── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    names = list(results.keys())
    ods_vals = [results[n]["ods"] for n in names]
    colors_list = []
    for n in names:
        if "Elliptical" in n:
            colors_list.append("#466A9F")
        else:
            colors_list.append("#1F414D")

    x = np.arange(len(names))
    bars = ax.bar(x, ods_vals, color=colors_list, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ODS F-score", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, ods_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#466A9F", label="Elliptical"),
                       Patch(facecolor="#1F414D", label="Rectangular")],
              frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "matched_ablation_ods.png"),
                dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "matched_ablation_ods.pdf"),
                bbox_inches="tight")
    plt.close()

    print(f"\nSaved: matched_ablation_ods.png/pdf")


if __name__ == "__main__":
    main()
