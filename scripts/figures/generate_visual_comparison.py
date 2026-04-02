"""Generate side-by-side visual comparisons: Original LF vs Anisotropic Stencil Filter.

Produces gradient magnitude maps and thresholded edge maps for a few images
from each dataset, saving publication-quality comparison figures.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import scipy.io as sio

from edgecritic.lf import lf_image
from edgecritic.aniso_lf import aniso_lf_triton

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "anisotropic-lf-proposal" / "figures" / "visual_results"
OUT.mkdir(parents=True, exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")


# --- Dataset loaders (2 images each) ---
def load_bsds500(n=2):
    base1 = ROOT / "datasets" / "BSDS500" / "BSDS500" / "data"
    base2 = ROOT / "datasets" / "BSDS500" / "BSDS500"
    base = base1 if (base1 / "images" / "test").exists() else base2
    img_dir = base / "images" / "test"
    gt_dir = base / "groundTruth" / "test"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt_mat = sio.loadmat(str(gt_dir / f"{f.stem}.mat"))
        gt_cell = gt_mat["groundTruth"]
        gt_union = np.zeros(img.shape[:2], dtype=bool)
        for i in range(gt_cell.shape[1]):
            bdry = gt_cell[0, i]["Boundaries"][0, 0]
            bdry = bdry.toarray() if hasattr(bdry, "toarray") else np.asarray(bdry)
            gt_union |= (bdry > 0)
        images.append(img)
        gts.append(gt_union)
        names.append(f.stem)
        if len(images) >= n:
            break
    return images, gts, names


def load_biped_v1(n=2):
    base = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
    img_dir = base / "imgs" / "test" / "rgbr"
    gt_dir = base / "edge_maps" / "test" / "rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
        if len(images) >= n:
            break
    return images, gts, names


def load_uded(n=2):
    img_dir = ROOT / "datasets" / "UDED" / "imgs"
    gt_dir = ROOT / "datasets" / "UDED" / "gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f.name).convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
        if len(images) >= n:
            break
    return images, gts, names


def estimate_vram_budget(H, W, half_width, np_count, total_vram_gb):
    border = int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2
    n_pixels = max((H - 2 * border) * (W - 2 * border), 1)
    L = 2 * half_width + 1
    per_pixel = L * (np_count * 24 + 20)
    total_needed_gb = n_pixels * per_pixel / 1e9
    if total_needed_gb < total_vram_gb * 0.4:
        return total_vram_gb * 0.5
    else:
        return total_vram_gb * 0.35


def auto_threshold(mag, percentile=85):
    """Pick threshold at given percentile of nonzero values."""
    nonzero = mag[mag > 0]
    if len(nonzero) == 0:
        return 0.5
    return np.percentile(nonzero, percentile)


def make_comparison(img, gt, name, dataset, configs):
    """Generate comparison figure for one image across multiple configs."""

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    n_configs = len(configs)

    # Run all methods
    results = {}
    for cfg in configs:
        label = cfg["label"]
        m = cfg["m"]
        Np = cfg["Np"]
        Ns = cfg["Ns"]
        d = cfg["d"]

        print(f"  {label}...", end=" ", flush=True)

        # Original LF
        H, W = img.shape
        vram = estimate_vram_budget(H, W, m, Np, total_vram)
        orig = lf_image(img, half_width=m, np_count=Np, order=d,
                        n_orientations=Ns, backend="cuda", max_vram_gb=vram)
        orig_mag = orig.gradient_mag

        # Anisotropic (Triton)
        aniso_mag, aniso_angle, _ = aniso_lf_triton(
            img, half_width=m, np_count=Np, order=d,
            n_orientations=Ns)

        results[label] = {"orig": orig_mag, "aniso": aniso_mag}
        print("done")

    # --- Figure: gradient magnitude comparison ---
    for cfg in configs:
        label = cfg["label"]
        orig_mag = results[label]["orig"]
        aniso_mag = results[label]["aniso"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Row 1: gradient magnitudes
        axes[0, 0].imshow(img, cmap="gray")
        axes[0, 0].set_title("Input Image", fontsize=12)
        axes[0, 0].axis("off")

        vmax = max(np.percentile(orig_mag[orig_mag > 0], 99) if np.any(orig_mag > 0) else 1,
                   np.percentile(aniso_mag[aniso_mag > 0], 99) if np.any(aniso_mag > 0) else 1)

        axes[0, 1].imshow(orig_mag, cmap="hot", vmin=0, vmax=vmax)
        axes[0, 1].set_title(f"Original LF  (m={cfg['m']}, Np={cfg['Np']})", fontsize=12)
        axes[0, 1].axis("off")

        axes[0, 2].imshow(aniso_mag, cmap="hot", vmin=0, vmax=vmax)
        axes[0, 2].set_title(f"Anisotropic Stencil Filter", fontsize=12)
        axes[0, 2].axis("off")

        # Row 2: thresholded edges + ground truth
        axes[1, 0].imshow(gt, cmap="gray")
        axes[1, 0].set_title("Ground Truth", fontsize=12)
        axes[1, 0].axis("off")

        thresh_orig = auto_threshold(orig_mag, 80)
        thresh_aniso = auto_threshold(aniso_mag, 80)

        axes[1, 1].imshow(orig_mag > thresh_orig, cmap="gray")
        axes[1, 1].set_title(f"LF Edges (t={thresh_orig:.1f})", fontsize=12)
        axes[1, 1].axis("off")

        axes[1, 2].imshow(aniso_mag > thresh_aniso, cmap="gray")
        axes[1, 2].set_title(f"ASF Edges (t={thresh_aniso:.1f})", fontsize=12)
        axes[1, 2].axis("off")

        plt.suptitle(f"{dataset} / {name} — {label}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = OUT / f"{dataset}_{name}_{label}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path.name}")

    # --- Figure: zoomed detail comparison ---
    # Pick the best config for a zoomed view
    best_cfg = configs[0]
    label = best_cfg["label"]
    orig_mag = results[label]["orig"]
    aniso_mag = results[label]["aniso"]

    H, W = img.shape
    # Crop center 30% of image
    cy, cx = H // 2, W // 2
    ch, cw = H // 3, W // 3
    y0, y1 = cy - ch // 2, cy + ch // 2
    x0, x1 = cx - cw // 2, cx + cw // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img[y0:y1, x0:x1], cmap="gray")
    axes[0].set_title("Input (center crop)", fontsize=12)
    axes[0].axis("off")

    vmax_crop = max(
        np.percentile(orig_mag[y0:y1, x0:x1][orig_mag[y0:y1, x0:x1] > 0], 99)
        if np.any(orig_mag[y0:y1, x0:x1] > 0) else 1,
        np.percentile(aniso_mag[y0:y1, x0:x1][aniso_mag[y0:y1, x0:x1] > 0], 99)
        if np.any(aniso_mag[y0:y1, x0:x1] > 0) else 1,
    )

    axes[1].imshow(orig_mag[y0:y1, x0:x1], cmap="hot", vmin=0, vmax=vmax_crop)
    axes[1].set_title("Original LF (zoomed)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(aniso_mag[y0:y1, x0:x1], cmap="hot", vmin=0, vmax=vmax_crop)
    axes[2].set_title("ASF (zoomed)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(f"{dataset} / {name} — Detail View", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = OUT / f"{dataset}_{name}_zoom.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# --- Main ---
CONFIGS = [
    {"m": 2, "Np": 100, "Ns": 18, "d": 4, "label": "m2-Np100"},
    {"m": 7, "Np": 100, "Ns": 18, "d": 4, "label": "m7-Np100"},
]

print("Loading datasets...")
for ds_name, loader in [("BSDS500", load_bsds500),
                         ("BIPED", load_biped_v1),
                         ("UDED", load_uded)]:
    try:
        imgs, gts, names = loader(n=2)
        print(f"\n{ds_name}: {len(imgs)} images")
        for img, gt, name in zip(imgs, gts, names):
            print(f"\n  Image: {name} ({img.shape})")
            make_comparison(img, gt, name, ds_name, CONFIGS)
    except Exception as e:
        print(f"\n{ds_name}: FAILED ({e})")

print("\nDone! All figures saved to", OUT)
