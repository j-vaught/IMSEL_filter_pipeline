"""Figure A: per-pixel fused output of the synthetic test image.

Four panels in a 2x2 grid:
    (a) input image
    (b) fused magnitude M_fused (signal cluster's validity-aware weight)
    (c) primary fused orientation theta_fused
    (d) secondary fused orientation theta_fused_sec, with suppressed
        pixels showing as background.

Pixels with v_fused=0 in (b)-(d) get pure black (background).
Orientation panels use 'hsv' (full saturation across the cycle, visible
on black); the magnitude panel uses 'inferno' (sequential, dark->hot).

Output: cetz_figures/pdfs/fig_cgmm_fused_output.pdf
Data:   outputs/cgmm_fusion_dump/<condition>_K3_hardEM.npz  (+ source image)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


BG = np.array([0x00, 0x00, 0x00]) / 255.0   # pure black background


def _dilate(values, valid_mask, dilate_px):
    """Distance-transform-based nearest-pixel value fill for dilation.
    Returns (filled_values, rendered_mask)."""
    if dilate_px <= 0:
        return values, valid_mask
    dist, (yi, xi) = ndimage.distance_transform_edt(
        ~valid_mask, return_indices=True)
    return values[yi, xi], dist <= dilate_px


def render_orientation_image(theta_rad, valid_mask, cmap, dilate_px=6):
    """theta_rad in [0, pi). Returns (H, W, 3) RGB float in [0, 1]."""
    H, W = theta_rad.shape
    out = np.broadcast_to(BG, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    theta_filled, rendered_mask = _dilate(theta_rad, valid_mask, dilate_px)
    rgba = cmap(theta_filled[rendered_mask] / np.pi)
    out[rendered_mask] = rgba[..., :3]
    return out


def render_magnitude_image(M, valid_mask, cmap, dilate_px=6, vmax=None):
    """M >= 0. Returns (H, W, 3) RGB float in [0, 1].  M is normalised
    by `vmax` (defaults to the 99th percentile of valid pixels)."""
    H, W = M.shape
    out = np.broadcast_to(BG, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    if vmax is None:
        vmax = float(np.percentile(M[valid_mask], 99))
    M_filled, rendered_mask = _dilate(M, valid_mask, dilate_px)
    norm = np.clip(M_filled[rendered_mask] / max(vmax, 1e-6), 0.0, 1.0)
    rgba = cmap(norm)
    out[rendered_mask] = rgba[..., :3]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--source-image", type=Path, required=True,
                   help="path to the input RGB image used for panel (a)")
    args = p.parse_args()

    d = np.load(args.dump, allow_pickle=False)
    theta_p = d["theta_fused"]
    theta_s = d["theta_fused_sec"]
    M_fused = d["M_fused"]
    v       = d["v_fused"].astype(bool)
    suppr   = d["suppressed"].astype(bool)
    H, W = theta_p.shape

    cmap_orient = matplotlib.colormaps["hsv"]
    cmap_mag    = matplotlib.colormaps["inferno"]

    primary_valid   = v & ~np.isnan(theta_p)
    secondary_valid = v & ~np.isnan(theta_s) & ~suppr
    M_valid         = v & (M_fused > 0)

    img_input = np.asarray(Image.open(args.source_image).convert("RGB"))

    img_M = render_magnitude_image(M_fused, M_valid, cmap_mag,
                                   dilate_px=6)
    img_p = render_orientation_image(np.where(np.isnan(theta_p), 0.0, theta_p),
                                      primary_valid, cmap_orient,
                                      dilate_px=6)
    img_s = render_orientation_image(np.where(np.isnan(theta_s), 0.0, theta_s),
                                      secondary_valid, cmap_orient,
                                      dilate_px=15)

    fig = plt.figure(figsize=(14.0, 4.0))
    gs = fig.add_gridspec(1, 4, wspace=0.04,
                          left=0.01, right=0.99,
                          top=0.99, bottom=0.10)

    panels = [
        (gs[0, 0], img_input, "(a) input"),
        (gs[0, 1], img_M,     "(b) fused magnitude"),
        (gs[0, 2], img_p,     "(c) primary"),
        (gs[0, 3], img_s,     "(d) secondary"),
    ]

    for slot, img, label in panels:
        ax = fig.add_subplot(slot)
        ax.imshow(img, interpolation="nearest", origin="upper",
                  extent=(0, W, H, 0))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("black")
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlabel(label, fontsize=10, color="#363636", labelpad=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  "
          f"M_valid={int(M_valid.sum())}  "
          f"primary={int(primary_valid.sum())}  "
          f"secondary={int(secondary_valid.sum())}")


if __name__ == "__main__":
    main()
