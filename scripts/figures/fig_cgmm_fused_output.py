"""Render the four raster panels for fig_cgmm_fused_output.typ.

Writes 4 PNGs into the paper's cetz_figures data directory:
    panel_a_input.png       - source RGB image
    panel_b_magnitude.png   - fused magnitude M_fused (inferno colormap)
    panel_c_primary.png     - primary fused orientation (hsv colormap)
    panel_d_secondary.png   - secondary fused orientation (hsv colormap)

The Typst file then assembles them into a 1x4 grid for the paper.

Usage:
    python3 scripts/figures/fig_cgmm_fused_output.py \\
        --dump outputs/cgmm_fusion_dump/clean_K3_hardEM.npz \\
        --source-image example_images/.../*_4096.png \\
        --out-dir "/path/to/cetz_figures/data/cgmm_fused_panels"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage


BG_RGB = np.array([0, 0, 0], dtype=np.uint8)   # pure black background


def _dilate(values, valid_mask, dilate_px):
    """Distance-transform-based nearest-pixel value fill for dilation."""
    if dilate_px <= 0:
        return values, valid_mask
    dist, (yi, xi) = ndimage.distance_transform_edt(
        ~valid_mask, return_indices=True)
    return values[yi, xi], dist <= dilate_px


def render_orientation(theta_rad, valid_mask, cmap, dilate_px=6):
    """theta_rad in [0, pi). Returns (H, W, 3) uint8 RGB."""
    H, W = theta_rad.shape
    out = np.broadcast_to(BG_RGB, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    theta_filled, rendered_mask = _dilate(theta_rad, valid_mask, dilate_px)
    rgba = cmap(theta_filled[rendered_mask] / np.pi)
    out[rendered_mask] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def render_magnitude(M, valid_mask, cmap, dilate_px=6, vmax=None):
    """M >= 0. Returns (H, W, 3) uint8 RGB."""
    H, W = M.shape
    out = np.broadcast_to(BG_RGB, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    if vmax is None:
        vmax = float(np.percentile(M[valid_mask], 99))
    M_filled, rendered_mask = _dilate(M, valid_mask, dilate_px)
    norm = np.clip(M_filled[rendered_mask] / max(vmax, 1e-6), 0.0, 1.0)
    rgba = cmap(norm)
    out[rendered_mask] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--source-image", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="directory to write the four panel_*.png files")
    p.add_argument("--primary-dilate-px",   type=int, default=6)
    p.add_argument("--secondary-dilate-px", type=int, default=15)
    p.add_argument("--magnitude-dilate-px", type=int, default=6)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    # Panel (a): input RGB image.
    img_input = np.asarray(Image.open(args.source_image).convert("RGB"))
    Image.fromarray(img_input).save(args.out_dir / "panel_a_input.png",
                                    optimize=True)

    # Panel (b): fused magnitude.
    img_M = render_magnitude(M_fused, M_valid, cmap_mag,
                             dilate_px=args.magnitude_dilate_px)
    Image.fromarray(img_M).save(args.out_dir / "panel_b_magnitude.png",
                                optimize=True)

    # Panel (c): primary fused orientation.
    img_p = render_orientation(np.where(np.isnan(theta_p), 0.0, theta_p),
                                primary_valid, cmap_orient,
                                dilate_px=args.primary_dilate_px)
    Image.fromarray(img_p).save(args.out_dir / "panel_c_primary.png",
                                optimize=True)

    # Panel (d): secondary fused orientation.
    img_s = render_orientation(np.where(np.isnan(theta_s), 0.0, theta_s),
                                secondary_valid, cmap_orient,
                                dilate_px=args.secondary_dilate_px)
    Image.fromarray(img_s).save(args.out_dir / "panel_d_secondary.png",
                                optimize=True)

    print(f"wrote 4 panels to {args.out_dir}")
    print(f"  M_valid={int(M_valid.sum())}  "
          f"primary={int(primary_valid.sum())}  "
          f"secondary={int(secondary_valid.sum())}")


if __name__ == "__main__":
    main()
