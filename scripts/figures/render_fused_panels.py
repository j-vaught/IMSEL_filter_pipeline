"""Render the c-GMM fused-output panels (magnitude, primary theta,
secondary theta) for any dump.

Writes three PNGs into the given output directory:
    panel_b_magnitude.png  (inferno colormap on M_fused)
    panel_c_primary.png    (HSV colormap on theta_fused)
    panel_d_secondary.png  (HSV colormap on theta_fused_sec)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

BG_RGB = np.array([0, 0, 0], dtype=np.uint8)


def _dilate(values, valid_mask, dilate_px):
    if dilate_px <= 0:
        return values, valid_mask
    dist, (yi, xi) = ndimage.distance_transform_edt(
        ~valid_mask, return_indices=True)
    return values[yi, xi], dist <= dilate_px


def render_orientation(theta_rad, valid_mask, cmap, dilate_px=0):
    H, W = theta_rad.shape
    out = np.broadcast_to(BG_RGB, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    th_filled, rendered = _dilate(theta_rad, valid_mask, dilate_px)
    rgba = cmap(th_filled[rendered] / np.pi)
    out[rendered] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def render_magnitude(M, valid_mask, cmap, dilate_px=0, vmax=None):
    H, W = M.shape
    out = np.broadcast_to(BG_RGB, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    if vmax is None:
        vmax = float(np.percentile(M[valid_mask], 99))
    M_filled, rendered = _dilate(M, valid_mask, dilate_px)
    norm = np.clip(M_filled[rendered] / max(vmax, 1e-6), 0.0, 1.0)
    rgba = cmap(norm)
    out[rendered] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--magnitude-dilate-px", type=int, default=0)
    p.add_argument("--orientation-dilate-px", type=int, default=0)
    p.add_argument("--secondary-dilate-px",   type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.dump}")
    d = np.load(args.dump, allow_pickle=False)
    th_p = d["theta_fused"]
    th_s = d["theta_fused_sec"]
    M    = d["M_fused"]
    v    = d["v_fused"].astype(bool)
    suppr = d["suppressed"].astype(bool)
    print(f"  size {M.shape}  v=1: {v.mean()*100:.1f}%")

    primary_valid   = v & ~np.isnan(th_p)
    secondary_valid = v & ~np.isnan(th_s) & ~suppr
    M_valid         = v & (M > 0)

    cmap_orient = matplotlib.colormaps["hsv"]
    cmap_mag    = matplotlib.colormaps["inferno"]

    img_M = render_magnitude(M, M_valid, cmap_mag,
                             dilate_px=args.magnitude_dilate_px)
    Image.fromarray(img_M).save(args.out_dir / "panel_b_magnitude.png",
                                 optimize=True)
    print(f"  wrote panel_b_magnitude.png  M_valid={int(M_valid.sum()):,}")

    img_p = render_orientation(np.where(np.isnan(th_p), 0.0, th_p),
                                primary_valid, cmap_orient,
                                dilate_px=args.orientation_dilate_px)
    Image.fromarray(img_p).save(args.out_dir / "panel_c_primary.png",
                                 optimize=True)
    print(f"  wrote panel_c_primary.png    primary={int(primary_valid.sum()):,}")

    img_s = render_orientation(np.where(np.isnan(th_s), 0.0, th_s),
                                secondary_valid, cmap_orient,
                                dilate_px=args.secondary_dilate_px)
    Image.fromarray(img_s).save(args.out_dir / "panel_d_secondary.png",
                                 optimize=True)
    print(f"  wrote panel_d_secondary.png  secondary={int(secondary_valid.sum()):,}")


if __name__ == "__main__":
    main()
