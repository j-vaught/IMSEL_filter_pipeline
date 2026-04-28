"""Render per-variant NMS edge maps.

Loads a fusion dump, runs each of the 9 NMS variants, dilates the
NMS-positive mask for visibility (the ridge is 1 px wide), and saves
each as a PNG into cetz_figures/data/cgmm_nms_panels/.

A companion typst figure (cetz_figures/fig_nms_variants.typ) lays them
out in a 3x3 grid for the §8 figure.

Also writes a side-by-side baseline (N1/A8) vs best (recommended
operating point passed in via --best) for the §8 hero figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from cgmm_nms import enhanced_nms


HSV = matplotlib.colormaps["hsv"]
BG = np.array([0x00, 0x00, 0x00], dtype=np.uint8)


def render_orientation(theta_rad, valid_mask, dilate_px=4):
    """theta in [0, pi) -> HSV-coloured (H, W, 3) RGB on black bg."""
    H, W = theta_rad.shape
    out = np.broadcast_to(BG, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    if dilate_px > 0:
        dist, (yi, xi) = ndimage.distance_transform_edt(
            ~valid_mask, return_indices=True)
        theta_filled = theta_rad[yi, xi]
        rendered_mask = dist <= dilate_px
    else:
        theta_filled = theta_rad
        rendered_mask = valid_mask
    rgba = HSV(theta_filled[rendered_mask] / np.pi)
    out[rendered_mask] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",     required=True, type=Path)
    p.add_argument("--out-dir",  required=True, type=Path)
    p.add_argument("--dilate-px", type=int, default=4)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dump, allow_pickle=False)
    M_p = d["M_primary"]; th_p = d["theta_primary"]
    M_s = d["M_sec"];     th_s = d["theta_sec"]
    v   = d["v_fused"]

    for nbh in (1, 2, 3):
        for fid in ("A8", "A16", "Acont"):
            out = enhanced_nms(th_p, M_p, th_s, M_s, v,
                               neighborhood=nbh, angular_fidelity=fid)
            kept = out > 0
            img = render_orientation(np.where(np.isnan(th_p), 0.0, th_p),
                                      kept, dilate_px=args.dilate_px)
            n = int(kept.sum())
            label = f"N{nbh}_{fid}"
            png = args.out_dir / f"panel_{label}.png"
            Image.fromarray(img).save(png, optimize=True)
            print(f"  {label}: kept={n:>7}  -> {png.name}")


if __name__ == "__main__":
    main()
