"""Figure A: per-pixel fused-orientation map of the synthetic test image.

Two panels at the test image's full resolution:
    - Left:  theta_fused (primary)
    - Right: theta_fused_sec (secondary), grey where suppressed.

Pixels with v_fused=0 in either panel get neutral grey (#ECECEC) so
they read as background.  Color-coded with a perceptually uniform
circular colormap (matplotlib 'twilight') over [0 deg, 180 deg).

Output: cetz_figures/pdfs/fig_cgmm_fused_output.pdf
Data:   outputs/cgmm_fusion_dump/<condition>_K3_hardEM.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


BG = np.array([0x00, 0x00, 0x00]) / 255.0   # pure black background


def render_orientation_image(theta_rad, valid_mask, cmap, dilate_px=2):
    """theta_rad in [0, pi).  valid_mask True where colored.
    The colored region is dilated by dilate_px pixels for visibility
    at print size (the GT edge mask is only 1-2 px wide).  Each
    dilated pixel inherits its nearest valid pixel's theta value via
    a distance-transform lookup.
    Returns (H, W, 3) RGB float in [0, 1]."""
    H, W = theta_rad.shape
    out = np.broadcast_to(BG, (H, W, 3)).copy()
    if not valid_mask.any():
        return out

    if dilate_px > 0:
        # Distance-transform-based nearest-pixel theta lookup so dilated
        # neighbours pick up the same orientation as the source pixel.
        dist, (yi, xi) = ndimage.distance_transform_edt(
            ~valid_mask, return_indices=True)
        theta_filled = theta_rad[yi, xi]
        rendered_mask = dist <= dilate_px
    else:
        theta_filled = theta_rad
        rendered_mask = valid_mask

    norm = theta_filled[rendered_mask] / np.pi
    rgba = cmap(norm)
    out[rendered_mask] = rgba[..., :3]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True,
                   help=".npz produced by cgmm_fusion_image_dump.py")
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--source-image", type=Path, default=None,
                   help="optional: underlay the source image at low alpha")
    args = p.parse_args()

    d = np.load(args.dump, allow_pickle=False)
    theta_p = d["theta_fused"]            # (H, W) float32 in [0, pi) or NaN
    theta_s = d["theta_fused_sec"]
    v       = d["v_fused"].astype(bool)
    suppr   = d["suppressed"].astype(bool)
    H, W = theta_p.shape

    # 'hsv' has full saturation across the cycle — visible on black bg.
    # 'twilight' is perceptually uniform but has near-black midtones
    # which would blend into the background.
    cmap = matplotlib.colormaps["hsv"]

    primary_valid   = v & ~np.isnan(theta_p)
    secondary_valid = v & ~np.isnan(theta_s) & ~suppr

    img_p = render_orientation_image(np.where(np.isnan(theta_p), 0.0, theta_p),
                                      primary_valid, cmap, dilate_px=6)
    img_s = render_orientation_image(np.where(np.isnan(theta_s), 0.0, theta_s),
                                      secondary_valid, cmap, dilate_px=6)

    fig = plt.figure(figsize=(10.0, 5.6))
    grid = fig.add_gridspec(1, 2, wspace=0.06, left=0.02, right=0.98,
                            top=0.98, bottom=0.18)
    axl = fig.add_subplot(grid[0, 0])
    axr = fig.add_subplot(grid[0, 1])

    axl.imshow(img_p, interpolation="nearest", origin="upper",
               extent=(0, W, H, 0))
    axl.set_xticks([]); axl.set_yticks([])
    axl.set_facecolor("black")
    for s in axl.spines.values(): s.set_visible(False)
    axl.set_xlabel("primary", fontsize=11, color="#363636",
                   labelpad=8)

    axr.imshow(img_s, interpolation="nearest", origin="upper",
               extent=(0, W, H, 0))
    axr.set_xticks([]); axr.set_yticks([])
    axr.set_facecolor("black")
    for s in axr.spines.values(): s.set_visible(False)
    axr.set_xlabel("secondary", fontsize=11, color="#363636",
                   labelpad=8)

    # Shared colorbar at the bottom showing the orientation->color map.
    cbar_ax = fig.add_axes((0.30, 0.06, 0.40, 0.025))
    grad = np.linspace(0, 1, 360).reshape(1, -1)
    cbar_ax.imshow(np.tile(grad, (10, 1)), aspect="auto", cmap=cmap,
                   extent=(0, 180, 0, 1))
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([0, 45, 90, 135, 180])
    cbar_ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°"],
                            fontsize=8)
    cbar_ax.tick_params(length=2, color="#363636")
    for s in cbar_ax.spines.values(): s.set_color("#363636")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  "
          f"primary={int(primary_valid.sum())}  "
          f"secondary={int(secondary_valid.sum())}")


if __name__ == "__main__":
    main()
