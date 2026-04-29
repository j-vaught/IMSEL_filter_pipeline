"""Regenerate every paper panel that depends on the full-image c-GMM dump.

Produces, for the noisy low_contrast_mixed_chroma 4096 image:
  cgmm_fused_panels/
    panel_a_input.png            (full)
    panel_b_magnitude.png        (full)
    panel_c_primary.png          (full)
    panel_d_secondary.png        (full)
  cgmm_nms_binary_panels/
    nested_low_contrast_mixed_chroma_4096_noisy_sigma13.png        (full input)
    nested_low_contrast_mixed_chroma_4096_noisy_sigma13_c512.png   (cropped input)
    panel_b_magnitude_c512.png                                     (cropped M_fused)
    binary_standard_N1_A8.png       (full traditional NMS)
    binary_standard_N1_A8_c512.png  (cropped)
    binary_N4_Acont.png             (full enhanced NMS at N4/Acont, corner-OR)
    binary_N4_Acont_c512.png        (cropped)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from cgmm_nms import enhanced_nms, standard_nms


BG_RGB = np.array([0, 0, 0], dtype=np.uint8)


def _dilate(values, valid_mask, dilate_px):
    if dilate_px <= 0:
        return values, valid_mask
    dist, (yi, xi) = ndimage.distance_transform_edt(
        ~valid_mask, return_indices=True)
    return values[yi, xi], dist <= dilate_px


def render_orientation(theta_rad, valid_mask, cmap, dilate_px=6):
    H, W = theta_rad.shape
    out = np.broadcast_to(BG_RGB, (H, W, 3)).copy()
    if not valid_mask.any():
        return out
    theta_filled, rendered = _dilate(theta_rad, valid_mask, dilate_px)
    rgba = cmap(theta_filled[rendered] / np.pi)
    out[rendered] = (rgba[..., :3] * 255).astype(np.uint8)
    return out


def render_magnitude(M, valid_mask, cmap, dilate_px=6, vmax=None):
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


def crop_center(arr, side):
    H, W = arr.shape[:2]
    r0 = (H - side) // 2
    c0 = (W - side) // 2
    return arr[r0:r0 + side, c0:c0 + side]


def save_binary(mask, path, fg=255, bg=0):
    img = np.full(mask.shape, bg, dtype=np.uint8)
    img[mask] = fg
    Image.fromarray(img, mode="L").save(path, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--source-image", required=True, type=Path,
                   help="clean RGB; the noisy version is generated with the "
                        "same RNG seed as the dump")
    p.add_argument("--out-dir-fused", required=True, type=Path,
                   help="cetz_figures/data/cgmm_fused_panels")
    p.add_argument("--out-dir-nms", required=True, type=Path,
                   help="cetz_figures/data/cgmm_nms_binary_panels")
    p.add_argument("--sigma", type=float, default=13.0)
    p.add_argument("--seed",  type=int,   default=0)
    p.add_argument("--crop",  type=int,   default=512)
    p.add_argument("--magnitude-dilate-px", type=int, default=0)
    p.add_argument("--orientation-dilate-px", type=int, default=0)
    p.add_argument("--secondary-dilate-px",   type=int, default=0)
    p.add_argument("--basename", default="nested_low_contrast_mixed_chroma_4096_noisy_sigma13")
    args = p.parse_args()

    args.out_dir_fused.mkdir(parents=True, exist_ok=True)
    args.out_dir_nms.mkdir(parents=True, exist_ok=True)

    # ---- load dump ----
    print(f"loading {args.dump}...")
    d = np.load(args.dump, allow_pickle=False)
    theta_p = d["theta_fused"]
    theta_s = d["theta_fused_sec"]
    M_fused = d["M_fused"]
    M_sec   = d["M_fused_sec"]
    v       = d["v_fused"].astype(bool)
    suppr   = d["suppressed"].astype(bool)
    H, W = theta_p.shape
    print(f"  size {H}x{W}  v=1: {v.mean()*100:.1f}%")

    # ---- (1) noisy input image, full + cropped ----
    rng = np.random.default_rng(args.seed)
    rgb = np.asarray(Image.open(args.source_image).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(rgb + rng.normal(0.0, args.sigma, rgb.shape).astype(np.float32),
                    0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb_n).save(args.out_dir_fused / "panel_a_input.png",
                                 optimize=True)
    Image.fromarray(rgb_n).save(args.out_dir_nms / f"{args.basename}.png",
                                 optimize=True)
    rgb_c = crop_center(rgb_n, args.crop)
    Image.fromarray(rgb_c).save(args.out_dir_nms / f"{args.basename}_c{args.crop}.png",
                                 optimize=True)
    print(f"  wrote panel_a_input.png and {args.basename}.png + _c{args.crop}.png")

    # ---- (2) M_fused panel (full + crop) ----
    cmap_mag = matplotlib.colormaps["inferno"]
    primary_valid   = v & ~np.isnan(theta_p)
    secondary_valid = v & ~np.isnan(theta_s) & ~suppr
    M_valid         = v & (M_fused > 0)
    img_M = render_magnitude(M_fused, M_valid, cmap_mag,
                             dilate_px=args.magnitude_dilate_px)
    Image.fromarray(img_M).save(args.out_dir_fused / "panel_b_magnitude.png",
                                 optimize=True)
    img_M_c = crop_center(img_M, args.crop)
    Image.fromarray(img_M_c).save(
        args.out_dir_nms / f"panel_b_magnitude_c{args.crop}.png",
        optimize=True)
    print("  wrote panel_b_magnitude.png + cropped")

    # ---- (3) primary + secondary orientation panels ----
    cmap_orient = matplotlib.colormaps["hsv"]
    img_p = render_orientation(np.where(np.isnan(theta_p), 0.0, theta_p),
                                primary_valid, cmap_orient,
                                dilate_px=args.orientation_dilate_px)
    Image.fromarray(img_p).save(args.out_dir_fused / "panel_c_primary.png",
                                 optimize=True)
    img_s = render_orientation(np.where(np.isnan(theta_s), 0.0, theta_s),
                                secondary_valid, cmap_orient,
                                dilate_px=args.secondary_dilate_px)
    Image.fromarray(img_s).save(args.out_dir_fused / "panel_d_secondary.png",
                                 optimize=True)
    print("  wrote panel_c_primary.png and panel_d_secondary.png")

    # ---- (4) traditional NMS at N1/A8 (Canny reference) ----
    print("  running standard NMS N1/A8...")
    out_std = standard_nms(theta_p, M_fused, v.astype(np.uint8),
                           neighborhood=1, angular_fidelity="A8")
    kept_std = out_std > 0
    save_binary(kept_std, args.out_dir_nms / "binary_standard_N1_A8.png")
    save_binary(crop_center(kept_std, args.crop),
                args.out_dir_nms / f"binary_standard_N1_A8_c{args.crop}.png")
    print(f"    standard kept: {int(kept_std.sum()):,}")

    # ---- (5) enhanced NMS at N4/Acont (corner-OR rule) ----
    print("  running enhanced NMS N4/Acont with corner-OR...")
    out_enh = enhanced_nms(theta_p, M_fused, theta_s, M_sec,
                           v.astype(np.uint8),
                           neighborhood=4, angular_fidelity="Acont",
                           corner_method="or")
    kept_enh = out_enh > 0
    save_binary(kept_enh, args.out_dir_nms / "binary_N4_Acont.png")
    save_binary(crop_center(kept_enh, args.crop),
                args.out_dir_nms / f"binary_N4_Acont_c{args.crop}.png")
    print(f"    enhanced kept: {int(kept_enh.sum()):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
