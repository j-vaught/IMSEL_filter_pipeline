"""Run vMM hard-EM K=3 fusion at every analytical edge pixel of the
synthetic test image and dump per-pixel outputs to .npz.

Runs on all_mask = (smooth_mask | junction_mask). Off-edge pixels are
skipped: they have no GT edge, so we set v_fused=0 and grey them out
in the figure.

Output schema (per pixel of the 4096x4096 grid):
    theta_fused        float32  [0, pi)        primary orientation, NaN where v=0
    theta_fused_sec    float32  [0, pi) or NaN secondary orientation, NaN where suppressed or v=0
    M_fused            float32                 primary signal weight
    M_fused_sec        float32                 secondary weight (0 if suppressed)
    v_fused            uint8    {0, 1}         pixel-level validity
    suppressed         uint8    {0, 1}         1 where v=1 but secondary slot was suppressed

Usage:
    PYTHONPATH=src:scripts/eval python3 \\
        scripts/eval/cgmm_fusion_image_dump.py \\
        --clean-rgb <path> --noisy-dir <path> --manifest <path> \\
        --out outputs/cgmm_fusion_dump/<condition>.npz
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cgmm_vmm import vmm_fuse, theta_M_to_phi_w
from cgmm_image_wide_eval import (
    build_gt_orientation, find_image_spec, load_channels_clean,
    load_channels_noisy, evaluate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-rgb",  required=True, type=Path)
    p.add_argument("--noisy-dir",  required=True, type=Path)
    p.add_argument("--manifest",   required=True, type=Path)
    p.add_argument("--image-key",  default="garnet_atlantic_grass")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--condition", choices=["clean", "noisy"], default="clean")
    p.add_argument("--r", type=int, default=9)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--m-values", default="0,5,10,20,30,40,50,60,70,80")
    p.add_argument("--n-orientations", type=int, default=64)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--vmm-tau-M-rel", type=float, default=0.10)
    p.add_argument("--vmm-rho",       type=float, default=0.40)
    p.add_argument("--vmm-theta-min-deg", type=float, default=10.0)
    p.add_argument("--vmm-n-iters", type=int, default=30)
    p.add_argument("--vertex-exclude-px", type=int, default=24)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    spec = find_image_spec(args.manifest, args.image_key, args.size)
    print(f"image: {spec['palette']} size={spec['size']}")

    t0 = time.perf_counter()
    gt_normal, all_mask, smooth_mask = build_gt_orientation(
        spec, edge_band_px=0,
        vertex_exclude_px=args.vertex_exclude_px)
    junction_mask = all_mask & ~smooth_mask
    print(f"GT built: smooth={smooth_mask.sum():,} "
          f"junction={junction_mask.sum():,} "
          f"({time.perf_counter()-t0:.1f}s)")

    # Run fusion at every analytical edge pixel.
    ys, xs = np.where(all_mask)
    sample_pixels = np.column_stack([xs, ys])
    print(f"running LF + vMM fusion at {len(ys):,} pixels")

    if args.condition == "clean":
        channels = load_channels_clean(args.clean_rgb)
    else:
        channels = load_channels_noisy(args.noisy_dir)

    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]
    # Pre-compute per-pixel GT just for the dump (not used in fusion).
    gt_normal_at = gt_normal[ys, xs].astype(np.float64)
    gt_tangent_deg = np.degrees((gt_normal_at + math.pi / 2.0)
                                 % math.pi).astype(np.float32)

    t0 = time.perf_counter()
    primary_t, primary_m = evaluate(args.condition, channels,
                                     sample_pixels, gt_tangent_deg,
                                     m_values, args.n_orientations,
                                     args.r, args.d)
    print(f"LF stage: {time.perf_counter() - t0:.1f}s "
          f"-> primary_t shape {primary_t.shape}")

    phi, w, _ = theta_M_to_phi_w(primary_t, primary_m)
    t0 = time.perf_counter()
    out = vmm_fuse(phi, w, K=args.K, n_iters=args.vmm_n_iters,
                   hard_em=True,
                   tau_M_rel=args.vmm_tau_M_rel,
                   rho=args.vmm_rho,
                   theta_min_deg=args.vmm_theta_min_deg)
    print(f"vMM K={args.K} hard-EM fusion: {time.perf_counter()-t0:.2f}s")

    # Build full 4096^2 arrays (off-edge pixels stay at sentinels).
    H = W = args.size
    theta_fused_img    = np.full((H, W), np.nan,  dtype=np.float32)
    theta_fused_sec_img = np.full((H, W), np.nan, dtype=np.float32)
    M_fused_img        = np.zeros((H, W), dtype=np.float32)
    M_fused_sec_img    = np.zeros((H, W), dtype=np.float32)
    v_fused_img        = np.zeros((H, W), dtype=np.uint8)
    suppressed_img     = np.zeros((H, W), dtype=np.uint8)
    is_junction_img    = junction_mask.astype(np.uint8)

    theta_fused_img[ys, xs]     = out["theta_fused"].astype(np.float32)
    theta_fused_sec_img[ys, xs] = out["theta_fused_sec"].astype(np.float32)
    M_fused_img[ys, xs]         = out["M_fused"].astype(np.float32)
    M_fused_sec_img[ys, xs]     = out["M_fused_sec"].astype(np.float32)
    v_fused_img[ys, xs]         = out["v_fused"]

    # Suppressed = pixel was valid (v_fused=1) but theta_fused_sec is NaN.
    sec_nan = np.isnan(out["theta_fused_sec"])
    suppressed_pixels = (out["v_fused"] == 1) & sec_nan
    suppressed_img[ys, xs] = suppressed_pixels.astype(np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
        condition       = args.condition,
        size            = args.size,
        theta_fused     = theta_fused_img,
        theta_fused_sec = theta_fused_sec_img,
        M_fused         = M_fused_img,
        M_fused_sec     = M_fused_sec_img,
        v_fused         = v_fused_img,
        suppressed      = suppressed_img,
        is_junction     = is_junction_img,
        gt_tangent_deg  = gt_tangent_deg.astype(np.float32),
        all_mask        = all_mask,
        smooth_mask     = smooth_mask,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
