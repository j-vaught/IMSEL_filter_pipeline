"""End-to-end timing of WVF + LF + orientation recovery, full image.

Loads the noisy synthetic 4096x4096 image (AWGN sigma=13), splits into
L/R/G/B, then for each (channel, radius, degree, lf_half_length) combo:
    1. WVF gradients (Metal)
    2. LF orientation stack at n_orientations (Metal)
    3. Reshape (n_orient, H, W) -> (H*W, n_orient) and run Metal
       orientation recovery -> (theta_p, M_p, theta_s, M_s, v)

Reports per-stage and total wall time.  Saves nothing -- pure timing.
This is the upper bound on dump generation if vMM fusion were free.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.lf._metal import lf_stack
from edgecritic.recovery._metal import (
    recover_two_peaks_metal, recovery_backend_available)


def add_awgn(rgb_u8, sigma, rng):
    arr = rgb_u8.astype(np.float32)
    arr += rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr, 0.0, 255.0)


def split_channels(rgb_f32):
    R = rgb_f32[..., 0]
    G = rgb_f32[..., 1]
    B = rgb_f32[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return {"L": L, "R": R, "G": G, "B": B}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--sigma", type=float, default=13.0)
    p.add_argument("--radii",   default="5,9")
    p.add_argument("--degrees", default="1,3")
    p.add_argument("--lf-half-lengths", default="40,60,80,100")
    p.add_argument("--n-orientations", type=int, default=64)
    p.add_argument("--method", default="box")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not recovery_backend_available():
        print("FATAL: Metal recovery backend not available.")
        sys.exit(1)

    radii   = [int(s) for s in args.radii.split(",")]
    degrees = [int(s) for s in args.degrees.split(",")]
    ms      = [int(s) for s in args.lf_half_lengths.split(",")]

    rng = np.random.default_rng(args.seed)
    rgb_u8 = np.asarray(Image.open(args.image).convert("RGB"))
    H, W = rgb_u8.shape[:2]
    print(f"image  : {args.image.name}  size={H}x{W}")
    print(f"sigma  : {args.sigma}")
    print(f"radii  : {radii}   degrees: {degrees}   "
          f"lf_half_lengths: {ms}   n_orient: {args.n_orientations}")
    rgb_n = add_awgn(rgb_u8, args.sigma, rng)
    channels = split_channels(rgb_n)

    angles = np.linspace(0.0, math.pi, args.n_orientations, endpoint=False)

    n_combos = len(channels) * len(radii) * len(degrees) * len(ms)
    print(f"total combos: {n_combos}")
    print()

    # warmup recovery once on a tiny slab
    warm = np.zeros((1024, args.n_orientations), dtype=np.float32)
    _ = recover_two_peaks_metal(angles, warm,
                                tau_sec_floor=0.40,
                                tau_validity=0.10,
                                dense_n=500,
                                min_sep_frac=0.125)

    t_wvf = 0.0
    t_lf  = 0.0
    t_rec = 0.0
    t_reshape = 0.0
    t_wall0 = time.perf_counter()
    combo = 0

    for ch_name, img in channels.items():
        for r in radii:
            for d in degrees:
                kernels = build_wvf_radius_kernels(radius=r, order=d)
                t0 = time.perf_counter()
                gx, gy = wvf_radius_gradients_metal(
                    img.astype(np.float32), kernels,
                    output_dtype=np.float32)
                t_wvf += time.perf_counter() - t0

                for m in ms:
                    combo += 1
                    t0 = time.perf_counter()
                    stack = lf_stack(gx, gy,
                                     lf_half_length=m,
                                     n_orientations=args.n_orientations,
                                     output_dtype=np.float32,
                                     method=args.method)
                    t_lf += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    resp = stack.transpose(1, 2, 0).reshape(
                        H * W, args.n_orientations).copy()
                    t_reshape += time.perf_counter() - t0
                    del stack

                    t0 = time.perf_counter()
                    th_p, M_p, th_s, M_s, v = recover_two_peaks_metal(
                        angles, resp,
                        tau_sec_floor=0.40,
                        tau_validity=0.10,
                        dense_n=500,
                        min_sep_frac=0.125)
                    t_rec += time.perf_counter() - t0

                    n_v = int(v.sum())
                    n_sec = int((~np.isnan(th_s)).sum())
                    print(f"  [{combo:>2}/{n_combos}] ch={ch_name}  r={r}  "
                          f"d={d}  m={m:>3}  "
                          f"v=1: {n_v/(H*W)*100:5.1f}%  "
                          f"sec_kept: {n_sec/(H*W)*100:5.1f}%")
                    del resp, th_p, M_p, th_s, M_s, v

    t_total = time.perf_counter() - t_wall0
    print()
    print("=" * 60)
    print(f"WVF (Metal)              : {t_wvf:>6.2f} s "
          f"({len(channels)*len(radii)*len(degrees)} calls)")
    print(f"LF stack (Metal)         : {t_lf:>6.2f} s "
          f"({n_combos} calls, avg {t_lf/n_combos*1000:.0f} ms)")
    print(f"reshape (transpose+copy) : {t_reshape:>6.2f} s")
    print(f"orientation recovery     : {t_rec:>6.2f} s "
          f"({n_combos} calls, avg {t_rec/n_combos*1000:.0f} ms)")
    print(f"wall time total          : {t_total:>6.2f} s")


if __name__ == "__main__":
    main()
