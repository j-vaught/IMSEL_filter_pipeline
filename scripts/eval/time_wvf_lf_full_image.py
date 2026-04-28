"""Time WVF + LF (Metal) over full image, no GT mask, no fusion.

Adds AWGN with given sigma, splits into L/R/G/B, then for each
(channel, radius, degree, m) computes Metal WVF gradients and a Metal
full-frame LF orientation stack at n_orientations.

Reports per-call and total wall-clock time.  Does NOT do orientation
recovery or vMM fusion -- just the GPU stages so we can see what is
actually fast.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make src/ importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.lf._metal import lf_orientation_stack_metal


def add_awgn(rgb_u8, sigma, rng):
    arr = rgb_u8.astype(np.float32)
    arr = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
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
    p.add_argument("--sigma", type=float, default=13.0,
                   help="AWGN sigma in 0-255 units")
    p.add_argument("--radii",   default="5,9")
    p.add_argument("--degrees", default="1,3")
    p.add_argument("--m-values", default="40,60,80,100")
    p.add_argument("--n-orientations", type=int, default=64)
    p.add_argument("--method", default="box",
                   choices=["exact", "box", "scanline"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    radii   = [int(s) for s in args.radii.split(",")]
    degrees = [int(s) for s in args.degrees.split(",")]
    ms      = [int(s) for s in args.m_values.split(",")]

    rng = np.random.default_rng(args.seed)
    rgb_u8 = np.asarray(Image.open(args.image).convert("RGB"))
    H, W = rgb_u8.shape[:2]
    print(f"image: {args.image.name}  size={H}x{W}")
    print(f"sigma={args.sigma}  radii={radii}  degrees={degrees}  "
          f"m_values={ms}  n_orient={args.n_orientations}  "
          f"method={args.method}")

    rgb_noisy = add_awgn(rgb_u8, args.sigma, rng)
    channels = split_channels(rgb_noisy)

    n_wvf_total = len(channels) * len(radii) * len(degrees)
    n_lf_total  = n_wvf_total * len(ms)
    print(f"WVF calls: {n_wvf_total}   LF orientation stacks: {n_lf_total}")
    print(f"per-stack output: ({args.n_orientations}, {H}, {W}) float32 "
          f"= {args.n_orientations * H * W * 4 / 1e9:.2f} GB")
    print()

    wvf_total = 0.0
    lf_total  = 0.0
    t_wall0   = time.perf_counter()

    for ch_name, img in channels.items():
        for r in radii:
            for d in degrees:
                kernels = build_wvf_radius_kernels(radius=r, order=d)
                t0 = time.perf_counter()
                gx, gy = wvf_radius_gradients_metal(
                    img.astype(np.float32), kernels,
                    output_dtype=np.float32)
                t_wvf = time.perf_counter() - t0
                wvf_total += t_wvf
                print(f"  WVF  ch={ch_name}  r={r}  d={d}  "
                      f"-> ({gx.shape}, {gy.shape})  {t_wvf*1000:.0f} ms")

                for m in ms:
                    t0 = time.perf_counter()
                    stack = lf_orientation_stack_metal(
                        gx, gy, m=m,
                        n_orientations=args.n_orientations,
                        output_dtype=np.float32,
                        method=args.method)
                    t_lf = time.perf_counter() - t0
                    lf_total += t_lf
                    print(f"    LF  m={m:>3}  -> {stack.shape}  "
                          f"{t_lf*1000:.0f} ms")
                    del stack

    t_wall = time.perf_counter() - t_wall0
    print()
    print(f"WVF total : {wvf_total:6.2f} s  ({n_wvf_total} calls, "
          f"avg {wvf_total/n_wvf_total*1000:.0f} ms)")
    print(f"LF total  : {lf_total:6.2f} s  ({n_lf_total} calls, "
          f"avg {lf_total/n_lf_total*1000:.0f} ms)")
    print(f"WVF+LF    : {wvf_total + lf_total:6.2f} s")
    print(f"wall time : {t_wall:6.2f} s")


if __name__ == "__main__":
    main()
