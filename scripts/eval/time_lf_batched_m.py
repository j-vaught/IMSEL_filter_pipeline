"""Compare batched LF half-lengths with one full-frame LF call per length.

Reports wall time and peak RSS for each path, at several
n_orientations settings.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.lf._metal import (
    lf_stack,
    lf_length_stack,
)


def peak_mb():
    """macOS reports ru_maxrss in BYTES (linux reports KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e6
    return rss / 1e3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--lf-half-lengths", default="40,60,80,100")
    p.add_argument("--orientations", default="16,32,64")
    p.add_argument("--r", type=int, default=9)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--sigma", type=float, default=13.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-layout", default="theta_yx_m", choices=["theta_yx_m", "theta_m_yx"]
    )
    p.add_argument("--max-chunk-gb", type=float, default=2.0)
    p.add_argument("--chunk-pause-s", type=float, default=0.0)
    args = p.parse_args()

    lf_half_lengths = np.array(
        [int(s) for s in args.lf_half_lengths.split(",")], dtype=np.int32
    )
    orient_list = [int(s) for s in args.orientations.split(",")]

    rng = np.random.default_rng(args.seed)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    H, W = rgb.shape[:2]
    rgb_n = np.clip(
        rgb.astype(np.float32) + rng.normal(0.0, args.sigma, rgb.shape).astype(np.float32),
        0.0,
        255.0,
    )
    L = (
        0.2126 * rgb_n[..., 0] + 0.7152 * rgb_n[..., 1] + 0.0722 * rgb_n[..., 2]
    ).astype(np.float32)

    print(f"image    : {args.image.name}  size={H}x{W}")
    print(f"LF lengths: {list(lf_half_lengths)}  ({len(lf_half_lengths)} values)")
    print(f"orient.  : {orient_list}")
    print(f"baseline RSS: {peak_mb():.0f} MB")
    print()

    kernels = build_wvf_radius_kernels(radius=args.r, order=args.d)
    gx, gy = wvf_radius_gradients_metal(L, kernels, output_dtype=np.float32)

    for n_orient in orient_list:
        single_bytes = n_orient * H * W * 4
        batch_bytes = n_orient * len(lf_half_lengths) * H * W * 4
        print(f"--- n_orient = {n_orient} -----------------------------")
        print(f"  per-stack memory (single length): {single_bytes/1e9:.2f} GB")
        print(f"  full output  (batched lengths): {batch_bytes/1e9:.2f} GB")

        # ---- Path A: per-length loop ---------------------------------
        rss_pre_a = peak_mb()
        t0 = time.perf_counter()
        for m in lf_half_lengths:
            stack = lf_stack(
                gx,
                gy,
                lf_half_length=int(m),
                n_orientations=n_orient,
                output_dtype=np.float32,
                method="box",
            )
            del stack
        t_loop = time.perf_counter() - t0
        rss_post_a = peak_mb()

        # ---- Path B: batched LF half-lengths -------------------------
        rss_pre_b = peak_mb()
        t0 = time.perf_counter()
        try:
            big = lf_length_stack(
                gx,
                gy,
                lf_half_lengths=lf_half_lengths,
                n_orientations=n_orient,
                output_dtype=np.float32,
                method="box",
                output_layout=args.output_layout,
                max_chunk_bytes=None
                if args.max_chunk_gb <= 0
                else int(args.max_chunk_gb * 1024**3),
                chunk_pause_s=args.chunk_pause_s,
            )
            shape_b = big.shape
            del big
            err_b = None
        except Exception as ex:
            shape_b = None
            err_b = repr(ex)
            t_loop = t_loop  # unused, keep
        t_batch = time.perf_counter() - t0
        rss_post_b = peak_mb()

        print(
            f"  per-length loop (x{len(lf_half_lengths)} calls): "
            f"{t_loop*1000:>6.0f} ms  | RSS {rss_pre_a:.0f} -> "
            f"{rss_post_a:.0f} MB"
        )
        if err_b is None:
            print(
                f"  batched     (1 call)        : "
                f"{t_batch*1000:>6.0f} ms  | RSS {rss_pre_b:.0f} -> "
                f"{rss_post_b:.0f} MB  shape={shape_b}"
            )
            speedup = t_loop / max(t_batch, 1e-9)
            print(f"  speedup (loop / batched)    : x{speedup:.2f}")
        else:
            print(f"  batched     (1 call)        : ERROR  {err_b}")
        print()


if __name__ == "__main__":
    main()
