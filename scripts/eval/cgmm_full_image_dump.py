"""Full-image c-GMM K=3 fusion dump using the Metal pipeline.

Pipeline:
  1. Load the noisy synthetic image (sigma=13 AWGN) at the chosen size.
  2. Split into L, R, G, B channels.
  3. For each (channel, radius, degree, lf_half_length) config, run the
     fused Metal front end (WVF + LF + orientation recovery).
  4. Stack all configs into per-pixel (phi, w) primary + secondary
     measurement arrays.
  5. Run Metal c-GMM K=3 hard-EM fusion in one call.
  6. Save dump in the schema fig_cgmm_fused_output.py expects:
       theta_fused, theta_fused_sec, M_fused, M_fused_sec,
       v_fused, suppressed.

This is the production dump for the paper's c-GMM fused-output figure
on the noisy low_contrast_mixed_chroma image at 4096^2.
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
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "agent_workspaces" / "cgmm_metal"))

from edgecritic.pipeline import wvf_lf_recover_metal
from edgecritic.cgmm._metal import cgmm_fuse_two_pass_metal
from reference_impl import theta_M_to_phi_w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path)
    p.add_argument("--sigma", type=float, default=13.0)
    p.add_argument("--radii",   default="5,9")
    p.add_argument("--degrees", default="1,3")
    p.add_argument("--lf-half-lengths", default="40,60,80,100")
    p.add_argument("--channels",        default="L,R,G,B")
    p.add_argument("--n-orientations",  type=int, default=64)
    p.add_argument("--K",               type=int, default=3)
    p.add_argument("--seed",            type=int, default=0)
    args = p.parse_args()

    radii   = [int(s) for s in args.radii.split(",")]
    degrees = [int(s) for s in args.degrees.split(",")]
    ms      = [int(s) for s in args.lf_half_lengths.split(",")]
    sel_ch  = [c.strip() for c in args.channels.split(",")]

    rng = np.random.default_rng(args.seed)
    rgb = np.asarray(Image.open(args.image).convert("RGB")).astype(np.float32)
    H, W = rgb.shape[:2]
    rgb_n = np.clip(rgb + rng.normal(0.0, args.sigma, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    R = rgb_n[..., 0]; G = rgb_n[..., 1]; B = rgb_n[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    all_chans = {"L": L.astype(np.float32),
                 "R": R.astype(np.float32),
                 "G": G.astype(np.float32),
                 "B": B.astype(np.float32)}
    chans = {k: all_chans[k] for k in sel_ch}

    n_combos = len(chans) * len(radii) * len(degrees) * len(ms)
    print(f"image    : {args.image.name}  size={H}x{W}")
    print(f"sigma={args.sigma}  channels={sel_ch}  radii={radii}  "
          f"degrees={degrees}  ms={ms}  n_orient={args.n_orientations}")
    print(f"per-pixel inputs (N): {n_combos}")
    print()

    # Front end: 1 call per (channel, r, d, m).
    th_p_list, M_p_list, th_s_list, M_s_list, v_list = [], [], [], [], []
    t_front0 = time.perf_counter()
    for ch_name, img in chans.items():
        for r in radii:
            for d in degrees:
                for m in ms:
                    t0 = time.perf_counter()
                    th_p, M_p, th_s, M_s, v = wvf_lf_recover_metal(
                        img, radius=r, degree=d, lf_half_length=m,
                        n_orientations=args.n_orientations,
                        tau_sec_floor=0.40, tau_validity=0.10,
                        dense_n=500, min_sep_frac=0.125, method="box")
                    dt = time.perf_counter() - t0
                    th_p_list.append(th_p)
                    M_p_list.append(M_p)
                    th_s_list.append(th_s)
                    M_s_list.append(M_s)
                    v_list.append(v)
                    print(f"  ch={ch_name}  r={r}  d={d}  m={m:>3}  "
                          f"v=1: {v.mean()*100:5.1f}%  {dt*1000:.0f} ms")
    t_front = time.perf_counter() - t_front0
    print(f"\nfront end: {t_front:.1f} s ({n_combos} fused calls)")

    # Tile the (phi, w) build + c-GMM call over pixel chunks to keep peak
    # memory bounded.  At full image (P=16.7M, N=64) the full (P, N) f32
    # arrays are 4 GB each x4 streams = 16 GB upfront, which swaps on
    # 16 GB systems.  TILE = 1M keeps per-tile (chunk, N) f32 buffers at
    # ~256 MB each.
    P = H * W
    N = n_combos
    two_pi = np.float32(2.0 * math.pi)

    # Pre-flatten the per-config arrays once.  These stay alive for the
    # duration of the tile loop (5 GB total for N=64 at 4096^2).
    th_p_flat = [a.reshape(P) for a in th_p_list]
    M_p_flat  = [a.reshape(P) for a in M_p_list]
    th_s_flat = [a.reshape(P) for a in th_s_list]
    M_s_flat  = [a.reshape(P) for a in M_s_list]
    v_flat    = [a.reshape(P) for a in v_list]
    del th_p_list, M_p_list, th_s_list, M_s_list, v_list

    # Output buffers.
    theta_primary = np.full(P, np.nan, dtype=np.float32)
    M_primary     = np.zeros(P,        dtype=np.float32)
    theta_sec     = np.full(P, np.nan, dtype=np.float32)
    M_sec         = np.zeros(P,        dtype=np.float32)
    v_fused_flat  = np.zeros(P,        dtype=np.uint8)
    keep_sec_flat = np.zeros(P,        dtype=np.uint8)

    TILE = 1 << 20    # 1,048,576 pixels per tile
    n_tiles = (P + TILE - 1) // TILE
    print(f"\nrunning c-GMM K={args.K} hard-EM (Metal) on "
          f"(P={P}, N={N}), {n_tiles} tiles of <= {TILE:,} pixels ...")
    t_cgmm0 = time.perf_counter()
    for ti, start in enumerate(range(0, P, TILE)):
        end = min(start + TILE, P)
        c = end - start
        phi_p_c = np.empty((c, N), dtype=np.float32)
        w_p_c   = np.empty((c, N), dtype=np.float32)
        phi_s_c = np.empty((c, N), dtype=np.float32)
        w_s_c   = np.empty((c, N), dtype=np.float32)
        for j in range(N):
            th_p_j = th_p_flat[j][start:end]
            M_p_j  = M_p_flat[j][start:end]
            th_s_j = th_s_flat[j][start:end]
            M_s_j  = M_s_flat[j][start:end]
            v_j    = v_flat[j][start:end].astype(np.float32, copy=False)
            finite_p = np.isfinite(th_p_j) & np.isfinite(M_p_j)
            finite_s = np.isfinite(th_s_j) & np.isfinite(M_s_j)
            phi_p_c[:, j] = np.where(finite_p, (2.0 * th_p_j) % two_pi, 0.0)
            w_p_c[:,   j] = v_j * np.where(
                finite_p & (M_p_j > 0), np.maximum(M_p_j, 0.0), 0.0)
            phi_s_c[:, j] = np.where(finite_s, (2.0 * th_s_j) % two_pi, 0.0)
            w_s_c[:,   j] = v_j * np.where(
                finite_s & (M_s_j > 0), np.maximum(M_s_j, 0.0), 0.0)
        out_c = cgmm_fuse_two_pass_metal(
            phi_p_c, w_p_c, phi_s_c, w_s_c,
            K=args.K, n_iters=30,
            init_kappa=4.0, hard_em=True,
            tau_M_rel=0.05, theta_min_deg=10.0)
        theta_primary[start:end] = out_c["theta_primary"].astype(np.float32)
        M_primary[start:end]     = out_c["M_primary"].astype(np.float32)
        theta_sec[start:end]     = out_c["theta_sec"].astype(np.float32)
        M_sec[start:end]         = out_c["M_sec"].astype(np.float32)
        v_fused_flat[start:end]  = out_c["v_fused"]
        keep_sec_flat[start:end] = out_c["keep_secondary_mask"]
        del phi_p_c, w_p_c, phi_s_c, w_s_c, out_c
        print(f"  tile {ti+1:>2}/{n_tiles}  "
              f"[{start:>9}:{end:>9}]  "
              f"v=1: {v_fused_flat[start:end].mean()*100:5.1f}%")
    t_cgmm = time.perf_counter() - t_cgmm0
    print(f"c-GMM Metal total: {t_cgmm:.2f} s "
          f"({n_tiles} tiles, avg {t_cgmm/n_tiles*1000:.0f} ms)")
    print(f"v_fused = 1 fraction: {v_fused_flat.mean()*100:.1f}%")
    print(f"M_primary range: [{M_primary.min():.3f}, {M_primary.max():.3f}]")

    # Reshape to (H, W) and build the figure-script schema.
    theta_fused      = theta_primary.reshape(H, W)
    theta_fused_sec  = theta_sec.reshape(H, W)
    M_fused          = M_primary.reshape(H, W)
    M_fused_sec      = M_sec.reshape(H, W)
    v_fused          = v_fused_flat.reshape(H, W)
    keep_sec         = keep_sec_flat.reshape(H, W)
    # `suppressed` schema = 1 where v=1 but the secondary slot was zeroed.
    suppressed       = ((v_fused == 1) & (keep_sec == 0)).astype(np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
        condition       = "noisy",
        size            = np.int32(H),
        theta_fused     = theta_fused,
        theta_fused_sec = theta_fused_sec,
        M_fused         = M_fused,
        M_fused_sec     = M_fused_sec,
        v_fused         = v_fused,
        suppressed      = suppressed,
        is_junction     = np.zeros((H, W), dtype=np.uint8),  # no GT in real-world dump
        config_image    = str(args.image.name),
        config_sigma    = np.float64(args.sigma),
        config_channels = np.array(sel_ch),
        config_radii    = np.array(radii, dtype=np.int32),
        config_degrees  = np.array(degrees, dtype=np.int32),
        config_lf_half_lengths = np.array(ms, dtype=np.int32),
        config_n_orientations  = np.int32(args.n_orientations),
        config_K        = np.int32(args.K),
    )
    print(f"\nwrote {args.out}")
    print(f"front end + c-GMM total: {t_front + t_cgmm:.1f} s")


if __name__ == "__main__":
    main()
