"""tau_M_rel sweep for the secondary-suppression rule.

Runs the LF + vMM fusion stage once on each condition, then varies
tau_M_rel post-hoc to compute the TP/FP table. theta_min_deg = 10
deg is fixed (the geometric criterion).

The rule under audit (current production):
    keep_sec = (M_sec_w > tau_M_rel * M_signal)
               AND (sep_theta_deg > theta_min_deg)

Definitions:
    edges     := smooth_mask & v_fused == 1     (well-defined edge)
    junctions := junction_mask & v_fused == 1   (within vertex_exclude_px)
    real corners (TP denominator) := junctions where the c-GMM
                                     reports sep_theta_deg > 10 deg
    edge FP rate := fraction of edges with keep_sec = True
    junction TP rate := fraction of (junctions and real-corner) with
                        keep_sec = True
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cgmm_vmm import vmm_fuse, theta_M_to_phi_w, circular_distance
from cgmm_image_wide_eval import (
    build_gt_orientation, find_image_spec, load_channels_clean,
    load_channels_noisy, evaluate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-rgb", required=True, type=Path)
    p.add_argument("--noisy-dir", required=True, type=Path)
    p.add_argument("--manifest",  required=True, type=Path)
    p.add_argument("--image-key", default="garnet_atlantic_grass")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--r", type=int, default=9)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--m-values", default="0,5,10,20,30,40,50,60,70,80")
    p.add_argument("--n-orientations", type=int, default=64)
    p.add_argument("--n-edge-pixels",     type=int, default=2000)
    p.add_argument("--n-junction-pixels", type=int, default=2000)
    p.add_argument("--vertex-exclude-px", type=int, default=24)
    p.add_argument("--theta-min-deg", type=float, default=10.0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    spec = find_image_spec(args.manifest, args.image_key, args.size)
    gt_normal, all_mask, smooth_mask = build_gt_orientation(
        spec, edge_band_px=0, vertex_exclude_px=args.vertex_exclude_px)
    junction_mask = all_mask & ~smooth_mask
    print(f"smooth_mask:   {smooth_mask.sum():>7,} pixels")
    print(f"junction_mask: {junction_mask.sum():>7,} pixels")

    rng = np.random.default_rng(args.seed)
    def sample(mask, n):
        ys, xs = np.where(mask)
        if len(ys) > n:
            idx = rng.choice(len(ys), size=n, replace=False)
            ys, xs = ys[idx], xs[idx]
        return np.column_stack([xs, ys])

    edge_pix = sample(smooth_mask, args.n_edge_pixels)
    junc_pix = sample(junction_mask, args.n_junction_pixels)
    all_pix = np.concatenate([edge_pix, junc_pix], axis=0)
    is_junction = np.concatenate([
        np.zeros(len(edge_pix), dtype=bool),
        np.ones(len(junc_pix),  dtype=bool),
    ])

    ys = all_pix[:, 1]; xs = all_pix[:, 0]
    gt_normal_at = gt_normal[ys, xs].astype(np.float64)
    gt_t = np.degrees((gt_normal_at + math.pi / 2.0) % math.pi)
    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]

    tau_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    cond_results = {}
    for cond_label, channels_loader, channels_arg in (
            ("clean", load_channels_clean, args.clean_rgb),
            ("noisy", load_channels_noisy, args.noisy_dir),
    ):
        print(f"\n========== {cond_label} ==========")
        channels = channels_loader(channels_arg)
        primary_t, primary_m = evaluate(
            cond_label, channels, all_pix, gt_t,
            m_values, args.n_orientations, args.r, args.d)
        phi, w, _ = theta_M_to_phi_w(primary_t, primary_m)

        # Run the fusion ONCE with a "keep all secondaries" rule (tau=0).
        # We extract the raw signal/secondary parameters and re-evaluate
        # the suppression rule post-hoc for each tau_M_rel in the sweep.
        out = vmm_fuse(phi, w, K=args.K, hard_em=True,
                       tau_M_rel=0.0, theta_min_deg=0.0)
        pi   = out["pi"]
        mu   = out["mu"]
        W    = out["W"]
        rng_p = np.arange(pi.shape[0])
        k_signal = np.argmax(pi, axis=1)
        pi_msk = pi.copy()
        pi_msk[rng_p, k_signal] = -np.inf
        k_sec = np.argmax(pi_msk, axis=1)
        M_signal = W[rng_p, k_signal]
        M_sec    = W[rng_p, k_sec]
        mu_signal = mu[rng_p, k_signal]
        mu_sec    = mu[rng_p, k_sec]
        sep_phi  = circular_distance(mu_signal, mu_sec)
        sep_theta_deg = np.degrees(sep_phi) / 2.0

        valid = out["v_fused"] == 1
        edge_ok = valid & ~is_junction
        junc_ok = valid &  is_junction
        # "Real corners" = junction pixels where the fit found two
        # well-separated modes.
        real_corner = junc_ok & (sep_theta_deg > args.theta_min_deg)

        rows = []
        for tau in tau_grid:
            keep = (
                (M_sec > tau * np.maximum(M_signal, 1e-30))
                & (sep_theta_deg > args.theta_min_deg)
            )
            edge_fp = float(keep[edge_ok].mean())
            junc_tp = (float(keep[real_corner].mean())
                       if real_corner.any() else float("nan"))
            rows.append(dict(
                tau_M_rel=tau,
                edge_FP=edge_fp,
                junction_TP=junc_tp,
                TP_FP=junc_tp / max(edge_fp, 1e-9),
                n_edges=int(edge_ok.sum()),
                n_real_corners=int(real_corner.sum()),
            ))
        cond_results[cond_label] = rows

    # ------------ print combined table ------------
    print("\n" + "=" * 120)
    print(f"tau_M_rel sweep, theta_min_deg = {args.theta_min_deg} deg, "
          f"K = {args.K}, hard-EM, single-pass")
    print("=" * 120)
    hdr = (f"{'tau_M_rel':>10}  "
           f"{'clean FP':>9} {'clean TP':>9} {'clean TP/FP':>12}  "
           f"{'noisy FP':>9} {'noisy TP':>9} {'noisy TP/FP':>12}")
    print(hdr)
    print("-" * len(hdr))
    for tau in tau_grid:
        c = next(r for r in cond_results["clean"] if r["tau_M_rel"] == tau)
        n = next(r for r in cond_results["noisy"] if r["tau_M_rel"] == tau)
        print(f"{tau:>10.2f}  "
              f"{c['edge_FP']:>9.4f} {c['junction_TP']:>9.4f} "
              f"{c['TP_FP']:>12.2f}  "
              f"{n['edge_FP']:>9.4f} {n['junction_TP']:>9.4f} "
              f"{n['TP_FP']:>12.2f}")
    print()
    print(f"  TP denominator (real corners): "
          f"clean n={cond_results['clean'][0]['n_real_corners']}  "
          f"noisy n={cond_results['noisy'][0]['n_real_corners']}")
    print(f"  FP denominator (smooth edges): "
          f"clean n={cond_results['clean'][0]['n_edges']}  "
          f"noisy n={cond_results['noisy'][0]['n_edges']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": dict(K=args.K,
                           theta_min_deg=args.theta_min_deg,
                           tau_grid=tau_grid,
                           n_edge_pixels=args.n_edge_pixels,
                           n_junction_pixels=args.n_junction_pixels,
                           vertex_exclude_px=args.vertex_exclude_px,
                           m_values=m_values, r=args.r, d=args.d,
                           n_orientations=args.n_orientations),
            "results": cond_results,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
