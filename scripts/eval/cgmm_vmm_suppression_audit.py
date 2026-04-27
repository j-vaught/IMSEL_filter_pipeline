"""Suppression-rule audit for the vMM K=3 hard-EM fusion.

Compares the distribution of pi[k_sec] / pi[k_signal] (and the related
M_sec / M_signal ratio) at:
    - smooth edge pixels  (smooth_mask, away from vertices)
    - junction pixels     (all_mask & ~smooth_mask, near vertices)

If junction pixels sit clearly above the rho threshold (default 0.40)
and smooth-edge pixels sit clearly below it, the defaults are right.
If there's overlap, the script prints suggested new thresholds at the
median and 90th percentiles of each population.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cgmm_vmm import vmm_fuse, theta_M_to_phi_w
from cgmm_image_wide_eval import (
    build_gt_orientation, find_image_spec, load_channels_clean,
    load_channels_noisy, evaluate)


def percentiles(arr, qs=(50, 90, 95, 99)):
    if len(arr) == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(arr, q)) for q in qs}


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
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--tau-M-rel", type=float, default=0.10)
    p.add_argument("--rho",       type=float, default=0.40)
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
        return np.column_stack([xs, ys]), ys, xs

    edge_pix, ye, xe   = sample(smooth_mask,   args.n_edge_pixels)
    junc_pix, yj, xj   = sample(junction_mask, args.n_junction_pixels)
    all_pix = np.concatenate([edge_pix, junc_pix], axis=0)

    is_junction = np.concatenate([
        np.zeros(len(edge_pix), dtype=bool),
        np.ones(len(junc_pix),  dtype=bool),
    ])

    gt_normal_at = gt_normal[
        np.concatenate([ye, yj]),
        np.concatenate([xe, xj]),
    ].astype(np.float64)
    gt_t = np.degrees((gt_normal_at + math.pi / 2.0) % math.pi)
    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]

    out_rows = {}
    for cond_label, channels_loader, channels_args in (
            ("clean", load_channels_clean, args.clean_rgb),
            ("noisy", load_channels_noisy, args.noisy_dir),
    ):
        print(f"\n========== {cond_label} ==========")
        channels = channels_loader(channels_args)
        primary_t, primary_m = evaluate(
            cond_label, channels, all_pix, gt_t,
            m_values, args.n_orientations, args.r, args.d)
        phi, w, _ = theta_M_to_phi_w(primary_t, primary_m)

        t0 = time.perf_counter()
        out = vmm_fuse(phi, w, K=args.K,
                       hard_em=True,
                       tau_M_rel=args.tau_M_rel, rho=args.rho)
        print(f"  vMM hard-EM K={args.K}: {time.perf_counter()-t0:.2f}s")

        # Per-pixel ratios.
        pi    = out["pi"]
        W     = out["W"]
        mu    = out["mu"]
        kappa = out["kappa"]
        rng_p = np.arange(pi.shape[0])
        k_signal = np.argmax(pi, axis=1)
        pi_msk = pi.copy()
        pi_msk[rng_p, k_signal] = -np.inf
        k_sec = np.argmax(pi_msk, axis=1)

        pi_signal    = pi[rng_p,    k_signal]
        pi_sec       = pi[rng_p,    k_sec]
        M_signal     = W[rng_p,     k_signal]
        M_sec        = W[rng_p,     k_sec]
        mu_signal    = mu[rng_p,    k_signal]
        mu_sec       = mu[rng_p,    k_sec]
        kappa_signal = kappa[rng_p, k_signal]
        kappa_sec    = kappa[rng_p, k_sec]
        v_fused      = out["v_fused"]

        ratio_pi = pi_sec / np.maximum(pi_signal, 1e-30)
        ratio_M  = M_sec  / np.maximum(M_signal,  1e-30)

        # Angular separation between signal and secondary mu, on the
        # ORIGINAL theta = phi/2 half-circle (so a 90 deg separation in
        # theta = 180 deg in phi == pi rad).
        from cgmm_vmm import circular_distance
        d_phi = circular_distance(mu_signal, mu_sec)              # rad in [0, pi]
        sep_theta_deg = np.degrees(d_phi) / 2.0                    # deg in [0, 90]

        keep_sec = (
            (M_sec > args.tau_M_rel * np.maximum(M_signal, 1e-30))
            & (pi_sec > args.rho * np.maximum(pi_signal, 1e-30))
        )

        valid = v_fused == 1
        edge_ok = valid & ~is_junction
        junc_ok = valid &  is_junction

        def stats_for(mask):
            return dict(
                n              = int(mask.sum()),
                ratio_pi_pcts  = percentiles(ratio_pi[mask]),
                sep_theta_pcts = percentiles(sep_theta_deg[mask]),
                kappa_sec_pcts = percentiles(kappa_sec[mask]),
                keep_sec_frac  = float(keep_sec[mask].mean()),
            )
        edge_stats = stats_for(edge_ok)
        junc_stats = stats_for(junc_ok)

        def print_metric(name, key):
            print(f"\n  {name}:")
            print(f"    {'pop':<10} {'n':>5} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7}")
            for label, st in [("edges", edge_stats), ("junctions", junc_stats)]:
                r = st[key]
                print(f"    {label:<10} {st['n']:>5d} "
                      f"{r[50]:>7.3f} {r[90]:>7.3f} {r[95]:>7.3f} {r[99]:>7.3f}")

        print_metric("pi[k_sec] / pi[k_signal]  (== M ratio)", "ratio_pi_pcts")
        print_metric("|theta_signal - theta_sec|  (deg, "
                     "[0, 90])",                                "sep_theta_pcts")
        print_metric("kappa[k_sec]",                            "kappa_sec_pcts")

        print(f"\n  Suppression keep-sec fraction at SPEC defaults "
              f"(rho={args.rho}, tau_M_rel={args.tau_M_rel}):")
        print(f"    edges:     {edge_stats['keep_sec_frac']:.4f}  "
              f"(want low; FP rate)")
        print(f"    junctions: {junc_stats['keep_sec_frac']:.4f}  "
              f"(want high; TP rate)")

        # Sweep alternative rules to find a better edge/junction split.
        print(f"\n  Sweep keep-sec under alternative rules:")
        rules = [
            ("rho>=0.40 alone (spec)",
             lambda: ratio_pi >= 0.40),
            ("mu_sep>=2 deg",
             lambda: sep_theta_deg >= 2.0),
            ("mu_sep>=5 deg",
             lambda: sep_theta_deg >= 5.0),
            ("mu_sep>=10 deg",
             lambda: sep_theta_deg >= 10.0),
            ("rho>=0.40 AND mu_sep>=5 deg",
             lambda: (ratio_pi >= 0.40) & (sep_theta_deg >= 5.0)),
            ("rho>=0.40 AND mu_sep>=10 deg",
             lambda: (ratio_pi >= 0.40) & (sep_theta_deg >= 10.0)),
        ]
        print(f"    {'rule':<35} {'edge FP':>9} {'junc TP':>9} {'TP/FP':>9}")
        for name, predf in rules:
            keep = predf()
            efp = keep[edge_ok].mean()
            jtp = keep[junc_ok].mean()
            tpfp = jtp / max(efp, 1e-6)
            print(f"    {name:<35} {efp:>9.4f} {jtp:>9.4f} {tpfp:>9.2f}")

        out_rows[cond_label] = dict(edges=edge_stats, junctions=junc_stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "K": args.K, "tau_M_rel": args.tau_M_rel, "rho": args.rho,
                "n_edge_pixels": args.n_edge_pixels,
                "n_junction_pixels": args.n_junction_pixels,
                "vertex_exclude_px": args.vertex_exclude_px,
            },
            "results": out_rows,
        }, f, indent=2)
    print(f"\nWrote {args.out}")

    # Suggest thresholds.
    print("\n----------- threshold suggestions ------------")
    for cond, rows in out_rows.items():
        e = rows["edges"]["ratio_pi_pcts"]
        j = rows["junctions"]["ratio_pi_pcts"]
        print(f"\n{cond}:")
        print(f"  edge p90 ratio_pi    = {e[90]:.3f}")
        print(f"  junction p10 ratio_pi (proxy via 100-p90) = "
              f"{j[90]:.3f}  (junctions show high ratios "
              f"so this is the upper end)")
        gap_lo = e[90]
        gap_hi = j[50]
        if gap_lo < gap_hi:
            mid = 0.5 * (gap_lo + gap_hi)
            print(f"  -> defendable rho ~ {mid:.3f} "
                  f"(midpoint of edge p90 = {gap_lo:.3f} "
                  f"and junction p50 = {gap_hi:.3f})")
        else:
            print(f"  -> populations OVERLAP: edge p90 ({gap_lo:.3f}) "
                  f"> junction p50 ({gap_hi:.3f}); rho cannot cleanly "
                  f"separate. Consider a multi-criterion rule (M ratio "
                  f"AND pi ratio AND mu separation).")


if __name__ == "__main__":
    main()
