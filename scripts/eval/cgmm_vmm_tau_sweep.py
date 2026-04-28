"""tau_M_rel sweep for the TWO-PASS secondary-suppression rule.

Runs two-pass vMM fusion (independent K=3 hard-EM on primary and on
secondary streams) once per (condition, pixel set), then varies
tau_M_rel post-hoc to compute the TP/FP table.  theta_min_deg = 10
deg is fixed (the geometric criterion).

Production rule under audit:
    keep_sec = (M_sec / M_primary > tau_M_rel)
               AND (|theta_primary - theta_sec| > theta_min_deg)
where M_primary and M_sec are absolute weights from SEPARATE fits.

Definitions:
    edges     := smooth_mask & v_fused == 1
    junctions := junction_mask & v_fused == 1
    real corners (TP denominator) := junctions where the TWO-PASS
                                     output reports angular separation
                                     > theta_min_deg between primary
                                     and secondary fits.
    edge FP rate := fraction of edges with keep_sec = True
    junction TP rate := fraction of (junctions and real-corner) with
                        keep_sec = True

Junction TP breakdown by within-pass tiebreak behavior:
    swap         := junction pixel where the per-config (theta_n)
                    primary peaks span both edges of the corner
                    (circular range of primary stream > 30 deg)
    always_A     := junction pixel where all per-config primaries
                    cluster near a single direction (range <= 30 deg)
    These should be disjoint and exhaustive among junctions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cgmm_vmm import (
    vmm_fuse_two_pass, theta_M_to_phi_w, circular_distance)
from cgmm_image_wide_eval import (
    build_gt_orientation, find_image_spec, load_channels_clean,
    load_channels_noisy, evaluate)


def circular_range_deg(theta_deg, axis=1):
    """Maximum unsigned line-orientation distance between any pair of
    samples along `axis`.  theta_deg in [0, 180); returns deg in [0, 90]."""
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    phi = (2.0 * np.deg2rad(theta_deg)) % (2.0 * np.pi)
    # Cluster mean direction, then per-sample distance to the mean.
    cos_mean = np.cos(phi).mean(axis=axis, keepdims=True)
    sin_mean = np.sin(phi).mean(axis=axis, keepdims=True)
    mu_phi = np.arctan2(sin_mean, cos_mean)
    d = circular_distance(phi, mu_phi)
    # 2x the maximum sample-to-mean distance in phi -> max pairwise in phi.
    span_phi = 2.0 * d.max(axis=axis)
    span_phi = np.minimum(span_phi, 2.0 * np.pi)
    return np.degrees(span_phi) / 2.0


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
    p.add_argument("--swap-spread-deg", type=float, default=30.0,
                   help="primary-stream circular range threshold "
                        "to classify a junction as 'swap' "
                        "(default 30 deg)")
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
        primary_t, primary_m, secondary_t, secondary_m = evaluate(
            cond_label, channels, all_pix, gt_t,
            m_values, args.n_orientations, args.r, args.d)

        phi_p, w_p, _ = theta_M_to_phi_w(primary_t, primary_m)
        phi_s, w_s, _ = theta_M_to_phi_w(secondary_t, secondary_m)

        # Run two-pass fusion with permissive thresholds (tau=0,
        # theta_min=0) so we get raw signal/secondary parameters and can
        # apply suppression rules post-hoc per tau_M_rel value.
        t0 = time.perf_counter()
        out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s,
                                K=args.K, hard_em=True,
                                tau_M_rel=0.0, theta_min_deg=0.0)
        print(f"  two-pass fusion ({cond_label}): "
              f"{time.perf_counter()-t0:.2f}s")

        # Re-extract the raw signal/secondary parameters.  Since we
        # passed tau=0, theta_min=0 above, every secondary survived;
        # but to apply different tau values cleanly we recompute mu_p
        # and mu_s directly from the diagnostic outputs.
        Pall = out["v_fused"].shape[0]
        rng_p = np.arange(Pall)
        # Indices of signal component per pixel from each fit.
        # (NaN mu where the fit was skipped.)
        primary_pi  = out["primary_pi"]
        primary_mu  = out["primary_mu"]
        secondary_pi = out["secondary_pi"]
        secondary_mu = out["secondary_mu"]
        # argmax-pi within each row, NaN-safe.
        def safe_argmax(arr):
            arr_f = np.where(np.isnan(arr), -np.inf, arr)
            return np.argmax(arr_f, axis=1)
        k_p = safe_argmax(primary_pi)
        k_s = safe_argmax(secondary_pi)
        mu_p_phi = np.where(np.isnan(primary_mu[rng_p, k_p]),
                             np.nan, primary_mu[rng_p, k_p])
        mu_s_phi = np.where(np.isnan(secondary_mu[rng_p, k_s]),
                             np.nan, secondary_mu[rng_p, k_s])
        M_p = out["M_primary"]
        # M_sec under permissive run: stored in M_sec when keep was True;
        # recompute from secondary fit's W to get the underlying value
        # regardless of suppression.  Use the diagnostic W via reconstruction:
        # W = pi * sum(W_total)? No - we need W_kappa directly. Re-run
        # secondary inspection from out["secondary_pi"] is insufficient.
        # Solution: also expose W_secondary in the diagnostics. Workaround:
        # the permissive run wrote M_sec directly into out["M_sec"]
        # (since keep was true wherever secondary_valid is true and tau=0,
        # theta_min=0).
        M_s = out["M_sec"]
        # But suppress_keep_secondary_mask reflects the permissive rule,
        # not the per-tau rule, so M_s is the absolute secondary weight
        # whenever keep was True.  For pixels where secondary_valid was
        # False, M_sec is 0 -> they cannot pass any tau. Fine.

        sep_phi = circular_distance(mu_p_phi, mu_s_phi)
        sep_theta_deg = np.degrees(sep_phi) / 2.0

        # Validity for the metrics.
        v = out["v_fused"] == 1
        edge_ok = v & ~is_junction
        junc_ok = v &  is_junction
        # "Real corners" = junctions where the two-pass fits found two
        # well-separated mode locations.
        sec_finite = np.isfinite(M_s) & (M_s > 0)
        real_corner = junc_ok & sec_finite & (sep_theta_deg > args.theta_min_deg)

        # Junction sub-classification by within-config primary-stream spread.
        jn_spread = circular_range_deg(primary_t)
        swap_mask = junc_ok & (jn_spread > args.swap_spread_deg)
        alwaysA_mask = junc_ok & (jn_spread <= args.swap_spread_deg)

        rows = []
        for tau in tau_grid:
            mass_ok = M_s > tau * np.maximum(M_p, 1e-30)
            sep_ok  = sep_theta_deg > args.theta_min_deg
            keep    = mass_ok & sep_ok

            edge_fp = float(keep[edge_ok].mean()) if edge_ok.any() else float("nan")
            junc_tp = (float(keep[real_corner].mean())
                       if real_corner.any() else float("nan"))
            tp_fp   = junc_tp / max(edge_fp, 1e-9) if np.isfinite(junc_tp) else float("nan")
            # Breakdown
            swap_tp = (float(keep[swap_mask & sec_finite & sep_ok].mean())
                       if (swap_mask & sec_finite & sep_ok).any()
                       else float("nan"))
            alwaysA_tp = (
                float(keep[alwaysA_mask & sec_finite & sep_ok].mean())
                if (alwaysA_mask & sec_finite & sep_ok).any()
                else float("nan"))
            # Same denominators but not gated on sep_ok (so we can also
            # see how often we get to the final keep step on the subgroups).
            swap_tp_all = (float(keep[swap_mask].mean())
                           if swap_mask.any() else float("nan"))
            alwaysA_tp_all = (float(keep[alwaysA_mask].mean())
                              if alwaysA_mask.any() else float("nan"))
            rows.append(dict(
                tau_M_rel=tau,
                edge_FP=edge_fp,
                junction_TP=junc_tp,
                TP_FP=tp_fp,
                swap_TP_at_real_corners=swap_tp,
                alwaysA_TP_at_real_corners=alwaysA_tp,
                swap_TP_all_swap=swap_tp_all,
                alwaysA_TP_all_alwaysA=alwaysA_tp_all,
            ))
        cond_results[cond_label] = dict(
            rows=rows,
            n_edges=int(edge_ok.sum()),
            n_junctions=int(junc_ok.sum()),
            n_real_corners=int(real_corner.sum()),
            n_swap_junctions=int(swap_mask.sum()),
            n_alwaysA_junctions=int(alwaysA_mask.sum()),
        )

    # ------------ print combined table ------------
    print("\n" + "=" * 120)
    print(f"tau_M_rel sweep, theta_min_deg = {args.theta_min_deg} deg, "
          f"K = {args.K}, hard-EM, TWO-PASS architecture")
    print("=" * 120)
    hdr = (f"{'tau':>6}  "
           f"{'clean FP':>9} {'clean TP':>9} {'clean TP/FP':>12}  "
           f"{'noisy FP':>9} {'noisy TP':>9} {'noisy TP/FP':>12}")
    print(hdr)
    print("-" * len(hdr))
    for tau in tau_grid:
        c = next(r for r in cond_results["clean"]["rows"]
                 if r["tau_M_rel"] == tau)
        n = next(r for r in cond_results["noisy"]["rows"]
                 if r["tau_M_rel"] == tau)
        print(f"{tau:>6.2f}  "
              f"{c['edge_FP']:>9.4f} {c['junction_TP']:>9.4f} "
              f"{c['TP_FP']:>12.2f}  "
              f"{n['edge_FP']:>9.4f} {n['junction_TP']:>9.4f} "
              f"{n['TP_FP']:>12.2f}")
    print()
    for cond in ("clean", "noisy"):
        r = cond_results[cond]
        print(f"  [{cond}] denominators: "
              f"edges={r['n_edges']}  junctions={r['n_junctions']}  "
              f"real_corners={r['n_real_corners']}  "
              f"swap={r['n_swap_junctions']}  "
              f"alwaysA={r['n_alwaysA_junctions']}")

    # Junction TP breakdown at every tau (helps pick a default).
    print("\n" + "=" * 120)
    print("Junction TP breakdown: swap vs always-A-wins (at real-corner "
          "denom AND with sep_ok), per condition")
    print("=" * 120)
    for cond in ("clean", "noisy"):
        print(f"\n  [{cond}]   {'tau':>6}  {'swap_TP':>10}  "
              f"{'alwaysA_TP':>12}  ({'TP_swap_all':>13} / {'TP_alwaysA_all':>16})")
        for r in cond_results[cond]["rows"]:
            print(f"           {r['tau_M_rel']:>6.2f}  "
                  f"{r['swap_TP_at_real_corners']:>10.4f}  "
                  f"{r['alwaysA_TP_at_real_corners']:>12.4f}  "
                  f"({r['swap_TP_all_swap']:>13.4f} / "
                  f"{r['alwaysA_TP_all_alwaysA']:>16.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": dict(K=args.K,
                           theta_min_deg=args.theta_min_deg,
                           swap_spread_deg=args.swap_spread_deg,
                           tau_grid=tau_grid,
                           n_edge_pixels=args.n_edge_pixels,
                           n_junction_pixels=args.n_junction_pixels,
                           vertex_exclude_px=args.vertex_exclude_px,
                           m_values=m_values, r=args.r, d=args.d,
                           n_orientations=args.n_orientations,
                           architecture="two-pass"),
            "results": cond_results,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
