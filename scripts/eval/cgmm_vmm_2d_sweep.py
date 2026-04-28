"""2D sweep over (tau_sec_floor, tau_M_rel) for the two-pass fusion
with §6.3 sentinel.

Strategy: run LF + orientation recovery ONCE per condition with
tau_sec_floor = 0 (the "no-local-max" sentinel still fires; only the
weak-floor sentinel is disabled).  This yields the raw secondary
stream.  Then for each tau_sec_floor in the sweep we apply the
weak-floor post-hoc on the cached arrays before feeding the secondary
stream into two-pass fusion.

For each tau_sec_floor: fusion runs once with permissive tau_M_rel = 0
and theta_min_deg = 10; then we sweep tau_M_rel in {0.05, 0.10, 0.20,
0.30} post-hoc by re-applying the suppression rule.

Reports a 4x4 cell table (clean) and a 4x4 cell table (noisy) of
(edge_FP, junc_TP, TP/FP) tuples, plus the swap vs always-A-wins
breakdown at the recommended operating point.
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
from cgmm_orientation_recovery import find_two_peaks
from cgmm_image_wide_eval import (
    build_gt_orientation, find_image_spec, load_channels_clean,
    load_channels_noisy, lf_response_at_pixels, compute_wvf)


def circular_range_deg(theta_deg, axis=1):
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    phi = (2.0 * np.deg2rad(theta_deg)) % (2.0 * np.pi)
    cos_mean = np.cos(phi).mean(axis=axis, keepdims=True)
    sin_mean = np.sin(phi).mean(axis=axis, keepdims=True)
    mu_phi = np.arctan2(sin_mean, cos_mean)
    d = circular_distance(phi, mu_phi)
    span_phi = 2.0 * d.max(axis=axis)
    span_phi = np.minimum(span_phi, 2.0 * np.pi)
    return np.degrees(span_phi) / 2.0


def evaluate_lf_streams(label, channels, sample_pixels,
                        m_values, n_orientations, r, d):
    """Run LF response at sample pixels and return RAW orientation
    recovery output (with tau_sec_floor=0; only the no-local-max
    sentinel fires)."""
    angles = np.linspace(0, math.pi, n_orientations, endpoint=False)
    px = sample_pixels[:, 0].astype(np.int32)
    py = sample_pixels[:, 1].astype(np.int32)
    N = len(px)
    n_m = len(m_values)
    n_ch = len(channels)

    primary_t   = np.zeros((N, n_ch * n_m), dtype=np.float64)
    primary_m   = np.zeros((N, n_ch * n_m), dtype=np.float64)
    secondary_t = np.zeros((N, n_ch * n_m), dtype=np.float64)
    secondary_m = np.zeros((N, n_ch * n_m), dtype=np.float64)

    col = 0
    for ch_name, img in channels.items():
        t0 = time.perf_counter()
        g_x, g_y = compute_wvf(img, r, d)
        print(f"  [{label}] WVF channel {ch_name}: "
              f"{time.perf_counter()-t0:.1f}s")
        for m in m_values:
            t1 = time.perf_counter()
            resp = np.zeros((N, n_orientations), dtype=np.float64)
            for k, theta in enumerate(angles):
                resp[:, k] = lf_response_at_pixels(g_x, g_y, px, py,
                                                   float(theta), int(m))
            t_p, m_p, t_s, m_s = find_two_peaks(
                angles, resp, tau_sec_floor=0.0)
            primary_t[:, col]   = np.degrees(t_p)
            primary_m[:, col]   = m_p
            secondary_t[:, col] = np.degrees(t_s)
            secondary_m[:, col] = m_s
            col += 1
            print(f"    m={m:>3}: {time.perf_counter()-t1:.1f}s")
    return primary_t, primary_m, secondary_t, secondary_m


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
    p.add_argument("--swap-spread-deg", type=float, default=30.0)
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

    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]
    tau_sec_floors = [0.20, 0.30, 0.40, 0.50]
    tau_M_rels     = [0.05, 0.10, 0.20, 0.30]

    sweep_out = {}
    for cond_label, channels_loader, channels_arg in (
            ("clean", load_channels_clean, args.clean_rgb),
            ("noisy", load_channels_noisy, args.noisy_dir),
    ):
        print(f"\n========== {cond_label} ==========")
        channels = channels_loader(channels_arg)
        primary_t, primary_m, secondary_t_raw, secondary_m_raw = \
            evaluate_lf_streams(cond_label, channels, all_pix,
                                m_values, args.n_orientations,
                                args.r, args.d)

        # Phi/w for primary stream stays constant across the sweep.
        phi_p, w_p, _ = theta_M_to_phi_w(primary_t, primary_m)

        # Junction sub-classification by primary-stream circular range.
        jn_spread = circular_range_deg(primary_t)

        cells = {}      # (tsf, tau_M_rel) -> dict
        for tsf in tau_sec_floors:
            # Apply weak-floor sentinel post-hoc on the raw secondary
            # output.  Note: the no-local-max sentinel was already
            # applied during evaluate_lf_streams (tau_sec_floor=0
            # disables only the magnitude floor, not that sentinel).
            ratio = (secondary_m_raw
                     / np.maximum(primary_m, 1e-30))
            weak  = ratio < tsf
            sec_m_filt = np.where(weak | np.isnan(secondary_t_raw),
                                  0.0, secondary_m_raw)
            sec_t_filt = np.where(weak | np.isnan(secondary_t_raw),
                                  np.nan, secondary_t_raw)
            phi_s, w_s, _ = theta_M_to_phi_w(sec_t_filt, sec_m_filt)

            # Run two-pass fusion with permissive thresholds (we apply
            # tau_M_rel post-hoc).
            t0 = time.perf_counter()
            out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s,
                                    K=args.K, hard_em=True,
                                    tau_M_rel=0.0,
                                    theta_min_deg=0.0)
            print(f"  [{cond_label}] tsf={tsf:.2f}: fusion "
                  f"{time.perf_counter()-t0:.2f}s")

            # Extract signal mu/M from each fit's argmax-pi component.
            P = out["v_fused"].shape[0]
            rng_p = np.arange(P)
            primary_pi  = out["primary_pi"]
            primary_mu  = out["primary_mu"]
            secondary_pi = out["secondary_pi"]
            secondary_mu = out["secondary_mu"]
            def safe_argmax(arr):
                arr_f = np.where(np.isnan(arr), -np.inf, arr)
                return np.argmax(arr_f, axis=1)
            k_p = safe_argmax(primary_pi)
            k_s = safe_argmax(secondary_pi)
            mu_p_phi = primary_mu[rng_p, k_p]
            mu_s_phi = secondary_mu[rng_p, k_s]
            M_p = out["M_primary"]
            M_s = out["M_sec"]   # under permissive run = absolute W_ks
                                  # whenever secondary_valid, else 0
            sep_phi = circular_distance(mu_p_phi, mu_s_phi)
            sep_theta_deg = np.degrees(sep_phi) / 2.0

            v = out["v_fused"] == 1
            edge_ok = v & ~is_junction
            junc_ok = v &  is_junction
            sec_valid = np.isfinite(M_s) & (M_s > 0)
            real_corner = junc_ok & sec_valid \
                          & (sep_theta_deg > args.theta_min_deg)
            swap_mask = junc_ok & (jn_spread > args.swap_spread_deg)
            alwaysA_mask = junc_ok & (jn_spread <= args.swap_spread_deg)

            for tau in tau_M_rels:
                mass_ok = M_s > tau * np.maximum(M_p, 1e-30)
                sep_ok  = sep_theta_deg > args.theta_min_deg
                keep = mass_ok & sep_ok

                edge_FP = float(keep[edge_ok].mean()) if edge_ok.any() else float("nan")
                junc_TP = float(keep[real_corner].mean()) if real_corner.any() else float("nan")
                tp_fp = junc_TP / max(edge_FP, 1e-9) if np.isfinite(junc_TP) else float("nan")
                swap_TP_real = (float(keep[swap_mask & real_corner].mean())
                                if (swap_mask & real_corner).any() else float("nan"))
                alwaysA_TP_real = (float(keep[alwaysA_mask & real_corner].mean())
                                   if (alwaysA_mask & real_corner).any() else float("nan"))
                swap_TP_all = (float(keep[swap_mask].mean())
                               if swap_mask.any() else float("nan"))
                alwaysA_TP_all = (float(keep[alwaysA_mask].mean())
                                  if alwaysA_mask.any() else float("nan"))
                cells[(tsf, tau)] = dict(
                    edge_FP=edge_FP, junc_TP=junc_TP, TP_FP=tp_fp,
                    swap_TP_real=swap_TP_real,
                    alwaysA_TP_real=alwaysA_TP_real,
                    swap_TP_all=swap_TP_all,
                    alwaysA_TP_all=alwaysA_TP_all,
                    n_real_corners=int(real_corner.sum()),
                    n_swap=int(swap_mask.sum()),
                    n_alwaysA=int(alwaysA_mask.sum()),
                    n_edges=int(edge_ok.sum()),
                )

        sweep_out[cond_label] = cells

    # ---------------- print 4x4 tables ----------------
    print("\n" + "=" * 100)
    for cond in ("clean", "noisy"):
        print(f"\n[{cond}] 2D sweep   (cells: edge_FP / junc_TP / TP_FP)")
        hdr = "tau_sec_floor \\ tau_M_rel  " + "  ".join(
            f"{tau:>20.2f}" for tau in tau_M_rels)
        print(hdr)
        print("-" * len(hdr))
        for tsf in tau_sec_floors:
            row = f"      {tsf:>5.2f}                  "
            for tau in tau_M_rels:
                c = sweep_out[cond][(tsf, tau)]
                row += f"  {c['edge_FP']:.3f}/{c['junc_TP']:.3f}/{c['TP_FP']:>5.1f} "
            print(row)

    # ---------------- pick a recommendation ----------------
    # Goal: junc_TP > 0.90, edge_FP < 0.10 (ideally < 0.05),
    # alwaysA_TP_real > 0.70, in BOTH conditions.
    print("\n" + "=" * 100)
    print("Operating-point search (junc_TP > 0.90 in both, lowest edge_FP)")
    best = None
    for tsf in tau_sec_floors:
        for tau in tau_M_rels:
            c = sweep_out["clean"][(tsf, tau)]
            n = sweep_out["noisy"][(tsf, tau)]
            if c["junc_TP"] > 0.90 and n["junc_TP"] > 0.90:
                key = max(c["edge_FP"], n["edge_FP"])
                if best is None or key < best[0]:
                    best = (key, tsf, tau, c, n)
    if best is None:
        print("  No operating point clears junc_TP > 0.90 in both conditions.")
        print("  Falling back to: highest junc_TP for any (tsf, tau).")
        best_score = -1
        for tsf in tau_sec_floors:
            for tau in tau_M_rels:
                c = sweep_out["clean"][(tsf, tau)]
                n = sweep_out["noisy"][(tsf, tau)]
                score = min(c["junc_TP"], n["junc_TP"])
                if score > best_score:
                    best_score = score
                    best = (None, tsf, tau, c, n)
    _, tsf, tau, c, n = best
    print(f"  Recommended: tau_sec_floor = {tsf:.2f}, "
          f"tau_M_rel = {tau:.2f}, theta_min_deg = {args.theta_min_deg}")
    print(f"  clean: edge_FP={c['edge_FP']:.4f}  junc_TP={c['junc_TP']:.4f}  "
          f"TP/FP={c['TP_FP']:.1f}")
    print(f"         swap_TP={c['swap_TP_real']:.4f}  "
          f"alwaysA_TP={c['alwaysA_TP_real']:.4f}")
    print(f"  noisy: edge_FP={n['edge_FP']:.4f}  junc_TP={n['junc_TP']:.4f}  "
          f"TP/FP={n['TP_FP']:.1f}")
    print(f"         swap_TP={n['swap_TP_real']:.4f}  "
          f"alwaysA_TP={n['alwaysA_TP_real']:.4f}")

    # ---------------- save JSON ----------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat = {cond: [
        dict(tau_sec_floor=tsf, tau_M_rel=tau, **cells[(tsf, tau)])
        for tsf in tau_sec_floors for tau in tau_M_rels
    ] for cond, cells in sweep_out.items()}
    with open(args.out, "w") as f:
        json.dump(dict(
            config=dict(K=args.K,
                        theta_min_deg=args.theta_min_deg,
                        swap_spread_deg=args.swap_spread_deg,
                        tau_sec_floors=tau_sec_floors,
                        tau_M_rels=tau_M_rels,
                        n_edge_pixels=args.n_edge_pixels,
                        n_junction_pixels=args.n_junction_pixels,
                        vertex_exclude_px=args.vertex_exclude_px,
                        m_values=m_values,
                        r=args.r, d=args.d,
                        n_orientations=args.n_orientations,
                        architecture="two-pass+sec-sentinel"),
            results=flat,
        ), f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
