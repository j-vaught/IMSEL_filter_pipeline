"""Compare 4 corner-recovery methods at N3/Acont:
    OR         - default spec (primary OR secondary check, both on M_primary)
    bypass     - force-keep every M_sec>0 pixel
    sec_mag    - secondary check uses M_sec instead of M_primary
    hysteresis - standard Canny hysteresis on the OR-method's NMS magnitudes

Reports the same metrics as the 9-variant audit (P, R, F1, corner-TP,
ridge thickness) plus a "kept" pixel count.  Both clean and noisy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from cgmm_nms import enhanced_nms, hysteresis
from cgmm_image_wide_eval import build_gt_orientation, find_image_spec
from cgmm_nms_audit import precision_recall_f1, corner_tp_rate, ridge_thickness


def evaluate_method(name, nms_mask, gt_edge_mask, real_corner_mask,
                    theta_primary, elapsed_s, n_valid):
    P, R, F1 = precision_recall_f1(nms_mask, gt_edge_mask)
    cTP = corner_tp_rate(nms_mask, real_corner_mask)
    thick, n_comp = ridge_thickness(nms_mask, theta_primary)
    return dict(
        method=name,
        precision=P, recall=R, F1=F1, corner_TP=cTP,
        ridge_thickness=thick, n_components=n_comp,
        n_kept=int(nms_mask.sum()),
        elapsed_s=elapsed_s,
        per_pixel_us=elapsed_s / max(n_valid, 1) * 1e6,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dump", required=True, type=Path)
    p.add_argument("--noisy-dump", required=True, type=Path)
    p.add_argument("--manifest",   required=True, type=Path)
    p.add_argument("--image-key",  default="garnet_atlantic_grass")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--vertex-exclude-px", type=int, default=24)
    p.add_argument("--neighborhood", type=int, default=3,
                   help="N value to evaluate (default N3)")
    p.add_argument("--angular-fidelity", default="Acont",
                   choices=["A8", "A16", "Acont"])
    p.add_argument("--hyst-low-pct",  type=float, default=10.0,
                   help="hysteresis LOW threshold percentile of "
                        "NMS magnitudes (default 10)")
    p.add_argument("--hyst-high-pct", type=float, default=50.0,
                   help="hysteresis HIGH threshold percentile of "
                        "NMS magnitudes (default 50)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    spec = find_image_spec(args.manifest, args.image_key, args.size)
    gt_normal, all_mask, smooth_mask = build_gt_orientation(
        spec, edge_band_px=0, vertex_exclude_px=args.vertex_exclude_px)
    junction_mask = all_mask & ~smooth_mask
    print(f"GT: smooth={smooth_mask.sum():,}  "
          f"junctions={junction_mask.sum():,}  "
          f"all={all_mask.sum():,}")

    out_rows = []
    for cond_label, path in (("clean", args.clean_dump),
                             ("noisy", args.noisy_dump)):
        print(f"\n========== {cond_label} ==========")
        d = np.load(path, allow_pickle=False)
        M_p = d["M_fused"]; th_p = d["theta_fused"]
        M_s = d["M_fused_sec"]; th_s = d["theta_fused_sec"]
        v   = d["v_fused"]
        sec_kept = (M_s > 0) & np.isfinite(th_s)
        real_corner = junction_mask & (v == 1) & sec_kept
        n_valid = int((v == 1).sum())
        print(f"  valid pixels={n_valid:,}  "
              f"real corners={int(real_corner.sum()):,}")

        # --- (a) baseline: corner_method='or' (the existing rule) -----
        t0 = time.perf_counter()
        nms_or = enhanced_nms(th_p, M_p, th_s, M_s, v,
                              neighborhood=args.neighborhood,
                              angular_fidelity=args.angular_fidelity,
                              corner_method="or")
        t_or = time.perf_counter() - t0
        out_rows.append({"condition": cond_label,
                         **evaluate_method("OR (baseline)", nms_or > 0,
                                          all_mask, real_corner,
                                          th_p, t_or, n_valid)})

        # --- (b) bypass NMS at flagged corners ------------------------
        t0 = time.perf_counter()
        nms_byp = enhanced_nms(th_p, M_p, th_s, M_s, v,
                               neighborhood=args.neighborhood,
                               angular_fidelity=args.angular_fidelity,
                               corner_method="bypass")
        t_byp = time.perf_counter() - t0
        out_rows.append({"condition": cond_label,
                         **evaluate_method("bypass at M_sec>0",
                                          nms_byp > 0,
                                          all_mask, real_corner,
                                          th_p, t_byp, n_valid)})

        # --- (c) compare on M_sec for the secondary check -------------
        t0 = time.perf_counter()
        nms_sec = enhanced_nms(th_p, M_p, th_s, M_s, v,
                               neighborhood=args.neighborhood,
                               angular_fidelity=args.angular_fidelity,
                               corner_method="sec_mag")
        t_sec = time.perf_counter() - t0
        out_rows.append({"condition": cond_label,
                         **evaluate_method("sec-check on M_sec",
                                          nms_sec > 0,
                                          all_mask, real_corner,
                                          th_p, t_sec, n_valid)})

        # --- (e) localmax corner detection (5 score-field variants) ---
        for sig_label, cm in (
                ("localmax M_sec",         "localmax_M_sec"),
                ("localmax M_p*M_s",       "localmax_corner_energy"),
                ("localmax M_s*musep",     "localmax_mu_sep"),
                ("localmax dt(M_s>0)",     "localmax_dt"),
                ("localmax dt*M_s",        "localmax_dt_M_sec"),
        ):
            t0 = time.perf_counter()
            nms_lm = enhanced_nms(th_p, M_p, th_s, M_s, v,
                                  neighborhood=args.neighborhood,
                                  angular_fidelity=args.angular_fidelity,
                                  corner_method=cm)
            t_lm = time.perf_counter() - t0
            out_rows.append({"condition": cond_label,
                             **evaluate_method(sig_label,
                                              nms_lm > 0,
                                              all_mask, real_corner,
                                              th_p, t_lm, n_valid)})

        # --- (d) hysteresis post-process on the OR baseline -----------
        # Pick HIGH/LOW thresholds as percentiles of the OR baseline's
        # surviving magnitudes (so the same threshold rule applies in
        # both clean and noisy without manual tuning).
        kept_vals = nms_or[nms_or > 0]
        high = float(np.percentile(kept_vals, args.hyst_high_pct))
        low  = float(np.percentile(kept_vals, args.hyst_low_pct))
        t0 = time.perf_counter()
        nms_hyst = hysteresis(nms_or, high_thresh=high, low_thresh=low)
        t_hyst = (time.perf_counter() - t0) + t_or  # include underlying NMS
        out_rows.append({"condition": cond_label,
                         "hyst_low": low, "hyst_high": high,
                         **evaluate_method("hysteresis (low={:.1f}, high={:.1f})".format(low, high),
                                          nms_hyst,
                                          all_mask, real_corner,
                                          th_p, t_hyst, n_valid)})

    # ---------------- print combined table ----------------
    print("\n" + "=" * 120)
    hdr = (f"{'cond':<6} {'method':<32} "
           f"{'P':>6} {'R':>6} {'F1':>6} {'cornerTP':>9} "
           f"{'thick':>6} {'kept':>9} {'ms':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in out_rows:
        print(f"{r['condition']:<6} {r['method']:<32} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['F1']:>6.4f} {r['corner_TP']:>9.4f} "
              f"{r['ridge_thickness']:>6.3f} {r['n_kept']:>9d} "
              f"{r['elapsed_s']*1000:>6.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            config=dict(neighborhood=args.neighborhood,
                        angular_fidelity=args.angular_fidelity,
                        hyst_low_pct=args.hyst_low_pct,
                        hyst_high_pct=args.hyst_high_pct,
                        vertex_exclude_px=args.vertex_exclude_px),
            results=out_rows,
        ), f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
