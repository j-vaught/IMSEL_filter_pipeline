"""9-variant NMS audit per the §8 brief.

Sweeps neighborhood in {1, 2, 3} x angular_fidelity in {A8, A16, Acont}
on clean and noisy fusion dumps.  Reports edge precision, recall, F1,
corner-preservation TP, ridge thickness, and per-pixel runtime.

Usage::

    PYTHONPATH=src:scripts/eval python3 scripts/eval/cgmm_nms_audit.py \\
        --clean-dump  outputs/cgmm_fusion_dump/clean_K3_dilated7.npz \\
        --noisy-dump  outputs/cgmm_fusion_dump/noisy_K3_dilated7.npz \\
        --manifest    example_images/synthetic_nested_shapes/manifest.json \\
        --image-key   garnet_atlantic_grass --size 4096 \\
        --out         outputs/cgmm_image_wide_eval/nms_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from cgmm_nms import enhanced_nms
from cgmm_image_wide_eval import build_gt_orientation, find_image_spec


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def precision_recall_f1(nms_mask, gt_edge_mask):
    """Per-pixel precision and recall with 1-pixel tolerance.

    precision = |NMS-positive within 1px of any GT edge| / |NMS-positive|
    recall    = |GT edge with any NMS-positive within 1px| / |GT edge|
    """
    if not nms_mask.any():
        return 0.0, 0.0, 0.0
    if not gt_edge_mask.any():
        return 0.0, 0.0, 0.0
    gt_dilated  = ndimage.binary_dilation(gt_edge_mask,
                                          structure=np.ones((3, 3)),
                                          iterations=1)
    nms_dilated = ndimage.binary_dilation(nms_mask,
                                          structure=np.ones((3, 3)),
                                          iterations=1)
    tp_p = (nms_mask & gt_dilated).sum()
    tp_r = (gt_edge_mask & nms_dilated).sum()
    P = tp_p / max(nms_mask.sum(), 1)
    R = tp_r / max(gt_edge_mask.sum(), 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    return float(P), float(R), float(F1)


def corner_tp_rate(nms_mask, real_corner_mask):
    """Fraction of real-corner pixels that are NMS-positive."""
    if not real_corner_mask.any():
        return float("nan")
    return float((nms_mask & real_corner_mask).sum()
                 / real_corner_mask.sum())


def ridge_thickness(nms_mask, theta_primary):
    """Mean of the maximum perpendicular extent of connected
    NMS-positive components.

    For each connected component we estimate the local edge tangent
    direction as the magnitude-weighted circular mean of theta_primary
    across component pixels (in doubled-angle space).  We then project
    component pixel coordinates onto the perpendicular direction and
    return the range (max - min) of the projection.  The mean over all
    components is the reported thickness.

    A perfectly thinned 1-pixel ridge gives a thickness ~ 0 to 1
    (range of integer projections of co-linear pixels).  Anything
    much above 1 indicates under-thinning.
    """
    if not nms_mask.any():
        return float("nan"), 0
    labels, n_comp = ndimage.label(nms_mask, structure=np.ones((3, 3)))
    thicknesses = []
    for cid in range(1, n_comp + 1):
        ys, xs = np.where(labels == cid)
        if len(ys) < 2:
            continue
        # Mean tangent direction in doubled-angle space.
        th = theta_primary[ys, xs]
        # NaN-safe (theta might be invalid in some edge cases).
        good = np.isfinite(th)
        if not good.any():
            continue
        cos2 = np.cos(2 * th[good]).mean()
        sin2 = np.sin(2 * th[good]).mean()
        mu_t = math.atan2(sin2, cos2) / 2.0
        # Perpendicular unit vector.
        perp_x = -math.sin(mu_t)
        perp_y =  math.cos(mu_t)
        proj = xs.astype(np.float64) * perp_x + ys.astype(np.float64) * perp_y
        thicknesses.append(float(proj.max() - proj.min()))
    if not thicknesses:
        return float("nan"), 0
    return float(np.mean(thicknesses)), len(thicknesses)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

VARIANTS = [
    (n, fid)
    for n in (1, 2, 3)
    for fid in ("A8", "A16", "Acont")
]


def run_variants(label, dump, gt_edge_mask, real_corner_mask,
                 fid_to_dump_keys):
    """Run all 9 variants on the given dump.  Returns list of dicts."""
    M_p     = dump["M_primary"]
    th_p    = dump["theta_primary"]
    M_s     = dump["M_sec"]
    th_s    = dump["theta_sec"]
    v       = dump["v_fused"]
    n_valid = int((v == 1).sum())
    print(f"  [{label}] valid pixels = {n_valid:,}")

    rows = []
    for nbh, fid in VARIANTS:
        t0 = time.perf_counter()
        out = enhanced_nms(th_p, M_p, th_s, M_s, v,
                           neighborhood=nbh, angular_fidelity=fid)
        elapsed = time.perf_counter() - t0
        nms_mask = out > 0
        n_kept = int(nms_mask.sum())

        P, R, F1 = precision_recall_f1(nms_mask, gt_edge_mask)
        corner_TP = corner_tp_rate(nms_mask, real_corner_mask)
        thickness, n_comp = ridge_thickness(nms_mask, th_p)
        per_pixel_us = (elapsed / max(n_valid, 1)) * 1e6

        rows.append(dict(
            condition=label, neighborhood=nbh, angular_fidelity=fid,
            n_kept=n_kept,
            precision=P, recall=R, F1=F1,
            corner_TP=corner_TP,
            ridge_thickness=thickness, n_components=n_comp,
            elapsed_s=elapsed, per_pixel_us=per_pixel_us,
        ))
        print(f"    N{nbh}/{fid:>5}: P={P:.4f}  R={R:.4f}  F1={F1:.4f}  "
              f"cornerTP={corner_TP:.4f}  thick={thickness:.3f}  "
              f"comps={n_comp:>4}  kept={n_kept:>6}  "
              f"{elapsed*1000:.1f}ms  ({per_pixel_us:.2f} us/pix)")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dump", required=True, type=Path)
    p.add_argument("--noisy-dump", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--image-key", default="garnet_atlantic_grass")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--vertex-exclude-px", type=int, default=24)
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
    for label, path in (("clean", args.clean_dump),
                        ("noisy", args.noisy_dump)):
        print(f"\n========== {label} ==========")
        d = np.load(path, allow_pickle=False)
        # Real-corner mask: junction pixels where the c-GMM kept a
        # secondary slot (i.e., spatially-separated bimodal evidence).
        v = d["v_fused"]
        sec_kept = (d["M_sec"] > 0) & np.isfinite(d["theta_sec"])
        real_corner_mask = junction_mask & (v == 1) & sec_kept

        rows = run_variants(label, d, all_mask, real_corner_mask,
                             fid_to_dump_keys=None)
        out_rows.extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            config=dict(image_key=args.image_key, size=args.size,
                        vertex_exclude_px=args.vertex_exclude_px,
                        variants=[dict(neighborhood=n, angular_fidelity=fid)
                                  for n, fid in VARIANTS]),
            results=out_rows,
        ), f, indent=2)
    print(f"\nWrote {args.out}")

    # Summary tables.
    print("\n" + "=" * 110)
    for cond in ("clean", "noisy"):
        print(f"\n[{cond}]   N x A     "
              f"{'P':>6} {'R':>6} {'F1':>6}  "
              f"{'corner_TP':>10}  {'thickness':>9}  {'us/pix':>8}")
        for r in out_rows:
            if r["condition"] != cond:
                continue
            print(f"     N{r['neighborhood']}/{r['angular_fidelity']:>5}  "
                  f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['F1']:>6.4f}  "
                  f"{r['corner_TP']:>10.4f}  {r['ridge_thickness']:>9.3f}  "
                  f"{r['per_pixel_us']:>8.2f}")


if __name__ == "__main__":
    main()
