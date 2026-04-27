"""Compute the orientation-recovery quintuple per N from exported peaks.

For each N in pixel_peak_analysis.json, the quintuple
    (theta_hat, M_hat, theta_sec, M_sec, v)
is extracted by reading the spline peaks, looking up their magnitudes
in the dense response, and labelling the higher-magnitude peak as the
primary one. The validity flag v is set with a simple range rule

    v = 1  iff  R(pixel) >= tau * R_ref

where R is the range of the dense response and R_ref is supplied by
the caller (e.g. an image-wide max range or a robust percentile).
Without a reference the script defaults to tau = 0 / R_ref = R, so
v is always 1 - the quintuple is then just the four numerical fields.

Usage::

    python scripts/eval/compute_pixel_quintuple.py \\
        --pixel-dir cetz_figures/data/lf_quintuple_L_4096/corner \\
        --r-ref 0   --tau 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-dir", required=True, type=Path)
    parser.add_argument("--r-ref", type=float, default=0.0,
                        help="Image-wide reference range. 0 = use this "
                             "pixel's own range (v always 1).")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="Validity threshold (R > tau * R_ref).")
    args = parser.parse_args()

    peaks = json.loads(
        (args.pixel_dir / "pixel_peak_analysis.json").read_text())

    out_rows = []
    for row in peaks["rows"]:
        n = int(row["n_orientations"])
        dense_a = np.asarray(row["spline_dense_angles_deg"])
        dense_r = np.asarray(row["spline_dense_response"])
        peak_angles = list(row["spline_peaks_deg"])

        # Look up magnitude at each peak angle by nearest-index lookup
        # in the dense curve (10000 points evenly on [0, 180)).
        peak_mags = []
        for pa in peak_angles:
            i = int(round(pa / 180.0 * (len(dense_r) - 1)))
            i = max(0, min(i, len(dense_r) - 1))
            peak_mags.append(float(dense_r[i]))

        if len(peak_angles) == 0:
            theta_hat = float("nan")
            mag_hat = 0.0
            theta_sec = float("nan")
            mag_sec = 0.0
        elif len(peak_angles) == 1:
            theta_hat = float(peak_angles[0])
            mag_hat = float(peak_mags[0])
            theta_sec = float("nan")
            mag_sec = 0.0
        else:
            order = sorted(range(len(peak_mags)),
                           key=lambda i: -peak_mags[i])
            theta_hat = float(peak_angles[order[0]])
            mag_hat = float(peak_mags[order[0]])
            theta_sec = float(peak_angles[order[1]])
            mag_sec = float(peak_mags[order[1]])

        # Range rule for validity (R = response range over [0, pi)).
        r_pixel = float(dense_r.max() - dense_r.min())
        r_ref = args.r_ref if args.r_ref > 0 else r_pixel
        v = int(r_pixel > args.tau * r_ref)

        out_rows.append({
            "n_orientations": n,
            "theta_hat": theta_hat,
            "M_hat": mag_hat,
            "theta_sec": theta_sec,
            "M_sec": mag_sec,
            "v": v,
            "R_pixel": r_pixel,
        })

    out_path = args.pixel_dir / "quintuple.json"
    with open(out_path, "w") as f:
        json.dump({
            "pixel_xy": peaks.get("pixel_xy"),
            "gt_angles_deg": peaks.get("gt_angles_deg"),
            "tau": args.tau,
            "r_ref": args.r_ref,
            "rows": out_rows,
        }, f, indent=2)

    print(f"Quintuples for {args.pixel_dir.name}:")
    print(f"  {'N':>3}  {'theta_hat':>9}  {'M_hat':>7}  "
          f"{'theta_sec':>9}  {'M_sec':>7}  {'v':>1}  {'R':>7}")
    for r in out_rows:
        ts = f"{r['theta_sec']:>9.3f}" if not np.isnan(r['theta_sec']) else f"{'-':>9}"
        ms = f"{r['M_sec']:>7.3f}"
        print(f"  {r['n_orientations']:>3}  {r['theta_hat']:>9.3f}  "
              f"{r['M_hat']:>7.3f}  {ts}  {ms}  {r['v']:>1}  {r['R_pixel']:>7.3f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
