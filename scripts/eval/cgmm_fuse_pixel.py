"""Fit a 2-component magnitude-weighted c-GMM on a single pixel's
configuration stack and emit the fused orientation + per-component
diagnostics.

The c-GMM works on the *doubled-angle* unit circle (phi = 2*theta) so
that pi-periodic orientations become 2pi-periodic and the fitting is
ordinary Gaussian on (cos phi, sin phi). Each (theta_i, M_i) measurement
is weighted by M_i; the validity flag v selects which measurements
enter the fit. The fused orientation is the mean direction of the
component carrying the larger mixing weight.

Usage::

    python scripts/eval/cgmm_fuse_pixel.py \\
        --pixel-dir cetz_figures/data/lf_quintuple_corners/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


def cgmm_fuse(thetas_deg: np.ndarray, mags: np.ndarray,
              n_components: int = 2,
              n_samples: int = 2000,
              random_state: int = 0) -> dict:
    """Magnitude-weighted GMM on the doubled-angle unit circle.

    Each measurement is replicated proportionally to its magnitude so
    a stock unweighted EM (sklearn.mixture.GaussianMixture) can fit
    the weighted mixture. Returns the components sorted by descending
    mixing weight; the first is the consensus / fused estimate.
    """
    thetas_deg = np.asarray(thetas_deg, dtype=float)
    mags = np.asarray(mags, dtype=float)
    assert thetas_deg.shape == mags.shape

    if mags.sum() <= 0 or len(thetas_deg) == 0:
        return {
            "theta_fused_deg": float("nan"),
            "weight_fused": 0.0,
            "components": [],
            "n_in": int(len(thetas_deg)),
        }

    phi = 2 * np.radians(thetas_deg)
    pts = np.column_stack([np.cos(phi), np.sin(phi)])
    w = mags / mags.sum()
    counts = np.round(w * n_samples).astype(int)
    counts[counts < 1] = 1
    samples = np.repeat(pts, counts, axis=0)

    n_eff = min(n_components, len(np.unique(samples, axis=0)))
    gm = GaussianMixture(n_components=n_eff,
                         covariance_type="full",
                         random_state=random_state).fit(samples)

    components = []
    for i in range(n_eff):
        mu = gm.means_[i]
        ang_phi = float(np.arctan2(mu[1], mu[0]))
        ang_theta = float(np.degrees(ang_phi / 2.0)) % 180.0
        rbar = float(np.linalg.norm(mu))
        components.append({
            "weight": float(gm.weights_[i]),
            "theta_deg": ang_theta,
            "R": rbar,
        })

    components.sort(key=lambda c: -c["weight"])
    return {
        "theta_fused_deg": components[0]["theta_deg"],
        "weight_fused": components[0]["weight"],
        "components": components,
        "n_in": int(len(thetas_deg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel-dir", required=True, type=Path)
    parser.add_argument("--include-secondary", action="store_true",
                        help="Also include (theta_sec, M_sec) as a "
                             "second measurement per config.")
    parser.add_argument("--validity-tau", type=float, default=0.0,
                        help="Drop configs whose M_hat is below this "
                             "fraction of the per-pixel max M_hat.")
    args = parser.parse_args()

    sweep = json.loads(
        (args.pixel_dir / "sweep_quintuples.json").read_text())
    rows = sweep["rows"]

    primary_t = [r["theta_hat"] for r in rows]
    primary_m = [r["M_hat"] for r in rows]
    if args.include_secondary:
        secondary_t = [r["theta_sec"] for r in rows
                       if (r["theta_sec"] == r["theta_sec"])
                       and r["M_sec"] > 0]
        secondary_m = [r["M_sec"] for r in rows
                       if (r["theta_sec"] == r["theta_sec"])
                       and r["M_sec"] > 0]
        all_t = primary_t + secondary_t
        all_m = primary_m + secondary_m
    else:
        all_t = primary_t
        all_m = primary_m

    if args.validity_tau > 0:
        m_max = max(all_m)
        keep = [(t, m) for t, m in zip(all_t, all_m)
                if m > args.validity_tau * m_max]
        all_t, all_m = ([t for t, _ in keep], [m for _, m in keep])

    result = cgmm_fuse(all_t, all_m, n_components=2)
    result["pixel_xy"] = sweep.get("pixel_xy")
    result["include_secondary"] = bool(args.include_secondary)
    result["validity_tau"] = float(args.validity_tau)

    out = args.pixel_dir / "cgmm_fusion.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"c-GMM fusion at pixel {result['pixel_xy']} "
          f"(n_in={result['n_in']}):")
    print(f"  fused theta = {result['theta_fused_deg']:.2f} deg "
          f"(component weight = {result['weight_fused']:.3f})")
    for i, c in enumerate(result["components"]):
        print(f"    component {i}: w={c['weight']:.3f}  "
              f"theta={c['theta_deg']:.2f}  R={c['R']:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
