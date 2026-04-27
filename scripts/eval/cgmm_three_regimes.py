"""Fit the 2-component magnitude-weighted c-GMM at the three canonical
regime pixels (corner v0, edge, blank) using their noisy 4-channel
m-sweep data.

Pools (theta, M) primary peaks across all 4 channels and all m values
into one configuration stack per pixel; doubles the angle (phi = 2 theta)
to lift to a 2pi-periodic representation; fits a magnitude-weighted
2-component Gaussian mixture in (cos phi, sin phi) by replicating each
point proportional to its M weight; recovers the two vM-equivalent
mean directions back in theta-space.

Output JSON for each pixel:
    {
        "label": "corner" | "edge" | "blank",
        "pixel_xy": [x, y],
        "samples": [{"theta": ..., "M": ..., "phi": ...}, ...],
        "components": [{"weight": ..., "theta_mean": ..., "phi_mean": ..., "R": ...}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture


CASES = (
    ("corner", "corner_v0_m_sweep_4096_noisy"),
    ("edge",   "edge_m_sweep_4096_noisy"),
    ("blank",  "blank_m_sweep_4096_noisy"),
)


def cgmm_fit(thetas_deg: np.ndarray, mags: np.ndarray,
             n_components: int = 2,
             n_samples: int = 4000,
             random_state: int = 0) -> list[dict]:
    """Magnitude-weighted GMM in doubled-angle unit-circle coords.
    Returns the components sorted by descending mixing weight."""
    if mags.sum() <= 1e-9 or len(thetas_deg) == 0:
        return []
    phi = 2 * np.radians(thetas_deg)
    pts = np.column_stack([np.cos(phi), np.sin(phi)])
    w = mags / mags.sum()
    counts = np.round(w * n_samples).astype(int)
    counts[counts < 1] = 1
    samples = np.repeat(pts, counts, axis=0)
    n_eff = min(n_components, len(np.unique(samples, axis=0)))
    if n_eff < 1:
        return []
    gm = GaussianMixture(n_components=n_eff,
                         covariance_type="full",
                         random_state=random_state).fit(samples)

    components = []
    for i in range(n_eff):
        mu = gm.means_[i]
        phi_mean = float(np.arctan2(mu[1], mu[0]))
        theta_mean = float(np.degrees(phi_mean / 2.0)) % 180.0
        R = float(np.linalg.norm(mu))
        components.append({
            "weight": float(gm.weights_[i]),
            "theta_mean": theta_mean,
            "phi_mean": float(phi_mean),
            "R": R,
        })
    components.sort(key=lambda c: -c["weight"])
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Dir containing the {corner_v0,edge,blank}_*.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out = {"cases": []}
    for label, prefix in CASES:
        all_thetas = []
        all_mags = []
        pixel_xy = None
        for ch in ("L", "R", "G", "B"):
            path = args.data_dir / f"{prefix}_{ch}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            v = data["vertices"][0]
            if pixel_xy is None:
                pixel_xy = v["pixel_xy"]
            for m_str, row in v["by_m"].items():
                all_thetas.append(row["theta_hat"])
                all_mags.append(row["M_hat"])
                # Include the secondary peak too so corner pixels expose
                # both tangent directions in the configuration stack.
                if (row["theta_sec"] == row["theta_sec"]
                        and row["M_sec"] > 0.0):
                    all_thetas.append(row["theta_sec"])
                    all_mags.append(row["M_sec"])
        thetas = np.asarray(all_thetas)
        mags = np.asarray(all_mags)
        components = cgmm_fit(thetas, mags)

        samples = [
            {"theta": float(t), "M": float(m),
             "phi": float(2 * np.radians(t))}
            for t, m in zip(thetas, mags)
        ]
        out["cases"].append({
            "label": label,
            "pixel_xy": pixel_xy,
            "samples": samples,
            "components": components,
        })
        n_real = sum(1 for s in samples if s["M"] > 1e-6)
        print(f"  {label}: {len(samples)} samples ({n_real} non-zero), "
              f"{len(components)} components")
        for c in components:
            print(f"    w={c['weight']:.3f}  "
                  f"theta_mean={c['theta_mean']:.2f}  "
                  f"R={c['R']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
