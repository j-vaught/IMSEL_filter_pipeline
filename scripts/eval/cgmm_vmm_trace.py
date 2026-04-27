"""Per-iteration EM trace dump for three canonical pixels (edge,
corner, off-edge) for the §7 figure / debugging.

Pulls noisy LF samples from the existing m-sweep JSON files at the
three pre-selected pixel coordinates, runs vmm_em_with_trace at K=3,
and writes the (mu, kappa, pi, gamma, log_lik) trajectory to an
HDF5-style npz file plus a human-readable summary print.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cgmm_vmm import vmm_em_with_trace, theta_M_to_phi_w


REGIMES = [
    ("edge",   [2106, 2063]),
    ("corner", [2157, 1711]),
    ("blank",  [2000, 1000]),
]


def load_real_pixel_streams():
    base = Path("/Users/user/Documents/New project/cetz_figures/data")
    out = {}
    for label, pxy in REGIMES:
        prefix = f"{'corner_v0' if label == 'corner' else label}_m_sweep_4096_noisy"
        ts, ms = [], []
        for ch in "LRGB":
            d = json.loads((base / f"{prefix}_{ch}.json").read_text())
            v = d["vertices"][0]
            for _, row in v["by_m"].items():
                ts.append(row["theta_hat"])
                ms.append(row["M_hat"])
        out[label] = (np.asarray(ts), np.asarray(ms), pxy)
    return out


def main():
    out_dir = Path("outputs/cgmm_vmm_trace")
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = load_real_pixel_streams()
    K, n_iters = 3, 30

    # Stack pixels as (P=3, N=40)
    labels = [lbl for lbl, _ in REGIMES]
    pix_xy = np.stack([streams[l][2] for l in labels])
    theta  = np.stack([streams[l][0] for l in labels])
    mag    = np.stack([streams[l][1] for l in labels])

    phi, w, v = theta_M_to_phi_w(theta, mag)
    tr = vmm_em_with_trace(phi, w, K=K, n_iters=n_iters)

    # Save .npz
    npz_path = out_dir / "trace_K3.npz"
    np.savez_compressed(npz_path,
        labels      = np.asarray(labels),
        pixel_xy    = pix_xy,
        theta_deg   = theta,
        magnitude   = mag,
        phi         = phi,
        w           = w,
        v           = v,
        mu_trace    = tr["mu_trace"],
        kappa_trace = tr["kappa_trace"],
        pi_trace    = tr["pi_trace"],
        gamma_trace = tr["gamma_trace"],
        W_trace     = tr["W_trace"],
        log_lik_trace = tr["log_lik_trace"],
    )
    print(f"Wrote {npz_path}")

    # Human-readable per-pixel summary
    for p, lbl in enumerate(labels):
        print(f"\n=== {lbl} pixel @ {pix_xy[p].tolist()} ===")
        print("iter   pi[0]  pi[1]  pi[2]   mu_deg[0/1/2]                    "
              "kappa[0/1/2]                  log_lik")
        for it in range(n_iters + 1):
            mu_deg = (tr["mu_trace"][it, p] % (2 * np.pi)) / 2 * 180 / np.pi
            ka     = tr["kappa_trace"][it, p]
            pi_    = tr["pi_trace"][it, p]
            ll = (tr["log_lik_trace"][it - 1, p]
                  if it > 0 else float("nan"))
            print(f"{it:>4}   "
                  f"{pi_[0]:.3f}  {pi_[1]:.3f}  {pi_[2]:.3f}   "
                  f"{mu_deg[0]:>7.3f} / {mu_deg[1]:>7.3f} / {mu_deg[2]:>7.3f}   "
                  f"{ka[0]:>7.1f} / {ka[1]:>7.1f} / {ka[2]:>7.1f}   "
                  f"{ll:>10.3f}")


if __name__ == "__main__":
    main()
