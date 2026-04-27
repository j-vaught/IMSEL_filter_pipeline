"""Validation harness for the weighted vM mixture EM (cgmm_vmm.py).

Runs the five checks from the implementation spec:

  1. vM density integrates to 1 numerically.
  2. EM weighted log-likelihood is monotone non-decreasing across iterations
     on a few real pixels.
  3. At a clean edge pixel the signal component must have pi > 0.6 and
     kappa > 5.
  4. An all-zero pixel (W_total = 0) must short-circuit and emit
     v_fused = 0 without running EM.
  5. theta_fused == (mu[k_signal] mod 2*pi) / 2 (consistency between the
     half-circle output and the doubled-angle internal representation).

Usage::

    PYTHONPATH=src:scripts/eval python3 scripts/eval/cgmm_vmm_validate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from cgmm_vmm import (
    log_I0_safe, vmm_em, vmm_em_with_trace, vmm_fuse, theta_M_to_phi_w,
)


# -- Real-pixel inputs (3 canonical regimes used elsewhere in the paper) --

REGIMES = [
    ("edge",   [2106, 2063]),
    ("corner", [2157, 1711]),
    ("blank",  [2000, 1000]),
]


def load_real_pixel_streams():
    """Pull (theta_deg, M) pooled across 4 channels x 10 m for each
    canonical pixel, from the noisy data dir used for fig_cgmm_three_regimes."""
    import json
    base = Path("/Users/user/Documents/New project/cetz_figures/data")
    out = {}
    for label, _ in REGIMES:
        prefix = f"{'corner_v0' if label == 'corner' else label}_m_sweep_4096_noisy"
        ts, ms = [], []
        for ch in "LRGB":
            d = json.loads((base / f"{prefix}_{ch}.json").read_text())
            v = d["vertices"][0]
            for _, row in v["by_m"].items():
                ts.append(row["theta_hat"])
                ms.append(row["M_hat"])
        out[label] = (np.asarray(ts), np.asarray(ms))
    return out


# -- Check 1 -----------------------------------------------------------

def check_density_integrates_to_one():
    """sum(vM(grid; mu, kappa)) * (2pi/grid_n) ~ 1 for several kappas."""
    from scipy.special import ive
    grid_n = 4096
    grid = np.linspace(0.0, 2.0 * math.pi, grid_n, endpoint=False)
    print("\n[1] vM density integrates to 1 numerically:")
    print(f"    {'kappa':>10} {'integral':>12} {'|err|':>10}")
    ok = True
    for kappa in (0.5, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
        log_norm = float(log_I0_safe(np.array([kappa]))[0]) + math.log(2 * math.pi)
        log_dens = kappa * np.cos(grid - 1.234) - log_norm
        dens = np.exp(log_dens)
        integral = dens.sum() * (2 * math.pi / grid_n)
        err = abs(integral - 1.0)
        ok = ok and err < 1e-4
        print(f"    {kappa:>10.1f} {integral:>12.8f} {err:>10.2e}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- Check 2 -----------------------------------------------------------

def check_log_lik_monotone():
    """Run EM with record_log_lik=True on the 3 canonical pixels and
    confirm the trace is non-decreasing (within numerical noise)."""
    streams = load_real_pixel_streams()
    print("\n[2] EM log-likelihood monotone non-decreasing:")
    ok = True
    for label, _ in REGIMES:
        ts, ms = streams[label]
        phi, w, _ = theta_M_to_phi_w(ts[None, :], ms[None, :])
        _, ll = vmm_em(phi, w, K=3, n_iters=30, record_log_lik=True)
        lls = ll[:, 0]
        diffs = np.diff(lls)
        worst_drop = float(diffs.min())
        cond = worst_drop > -1e-6
        ok = ok and cond
        print(f"    {label:<8} ll[0]={lls[0]:>10.3f} ll[-1]={lls[-1]:>10.3f} "
              f"worst step Δ={worst_drop:>+8.2e} {'OK' if cond else 'FAIL'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- Check 3 -----------------------------------------------------------

def check_clean_edge_concentration():
    """At the clean edge pixel, the signal component should be confident:
    pi > 0.6 (most mass) and kappa > 5 (tight)."""
    streams = load_real_pixel_streams()
    ts, ms = streams["edge"]
    phi, w, _ = theta_M_to_phi_w(ts[None, :], ms[None, :])
    out = vmm_fuse(phi, w, K=3, n_iters=30)
    pi = out["pi"][0]
    kappa = out["kappa"][0]
    k_signal = int(np.argmax(pi))
    pi_s = float(pi[k_signal])
    kappa_s = float(kappa[k_signal])
    print("\n[3] Clean edge pixel: pi[signal] > 0.6 AND kappa[signal] > 5:")
    print(f"    edge pixel  pi[signal]={pi_s:.3f}  kappa[signal]={kappa_s:.2f}")
    cond = pi_s > 0.6 and kappa_s > 5.0
    print(f"    -> {'PASS' if cond else 'FAIL'}")
    return cond


# -- Check 4 -----------------------------------------------------------

def check_all_zero_short_circuit():
    """Pixel with all w=0 must emit v_fused=0 and skip EM."""
    phi = np.zeros((1, 40), dtype=np.float64)
    w   = np.zeros((1, 40), dtype=np.float64)
    out = vmm_fuse(phi, w, K=3, n_iters=30)
    v = int(out["v_fused"][0])
    th = out["theta_fused"][0]
    M  = float(out["M_fused"][0])
    print("\n[4] All-zero pixel short-circuits:")
    print(f"    v_fused={v}  theta_fused={th}  M_fused={M}")
    cond = (v == 0) and (math.isnan(th)) and (M == 0.0)
    print(f"    -> {'PASS' if cond else 'FAIL'}")
    return cond


# -- Check 5 -----------------------------------------------------------

def check_doubled_angle_inverse():
    """theta_fused == (mu[k_signal] mod 2pi) / 2 for a real pixel."""
    streams = load_real_pixel_streams()
    ts, ms = streams["corner"]
    phi, w, _ = theta_M_to_phi_w(ts[None, :], ms[None, :])
    out = vmm_fuse(phi, w, K=3, n_iters=30)
    pi = out["pi"][0]
    k_signal = int(np.argmax(pi))
    mu_s = float(out["mu"][0, k_signal])
    th = float(out["theta_fused"][0])
    expected = (mu_s % (2 * math.pi)) / 2.0
    err = abs(th - expected)
    print("\n[5] theta_fused == (mu[k_signal] mod 2pi) / 2:")
    print(f"    mu[k_signal]={mu_s:.6f}  expected={expected:.6f}  "
          f"theta_fused={th:.6f}  |Δ|={err:.2e}")
    cond = err < 1e-9
    print(f"    -> {'PASS' if cond else 'FAIL'}")
    return cond


# ---------------------------------------------------------------------

def main():
    results = [
        check_density_integrates_to_one(),
        check_log_lik_monotone(),
        check_clean_edge_concentration(),
        check_all_zero_short_circuit(),
        check_doubled_angle_inverse(),
    ]
    print("\n=========================================================")
    print(f"VALIDATION: {sum(results)}/{len(results)} checks passed")
    print("=========================================================")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
