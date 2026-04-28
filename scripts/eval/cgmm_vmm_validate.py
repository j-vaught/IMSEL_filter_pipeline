"""Validation harness for the two-pass vMM fusion + §6.3 sentinel.

Runs the spec-section-7 checks from the latest brief:

  1. tau_sec_floor sentinel actually fires.
  2. Two distinct fits operate on independent streams.
  3. Always-A-wins junction case: secondary slot recovered.
  4. Regular-edge case after §6.3 fix: no spurious secondary slot.
  5. Degenerate-pixel guard: n_active < K -> v_fused = 0, secondary skipped.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from cgmm_vmm import vmm_fuse_two_pass, theta_M_to_phi_w
from cgmm_orientation_recovery import find_two_peaks


# -- Check 1 -----------------------------------------------------------

def check_sentinel_fires():
    """Synthetic LF curve with dominant peak at theta=0 (s=1.0) and
    small bump at theta=pi/2 (s=0.05).  tau_sec_floor=0.30 should
    suppress; tau_sec_floor=0.01 should retain."""
    K = 64
    angles = np.linspace(0.0, math.pi, K, endpoint=False)
    def gauss(x, mu, sigma=0.05):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    resp = (1.00 * gauss(angles, 0.0)
            + 0.05 * gauss(angles, math.pi / 2)).reshape(1, -1)

    print("\n[1] §6.3 sentinel fires correctly:")
    ok = True
    for tau, expect_zero in ((0.30, True), (0.01, False)):
        _, _, th_sec, M_sec = find_two_peaks(angles, resp,
                                             tau_sec_floor=tau)
        zero = (np.isnan(th_sec[0]) and M_sec[0] == 0.0)
        c = (zero == expect_zero)
        ok = ok and c
        print(f"    tau_sec_floor={tau:.2f}  -> "
              f"theta_sec={th_sec[0]:.4f}, M_sec={M_sec[0]:.4f}  "
              f"({'zeroed' if zero else 'kept'})  "
              f"{'OK' if c else 'FAIL'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- Check 2 -----------------------------------------------------------

def check_distinct_fits():
    """Primary stream all theta_n=0deg, secondary stream all theta_n_sec=
    90deg.  Primary fit signal mu_phi should be ~0; secondary fit signal
    mu_phi should be ~180 (= doubled-angle of 90deg)."""
    N = 40
    prim_t_deg = np.zeros((1, N))
    prim_m     = np.ones((1, N))
    sec_t_deg  = np.full((1, N), 90.0)
    sec_m      = np.ones((1, N))
    phi_p, w_p, _ = theta_M_to_phi_w(prim_t_deg, prim_m)
    phi_s, w_s, _ = theta_M_to_phi_w(sec_t_deg,  sec_m)
    out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, hard_em=True,
                            tau_M_rel=0.05, theta_min_deg=10.0)
    p_mu = (np.degrees(out["primary_mu"][0])
            [int(np.argmax(out["primary_pi"][0]))]) % 360
    s_mu = (np.degrees(out["secondary_mu"][0])
            [int(np.argmax(out["secondary_pi"][0]))]) % 360
    p_ok = abs(p_mu) < 0.5 or abs(p_mu - 360) < 0.5
    s_ok = abs(s_mu - 180) < 0.5
    print("\n[2] Two distinct fits on independent streams:")
    print(f"    primary  fit signal mu_phi = {p_mu:.2f} deg (expected ~0)")
    print(f"    secondary fit signal mu_phi = {s_mu:.2f} deg (expected ~180)")
    cond = p_ok and s_ok
    print(f"    -> {'PASS' if cond else 'FAIL'}")
    return cond


# -- Check 3 -----------------------------------------------------------

def check_always_A_wins():
    """Synthetic always-A-wins junction.  Primary all theta_n=0deg @ M=1,
    secondary all theta_n_sec=90deg @ M=0.5.  Both slots should be
    preserved by suppression."""
    N = 40
    prim_t = np.zeros((1, N))
    prim_m = np.ones((1, N))
    sec_t  = np.full((1, N), 90.0)
    sec_m  = np.full((1, N), 0.5)
    phi_p, w_p, _ = theta_M_to_phi_w(prim_t, prim_m)
    phi_s, w_s, _ = theta_M_to_phi_w(sec_t,  sec_m)
    out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, hard_em=True,
                            tau_M_rel=0.05, theta_min_deg=10.0)
    tp = float(out["theta_primary"][0])
    ts = float(out["theta_sec"][0])
    keep = int(out["keep_secondary_mask"][0])
    print("\n[3] Always-A-wins junction (synthetic):")
    print(f"    theta_primary = {math.degrees(tp):.3f} deg "
          f"(expected ~0)")
    print(f"    theta_sec    = {math.degrees(ts):.3f} deg "
          f"(expected ~90)")
    print(f"    keep_secondary = {keep}  (expected 1)")
    ok = (abs(math.degrees(tp)) < 0.5
          and abs(math.degrees(ts) - 90) < 0.5
          and keep == 1)
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- Check 4 -----------------------------------------------------------

def check_regular_edge_post_sentinel():
    """Regular edge: primary all theta=0 deg @ M=1.  Per-config secondary
    peaks at random angles, all M_sec=0.10.  With tau_sec_floor=0.30
    these are all zeroed upstream, so the secondary stream has 0 active
    configs.  The two-pass fusion's degenerate guard then short-circuits
    the secondary fit; theta_sec must be NaN, M_sec=0."""
    N = 40
    prim_t = np.zeros((1, N))
    prim_m = np.ones((1, N))
    rng = np.random.default_rng(0)
    sec_t_raw = rng.uniform(0, 180, size=(1, N))
    sec_m_raw = np.full((1, N), 0.10)
    # Apply tau_sec_floor = 0.30 sentinel post-hoc.
    weak = (sec_m_raw / np.maximum(prim_m, 1e-30)) < 0.30
    sec_t = np.where(weak, np.nan, sec_t_raw)
    sec_m = np.where(weak, 0.0,    sec_m_raw)
    phi_p, w_p, _ = theta_M_to_phi_w(prim_t, prim_m)
    phi_s, w_s, _ = theta_M_to_phi_w(sec_t,  sec_m)
    n_active_s = int((w_s > 1e-12).sum())
    out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, hard_em=True,
                            tau_M_rel=0.05, theta_min_deg=10.0)
    print("\n[4] Regular edge after §6.3 fix (sentinel zeroes secondaries):")
    print(f"    n_active_sec after sentinel = {n_active_s}  (expected 0)")
    print(f"    theta_sec = {out['theta_sec'][0]}  (expected NaN)")
    print(f"    M_sec    = {float(out['M_sec'][0]):.4f}  (expected 0)")
    print(f"    v_fused  = {int(out['v_fused'][0])}  (expected 1)")
    ok = (n_active_s == 0
          and np.isnan(out["theta_sec"][0])
          and float(out["M_sec"][0]) == 0.0
          and int(out["v_fused"][0]) == 1)
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- Check 5 -----------------------------------------------------------

def check_degenerate_guard():
    """n_active_p < K  -> v_fused=0, both fits skipped.
    n_active_p >= K and n_active_s < K  -> v_fused=1, secondary skipped."""
    N = 40
    print("\n[5] Per-pass degenerate-pixel guard (K=3):")
    ok = True

    # Case A: primary invalid (n_active_p = 2 < K = 3).
    prim_t = np.zeros((1, N))
    prim_m = np.zeros((1, N)); prim_m[0, :2] = 1.0
    sec_t  = np.zeros((1, N))
    sec_m  = np.ones((1, N))
    phi_p, w_p, _ = theta_M_to_phi_w(prim_t, prim_m)
    phi_s, w_s, _ = theta_M_to_phi_w(sec_t,  sec_m)
    out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, hard_em=True)
    cond_a = (int(out["v_fused"][0]) == 0
              and np.isnan(out["theta_primary"][0])
              and np.isnan(out["theta_sec"][0]))
    print(f"    primary invalid:    v_fused={int(out['v_fused'][0])}  "
          f"theta_p={out['theta_primary'][0]}  "
          f"theta_s={out['theta_sec'][0]}  "
          f"{'OK' if cond_a else 'FAIL'}")
    ok = ok and cond_a

    # Case B: primary valid, secondary invalid (n_active_s = 2 < K).
    prim_t = np.zeros((1, N))
    prim_m = np.ones((1, N))
    sec_t  = np.zeros((1, N))
    sec_m  = np.zeros((1, N)); sec_m[0, :2] = 1.0
    phi_p, w_p, _ = theta_M_to_phi_w(prim_t, prim_m)
    phi_s, w_s, _ = theta_M_to_phi_w(sec_t,  sec_m)
    out = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, hard_em=True)
    cond_b = (int(out["v_fused"][0]) == 1
              and np.isnan(out["theta_sec"][0])
              and float(out["M_sec"][0]) == 0.0)
    print(f"    secondary invalid:  v_fused={int(out['v_fused'][0])}  "
          f"theta_s={out['theta_sec'][0]}  "
          f"M_s={float(out['M_sec'][0]):.4f}  "
          f"{'OK' if cond_b else 'FAIL'}")
    ok = ok and cond_b
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------

def main():
    results = [
        check_sentinel_fires(),
        check_distinct_fits(),
        check_always_A_wins(),
        check_regular_edge_post_sentinel(),
        check_degenerate_guard(),
    ]
    print("\n=========================================================")
    print(f"VALIDATION: {sum(results)}/{len(results)} checks passed")
    print("=========================================================")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
