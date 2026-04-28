"""Acceptance tests for the Metal-accelerated c-GMM K=3 fusion.

Run from repo root:

    PYTHONPATH=src:agent_workspaces/cgmm_metal \\
        python3 agent_workspaces/cgmm_metal/tests/test_cgmm_metal.py

Four tests:
    1. test_correctness_small   -- 1M rows, N=4, outputs match reference.
    2. test_correctness_full    -- full 4096^2 (16.7M rows, N=16).
    3. test_speed_small         -- >= 20x speedup over Python at 1M, N=4.
    4. test_speed_full          -- full image (16.7M, N=16) under hard target.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference_impl import cgmm_fuse_two_pass, theta_M_to_phi_w  # source-of-truth

try:
    from edgecritic.cgmm._metal import (  # type: ignore
        cgmm_fuse_two_pass_metal,
        cgmm_backend_available,
    )
    METAL_OK = cgmm_backend_available()
except Exception as ex:
    METAL_OK = False
    _IMPORT_ERR = repr(ex)


REF = Path(__file__).resolve().parent.parent / "reference"
THETA_TOL_RAD = math.radians(1.0)   # 1 deg
M_TOL_REL     = 5e-2                # 5%
V_DISAGREE_FRAC   = 1e-3            # <=0.1% rows may disagree on v_fused
SEC_DISAGREE_FRAC = 5e-3            # <=0.5% rows may disagree on secondary kept-ness


def _check_outputs(expected, predicted, tag):
    th_p_g = expected["theta_primary"]
    M_p_g  = expected["M_primary"]
    th_s_g = expected["theta_sec"]
    M_s_g  = expected["M_sec"]
    v_g    = expected["v_fused"]

    th_p, M_p, th_s, M_s, v = (
        predicted["theta_primary"],
        predicted["M_primary"],
        predicted["theta_sec"],
        predicted["M_sec"],
        predicted["v_fused"],
    )
    n = th_p.size

    # ---- v_fused ----
    n_disagree_v = int((v.astype(bool) ^ v_g.astype(bool)).sum())
    print(f"  [{tag}] v_fused  : disagree={n_disagree_v}/{n} "
          f"({n_disagree_v/n*100:.3f}%)")
    if n_disagree_v / n > V_DISAGREE_FRAC:
        raise AssertionError(
            f"v_fused disagrees on {n_disagree_v}/{n} rows "
            f"(allowed {V_DISAGREE_FRAC*100:.1f}%)")

    # Compare on rows where both are valid.
    both_valid = (v == 1) & (v_g == 1)
    if both_valid.any():
        a = th_p[both_valid].astype(np.float64)
        b = th_p_g[both_valid].astype(np.float64)
        dth = np.minimum(np.abs(a - b), math.pi - np.abs(a - b))
        bad_th = dth > THETA_TOL_RAD
        ma = M_p[both_valid].astype(np.float64)
        mb = M_p_g[both_valid].astype(np.float64)
        dM = np.abs(ma - mb) / np.maximum(mb, 1e-12)
        bad_M = dM > M_TOL_REL
        n_bad_p = int((bad_th | bad_M).sum())
        print(f"  [{tag}] primary  : "
              f"max dtheta={np.degrees(dth.max()):.4f} deg  "
              f"max dM/M={dM.max():.2e}  bad={n_bad_p}/{int(both_valid.sum())}")
        if n_bad_p / int(both_valid.sum()) > 0.001:
            raise AssertionError(
                f"primary disagreement on {n_bad_p} rows "
                f"(allowed 0.1% of valid)")

    # ---- secondary kept-ness ----
    sec_kept_g = ~np.isnan(th_s_g) & (M_s_g > 0)
    sec_kept   = ~np.isnan(th_s)   & (M_s   > 0)
    n_disagree_s = int((sec_kept_g ^ sec_kept).sum())
    print(f"  [{tag}] sec kept : disagree={n_disagree_s}/{n} "
          f"({n_disagree_s/n*100:.3f}%)")
    if n_disagree_s / n > SEC_DISAGREE_FRAC:
        raise AssertionError(
            f"secondary kept-ness disagrees on {n_disagree_s}/{n} rows")

    both_kept = sec_kept & sec_kept_g
    if both_kept.any():
        a = th_s[both_kept].astype(np.float64)
        b = th_s_g[both_kept].astype(np.float64)
        ds = np.minimum(np.abs(a - b), math.pi - np.abs(a - b))
        ma = M_s[both_kept].astype(np.float64)
        mb = M_s_g[both_kept].astype(np.float64)
        dms = np.abs(ma - mb) / np.maximum(mb, 1e-12)
        n_bad_s = int(((ds > THETA_TOL_RAD) | (dms > M_TOL_REL)).sum())
        print(f"  [{tag}] secondary: "
              f"max dtheta={np.degrees(ds.max()):.4f} deg  "
              f"max dM/M={dms.max():.2e}  bad={n_bad_s}/{int(both_kept.sum())}")
        if n_bad_s / int(both_kept.sum()) > 0.005:
            raise AssertionError(
                f"secondary value disagreement on {n_bad_s} rows "
                f"(allowed 0.5% of both-kept)")


def _load_inputs():
    inp = np.load(REF / "inputs.npz")
    return {
        "phi_p": inp["phi_p"].astype(np.float64),
        "w_p":   inp["w_p"].astype(np.float64),
        "phi_s": inp["phi_s"].astype(np.float64),
        "w_s":   inp["w_s"].astype(np.float64),
    }


def _load_expected():
    exp = np.load(REF / "expected.npz")
    return {k: exp[k] for k in exp.files}


def test_correctness_small():
    if not METAL_OK:
        raise RuntimeError(f"Metal c-GMM backend unavailable: {_IMPORT_ERR}")
    ins = _load_inputs()
    exp = _load_expected()
    print(f"[small] P={ins['phi_p'].shape[0]:,}  N={ins['phi_p'].shape[1]}")

    pred = cgmm_fuse_two_pass_metal(
        ins["phi_p"], ins["w_p"], ins["phi_s"], ins["w_s"],
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)

    _check_outputs(exp, pred, "small")
    print("  PASS")


def test_correctness_full():
    """Run the fused front end on the noisy 4096^2 image with N=16
    inputs, then compare Metal c-GMM against the Python reference on a
    500K random subset (full reference is too slow to run on 16.7M)."""
    if not METAL_OK:
        raise RuntimeError("Metal c-GMM backend unavailable")
    from PIL import Image
    from edgecritic.pipeline import wvf_lf_recover_metal

    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/4096"
                / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png")
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(rgb + rng.normal(0.0, 13.0, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    R, G, B = rgb_n[..., 0], rgb_n[..., 1], rgb_n[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    chans = {"L": L.astype(np.float32),
             "R": R.astype(np.float32),
             "G": G.astype(np.float32),
             "B": B.astype(np.float32)}
    m_values = [40, 60, 80, 100]
    H, W = L.shape
    P = H * W

    th_p_list, M_p_list, th_s_list, M_s_list, v_list = [], [], [], [], []
    for ch_name, img in chans.items():
        for m in m_values:
            th_p, M_p, th_s, M_s, v = wvf_lf_recover_metal(
                img, radius=9, degree=3, lf_half_length=m,
                n_orientations=64,
                tau_sec_floor=0.40, tau_validity=0.10,
                dense_n=500, min_sep_frac=0.125, method="box")
            th_p_list.append(th_p); M_p_list.append(M_p)
            th_s_list.append(th_s); M_s_list.append(M_s)
            v_list.append(v)

    th_p_arr = np.stack([np.degrees(a).reshape(P) for a in th_p_list], axis=1)
    M_p_arr  = np.stack([a.reshape(P)             for a in M_p_list], axis=1)
    th_s_arr = np.stack([np.degrees(a).reshape(P) for a in th_s_list], axis=1)
    M_s_arr  = np.stack([a.reshape(P)             for a in M_s_list], axis=1)
    v_arr    = np.stack([a.reshape(P)             for a in v_list],   axis=1)
    M_p_g = M_p_arr * v_arr
    M_s_g = M_s_arr * v_arr
    phi_p, w_p, _ = theta_M_to_phi_w(th_p_arr, M_p_g)
    phi_s, w_s, _ = theta_M_to_phi_w(th_s_arr, M_s_g)
    print(f"[full]  P={P:,}  N={phi_p.shape[1]}")

    pred = cgmm_fuse_two_pass_metal(
        phi_p, w_p, phi_s, w_s,
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)

    rng2 = np.random.default_rng(7)
    idx = rng2.choice(P, size=500_000, replace=False)
    ref = cgmm_fuse_two_pass(
        phi_p[idx], w_p[idx], phi_s[idx], w_s[idx],
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)

    pred_subset = {k: pred[k][idx] for k in (
        "theta_primary", "M_primary", "theta_sec", "M_sec", "v_fused")}
    _check_outputs(ref, pred_subset, "full(500K subset)")
    print("  PASS")


def test_speed_small():
    if not METAL_OK:
        raise RuntimeError("Metal c-GMM backend unavailable")
    ins = _load_inputs()
    n = ins["phi_p"].shape[0]

    _ = cgmm_fuse_two_pass_metal(
        ins["phi_p"], ins["w_p"], ins["phi_s"], ins["w_s"],
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)

    t0 = time.perf_counter()
    _ = cgmm_fuse_two_pass_metal(
        ins["phi_p"], ins["w_p"], ins["phi_s"], ins["w_s"],
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)
    t_metal = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = cgmm_fuse_two_pass(
        ins["phi_p"], ins["w_p"], ins["phi_s"], ins["w_s"],
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)
    t_py = time.perf_counter() - t0

    print(f"[speed_small] P={n:,}  N={ins['phi_p'].shape[1]}")
    print(f"  Python reference: {t_py*1000:>8.0f} ms")
    print(f"  Metal           : {t_metal*1000:>8.0f} ms  "
          f"(speedup x{t_py/max(t_metal,1e-9):.1f})")
    if t_metal * 20 > t_py:
        raise AssertionError(
            f"speedup x{t_py/t_metal:.1f} below required x20")
    print("  PASS")


def test_speed_full():
    """Full image (16.7M rows, N=16).  Hard target 2 s, stretch 500 ms."""
    if not METAL_OK:
        raise RuntimeError("Metal c-GMM backend unavailable")
    from PIL import Image
    from edgecritic.pipeline import wvf_lf_recover_metal

    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/4096"
                / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png")
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(rgb + rng.normal(0.0, 13.0, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    R, G, B = rgb_n[..., 0], rgb_n[..., 1], rgb_n[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    chans = {"L": L.astype(np.float32),
             "R": R.astype(np.float32),
             "G": G.astype(np.float32),
             "B": B.astype(np.float32)}
    m_values = [40, 60, 80, 100]
    H, W = L.shape
    P = H * W

    th_p_list, M_p_list, th_s_list, M_s_list, v_list = [], [], [], [], []
    for ch_name, img in chans.items():
        for m in m_values:
            th_p, M_p, th_s, M_s, v = wvf_lf_recover_metal(
                img, radius=9, degree=3, lf_half_length=m,
                n_orientations=64,
                tau_sec_floor=0.40, tau_validity=0.10,
                dense_n=500, min_sep_frac=0.125, method="box")
            th_p_list.append(th_p); M_p_list.append(M_p)
            th_s_list.append(th_s); M_s_list.append(M_s)
            v_list.append(v)

    th_p_arr = np.stack([np.degrees(a).reshape(P) for a in th_p_list], axis=1)
    M_p_arr  = np.stack([a.reshape(P)             for a in M_p_list], axis=1)
    th_s_arr = np.stack([np.degrees(a).reshape(P) for a in th_s_list], axis=1)
    M_s_arr  = np.stack([a.reshape(P)             for a in M_s_list], axis=1)
    v_arr    = np.stack([a.reshape(P)             for a in v_list],   axis=1)
    M_p_g = M_p_arr * v_arr
    M_s_g = M_s_arr * v_arr
    phi_p, w_p, _ = theta_M_to_phi_w(th_p_arr, M_p_g)
    phi_s, w_s, _ = theta_M_to_phi_w(th_s_arr, M_s_g)

    # warmup
    _ = cgmm_fuse_two_pass_metal(
        phi_p, w_p, phi_s, w_s,
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)

    t0 = time.perf_counter()
    _ = cgmm_fuse_two_pass_metal(
        phi_p, w_p, phi_s, w_s,
        K=3, n_iters=30, init_kappa=4.0, hard_em=True,
        tau_M_rel=0.05, theta_min_deg=10.0)
    elapsed = time.perf_counter() - t0
    print(f"[speed_full] P={P:,}  N={phi_p.shape[1]}  "
          f"wall={elapsed*1000:.0f} ms")
    if elapsed > 2.0:
        raise AssertionError(
            f"full-image c-GMM {elapsed*1000:.0f} ms > 2000 ms hard target")
    if elapsed > 0.5:
        print(f"  PASS (hard) -- stretch (<500 ms) not met")
    else:
        print(f"  PASS (stretch)")


if __name__ == "__main__":
    failed = 0
    for name, fn in [("test_correctness_small", test_correctness_small),
                     ("test_correctness_full",  test_correctness_full),
                     ("test_speed_small",       test_speed_small),
                     ("test_speed_full",        test_speed_full)]:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as ex:
            print(f"  FAIL: {ex}")
            failed += 1
    sys.exit(failed)
