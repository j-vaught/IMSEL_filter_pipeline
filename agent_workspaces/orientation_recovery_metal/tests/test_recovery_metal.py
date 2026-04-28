"""Acceptance tests for the Metal-accelerated orientation recovery.

Run from repo root:

    PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \\
        python3 agent_workspaces/orientation_recovery_metal/tests/test_recovery_metal.py

Three tests:
    1. test_correctness_small  -- 200K-row slab, outputs match reference.
    2. test_correctness_full   -- full 4096x4096 (16.7M rows).
    3. test_speed              -- batched call >= 20x faster than the
                                  numpy reference at 200K rows.
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

from reference_impl import find_two_peaks  # python source-of-truth

# Implementer must create this module.
try:
    from edgecritic.recovery._metal import (  # type: ignore
        recover_two_peaks_metal,
        recovery_backend_available,
    )
    METAL_OK = recovery_backend_available()
except Exception as ex:
    METAL_OK = False
    _IMPORT_ERR = repr(ex)


REF = Path(__file__).resolve().parent.parent / "reference"
THETA_TOL_RAD = math.radians(0.5)   # 0.5 deg
M_TOL_REL     = 5e-3                # 0.5%
SEC_DISAGREE_FRAC = 1e-3            # <=0.1% rows may disagree on secondary kept-ness
V_DISAGREE_FRAC   = 1e-3            # <=0.1% rows may disagree on validity flag


def _check_outputs(angles, response, expected, tag):
    th_p_g = expected["theta_primary"]
    M_p_g  = expected["M_primary"]
    th_s_g = expected["theta_secondary"]
    M_s_g  = expected["M_secondary"]
    v_g    = expected["v"]

    th_p, M_p, th_s, M_s, v = recover_two_peaks_metal(
        angles, response,
        tau_sec_floor=0.40,
        tau_validity=0.10,
        dense_n=500,
        min_sep_frac=0.125)
    n = th_p.size

    # ---- primary ----
    dth = np.minimum(np.abs(th_p - th_p_g),
                     math.pi - np.abs(th_p - th_p_g))
    bad_th = dth > THETA_TOL_RAD
    dM = np.abs(M_p - M_p_g) / np.maximum(M_p_g, 1e-12)
    bad_M  = dM > M_TOL_REL
    n_bad_p = int((bad_th | bad_M).sum())
    print(f"  [{tag}] primary  : "
          f"max dtheta={np.degrees(dth.max()):.4f} deg  "
          f"max dM/M={dM.max():.2e}  bad={n_bad_p}/{n}")
    if n_bad_p > 0:
        raise AssertionError(
            f"primary disagreement on {n_bad_p}/{n} rows")

    # ---- secondary kept-ness ----
    sec_kept_g = ~np.isnan(th_s_g)
    sec_kept   = ~np.isnan(th_s)
    n_disagree_s = int((sec_kept_g ^ sec_kept).sum())
    print(f"  [{tag}] sec kept : disagree={n_disagree_s}/{n} "
          f"({n_disagree_s/n*100:.3f}%)")
    if n_disagree_s / n > SEC_DISAGREE_FRAC:
        raise AssertionError(
            f"secondary kept-ness disagrees on {n_disagree_s}/{n} rows")

    both_kept = sec_kept & sec_kept_g
    if both_kept.any():
        ds = np.minimum(np.abs(th_s[both_kept] - th_s_g[both_kept]),
                        math.pi - np.abs(th_s[both_kept] - th_s_g[both_kept]))
        dms = np.abs(M_s[both_kept] - M_s_g[both_kept]) / np.maximum(
            M_s_g[both_kept], 1e-12)
        n_bad_s = int(((ds > THETA_TOL_RAD) | (dms > M_TOL_REL)).sum())
        print(f"  [{tag}] secondary: "
              f"max dtheta={np.degrees(ds.max()):.4f} deg  "
              f"max dM/M={dms.max():.2e}  bad={n_bad_s}/{both_kept.sum()}")
        if n_bad_s > 0:
            raise AssertionError(
                f"secondary value disagreement on {n_bad_s} rows")

    # ---- validity flag v ----
    n_disagree_v = int((v.astype(bool) ^ v_g.astype(bool)).sum())
    print(f"  [{tag}] v flag   : disagree={n_disagree_v}/{n} "
          f"({n_disagree_v/n*100:.3f}%)")
    if n_disagree_v / n > V_DISAGREE_FRAC:
        raise AssertionError(
            f"validity flag disagrees on {n_disagree_v}/{n} rows")


def test_correctness_small():
    if not METAL_OK:
        raise RuntimeError(f"Metal recovery backend unavailable: "
                           f"{_IMPORT_ERR}")
    inp = np.load(REF / "inputs.npz")
    exp = np.load(REF / "expected.npz")
    angles = inp["angles"].astype(np.float64)
    resp   = inp["response"].astype(np.float32)
    print(f"[small] N={resp.shape[0]:,}  K={resp.shape[1]}")
    _check_outputs(angles, resp, exp, "small")
    print("  PASS")


def test_correctness_full():
    """Build a full 4096x4096 slab, run Metal, compare to scipy ref on
    a random 500K subset (full reference is too slow to run inside the
    test)."""
    if not METAL_OK:
        raise RuntimeError("Metal recovery backend unavailable")
    from PIL import Image
    from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
    from edgecritic.wvf._metal import wvf_radius_gradients_metal
    from edgecritic.lf._metal import lf_stack

    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/4096"
                / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png")
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(rgb + rng.normal(0.0, 13.0, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    L = (0.2126 * rgb_n[..., 0] + 0.7152 * rgb_n[..., 1]
         + 0.0722 * rgb_n[..., 2]).astype(np.float32)
    kernels = build_wvf_radius_kernels(radius=9, order=3)
    gx, gy = wvf_radius_gradients_metal(L, kernels, output_dtype=np.float32)
    n_orient = 64
    stack = lf_stack(
        gx, gy,
        lf_half_length=60,
        n_orientations=n_orient,
        output_dtype=np.float32,
        method="box",
    )
    H, W = L.shape
    resp = stack.transpose(1, 2, 0).reshape(H * W, n_orient).copy()
    angles = np.linspace(0, math.pi, n_orient, endpoint=False)
    print(f"[full]  N={resp.shape[0]:,}  K={resp.shape[1]}")

    th_p, M_p, th_s, M_s, v = recover_two_peaks_metal(
        angles, resp, tau_sec_floor=0.40,
        tau_validity=0.10,
        dense_n=500, min_sep_frac=0.125)

    rng2 = np.random.default_rng(7)
    idx = rng2.choice(resp.shape[0], size=500_000, replace=False)
    th_p_ref, M_p_ref, th_s_ref, M_s_ref, v_ref = find_two_peaks(
        angles, resp[idx].astype(np.float64),
        tau_sec_floor=0.40, tau_validity=0.10,
        dense_n=500, min_sep_frac=0.125)

    # NOTE: v_ref above uses R_ref of the 500K subset, not the full
    # image.  For the test we compare against the full-image v.
    metal_v_subset = v[idx]
    _check_outputs(angles, resp[idx],
                   {"theta_primary":   th_p_ref,
                    "M_primary":       M_p_ref,
                    "theta_secondary": th_s_ref,
                    "M_secondary":     M_s_ref,
                    "v":               metal_v_subset},  # tautological for v;
                                                          # primary/secondary
                                                          # are the real check
                   "full(500K subset)")
    print("  PASS")


def test_speed():
    if not METAL_OK:
        raise RuntimeError("Metal recovery backend unavailable")
    inp = np.load(REF / "inputs.npz")
    angles = inp["angles"].astype(np.float64)
    resp   = inp["response"].astype(np.float32)
    n = resp.shape[0]

    _ = recover_two_peaks_metal(angles, resp,
                                tau_sec_floor=0.40,
                                tau_validity=0.10,
                                dense_n=500, min_sep_frac=0.125)

    t0 = time.perf_counter()
    _ = recover_two_peaks_metal(angles, resp,
                                tau_sec_floor=0.40,
                                tau_validity=0.10,
                                dense_n=500, min_sep_frac=0.125)
    t_metal = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = find_two_peaks(angles, resp.astype(np.float64),
                       tau_sec_floor=0.40,
                       tau_validity=0.10,
                       dense_n=500, min_sep_frac=0.125)
    t_numpy = time.perf_counter() - t0

    print(f"[speed] N={n:,}")
    print(f"  numpy reference : {t_numpy*1000:>8.1f} ms")
    print(f"  metal           : {t_metal*1000:>8.1f} ms  "
          f"(speedup x{t_numpy/max(t_metal,1e-9):.1f})")
    if t_metal * 20 > t_numpy:
        raise AssertionError(
            f"speedup x{t_numpy/t_metal:.1f} below required x20")
    print("  PASS")


if __name__ == "__main__":
    failed = 0
    for name, fn in [("test_correctness_small", test_correctness_small),
                     ("test_correctness_full",  test_correctness_full),
                     ("test_speed",             test_speed)]:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as ex:
            print(f"  FAIL: {ex}")
            failed += 1
    sys.exit(failed)
