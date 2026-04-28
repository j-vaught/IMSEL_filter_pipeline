"""Independent verification of the fused WVF+LF+recovery Metal pipeline.

Does NOT use the implementer's test scaffold.  Instead:
  1. Calls wvf_lf_recover_metal directly.
  2. Recomputes the same outputs by running the three component kernels
     separately and reshaping by hand.
  3. Compares element-wise.
  4. Times both paths after warmup.

Reports mismatches and timings.  No assertions -- prints raw numbers
so we can read the verdict.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.lf._metal import lf_stack
from edgecritic.recovery._metal import recover_two_peaks_metal
from edgecritic.pipeline import wvf_lf_recover_metal, pipeline_backend_available


def unfused(L, r, d, m, n_orient, angles):
    """Reproduce the fused output by composing the three component kernels."""
    kernels = build_wvf_radius_kernels(radius=r, order=d)
    gx, gy = wvf_radius_gradients_metal(L, kernels, output_dtype=np.float32)
    stack = lf_stack(gx, gy,
                     lf_half_length=m,
                     n_orientations=n_orient,
                     output_dtype=np.float32,
                     method="box")
    H, W = L.shape
    resp = stack.transpose(1, 2, 0).reshape(H * W, n_orient).copy()
    th_p, M_p, th_s, M_s, v = recover_two_peaks_metal(
        angles, resp,
        tau_sec_floor=0.40, tau_validity=0.10,
        dense_n=500, min_sep_frac=0.125)
    return (th_p.reshape(H, W),
            M_p.reshape(H, W),
            th_s.reshape(H, W),
            M_s.reshape(H, W),
            v.reshape(H, W))


def compare(name, a, b, tol=1e-5):
    if a.dtype == np.uint8:
        diff = int((a != b).sum())
        print(f"  {name:<18}: dtype={a.dtype}  shape={a.shape}  "
              f"disagreements={diff}/{a.size}")
        return diff == 0
    # treat NaN as equal where both NaN
    nan_a = np.isnan(a); nan_b = np.isnan(b)
    nan_eq = (nan_a & nan_b).sum()
    nan_diff = int((nan_a ^ nan_b).sum())
    finite = (~nan_a) & (~nan_b)
    if finite.any():
        d = np.abs(a[finite] - b[finite])
        max_d = float(d.max())
        bad = int((d > tol).sum())
    else:
        max_d = 0.0; bad = 0
    print(f"  {name:<18}: max_abs_diff={max_d:.3e}  >tol({tol:.0e})={bad}/"
          f"{int(finite.sum())}  nan_pattern_diff={nan_diff}  "
          f"both_nan={int(nan_eq)}")
    return (max_d <= tol) and (nan_diff == 0)


def main():
    if not pipeline_backend_available():
        print("FATAL: pipeline backend not available")
        sys.exit(1)

    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/4096"
                / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png")
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(rgb + rng.normal(0.0, 13.0, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    L = (0.2126 * rgb_n[..., 0] + 0.7152 * rgb_n[..., 1]
         + 0.0722 * rgb_n[..., 2]).astype(np.float32)

    print(f"image: {img_path.name}  L shape={L.shape}")
    n_orient = 64
    angles = np.linspace(0.0, math.pi, n_orient, endpoint=False)

    # ===== correctness on (r=9, d=3, m=60) =====
    r, d, m = 9, 3, 60
    print(f"\n--- correctness check at (r={r}, d={d}, m={m}, "
          f"n_orient={n_orient}) ---")
    print("running fused path...")
    th_p_f, M_p_f, th_s_f, M_s_f, v_f = wvf_lf_recover_metal(
        L, radius=r, degree=d, lf_half_length=m,
        n_orientations=n_orient,
        tau_sec_floor=0.40, tau_validity=0.10,
        dense_n=500, min_sep_frac=0.125, method="box")
    print(f"  fused outputs: theta_p {th_p_f.shape} {th_p_f.dtype}, "
          f"v {v_f.shape} {v_f.dtype}")

    print("running unfused composition...")
    th_p_u, M_p_u, th_s_u, M_s_u, v_u = unfused(L, r, d, m, n_orient, angles)
    print(f"  unfused outputs match shapes")

    ok = True
    print("\nelement-wise comparison:")
    ok &= compare("theta_primary",   th_p_f, th_p_u)
    ok &= compare("M_primary",       M_p_f,  M_p_u)
    ok &= compare("theta_secondary", th_s_f, th_s_u)
    ok &= compare("M_secondary",     M_s_f,  M_s_u)
    ok &= compare("v",               v_f,    v_u)
    print(f"\noverall correctness: {'PASS' if ok else 'FAIL'}")

    # ===== speed: 3 warm + 3 timed runs =====
    print(f"\n--- speed check (warm 3, timed 3) at (r={r}, d={d}, m={m}) ---")
    for _ in range(3):
        _ = wvf_lf_recover_metal(L, radius=r, degree=d, lf_half_length=m,
                                  n_orientations=n_orient, method="box")

    times_fused = []
    for i in range(3):
        t0 = time.perf_counter()
        _ = wvf_lf_recover_metal(L, radius=r, degree=d, lf_half_length=m,
                                  n_orientations=n_orient, method="box")
        times_fused.append(time.perf_counter() - t0)

    print(f"fused timings    : {[f'{t*1000:.0f}' for t in times_fused]} ms")
    print(f"  median fused   : {np.median(times_fused)*1000:.0f} ms")

    # one unfused run timed
    t0 = time.perf_counter()
    _ = unfused(L, r, d, m, n_orient, angles)
    t_un = time.perf_counter() - t0
    print(f"unfused (1 timed): {t_un*1000:.0f} ms")
    print(f"speedup          : x{t_un/np.median(times_fused):.2f}")

    # ===== sweep correctness =====
    print(f"\n--- sweep correctness on 16 configs ---")
    n_bad = 0
    for r_ in (5, 9):
        for d_ in (1, 3):
            for m_ in (40, 60, 80, 100):
                f = wvf_lf_recover_metal(L, radius=r_, degree=d_,
                                          lf_half_length=m_,
                                          n_orientations=n_orient,
                                          method="box")
                u = unfused(L, r_, d_, m_, n_orient, angles)
                bad = 0
                for i, name in enumerate(("th_p", "M_p", "th_s", "M_s", "v")):
                    a, b = f[i], u[i]
                    if a.dtype == np.uint8:
                        if (a != b).any():
                            bad += int((a != b).sum())
                    else:
                        nan_a = np.isnan(a); nan_b = np.isnan(b)
                        if (nan_a ^ nan_b).any():
                            bad += int((nan_a ^ nan_b).sum())
                        finite = (~nan_a) & (~nan_b)
                        if finite.any():
                            md = float(np.abs(a[finite] - b[finite]).max())
                            if md > 1e-5:
                                bad += int((np.abs(a[finite] - b[finite]) > 1e-5).sum())
                tag = "OK " if bad == 0 else f"BAD ({bad})"
                print(f"  r={r_} d={d_} m={m_:>3}  {tag}")
                n_bad += int(bad > 0)

    print(f"\nsweep total bad configs: {n_bad}/16")
    print(f"\nVERDICT: "
          f"{'ALL CHECKS PASSED' if (ok and n_bad == 0) else 'FAILURES DETECTED'}")


if __name__ == "__main__":
    main()
