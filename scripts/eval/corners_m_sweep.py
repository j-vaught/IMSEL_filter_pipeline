"""Sweep LF half-length m at each inner-star corner. Single channel,
single (r, d). Uses the Metal GPU backend if available."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from edgecritic.wvf._radius_kernels import (
    wvf_radius_gradients_cpu, build_wvf_radius_kernels)

try:
    from edgecritic.wvf._metal import (
        wvf_radius_gradients_metal, metal_backend_available)
    _METAL_OK = metal_backend_available()
except Exception:
    _METAL_OK = False
    wvf_radius_gradients_metal = None


def compute_wvf(img, r, d):
    if _METAL_OK:
        kernels = build_wvf_radius_kernels(radius=r, order=d)
        return wvf_radius_gradients_metal(img, kernels)
    return wvf_radius_gradients_cpu(img, radius=r, order=d)


def find_n_peaks(response, n_peaks=2, min_separation_frac=0.125):
    n = len(response)
    distance = max(1, int(min_separation_frac * n))
    peaks, _ = find_peaks(response, distance=distance)
    if len(peaks) < n_peaks:
        order = np.argsort(-response)
        return np.sort(order[:n_peaks])
    heights = response[peaks]
    top = peaks[np.argsort(-heights)[:n_peaks]]
    return np.sort(top)


def spline_peaks(angles_rad, response, n_peaks=2, k_dense=10000):
    x = np.concatenate([angles_rad, [np.pi]])
    y = np.concatenate([response, [response[0]]])
    cs = CubicSpline(x, y, bc_type="periodic")
    dense_a = np.linspace(0, np.pi, k_dense, endpoint=False)
    dense_r = cs(dense_a)
    idx = find_n_peaks(dense_r, n_peaks)
    return np.degrees(dense_a[idx]), dense_r[idx]


def lf_at_pixel(g_x, g_y, px, py, theta_rad, m):
    H, W = g_x.shape
    cos_t, sin_t = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    if m <= 0:
        return abs(-sin_t * g_x[py, px] + cos_t * g_y[py, px])
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0
    sigma = m / 2.0
    j_offsets = np.arange(-m, m + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)
    acc, wsum = 0.0, 0.0
    for w_j, j in zip(weights, j_offsets):
        ix = int(round(j * step * cos_t))
        iy = int(round(j * step * sin_t))
        yy, xx = py + iy, px + ix
        if 0 <= yy < H and 0 <= xx < W:
            g_perp = -sin_t * g_x[yy, xx] + cos_t * g_y[yy, xx]
            acc += w_j * g_perp
            wsum += w_j
    return abs(acc / wsum) if wsum > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--vertices", required=True, type=str,
                        help="Semicolon-separated 'x,y' pairs.")
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--m-values", required=True, type=str,
                        help="Comma-separated m values to sweep.")
    parser.add_argument("--n-orientations", type=int, default=64)
    parser.add_argument("--n-peaks", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    print(f"WVF backend: {'Metal (GPU)' if _METAL_OK else 'CPU'}")

    vertices = []
    for v in args.vertices.split(";"):
        x, y = (int(s.strip()) for s in v.split(","))
        vertices.append((x, y))

    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]
    angles = np.linspace(0, np.pi, args.n_orientations, endpoint=False)

    img = np.asarray(Image.open(args.image).convert("L"),
                     dtype=np.float64)
    h, w = img.shape
    print(f"Loaded {args.image.name}: {h} x {w}")

    t0 = time.perf_counter()
    g_x, g_y = compute_wvf(
        img.astype(np.float32) if _METAL_OK else img,
        args.r, args.d)
    print(f"  WVF r={args.r} d={args.d}  "
          f"({time.perf_counter() - t0:.2f}s)")

    out = {
        "image": str(args.image),
        "config": {"r": args.r, "d": args.d,
                   "n_orientations": args.n_orientations},
        "m_values": m_values,
        "angles_deg": [float(v) for v in np.degrees(angles)],
        "vertices": [],
    }

    for vi, (px, py) in enumerate(vertices):
        v_entry = {"id": vi, "pixel_xy": [px, py], "by_m": {}}
        for m in m_values:
            t0 = time.perf_counter()
            resp = np.array([lf_at_pixel(g_x, g_y, px, py, t, m)
                             for t in angles])
            peaks_a, peaks_m = spline_peaks(angles, resp,
                                            n_peaks=args.n_peaks)
            order = np.argsort(-peaks_m)
            peaks_a = peaks_a[order]
            peaks_m = peaks_m[order]
            v_entry["by_m"][str(m)] = {
                "theta_hat": float(peaks_a[0]),
                "M_hat": float(peaks_m[0]),
                "theta_sec": (float(peaks_a[1]) if len(peaks_a) > 1
                              else float("nan")),
                "M_sec": (float(peaks_m[1]) if len(peaks_m) > 1
                          else 0.0),
                "response": [float(v) for v in resp],
            }
            print(f"  v{vi} m={m:>3}  "
                  f"theta={peaks_a[0]:6.2f}/{peaks_a[1] if len(peaks_a)>1 else float('nan'):>6.2f}  "
                  f"M={peaks_m[0]:.3f}/{peaks_m[1] if len(peaks_m)>1 else 0:.3f}  "
                  f"({time.perf_counter() - t0:.2f}s)")
        out["vertices"].append(v_entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
