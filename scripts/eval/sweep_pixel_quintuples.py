"""Sweep (r, d, m) at a single pixel and emit per-config quintuples.

For every (r, d, m) on the requested grid, compute the LF response
curve over `n_orientations` angles AT THE GIVEN PIXEL, fit a periodic
cubic spline, locate the top peaks, and record the orientation and
magnitude at each. The output is a JSON with one row per (r, d, m):
``{r, d, m, theta_hat, M_hat, theta_sec, M_sec, R}``.

Usage::

    PYTHONPATH=src python scripts/eval/sweep_pixel_quintuples.py \\
        --image cetz_figures/data/color_channels/4096_lowcon/channel_L.png \\
        --pixel 2123,2048 \\
        --out   cetz_figures/data/lf_quintuple_L_4096/corner/sweep_quintuples.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from edgecritic.wvf._radius_kernels import wvf_radius_gradients_cpu


def _find_n_peaks(response: np.ndarray, n_peaks: int,
                  min_separation_frac: float = 0.125) -> np.ndarray:
    n = len(response)
    distance = max(1, int(min_separation_frac * n))
    peaks, _ = find_peaks(response, distance=distance)
    if len(peaks) < n_peaks:
        order = np.argsort(-response)
        return np.sort(order[:n_peaks])
    heights = response[peaks]
    top = peaks[np.argsort(-heights)[:n_peaks]]
    return np.sort(top)


def spline_peaks(angles_rad: np.ndarray, response: np.ndarray,
                 n_peaks: int = 2,
                 k_dense: int = 10000) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.concatenate([angles_rad, [np.pi]])
    y = np.concatenate([response, [response[0]]])
    cs = CubicSpline(x, y, bc_type="periodic")
    dense_a = np.linspace(0, np.pi, k_dense, endpoint=False)
    dense_r = cs(dense_a)
    idx = _find_n_peaks(dense_r, n_peaks)
    return (np.degrees(dense_a[idx]),
            dense_r[idx],
            float(dense_r.max() - dense_r.min()))


def lf_pixel_response(g_x: np.ndarray, g_y: np.ndarray,
                      px: int, py: int,
                      theta: float, half_width: int) -> float:
    H, W = g_x.shape
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    if half_width <= 0:
        g_perp = -sin_t * g_x[py, px] + cos_t * g_y[py, px]
        return abs(float(g_perp))
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0
    sigma = half_width / 2.0
    j_offsets = np.arange(-half_width, half_width + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)
    acc = 0.0
    wsum = 0.0
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
    parser.add_argument("--pixel", required=True, type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--r-values", default="3,5,7,9")
    parser.add_argument("--d-values", default="1,3,5")
    parser.add_argument("--m-values", default="0,2,4,6")
    parser.add_argument("--n-orientations", type=int, default=64)
    parser.add_argument("--n-peaks", type=int, default=2)
    args = parser.parse_args()

    px, py = (int(v.strip()) for v in args.pixel.split(","))
    r_vals = [int(v) for v in args.r_values.split(",") if v.strip()]
    d_vals = [int(v) for v in args.d_values.split(",") if v.strip()]
    m_vals = [int(v) for v in args.m_values.split(",") if v.strip()]

    pil = Image.open(args.image).convert("L")
    image = np.asarray(pil, dtype=np.float64)
    h, w = image.shape
    print(f"Loaded {args.image.name}: {h} x {w}, pixel=({px}, {py})")

    angles = np.linspace(0, np.pi, args.n_orientations, endpoint=False)

    rows = []
    grad_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for r in r_vals:
        for d in d_vals:
            if (r, d) not in grad_cache:
                grad_cache[(r, d)] = wvf_radius_gradients_cpu(
                    image, radius=r, order=d)
            g_x, g_y = grad_cache[(r, d)]
            for m in m_vals:
                resp = np.array([
                    lf_pixel_response(g_x, g_y, px, py, t, m)
                    for t in angles])
                peak_a, peak_m, R = spline_peaks(
                    angles, resp, n_peaks=args.n_peaks)
                # Sort peaks by descending magnitude.
                order = np.argsort(-peak_m)
                peak_a = peak_a[order]
                peak_m = peak_m[order]
                rows.append({
                    "r": int(r), "d": int(d), "m": int(m),
                    "theta_hat": float(peak_a[0]),
                    "M_hat": float(peak_m[0]),
                    "theta_sec": float(peak_a[1])
                                  if len(peak_a) > 1 else float("nan"),
                    "M_sec": float(peak_m[1])
                              if len(peak_m) > 1 else 0.0,
                    "R": R,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "image": str(args.image),
            "pixel_xy": [px, py],
            "n_orientations": args.n_orientations,
            "rows": rows,
        }, f, indent=2)
    print(f"Wrote {len(rows)} quintuples to {args.out}")


if __name__ == "__main__":
    main()
