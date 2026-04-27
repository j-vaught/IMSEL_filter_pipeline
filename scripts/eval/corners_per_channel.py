"""For each corner pixel and each input channel, compute the LF
orientation peaks at a fixed best-config (r, d, m). Emits a single
JSON for all (vertex, channel) pairs.

Usage::

    PYTHONPATH=src python scripts/eval/corners_per_channel.py \\
        --channels-dir cetz_figures/data/color_channels/1024 \\
        --vertices "539,428;601,512;539,596;441,564;441,460" \\
        --r 3 --d 5 --m 2 \\
        --out cetz_figures/data/corners_per_channel.json
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


def lf_at_pixel(g_x, g_y, px, py, theta, m):
    H, W = g_x.shape
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels-dir", required=True, type=Path)
    parser.add_argument("--vertices", required=True, type=str,
                        help="Semicolon-separated 'x,y' pairs.")
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n-orientations", type=int, default=64)
    parser.add_argument("--n-peaks", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    vertices = []
    for v in args.vertices.split(";"):
        x, y = (int(s.strip()) for s in v.split(","))
        vertices.append((x, y))

    angles = np.linspace(0, np.pi, args.n_orientations, endpoint=False)
    out = {
        "config": {"r": args.r, "d": args.d, "m": args.m,
                   "n_orientations": args.n_orientations},
        "angles_deg": [float(v) for v in np.degrees(angles)],
        "vertices": [],
    }

    for ch in ("L", "R", "G", "B"):
        path = args.channels_dir / f"channel_{ch}.png"
        img = np.asarray(Image.open(path).convert("L"),
                         dtype=np.float64)
        g_x, g_y = wvf_radius_gradients_cpu(img, radius=args.r,
                                            order=args.d)
        for vi, (px, py) in enumerate(vertices):
            resp = np.array([lf_at_pixel(g_x, g_y, px, py, t, args.m)
                             for t in angles])
            peaks_a, peaks_m = spline_peaks(angles, resp,
                                            n_peaks=args.n_peaks)
            order = np.argsort(-peaks_m)
            peaks_a = peaks_a[order]
            peaks_m = peaks_m[order]

            if len(out["vertices"]) <= vi:
                out["vertices"].append({
                    "id": vi,
                    "pixel_xy": [px, py],
                    "channels": {},
                })
            out["vertices"][vi]["channels"][ch] = {
                "theta_hat": float(peaks_a[0]),
                "M_hat": float(peaks_m[0]),
                "theta_sec": (float(peaks_a[1]) if len(peaks_a) > 1
                              else float("nan")),
                "M_sec": (float(peaks_m[1]) if len(peaks_m) > 1
                          else 0.0),
                "response": [float(v) for v in resp],
            }
            print(f"  ch={ch} v{vi} ({px},{py}): "
                  f"theta_hat={peaks_a[0]:6.2f}  "
                  f"M={peaks_m[0]:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
