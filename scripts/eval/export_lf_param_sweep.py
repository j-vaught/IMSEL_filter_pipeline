"""Sweep WVF (r, d) and LF (m) on a single channel; export edge PNGs.

For a fixed input image, compute the LF orientation-max magnitude
|max_theta L_theta| at every pixel for a 3-row parameter sweep:

  Row 1: vary polynomial degree d (fixed r, m)
  Row 2: vary WVF radius r          (fixed d, m)
  Row 3: vary LF half-length m      (fixed r, d)

All output PNGs share a single global vmax (clip-percentile of the
union of all magnitude maps) and are saved with the gray_r convention
(high response = dark) so they are display-ready for a CeTZ figure.

Usage::

    PYTHONPATH=src python scripts/eval/export_lf_param_sweep.py \\
        --image cetz_figures/data/color_channels/1024/channel_L.png \\
        --out-dir cetz_figures/data/lf_param_sweep/L_1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from edgecritic.wvf._radius_kernels import wvf_radius_gradients_cpu


def lf_max_magnitude(g_x: np.ndarray, g_y: np.ndarray,
                     half_width: int,
                     n_orientations: int = 16) -> np.ndarray:
    """Return max over theta of |L_theta|(p) using the steered-WVF
    formulation. half_width = 0 returns |grad| from the WVF directly."""
    H, W = g_x.shape
    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)
    if half_width <= 0:
        return np.sqrt(g_x ** 2 + g_y ** 2).astype(np.float32)

    sigma = half_width / 2.0
    j_offsets = np.arange(-half_width, half_width + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)

    max_resp = np.zeros((H, W), dtype=np.float32)
    for theta in angles:
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        max_trig = max(abs(cos_t), abs(sin_t))
        step = 1.0 / max_trig if max_trig > 0 else 1.0
        g_perp = -sin_t * g_x + cos_t * g_y

        response = np.zeros((H, W), dtype=np.float64)
        weight_sum = np.zeros((H, W), dtype=np.float64)
        for w_j, j in zip(weights, j_offsets):
            ix = int(round(j * step * cos_t))
            iy = int(round(j * step * sin_t))
            y_src_lo = max(0, iy)
            y_src_hi = min(H, H + iy)
            x_src_lo = max(0, ix)
            x_src_hi = min(W, W + ix)
            y_dst_lo = max(0, -iy)
            y_dst_hi = y_dst_lo + (y_src_hi - y_src_lo)
            x_dst_lo = max(0, -ix)
            x_dst_hi = x_dst_lo + (x_src_hi - x_src_lo)
            if y_src_hi <= y_src_lo or x_src_hi <= x_src_lo:
                continue
            response[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] += (
                w_j * g_perp[y_src_lo:y_src_hi, x_src_lo:x_src_hi])
            weight_sum[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] += w_j

        valid = weight_sum > 0
        response[valid] = response[valid] / weight_sum[valid]
        np.maximum(max_resp, np.abs(response).astype(np.float32),
                   out=max_resp)
    return max_resp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--r-values", type=str, default="3,5,7,9",
                        help="Comma-separated WVF radii to sweep.")
    parser.add_argument("--d-values", type=str, default="2,3,4,5",
                        help="Comma-separated WVF polynomial degrees.")
    parser.add_argument("--m-values", type=str, default="3,5,7,11",
                        help="Comma-separated LF half-lengths.")
    parser.add_argument("--fixed-r", type=int, default=7)
    parser.add_argument("--fixed-d", type=int, default=4)
    parser.add_argument("--fixed-m", type=int, default=7)
    parser.add_argument("--n-orientations", type=int, default=16)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--crop-size", type=int, default=0,
                        help="Optional centered square crop of this side.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(args.image).convert("L")
    image = np.asarray(pil, dtype=np.float64)
    h, w = image.shape
    print(f"Loaded {args.image.name}: {h} x {w}")

    r_vals = [int(v) for v in args.r_values.split(",") if v.strip()]
    d_vals = [int(v) for v in args.d_values.split(",") if v.strip()]
    m_vals = [int(v) for v in args.m_values.split(",") if v.strip()]

    sweeps = [
        ("d", [(args.fixed_r, d, args.fixed_m) for d in d_vals]),
        ("r", [(r, args.fixed_d, args.fixed_m) for r in r_vals]),
        ("m", [(args.fixed_r, args.fixed_d, m) for m in m_vals]),
    ]

    grad_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    responses: dict[tuple[int, int, int], np.ndarray] = {}

    for axis, params in sweeps:
        for (r, d, m) in params:
            if (r, d, m) in responses:
                continue
            if (r, d) not in grad_cache:
                t0 = time.perf_counter()
                grad_cache[(r, d)] = wvf_radius_gradients_cpu(
                    image, radius=r, order=d)
                print(f"  WVF r={r} d={d}  "
                      f"({time.perf_counter() - t0:.1f}s)")
            g_x, g_y = grad_cache[(r, d)]
            t0 = time.perf_counter()
            mag = lf_max_magnitude(g_x, g_y, m,
                                   n_orientations=args.n_orientations)
            print(f"  LF  r={r} d={d} m={m}  "
                  f"({time.perf_counter() - t0:.1f}s)")
            responses[(r, d, m)] = mag

    # Optional centered crop applied uniformly to all maps.
    if args.crop_size > 0:
        c = args.crop_size
        cy, cx = h // 2, w // 2
        y0 = max(0, cy - c // 2)
        x0 = max(0, cx - c // 2)
        y1 = min(h, y0 + c)
        x1 = min(w, x0 + c)
        for k in list(responses.keys()):
            responses[k] = responses[k][y0:y1, x0:x1]
        print(f"Cropped to {y1 - y0} x {x1 - x0} centered on the image.")

    # Global vmax for cross-panel comparability.
    vmax = max(float(np.percentile(mag, args.clip_percentile))
               for mag in responses.values())
    print(f"Global vmax (p{args.clip_percentile}): {vmax:.3f}")

    # Save gray_r PNGs (high response = dark).
    for (r, d, m), mag in responses.items():
        if vmax > 0:
            u8 = 255 - (np.clip(mag / vmax, 0.0, 1.0) * 255.0
                        ).astype(np.uint8)
        else:
            u8 = np.full(mag.shape, 255, dtype=np.uint8)
        fname = f"edge_r{r}_d{d}_m{m}.png"
        Image.fromarray(u8, mode="L").save(args.out_dir / fname)

    manifest = {
        "source_image": str(args.image),
        "image_shape": [int(h), int(w)],
        "sweeps": [
            {"axis": axis, "params": list(map(list, params))}
            for axis, params in sweeps
        ],
        "fixed": {"r": args.fixed_r,
                  "d": args.fixed_d,
                  "m": args.fixed_m},
        "n_orientations": args.n_orientations,
        "vmax": vmax,
        "crop_size": int(args.crop_size),
        "convention": "gray_r (high response = dark)",
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exported {len(responses)} edge maps to {args.out_dir}")


if __name__ == "__main__":
    main()
