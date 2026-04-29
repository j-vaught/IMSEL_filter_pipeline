"""A1 reviewer-pass-5 ablation: LF index-space vs Euclidean weights.

The experiment builds an analytical step-edge bank with one image per
ground-truth tangent orientation, computes WVF gradients with the Metal
backend, evaluates LF response curves at true-edge pixels under the two
weighting rules, and recovers the primary orientation with the reference
periodic cubic spline estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent_workspaces" / "orientation_recovery_metal"))

from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from reference_impl import find_two_peaks


def _round_half_even(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(np.int32)


def _angular_error_deg(theta: np.ndarray, theta_gt: float) -> np.ndarray:
    delta = np.abs(theta - theta_gt)
    delta = np.minimum(delta, math.pi - delta)
    return np.degrees(delta)


def _make_step_edge(size: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    normal_x = -math.sin(theta)
    normal_y = math.cos(theta)
    signed = (xx - cx) * normal_x + (yy - cy) * normal_y
    image = 128.0 + 70.0 * np.tanh(signed / 1.25)
    return image.astype(np.float32), signed


def _edge_pixels(
    signed: np.ndarray,
    border: int,
    max_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = signed.shape
    mask = np.abs(signed) <= 0.6
    mask[:border, :] = False
    mask[h - border :, :] = False
    mask[:, :border] = False
    mask[:, w - border :] = False
    ys, xs = np.nonzero(mask)
    if xs.size > max_pixels:
        take = rng.choice(xs.size, size=max_pixels, replace=False)
        xs = xs[take]
        ys = ys[take]
    return xs.astype(np.int32), ys.astype(np.int32)


def _lf_response_rows(
    gx: np.ndarray,
    gy: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    angles: np.ndarray,
    m: int,
    sigma: float,
    scheme: str,
) -> np.ndarray:
    rows = np.empty((xs.size, angles.size), dtype=np.float64)
    h, w = gx.shape
    del h, w
    for ai, theta in enumerate(angles):
        cos_t = math.cos(float(theta))
        sin_t = math.sin(float(theta))
        delta = 1.0 / max(abs(cos_t), abs(sin_t), 1.0e-12)
        half = m if scheme == "index" else int(math.floor(m / delta))
        js = np.arange(-half, half + 1, dtype=np.float64)
        offsets = js * delta
        dx = _round_half_even(offsets * cos_t)
        dy = _round_half_even(offsets * sin_t)
        sample_x = xs[:, None] + dx[None, :]
        sample_y = ys[:, None] + dy[None, :]
        samples = (
            -sin_t * gx[sample_y, sample_x]
            + cos_t * gy[sample_y, sample_x]
        )
        if scheme == "index":
            weights = np.exp(-0.5 * (js / sigma) ** 2)
        elif scheme == "euclidean":
            weights = np.exp(-0.5 * ((js * delta) / sigma) ** 2)
        else:
            raise ValueError(f"unknown LF weighting scheme {scheme!r}")
        rows[:, ai] = np.abs(samples @ weights / weights.sum())
    return rows


def run_ablation(
    output_path: Path,
    image_size: int = 320,
    pixels_per_bin: int = 512,
    seed: int = 17,
) -> dict:
    radius = 5
    degree = 3
    m = 40
    sigma = m / 3.0
    n_orientations = 64
    gt_degrees = np.arange(0, 180, 5, dtype=np.float64)
    angles = np.linspace(0.0, math.pi, n_orientations, endpoint=False)
    kernels = build_wvf_radius_kernels(radius=radius, order=degree)
    rng = np.random.default_rng(seed)
    border = m + radius + 4
    rows = []

    for gt_deg in gt_degrees:
        theta_gt = math.radians(float(gt_deg))
        image, signed = _make_step_edge(image_size, theta_gt)
        xs, ys = _edge_pixels(signed, border, pixels_per_bin, rng)
        gx, gy = wvf_radius_gradients_metal(image, kernels, output_dtype=np.float32)
        for scheme in ("index", "euclidean"):
            response = _lf_response_rows(gx, gy, xs, ys, angles, m, sigma, scheme)
            theta_hat, mag, _, _, _ = find_two_peaks(
                angles,
                response,
                tau_sec_floor=0.40,
                tau_validity=0.10,
                dense_n=500,
                min_sep_frac=0.125,
            )
            err = _angular_error_deg(theta_hat, theta_gt)
            rows.append(
                {
                    "theta_deg": float(gt_deg),
                    "scheme": scheme,
                    "mean_abs_error_deg": float(err.mean()),
                    "p95_abs_error_deg": float(np.percentile(err, 95)),
                    "max_abs_error_deg": float(err.max()),
                    "mean_magnitude": float(mag.mean()),
                    "n_pixels": int(xs.size),
                }
            )

    index_rows = [row for row in rows if row["scheme"] == "index"]
    euclidean_rows = [row for row in rows if row["scheme"] == "euclidean"]
    worst_index = max(row["mean_abs_error_deg"] for row in index_rows)
    worst_euclidean = max(row["mean_abs_error_deg"] for row in euclidean_rows)
    if worst_index < 0.5:
        decision = "keep_index_default"
        decision_text = (
            "Keep the index-space default; the worst-bin mean absolute error "
            "is below 0.5 deg."
        )
    elif worst_index <= 1.0:
        decision = "keep_index_default_with_limitation"
        decision_text = (
            "Keep the index-space default but report the angular variation "
            "as a limitation."
        )
    else:
        decision = "switch_to_euclidean"
        decision_text = (
            "Switch to the Euclidean weighting variant; the index-space "
            "worst-bin error exceeds 1 deg."
        )

    output = {
        "ablation": "A1",
        "config": {
            "image_size": image_size,
            "pixels_per_bin": pixels_per_bin,
            "radius": radius,
            "degree": degree,
            "lf_half_length": m,
            "sigma": sigma,
            "n_orientations": n_orientations,
            "dense_n": 500,
            "gt_bin_step_deg": 5,
        },
        "rows": rows,
        "summary": {
            "worst_index_mean_error_deg": float(worst_index),
            "worst_euclidean_mean_error_deg": float(worst_euclidean),
            "decision": decision,
            "decision_text": decision_text,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT.parent / "New project",
        help="paper repository root",
    )
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pixels-per-bin", type=int, default=512)
    args = parser.parse_args()

    out = (
        args.paper_root
        / "cetz_figures"
        / "data"
        / "ablation_a1"
        / "results.json"
    )
    result = run_ablation(
        out,
        image_size=args.image_size,
        pixels_per_bin=args.pixels_per_bin,
    )
    print(f"wrote {out}")
    print(
        "A1 decision: "
        f"{result['summary']['decision']} "
        f"(worst index bin {result['summary']['worst_index_mean_error_deg']:.3f} deg, "
        f"worst Euclidean bin {result['summary']['worst_euclidean_mean_error_deg']:.3f} deg)"
    )


if __name__ == "__main__":
    main()
