"""A2 reviewer-pass-5 ablation: orientation estimator comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "agent_workspaces" / "orientation_recovery_metal"))

from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from reference_impl import find_two_peaks


ESTIMATORS = (
    ("cubic", "cubic interp"),
    ("parabolic", "parabolic"),
    ("smooth", "smoothing spline"),
    ("trig", "trig fit"),
)


def _round_half_even(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(np.int32)


def _theta_error_deg(theta: np.ndarray, gt: float) -> np.ndarray:
    delta = np.abs(theta - gt)
    return np.degrees(np.minimum(delta, math.pi - delta))


def _secondary_matches(theta_s: np.ndarray, gt: tuple[float, float]) -> np.ndarray:
    out = np.zeros(theta_s.shape, dtype=bool)
    finite = np.isfinite(theta_s)
    if not finite.any():
        return out
    sec = theta_s[finite]
    d0 = np.degrees(np.minimum(np.abs(sec - gt[0]), math.pi - np.abs(sec - gt[0])))
    d1 = np.degrees(np.minimum(np.abs(sec - gt[1]), math.pi - np.abs(sec - gt[1])))
    out[finite] = (d0 <= 5.0) | (d1 <= 5.0)
    return out


def _step_edge(size: int, theta: float, noise_db: float | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    normal_x = -math.sin(theta)
    normal_y = math.cos(theta)
    signed = (xx - cx) * normal_x + (yy - cy) * normal_y
    image = 128.0 + 70.0 * np.tanh(signed / 1.25)
    if noise_db is not None:
        sigma = float(image.std() / (10.0 ** (noise_db / 20.0)))
        image = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(image, 0.0, 255.0).astype(np.float32), signed


def _corner_image(size: int, theta_a: float, theta_b: float, noise_db: float | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    def signed(theta: float) -> np.ndarray:
        return (xx - cx) * (-math.sin(theta)) + (yy - cy) * math.cos(theta)
    d_a = signed(theta_a)
    d_b = signed(theta_b)
    image = 128.0 + 42.0 * np.tanh(d_a / 1.35) + 42.0 * np.tanh(d_b / 1.35)
    if noise_db is not None:
        sigma = float(image.std() / (10.0 ** (noise_db / 20.0)))
        image = image + rng.normal(0.0, sigma, image.shape)
    corner_dist = np.maximum(np.abs(d_a), np.abs(d_b))
    return np.clip(image, 0.0, 255.0).astype(np.float32), corner_dist


def _sample_mask(mask_value: np.ndarray, border: int, threshold: float, max_pixels: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = mask_value <= threshold
    h, w = mask.shape
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
) -> np.ndarray:
    rows = np.empty((xs.size, angles.size), dtype=np.float64)
    for ai, theta in enumerate(angles):
        cos_t = math.cos(float(theta))
        sin_t = math.sin(float(theta))
        delta = 1.0 / max(abs(cos_t), abs(sin_t), 1.0e-12)
        js = np.arange(-m, m + 1, dtype=np.float64)
        offsets = js * delta
        dx = _round_half_even(offsets * cos_t)
        dy = _round_half_even(offsets * sin_t)
        sx = xs[:, None] + dx[None, :]
        sy = ys[:, None] + dy[None, :]
        samples = -sin_t * gx[sy, sx] + cos_t * gy[sy, sx]
        weights = np.exp(-0.5 * (js / sigma) ** 2)
        rows[:, ai] = np.abs(samples @ weights / weights.sum())
    return rows


def _dense_two_peaks(dy: np.ndarray, dense_angles: np.ndarray, tau_sec_floor: float = 0.40, min_sep_frac: float = 0.125) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, dense_n = dy.shape
    left = np.roll(dy, 1, axis=1)
    right = np.roll(dy, -1, axis=1)
    is_peak = (dy >= left) & (dy >= right)
    masked = np.where(is_peak, dy, -np.inf)
    primary_idx = np.argmax(masked, axis=1)
    theta_p = dense_angles[primary_idx]
    M_p = dy[np.arange(n), primary_idx]
    sep = max(1, int(min_sep_frac * dense_n))
    grid = np.arange(dense_n)
    dist = np.abs(grid[None, :] - primary_idx[:, None])
    dist = np.minimum(dist, dense_n - dist)
    masked2 = np.where((dist > sep) & is_peak, dy, -np.inf)
    sec_idx = np.argmax(masked2, axis=1)
    sec_val = masked2[np.arange(n), sec_idx]
    has_sec = sec_val > -np.inf
    M_s_raw = dy[np.arange(n), sec_idx]
    suppress = (~has_sec) | (M_s_raw / np.maximum(M_p, 1.0e-30) < tau_sec_floor)
    theta_s = np.where(suppress, np.nan, dense_angles[sec_idx])
    M_s = np.where(suppress, 0.0, M_s_raw)
    y_max = dy.max(axis=1)
    return theta_p, np.minimum(M_p, y_max), theta_s, np.minimum(M_s, y_max)


def _parabolic(response: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k = response.shape
    idx = np.argmax(response, axis=1)
    left = response[np.arange(n), (idx - 1) % k]
    center = response[np.arange(n), idx]
    right = response[np.arange(n), (idx + 1) % k]
    denom = left - 2.0 * center + right
    denom = np.where(np.abs(denom) < 1.0e-30, -1.0e-30, denom)
    delta = np.clip(0.5 * (left - right) / denom, -0.5, 0.5)
    theta_p = ((idx + delta) % k) * (math.pi / k)
    M_p = center - 0.25 * (left - right) * delta

    sep = max(1, int(0.125 * k))
    is_peak = (response > np.roll(response, 1, axis=1)) & (response > np.roll(response, -1, axis=1))
    grid = np.arange(k)
    dist = np.abs(grid[None, :] - idx[:, None])
    dist = np.minimum(dist, k - dist)
    masked = np.where((dist > sep) & is_peak, response, -np.inf)
    sec_idx = np.argmax(masked, axis=1)
    sec_val = masked[np.arange(n), sec_idx]
    has_sec = sec_val > -np.inf
    sl = response[np.arange(n), (sec_idx - 1) % k]
    sc = response[np.arange(n), sec_idx]
    sr = response[np.arange(n), (sec_idx + 1) % k]
    sden = sl - 2.0 * sc + sr
    sden = np.where(np.abs(sden) < 1.0e-30, -1.0e-30, sden)
    sdelta = np.clip(0.5 * (sl - sr) / sden, -0.5, 0.5)
    theta_s_raw = ((sec_idx + sdelta) % k) * (math.pi / k)
    M_s_raw = sc - 0.25 * (sl - sr) * sdelta
    suppress = (~has_sec) | (M_s_raw / np.maximum(M_p, 1.0e-30) < 0.40)
    theta_s = np.where(suppress, np.nan, theta_s_raw)
    M_s = np.where(suppress, 0.0, M_s_raw)
    return theta_p, M_p, theta_s, M_s


def _smoothing_matrix(k: int, lam: float, weights: np.ndarray | None = None) -> np.ndarray:
    d2 = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        d2[i, (i - 1) % k] = 1.0
        d2[i, i] = -2.0
        d2[i, (i + 1) % k] = 1.0
    penalty = d2.T @ d2
    if weights is None:
        return np.eye(k) + lam * penalty
    return np.diag(weights) + lam * penalty


def _select_smoothing_lambda(response: np.ndarray) -> float:
    k = response.shape[1]
    candidates = (0.05, 0.15, 0.45, 1.35, 4.0)
    sample = response[: min(96, response.shape[0])]
    best = candidates[0]
    best_err = float("inf")
    for lam in candidates:
        err = 0.0
        for fold in range(4):
            weights = np.ones(k)
            weights[fold::4] = 0.0
            mat = _smoothing_matrix(k, lam, weights)
            smooth = np.linalg.solve(mat, (weights[None, :] * sample).T).T
            diff = smooth[:, fold::4] - sample[:, fold::4]
            err += float(np.mean(diff * diff))
        if err < best_err:
            best_err = err
            best = lam
    return best


def _smooth_estimator(response: np.ndarray, angles: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = response.shape[1]
    mat = _smoothing_matrix(k, lam)
    smooth = np.linalg.solve(mat, response.T).T
    x = np.concatenate([angles, [math.pi]])
    y = np.concatenate([smooth, smooth[:, :1]], axis=1)
    cs = CubicSpline(x, y, axis=1, bc_type="periodic")
    dense_angles = np.linspace(0.0, math.pi, 500, endpoint=False)
    return _dense_two_peaks(cs(dense_angles), dense_angles)


def _trig_estimator(response: np.ndarray, angles: np.ndarray, order: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = [np.ones_like(angles)]
    for n in range(1, order + 1):
        cols.append(np.cos(2.0 * n * angles))
        cols.append(np.sin(2.0 * n * angles))
    design = np.column_stack(cols)
    coef = response @ np.linalg.pinv(design).T
    dense_angles = np.linspace(0.0, math.pi, 500, endpoint=False)
    dcols = [np.ones_like(dense_angles)]
    for n in range(1, order + 1):
        dcols.append(np.cos(2.0 * n * dense_angles))
        dcols.append(np.sin(2.0 * n * dense_angles))
    dense = coef @ np.column_stack(dcols).T
    return _dense_two_peaks(dense, dense_angles)


def _estimate(name: str, response: np.ndarray, angles: np.ndarray, smooth_lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "cubic":
        theta_p, M_p, theta_s, M_s, _ = find_two_peaks(angles, response, dense_n=500)
        return theta_p, M_p, theta_s, M_s
    if name == "parabolic":
        return _parabolic(response, angles)
    if name == "smooth":
        return _smooth_estimator(response, angles, smooth_lam)
    if name == "trig":
        return _trig_estimator(response, angles)
    raise ValueError(name)


def run_ablation(output_path: Path, image_size: int = 256, pixels_per_bin: int = 64) -> dict:
    rng = np.random.default_rng(123)
    radius, degree, m = 5, 3, 40
    sigma = m / 3.0
    border = m + radius + 4
    gt_degrees = np.arange(0, 180, 15, dtype=np.float64)
    snr_levels = [None, 30.0, 20.0, 10.0]
    snr_labels = ["clean", "30", "20", "10"]
    angles = np.linspace(0.0, math.pi, 64, endpoint=False)
    kernels = build_wvf_radius_kernels(radius=radius, order=degree)

    primary_rows = []
    timing_response = None
    smoothing_lambda = 0.45
    for snr_label, snr in zip(snr_labels, snr_levels, strict=True):
        responses = []
        gt = []
        for gt_deg in gt_degrees:
            theta_gt = math.radians(float(gt_deg))
            image, signed = _step_edge(image_size, theta_gt, snr, rng)
            xs, ys = _sample_mask(np.abs(signed), border, 0.6, pixels_per_bin, rng)
            gx, gy = wvf_radius_gradients_metal(image, kernels, output_dtype=np.float32)
            resp = _lf_response_rows(gx, gy, xs, ys, angles, m, sigma)
            responses.append(resp)
            gt.append(np.full(resp.shape[0], theta_gt))
        response = np.vstack(responses)
        gt_theta = np.concatenate(gt)
        if snr_label == "20":
            smoothing_lambda = _select_smoothing_lambda(response)
            timing_response = response[: min(512, response.shape[0])]
        for name, label in ESTIMATORS:
            theta_p, _, _, _ = _estimate(name, response, angles, smoothing_lambda)
            err = _theta_error_deg(theta_p, gt_theta)
            primary_rows.append(
                {
                    "snr": snr_label,
                    "estimator": name,
                    "label": label,
                    "mean_primary_error_deg": float(err.mean()),
                    "p95_primary_error_deg": float(np.percentile(err, 95)),
                    "n_pixels": int(err.size),
                }
            )

    corner_theta = (math.radians(66.4), math.radians(149.5))
    corner_rows = []
    for snr_label, snr in zip(snr_labels, snr_levels, strict=True):
        image, corner_dist = _corner_image(image_size, corner_theta[0], corner_theta[1], snr, rng)
        xs, ys = _sample_mask(corner_dist, border, 3.0, 320, rng)
        gx, gy = wvf_radius_gradients_metal(image, kernels, output_dtype=np.float32)
        response = _lf_response_rows(gx, gy, xs, ys, angles, m, sigma)
        for name, label in ESTIMATORS:
            _, _, theta_s, _ = _estimate(name, response, angles, smoothing_lambda)
            recall = _secondary_matches(theta_s, corner_theta)
            corner_rows.append(
                {
                    "snr": snr_label,
                    "estimator": name,
                    "label": label,
                    "corner_recall": float(recall.mean()),
                    "n_pixels": int(recall.size),
                }
            )

    if timing_response is None:
        timing_response = response[: min(512, response.shape[0])]
    timing_rows = []
    for name, label in ESTIMATORS:
        t0 = time.perf_counter()
        _ = _estimate(name, timing_response, angles, smoothing_lambda)
        elapsed = time.perf_counter() - t0
        timing_rows.append(
            {
                "estimator": name,
                "label": label,
                "cost_us_per_pixel": float(elapsed / timing_response.shape[0] * 1.0e6),
            }
        )

    def row_for(rows: list[dict], estimator: str, snr: str) -> dict:
        return next(row for row in rows if row["estimator"] == estimator and row["snr"] == snr)

    cubic_wins = True
    for snr in snr_labels:
        cubic_err = row_for(primary_rows, "cubic", snr)["mean_primary_error_deg"]
        cubic_rec = row_for(corner_rows, "cubic", snr)["corner_recall"]
        for name, _ in ESTIMATORS:
            if name == "cubic":
                continue
            cubic_wins &= cubic_err <= row_for(primary_rows, name, snr)["mean_primary_error_deg"] + 1.0e-12
            cubic_wins &= cubic_rec >= row_for(corner_rows, name, snr)["corner_recall"] - 1.0e-12

    replacement = None
    for name in ("smooth", "trig"):
        for snr in ("20", "10"):
            err_gain = (
                row_for(primary_rows, "cubic", snr)["mean_primary_error_deg"]
                - row_for(primary_rows, name, snr)["mean_primary_error_deg"]
            )
            recall_gain = (
                row_for(corner_rows, name, snr)["corner_recall"]
                - row_for(corner_rows, "cubic", snr)["corner_recall"]
            )
            if err_gain > 0.2 or recall_gain > 0.05:
                replacement = name
                break
        if replacement:
            break
    parabolic_clean_gap = (
        row_for(primary_rows, "parabolic", "clean")["mean_primary_error_deg"]
        - row_for(primary_rows, "cubic", "clean")["mean_primary_error_deg"]
    )
    parabolic_noise_gap = (
        row_for(primary_rows, "parabolic", "10")["mean_primary_error_deg"]
        - row_for(primary_rows, "cubic", "10")["mean_primary_error_deg"]
    )
    if cubic_wins:
        decision = "keep_cubic"
        decision_text = "Keep cubic interpolation; it wins primary error and corner recall at every SNR."
    elif replacement:
        decision = f"replace_with_{replacement}"
        decision_text = f"Replace cubic interpolation with {replacement}; it clears the SNR 10/20 improvement rule."
    else:
        decision = "keep_cubic"
        decision_text = "Keep cubic interpolation; alternatives do not clear the replacement thresholds."

    output = {
        "ablation": "A2",
        "config": {
            "image_size": image_size,
            "pixels_per_bin": pixels_per_bin,
            "radius": radius,
            "degree": degree,
            "lf_half_length": m,
            "sigma": sigma,
            "n_orientations": 64,
            "dense_n": 500,
            "smoothing_lambda": smoothing_lambda,
        },
        "primary_rows": primary_rows,
        "corner_rows": corner_rows,
        "timing_rows": timing_rows,
        "summary": {
            "decision": decision,
            "decision_text": decision_text,
            "parabolic_clean_gap_deg": float(parabolic_clean_gap),
            "parabolic_snr10_gap_deg": float(parabolic_noise_gap),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT.parent / "New project")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pixels-per-bin", type=int, default=64)
    args = parser.parse_args()
    out = args.paper_root / "cetz_figures" / "data" / "ablation_a2" / "results.json"
    result = run_ablation(out, image_size=args.image_size, pixels_per_bin=args.pixels_per_bin)
    print(f"wrote {out}")
    print(f"A2 decision: {result['summary']['decision']}")
    for row in result["primary_rows"]:
        if row["snr"] in ("clean", "10"):
            print(
                f"  {row['snr']:>5} {row['estimator']:>9}: "
                f"primary error {row['mean_primary_error_deg']:.3f} deg"
            )
    for row in result["corner_rows"]:
        if row["snr"] == "10":
            print(
                f"  snr10 {row['estimator']:>9}: corner recall "
                f"{row['corner_recall']:.3f}"
            )


if __name__ == "__main__":
    main()
