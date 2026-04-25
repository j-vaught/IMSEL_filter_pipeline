"""Periodic cubic-spline orientation estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class SplineOrientationResult:
    """Spline-derived gradient magnitude and orientation maps."""

    magnitude: np.ndarray
    angle: np.ndarray
    response_range: np.ndarray


def spline_orientation(
    gradients: np.ndarray,
    angles: np.ndarray,
    period: float = np.pi,
    refine: bool = True,
) -> tuple[float, float]:
    """Estimate the orientation of the maximum LF response at one pixel."""
    values = np.abs(np.asarray(gradients, dtype=np.float64))
    theta = np.asarray(angles, dtype=np.float64)
    if values.ndim != 1 or theta.ndim != 1 or len(values) != len(theta):
        raise ValueError("gradients and angles must be one-dimensional arrays of the same length")
    if len(values) == 0 or not np.any(np.isfinite(values)):
        return 0.0, 0.0

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    best_index = int(np.argmax(values))
    best_angle = float(theta[best_index] % period)
    best_value = float(values[best_index])

    if not refine or len(values) < 4 or float(np.ptp(values)) <= np.finfo(np.float64).eps:
        return best_angle, best_value

    x = np.concatenate([theta, [period]])
    y = np.concatenate([values, [values[0]]])

    try:
        spline = CubicSpline(x, y, bc_type="periodic")
        derivative = spline.derivative()
        roots = derivative.roots(extrapolate=False)
    except (ValueError, np.linalg.LinAlgError):
        return best_angle, best_value

    candidates = [best_angle]
    left_start = theta[best_index - 1] if best_index > 0 else theta[-1]
    left_end = theta[best_index] if best_index > 0 else period
    right_start = theta[best_index]
    right_end = theta[best_index + 1] if best_index + 1 < len(theta) else period

    for root in roots:
        root = float(root)
        if left_start <= root <= left_end or right_start <= root <= right_end:
            candidates.append(root % period)

    scores = np.array([spline(candidate if candidate > 0.0 else 0.0) for candidate in candidates])
    winner = int(np.argmax(scores))
    return float(candidates[winner] % period), float(max(scores[winner], 0.0))


def spline_orientation_map(
    responses: np.ndarray,
    angles: np.ndarray,
    range_threshold_rel: float = 0.0,
    refine: bool = True,
) -> SplineOrientationResult:
    """Compute spline-derived magnitude and orientation for every pixel."""
    response_array = np.asarray(responses, dtype=np.float64)
    if response_array.ndim != 3:
        raise ValueError("responses must have shape (H, W, n_orientations)")

    h, w, _ = response_array.shape
    abs_responses = np.abs(response_array)
    ranges = np.ptp(abs_responses, axis=-1)
    threshold = float(range_threshold_rel) * float(np.max(ranges)) if ranges.size else 0.0

    magnitude = np.zeros((h, w), dtype=np.float64)
    angle = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            if ranges[y, x] <= threshold:
                continue
            theta, mag = spline_orientation(abs_responses[y, x, :], angles, refine=refine)
            angle[y, x] = theta
            magnitude[y, x] = mag

    return SplineOrientationResult(
        magnitude=magnitude,
        angle=angle,
        response_range=ranges,
    )
