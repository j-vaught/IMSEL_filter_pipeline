"""Enhanced non-maximum suppression and Canny-style hysteresis."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _suppression_mask(
    magnitude: np.ndarray,
    angle: np.ndarray,
    n_directions: int,
) -> np.ndarray:
    mag = np.asarray(magnitude, dtype=np.float64)
    theta = np.asarray(angle, dtype=np.float64) % np.pi
    if mag.shape != theta.shape:
        raise ValueError("magnitude and angle must have matching shapes")
    if n_directions < 4:
        raise ValueError("n_directions must be at least 4")

    h, w = mag.shape
    if h == 0 or w == 0:
        return np.zeros_like(mag, dtype=bool)

    y_grid, x_grid = np.indices(mag.shape, dtype=np.float64)
    direction_index = np.rint(theta / (np.pi / n_directions)).astype(np.int64) % n_directions
    keep = np.zeros_like(mag, dtype=bool)

    for index in range(n_directions):
        mask = direction_index == index
        if not np.any(mask):
            continue

        direction = index * np.pi / n_directions
        dx = np.cos(direction)
        dy = np.sin(direction)
        forward = ndimage.map_coordinates(
            mag,
            [y_grid + dy, x_grid + dx],
            order=1,
            mode="nearest",
        )
        backward = ndimage.map_coordinates(
            mag,
            [y_grid - dy, x_grid - dx],
            order=1,
            mode="nearest",
        )
        keep |= mask & (mag >= forward) & (mag >= backward) & (mag > 0.0)

    return keep


def enhanced_nonmax_suppression(
    magnitude: np.ndarray,
    angle: np.ndarray,
    n_directions: int = 8,
    secondary_magnitude: np.ndarray | None = None,
    secondary_angle: np.ndarray | None = None,
    secondary_ratio: float = 0.85,
) -> np.ndarray:
    """Thin a gradient map using denser NMS and optional corner support."""
    mag = np.asarray(magnitude, dtype=np.float64)
    keep = _suppression_mask(mag, angle, n_directions=n_directions)
    thinned = np.zeros_like(mag, dtype=np.float64)
    thinned[keep] = mag[keep]

    if secondary_magnitude is not None and secondary_angle is not None:
        secondary_mag = np.asarray(secondary_magnitude, dtype=np.float64)
        secondary_theta = np.asarray(secondary_angle, dtype=np.float64)
        if secondary_mag.shape != mag.shape or secondary_theta.shape != mag.shape:
            raise ValueError("secondary inputs must match magnitude shape")
        secondary_keep = _suppression_mask(
            secondary_mag,
            secondary_theta,
            n_directions=n_directions,
        )
        comparable = secondary_mag >= float(secondary_ratio) * mag
        secondary_keep &= comparable
        thinned[secondary_keep] = np.maximum(thinned[secondary_keep], secondary_mag[secondary_keep])

    return thinned


def automatic_hysteresis_thresholds(
    nms_magnitude: np.ndarray,
    high_quantile: float = 0.90,
    low_ratio: float = 0.40,
) -> tuple[float, float]:
    """Choose high and low hysteresis thresholds from positive NMS values."""
    values = np.asarray(nms_magnitude, dtype=np.float64)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 1.0, 0.4

    high = float(np.quantile(positive, high_quantile))
    low = float(low_ratio) * high
    return high, low


def hysteresis_threshold(
    nms_magnitude: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """Keep weak edge pixels only when connected to a strong edge pixel."""
    nms = np.asarray(nms_magnitude, dtype=np.float64)
    strong = nms >= float(high_threshold)
    candidates = nms >= float(low_threshold)
    if not np.any(candidates):
        return np.zeros_like(candidates, dtype=bool)
    if not np.any(strong):
        return np.zeros_like(candidates, dtype=bool)

    labels, count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(candidates, dtype=bool)

    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labels, strong_labels)
