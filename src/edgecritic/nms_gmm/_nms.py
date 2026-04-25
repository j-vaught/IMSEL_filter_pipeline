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


def remove_small_components(edges: np.ndarray, min_size: int = 8) -> np.ndarray:
    """Remove connected edge components smaller than ``min_size`` pixels."""
    edge_mask = np.asarray(edges, dtype=bool)
    if min_size <= 1 or not np.any(edge_mask):
        return edge_mask.copy()

    labels, count = ndimage.label(edge_mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return edge_mask.copy()

    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def _line_footprint(theta: float, max_gap: int) -> np.ndarray:
    """Build a straight binary footprint for directional gap closing."""
    radius = int(max_gap)
    coords = []
    for step in range(-radius, radius + 1):
        dx = int(round(step * np.cos(theta)))
        dy = int(round(step * np.sin(theta)))
        coords.append((dy, dx))

    coords = np.array(sorted(set(coords)), dtype=np.int32)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    footprint = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)
    footprint[coords[:, 0] - min_y, coords[:, 1] - min_x] = True
    return footprint


def link_short_gaps(
    edges: np.ndarray,
    angle: np.ndarray,
    candidate: np.ndarray | None = None,
    max_gap: int = 3,
    n_directions: int = 8,
    iterations: int = 1,
) -> np.ndarray:
    """Link short gaps along the local edge tangent direction.

    The gradient angle is normal to the edge, so the closing direction is
    rotated by 90 degrees. Linking is constrained to candidate pixels to avoid
    filling large low-evidence regions.
    """
    linked = np.asarray(edges, dtype=bool).copy()
    if max_gap <= 0 or not np.any(linked):
        return linked

    theta = np.asarray(angle, dtype=np.float64) % np.pi
    if theta.shape != linked.shape:
        raise ValueError("angle must have the same shape as edges")

    if candidate is None:
        candidate_mask = np.ones_like(linked, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate, dtype=bool)
        if candidate_mask.shape != linked.shape:
            raise ValueError("candidate must have the same shape as edges")

    tangent = (theta + np.pi / 2.0) % np.pi
    bin_width = np.pi / int(n_directions)
    direction_index = np.rint(tangent / bin_width).astype(np.int64) % int(n_directions)

    for _ in range(max(1, int(iterations))):
        grown = linked.copy()
        for index in range(int(n_directions)):
            direction_mask = direction_index == index
            if not np.any(linked & direction_mask):
                continue
            footprint = _line_footprint(index * bin_width, max_gap=max_gap)
            closed = ndimage.binary_closing(
                linked & direction_mask,
                structure=footprint,
            )
            grown |= closed & candidate_mask & direction_mask
        if np.array_equal(grown, linked):
            break
        linked = grown

    return linked
