"""Line Filter response stacks used by the NMS edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from edgecritic.core.taylor import (
    build_taylor_matrix,
    get_circular_neighbors,
    get_square_neighbors,
    rotate_coordinates,
)


@dataclass(frozen=True)
class LineFilterKernels:
    """Dense LF kernels for a fixed scale and orientation grid."""

    kernels: np.ndarray
    angles: np.ndarray
    border: int


@dataclass(frozen=True)
class LineFilterResponses:
    """Per-orientation LF response maps."""

    responses: np.ndarray
    angles: np.ndarray
    border: int


def _neighbors(np_count: int, neighbor_type: str) -> np.ndarray:
    if neighbor_type == "square":
        return get_square_neighbors(np_count)
    if neighbor_type == "circular":
        return get_circular_neighbors(np_count)
    raise ValueError("neighbor_type must be 'circular' or 'square'")


def _pfx_row(neighbors: np.ndarray, theta: float, order: int) -> np.ndarray:
    local = rotate_coordinates(neighbors, theta)
    design = build_taylor_matrix(local, order=order)
    return np.linalg.pinv(design)[1, :]


@lru_cache(maxsize=64)
def _cached_line_filter_kernels(
    half_width: int,
    np_count: int,
    order: int,
    n_orientations: int,
    neighbor_type: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    if np_count <= 0:
        raise ValueError("np_count must be positive")
    if n_orientations < 4:
        raise ValueError("n_orientations must be at least 4")

    neighbors = _neighbors(np_count, neighbor_type)
    angles = np.linspace(0.0, np.pi, n_orientations, endpoint=False)

    line_offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    if half_width == 0:
        line_weights = np.ones(1, dtype=np.float64)
    else:
        sigma = max(half_width / 2.0, np.finfo(np.float64).eps)
        line_weights = np.exp(-0.5 * (line_offsets / sigma) ** 2)
        line_weights /= line_weights.sum()

    sparse_offsets: list[np.ndarray] = []
    sparse_weights: list[np.ndarray] = []
    max_dy = 0
    max_dx = 0

    for theta in angles:
        pfx = _pfx_row(neighbors, theta, order)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        stencil: dict[tuple[int, int], float] = {}
        for line_index, offset in enumerate(line_offsets):
            vdx = int(round(offset * cos_t))
            vdy = int(round(offset * sin_t))
            line_weight = line_weights[line_index]

            for neighbor_index in range(len(neighbors)):
                ndx = int(neighbors[neighbor_index, 0])
                ndy = int(neighbors[neighbor_index, 1])
                key = (vdy + ndy, vdx + ndx)
                stencil[key] = stencil.get(key, 0.0) + line_weight * pfx[neighbor_index]

        offsets = np.array(list(stencil.keys()), dtype=np.int32)
        weights = np.array(list(stencil.values()), dtype=np.float64)
        sparse_offsets.append(offsets)
        sparse_weights.append(weights)

        if len(offsets):
            max_dy = max(max_dy, int(np.max(np.abs(offsets[:, 0]))))
            max_dx = max(max_dx, int(np.max(np.abs(offsets[:, 1]))))

    kernels = np.zeros((n_orientations, 2 * max_dy + 1, 2 * max_dx + 1), dtype=np.float64)
    for kernel_index, (offsets, weights) in enumerate(zip(sparse_offsets, sparse_weights)):
        for offset_index in range(len(offsets)):
            dy, dx = offsets[offset_index]
            kernels[kernel_index, dy + max_dy, dx + max_dx] += weights[offset_index]

    return kernels, angles, max(max_dy, max_dx)


def build_line_filter_kernels(
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
) -> LineFilterKernels:
    """Build dense LF kernels over ``[0, pi)`` for fast response maps."""
    kernels, angles, border = _cached_line_filter_kernels(
        int(half_width),
        int(np_count),
        int(order),
        int(n_orientations),
        str(neighbor_type),
    )
    return LineFilterKernels(kernels=kernels.copy(), angles=angles.copy(), border=border)


def line_filter_response_stack(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
    mode: str = "reflect",
    zero_border: bool = True,
) -> LineFilterResponses:
    """Apply LF kernels and return signed responses for each orientation."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("line_filter_response_stack expects a 2-D image domain")

    kernel_set = build_line_filter_kernels(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )

    responses = np.empty((*img.shape, len(kernel_set.angles)), dtype=np.float64)
    for index, kernel in enumerate(kernel_set.kernels):
        responses[..., index] = ndimage.correlate(img, kernel, mode=mode)

    if zero_border and kernel_set.border > 0:
        b = kernel_set.border
        responses[:b, :, :] = 0.0
        responses[-b:, :, :] = 0.0
        responses[:, :b, :] = 0.0
        responses[:, -b:, :] = 0.0

    return LineFilterResponses(
        responses=responses,
        angles=kernel_set.angles,
        border=kernel_set.border,
    )
