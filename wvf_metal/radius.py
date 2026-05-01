"""Radius-defined WVF derivative kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class WVFRadiusKernels:
    """Direct derivative kernels for one integer disk radius."""

    radius: int
    degree: int
    offsets_xy: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    kernel_x: np.ndarray
    kernel_y: np.ndarray

    @property
    def support_size(self) -> int:
        return int(self.offsets_xy.shape[0])


@dataclass(frozen=True)
class WVFAntipodalKernels:
    """Antipodal derivative pairs for one integer disk radius."""

    radius: int
    degree: int
    offsets_xy: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.offsets_xy.shape[0])


def taylor_exponents(degree: int) -> list[tuple[int, int]]:
    """Return WVF Taylor exponents through ``degree`` with stable ordering."""
    d = int(degree)
    if d < 0:
        raise ValueError("degree must be nonnegative")

    exponents: list[tuple[int, int]] = [(0, 0)]
    if d >= 1:
        exponents.extend([(1, 0), (0, 1)])

    for total_degree in range(2, d + 1):
        exponents.append((total_degree, 0))
        exponents.append((0, total_degree))
        for px in range(total_degree - 1, 0, -1):
            exponents.append((px, total_degree - px))

    return exponents


def build_taylor_matrix(coords_xy: np.ndarray, degree: int = 4) -> np.ndarray:
    """Build the scaled Taylor design matrix for local coordinates."""
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords_xy must have shape (n, 2)")

    x = coords[:, 0]
    y = coords[:, 1]
    columns = []
    for px, py in taylor_exponents(degree):
        scale = math.factorial(px) * math.factorial(py)
        columns.append((x**px) * (y**py) / scale)
    return np.column_stack(columns)


def disk_offsets(radius: int, include_center: bool = False) -> np.ndarray:
    """Return integer offsets inside a disk of ``radius``."""
    r = int(radius)
    if r < 1:
        raise ValueError("radius must be a positive integer")

    offsets: list[tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not include_center and dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= r * r:
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=np.float64)


def _dense_kernel(offsets_xy: np.ndarray, weights: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * int(radius) + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    for index, (dx, dy) in enumerate(offsets_xy.astype(np.int64)):
        kernel[dy + radius, dx + radius] += float(weights[index])
    return kernel


@lru_cache(maxsize=64)
def build_wvf_radius_kernels(radius: int, degree: int = 4) -> WVFRadiusKernels:
    """Build direct WVF derivative kernels for an integer disk."""
    r = int(radius)
    d = int(degree)
    offsets = disk_offsets(r, include_center=False)
    design = build_taylor_matrix(offsets, degree=d)
    if design.shape[0] < design.shape[1]:
        raise ValueError(
            f"radius {r} gives {design.shape[0]} samples, fewer than "
            f"the {design.shape[1]} Taylor coefficients for degree {d}"
        )

    pinv = np.linalg.pinv(design)
    weights_x = np.ascontiguousarray(pinv[1, :], dtype=np.float64)
    weights_y = np.ascontiguousarray(pinv[2, :], dtype=np.float64)
    kernel_x = _dense_kernel(offsets, weights_x, r)
    kernel_y = _dense_kernel(offsets, weights_y, r)

    return WVFRadiusKernels(
        radius=r,
        degree=d,
        offsets_xy=np.ascontiguousarray(offsets, dtype=np.float64),
        weights_x=weights_x,
        weights_y=weights_y,
        kernel_x=np.ascontiguousarray(kernel_x, dtype=np.float64),
        kernel_y=np.ascontiguousarray(kernel_y, dtype=np.float64),
    )


@lru_cache(maxsize=64)
def build_wvf_antipodal_kernels(radius: int, degree: int = 4) -> WVFAntipodalKernels:
    """Build antipodal WVF derivative pairs for the optimized Metal path."""
    kernels = build_wvf_radius_kernels(radius=radius, degree=degree)
    offsets = kernels.offsets_xy.astype(np.int64, copy=False)
    index = {tuple(offset): i for i, offset in enumerate(offsets)}
    used: set[int] = set()
    pair_offsets: list[tuple[int, int]] = []
    pair_weights_x: list[float] = []
    pair_weights_y: list[float] = []

    for i, (dx, dy) in enumerate(offsets):
        if i in used:
            continue
        opposite = (-int(dx), -int(dy))
        j = index.get(opposite)
        if j is None:
            raise ValueError(f"offset ({dx}, {dy}) has no antipodal partner")

        used.add(i)
        used.add(j)
        if int(dy) > 0 or (int(dy) == 0 and int(dx) > 0):
            pos, neg = i, j
            pair_dx, pair_dy = int(dx), int(dy)
        else:
            pos, neg = j, i
            pair_dx, pair_dy = opposite

        pair_offsets.append((pair_dx, pair_dy))
        pair_weights_x.append(
            0.5 * (float(kernels.weights_x[pos]) - float(kernels.weights_x[neg]))
        )
        pair_weights_y.append(
            0.5 * (float(kernels.weights_y[pos]) - float(kernels.weights_y[neg]))
        )

    return WVFAntipodalKernels(
        radius=int(radius),
        degree=int(degree),
        offsets_xy=np.ascontiguousarray(pair_offsets, dtype=np.int32),
        weights_x=np.ascontiguousarray(pair_weights_x, dtype=np.float64),
        weights_y=np.ascontiguousarray(pair_weights_y, dtype=np.float64),
    )


def _reflect_indices(values: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 1:
        return np.zeros_like(values, dtype=np.int64)

    reflected = np.asarray(values, dtype=np.int64).copy()
    outside = (reflected < 0) | (reflected >= limit)
    while bool(outside.any()):
        low = reflected < 0
        reflected[low] = -reflected[low] - 1
        high = reflected >= limit
        reflected[high] = 2 * limit - reflected[high] - 1
        outside = (reflected < 0) | (reflected >= limit)
    return reflected


def wvf_radius_gradients_cpu(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF derivative components with a pure NumPy reference path."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("wvf_radius_gradients_cpu expects a 2-D image")

    kernels = build_wvf_radius_kernels(radius=radius, degree=degree)
    h, w = img.shape
    x = np.arange(w, dtype=np.int64)
    y = np.arange(h, dtype=np.int64)
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)

    for (dx, dy), wx, wy in zip(
        kernels.offsets_xy.astype(np.int64),
        kernels.weights_x,
        kernels.weights_y,
        strict=True,
    ):
        ix = _reflect_indices(x + int(dx), w)
        iy = _reflect_indices(y + int(dy), h)
        sampled = img[np.ix_(iy, ix)]
        gx += sampled * float(wx)
        gy += sampled * float(wy)

    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float64):
        return gx, gy
    return gx.astype(dtype), gy.astype(dtype)
