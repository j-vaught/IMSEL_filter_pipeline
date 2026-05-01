"""Radius-defined isotropic WVF derivative kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from core.taylor import build_taylor_matrix


@dataclass(frozen=True)
class WVFRadiusKernels:
    """Two canonical derivative kernels for one integer disk radius."""

    radius: int
    order: int
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
    order: int
    offsets_xy: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.offsets_xy.shape[0])


def disk_offsets(radius: int, include_center: bool = False) -> np.ndarray:
    """Return all integer offsets inside a disk of ``radius``."""
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
def build_wvf_radius_kernels(radius: int, order: int = 4) -> WVFRadiusKernels:
    """Build isotropic WVF ``Gx`` and ``Gy`` kernels from an integer disk."""
    r = int(radius)
    d = int(order)
    offsets = disk_offsets(r, include_center=False)
    design = build_taylor_matrix(offsets, order=d)
    if design.shape[0] < design.shape[1]:
        raise ValueError(
            f"radius {r} gives {design.shape[0]} samples, fewer than "
            f"the {design.shape[1]} Taylor coefficients for order {d}"
        )

    pinv = np.linalg.pinv(design)
    weights_x = np.ascontiguousarray(pinv[1, :], dtype=np.float64)
    weights_y = np.ascontiguousarray(pinv[2, :], dtype=np.float64)
    kernel_x = _dense_kernel(offsets, weights_x, r)
    kernel_y = _dense_kernel(offsets, weights_y, r)

    return WVFRadiusKernels(
        radius=r,
        order=d,
        offsets_xy=np.ascontiguousarray(offsets, dtype=np.float64),
        weights_x=weights_x,
        weights_y=weights_y,
        kernel_x=np.ascontiguousarray(kernel_x, dtype=np.float64),
        kernel_y=np.ascontiguousarray(kernel_y, dtype=np.float64),
    )


@lru_cache(maxsize=64)
def build_wvf_antipodal_kernels(radius: int, order: int = 4) -> WVFAntipodalKernels:
    """Build symmetric WVF derivative pairs for Metal acceleration."""
    kernels = build_wvf_radius_kernels(radius=radius, order=order)
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
        order=int(order),
        offsets_xy=np.ascontiguousarray(pair_offsets, dtype=np.int32),
        weights_x=np.ascontiguousarray(pair_weights_x, dtype=np.float64),
        weights_y=np.ascontiguousarray(pair_weights_y, dtype=np.float64),
    )


def wvf_radius_gradients_cpu(
    image: np.ndarray,
    radius: int,
    order: int = 4,
    mode: str = "reflect",
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic WVF derivative components on CPU."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("wvf_radius_gradients_cpu expects a 2-D image")

    kernels = build_wvf_radius_kernels(radius=radius, order=order)
    gx = ndimage.correlate(img, kernels.kernel_x, mode=mode)
    gy = ndimage.correlate(img, kernels.kernel_y, mode=mode)
    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float64):
        return gx, gy
    return gx.astype(dtype), gy.astype(dtype)
