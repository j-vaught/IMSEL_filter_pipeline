#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _monomial_exponents(order: int) -> list[tuple[int, int]]:
    exponents = [(0, 0)]
    if order >= 1:
        exponents.extend([(1, 0), (0, 1)])
    for degree in range(2, order + 1):
        exponents.append((degree, 0))
        exponents.append((0, degree))
        for px in range(degree - 1, 0, -1):
            exponents.append((px, degree - px))
    return exponents


def disk_offsets(radius: float, include_center: bool = False) -> np.ndarray:
    r = float(radius)
    limit = int(math.ceil(r))
    coords: list[tuple[int, int]] = []
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            if not include_center and dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= r * r + 1.0e-12:
                coords.append((dx, dy))
    return np.asarray(coords, dtype=np.float64)


def build_design_matrix(
    offsets_xy: np.ndarray,
    degree: int,
    normalize_radius: float | None = None,
) -> np.ndarray:
    coords = np.asarray(offsets_xy, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    if normalize_radius is not None:
        x = x / float(normalize_radius)
        y = y / float(normalize_radius)
    columns = []
    for px, py in _monomial_exponents(int(degree)):
        scale = math.factorial(px) * math.factorial(py)
        columns.append((x**px) * (y**py) / scale)
    return np.column_stack(columns)


def _svd_cutoff(matrix_shape: tuple[int, int]) -> float:
    return max(int(matrix_shape[0]), int(matrix_shape[1])) * np.finfo(np.float64).eps


def gaussian_pixel_weights(offsets_xy: np.ndarray, sigma_w: float | None) -> np.ndarray:
    coords = np.asarray(offsets_xy, dtype=np.float64)
    if sigma_w is None or math.isinf(float(sigma_w)):
        return np.ones(coords.shape[0], dtype=np.float64)
    sigma = float(sigma_w)
    rho2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    return np.exp(-rho2 / (2.0 * sigma * sigma))


def compute_weighted_pseudoinverse(
    design: np.ndarray,
    pixel_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(design, dtype=np.float64)
    weights = np.asarray(pixel_weights, dtype=np.float64)
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    weighted_design = sqrt_w[:, None] * matrix
    cutoff = _svd_cutoff(weighted_design.shape)
    pinv_weighted = np.linalg.pinv(weighted_design, rcond=cutoff)
    pinv = pinv_weighted * sqrt_w[None, :]
    singular_values = np.linalg.svd(weighted_design, compute_uv=False)
    return np.asarray(pinv, dtype=np.float64), np.asarray(singular_values, dtype=np.float64)


def _dense_kernel(offsets_xy: np.ndarray, weights: np.ndarray, half_extent: int) -> np.ndarray:
    size = 2 * int(half_extent) + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    for index, (dx, dy) in enumerate(offsets_xy.astype(np.int64)):
        kernel[dy + half_extent, dx + half_extent] += float(weights[index])
    return kernel


@dataclass(frozen=True)
class WeightedDiskKernels:
    radius: float
    degree: int
    normalize_coords: bool
    sigma_w: float | None
    offsets_xy: np.ndarray
    pixel_weights: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    support_cardinality: int
    design_matrix_shape: tuple[int, int]
    kappa_design_matrix: float
    sigma_min: float
    rank_deficient_count: int
    kernel_half_extent: int

    @property
    def sigma_label(self) -> str:
        if self.sigma_w is None or math.isinf(float(self.sigma_w)):
            return "uniform"
        return f"{float(self.sigma_w):g}px"


def build_weighted_disk_kernels(
    radius: float,
    degree: int = 3,
    normalize_coords: bool = False,
    sigma_w: float | None = None,
) -> WeightedDiskKernels:
    r = float(radius)
    degree_i = int(degree)
    normalized = bool(normalize_coords)
    offsets_xy = disk_offsets(r, include_center=False)
    design = build_design_matrix(
        offsets_xy,
        degree=degree_i,
        normalize_radius=r if normalized else None,
    )
    pixel_weights = gaussian_pixel_weights(offsets_xy, sigma_w)
    pinv, singular_values = compute_weighted_pseudoinverse(design, pixel_weights)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    cutoff = _svd_cutoff(design.shape) * sigma_max
    derivative_scale = 1.0 / r if normalized else 1.0
    weights_x = np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)
    weights_y = np.asarray(pinv[2, :] * derivative_scale, dtype=np.float64)
    half_extent = int(math.ceil(r))
    return WeightedDiskKernels(
        radius=r,
        degree=degree_i,
        normalize_coords=normalized,
        sigma_w=None if sigma_w is None or math.isinf(float(sigma_w)) else float(sigma_w),
        offsets_xy=np.ascontiguousarray(offsets_xy, dtype=np.float64),
        pixel_weights=np.ascontiguousarray(pixel_weights, dtype=np.float64),
        weights_x=np.ascontiguousarray(weights_x, dtype=np.float64),
        weights_y=np.ascontiguousarray(weights_y, dtype=np.float64),
        kernel_x=np.ascontiguousarray(_dense_kernel(offsets_xy, weights_x, half_extent), dtype=np.float64),
        kernel_y=np.ascontiguousarray(_dense_kernel(offsets_xy, weights_y, half_extent), dtype=np.float64),
        support_cardinality=int(offsets_xy.shape[0]),
        design_matrix_shape=(int(design.shape[0]), int(design.shape[1])),
        kappa_design_matrix=float("inf") if sigma_min <= 0.0 else sigma_max / sigma_min,
        sigma_min=sigma_min,
        rank_deficient_count=int(np.count_nonzero(singular_values <= cutoff)),
        kernel_half_extent=half_extent,
    )
