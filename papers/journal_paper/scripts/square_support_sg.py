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


def compute_pseudoinverse(design: np.ndarray) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    cutoff = max(matrix.shape) * np.finfo(np.float64).eps
    return np.linalg.pinv(matrix, rcond=cutoff)


def design_condition_number(design: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(design, dtype=np.float64), compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    return float("inf") if sigma_min <= 0.0 else sigma_max / sigma_min


def square_offsets(half_side: int, include_center: bool = False) -> np.ndarray:
    h = int(half_side)
    coords = [(dx, dy) for dy in range(-h, h + 1) for dx in range(-h, h + 1)]
    offsets_xy = np.asarray(coords, dtype=np.float64)
    if include_center:
        return offsets_xy
    keep = ~((offsets_xy[:, 0] == 0.0) & (offsets_xy[:, 1] == 0.0))
    return offsets_xy[keep]


def _dense_kernel(offsets_xy: np.ndarray, weights: np.ndarray, half_side: int) -> np.ndarray:
    size = 2 * int(half_side) + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    for index, (dx, dy) in enumerate(offsets_xy.astype(np.int64)):
        kernel[dy + half_side, dx + half_side] += float(weights[index])
    return kernel


@dataclass(frozen=True)
class SquareSupportKernels:
    half_side: int
    degree: int
    normalize_coords: bool
    offsets_xy: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    kappa_design_matrix: float


def build_square_support_kernels(
    half_side: int,
    degree: int = 3,
    normalize_coords: bool = False,
) -> SquareSupportKernels:
    h = int(half_side)
    normalized = bool(normalize_coords)
    offsets_xy = square_offsets(h, include_center=False)
    design = build_design_matrix(
        offsets_xy,
        degree=int(degree),
        normalize_radius=float(h) if normalized else None,
    )
    pinv = compute_pseudoinverse(design)
    derivative_scale = 1.0 / float(h) if normalized else 1.0
    weights_x = np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)
    weights_y = np.asarray(pinv[2, :] * derivative_scale, dtype=np.float64)
    return SquareSupportKernels(
        half_side=h,
        degree=int(degree),
        normalize_coords=normalized,
        offsets_xy=np.ascontiguousarray(offsets_xy, dtype=np.float64),
        weights_x=np.ascontiguousarray(weights_x, dtype=np.float64),
        weights_y=np.ascontiguousarray(weights_y, dtype=np.float64),
        kernel_x=np.ascontiguousarray(_dense_kernel(offsets_xy, weights_x, h), dtype=np.float64),
        kernel_y=np.ascontiguousarray(_dense_kernel(offsets_xy, weights_y, h), dtype=np.float64),
        kappa_design_matrix=design_condition_number(design),
    )
