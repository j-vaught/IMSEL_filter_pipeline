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


def square_offsets(half_side: float, include_center: bool = False) -> np.ndarray:
    h = float(half_side)
    limit = int(math.ceil(h))
    coords: list[tuple[int, int]] = []
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            if abs(dx) <= h + 1.0e-12 and abs(dy) <= h + 1.0e-12:
                coords.append((dx, dy))
    offsets_xy = np.asarray(coords, dtype=np.float64)
    if include_center:
        return offsets_xy
    keep = ~((offsets_xy[:, 0] == 0.0) & (offsets_xy[:, 1] == 0.0))
    return offsets_xy[keep]


def regular_polygon_vertices(
    side_count: int,
    circumradius: float,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    count = int(side_count)
    radius = float(circumradius)
    rotation_rad = math.radians(float(rotation_deg))
    vertices = []
    for index in range(count):
        angle = rotation_rad + 2.0 * math.pi * index / count
        vertices.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.asarray(vertices, dtype=np.float64)


def _point_in_convex_polygon(point_xy: np.ndarray, vertices_xy: np.ndarray, tol: float = 1.0e-12) -> bool:
    vertices = np.asarray(vertices_xy, dtype=np.float64)
    point = np.asarray(point_xy, dtype=np.float64)
    rolled = np.roll(vertices, -1, axis=0)
    edge_vectors = rolled - vertices
    point_vectors = point[None, :] - vertices
    cross = edge_vectors[:, 0] * point_vectors[:, 1] - edge_vectors[:, 1] * point_vectors[:, 0]
    return bool(np.all(cross >= -tol) or np.all(cross <= tol))


def convex_polygon_offsets(
    vertices_xy: np.ndarray,
    bounding_radius: int,
    include_center: bool = False,
) -> np.ndarray:
    radius = int(math.ceil(float(bounding_radius)))
    coords: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not include_center and dx == 0 and dy == 0:
                continue
            if _point_in_convex_polygon(np.asarray((dx, dy), dtype=np.float64), vertices_xy):
                coords.append((dx, dy))
    return np.asarray(coords, dtype=np.float64)


def _dense_kernel(offsets_xy: np.ndarray, weights: np.ndarray, half_extent: int) -> np.ndarray:
    size = 2 * int(half_extent) + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    for index, (dx, dy) in enumerate(offsets_xy.astype(np.int64)):
        kernel[dy + half_extent, dx + half_extent] += float(weights[index])
    return kernel


@dataclass(frozen=True)
class SupportKernels:
    support_name: str
    support_value: float
    degree: int
    normalize_coords: bool
    offsets_xy: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    support_cardinality: int
    design_matrix_shape: tuple[int, int]
    kappa_design_matrix: float


SquareSupportKernels = SupportKernels


def build_support_kernels(
    offsets_xy: np.ndarray,
    support_name: str,
    support_value: float,
    degree: int = 3,
    normalize_coords: bool = False,
    kernel_half_extent: int | None = None,
) -> SupportKernels:
    coords = np.asarray(offsets_xy, dtype=np.float64)
    degree_i = int(degree)
    normalized = bool(normalize_coords)
    if kernel_half_extent is None:
        kernel_half_extent = int(np.max(np.abs(coords.astype(np.int64)))) if coords.size else 0
    design = build_design_matrix(
        coords,
        degree=degree_i,
        normalize_radius=float(support_value) if normalized else None,
    )
    pinv = compute_pseudoinverse(design)
    derivative_scale = 1.0 / float(support_value) if normalized else 1.0
    weights_x = np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)
    weights_y = np.asarray(pinv[2, :] * derivative_scale, dtype=np.float64)
    return SupportKernels(
        support_name=str(support_name),
        support_value=float(support_value),
        degree=degree_i,
        normalize_coords=normalized,
        offsets_xy=np.ascontiguousarray(coords, dtype=np.float64),
        weights_x=np.ascontiguousarray(weights_x, dtype=np.float64),
        weights_y=np.ascontiguousarray(weights_y, dtype=np.float64),
        kernel_x=np.ascontiguousarray(_dense_kernel(coords, weights_x, int(kernel_half_extent)), dtype=np.float64),
        kernel_y=np.ascontiguousarray(_dense_kernel(coords, weights_y, int(kernel_half_extent)), dtype=np.float64),
        support_cardinality=int(coords.shape[0]),
        design_matrix_shape=(int(design.shape[0]), int(design.shape[1])),
        kappa_design_matrix=design_condition_number(design),
    )


def build_square_support_kernels(
    half_side: float,
    degree: int = 3,
    normalize_coords: bool = False,
) -> SquareSupportKernels:
    h = float(half_side)
    return build_support_kernels(
        offsets_xy=square_offsets(h, include_center=False),
        support_name="square",
        support_value=h,
        degree=int(degree),
        normalize_coords=bool(normalize_coords),
        kernel_half_extent=int(math.ceil(h)),
    )


def build_polygon_support_kernels(
    name: str,
    vertices_xy: np.ndarray,
    bounding_radius: float,
    degree: int = 3,
    normalize_coords: bool = False,
) -> SupportKernels:
    radius = float(bounding_radius)
    half_extent = int(math.ceil(radius))
    offsets_xy = convex_polygon_offsets(vertices_xy, bounding_radius=half_extent, include_center=False)
    return build_support_kernels(
        offsets_xy=offsets_xy,
        support_name=str(name),
        support_value=radius,
        degree=int(degree),
        normalize_coords=bool(normalize_coords),
        kernel_half_extent=half_extent,
    )
