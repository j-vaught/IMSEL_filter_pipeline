"""Portable CPU FFT fallback for standalone WVF."""

from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np


def _taylor_exponents(degree: int) -> list[tuple[int, int]]:
    exponents = [(0, 0)]
    if degree >= 1:
        exponents.extend(((1, 0), (0, 1)))
    for total_degree in range(2, degree + 1):
        exponents.append((total_degree, 0))
        exponents.append((0, total_degree))
        for px in range(total_degree - 1, 0, -1):
            exponents.append((px, total_degree - px))
    return exponents


def _disk_offsets(radius: int) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be positive")
    offsets: list[tuple[int, int]] = []
    radius2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius2:
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=np.int32)


@lru_cache(maxsize=64)
def dense_kernels(radius: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    if degree < 1:
        raise ValueError("degree must be at least 1 for WVF gradients")

    offsets = _disk_offsets(radius)
    exponents = _taylor_exponents(degree)
    design = np.empty((offsets.shape[0], len(exponents)), dtype=np.float64)
    for row, (dx, dy) in enumerate(offsets):
        x = float(dx)
        y = float(dy)
        for col, (px, py) in enumerate(exponents):
            scale = factorial(px) * factorial(py)
            design[row, col] = (x**px) * (y**py) / scale

    if design.shape[0] < design.shape[1]:
        raise ValueError(
            f"radius {radius} gives {design.shape[0]} samples, fewer than "
            f"{design.shape[1]} Taylor coefficients for degree {degree}"
        )

    ata = design.T @ design
    inv = np.linalg.inv(ata)
    pseudo = inv @ design.T
    weights_x = pseudo[1]
    weights_y = pseudo[2]

    kernel_width = 2 * radius + 1
    kernel_x = np.zeros((kernel_width, kernel_width), dtype=np.float32)
    kernel_y = np.zeros((kernel_width, kernel_width), dtype=np.float32)
    for index, (dx, dy) in enumerate(offsets):
        kernel_x[radius - dy, radius - dx] += np.float32(weights_x[index])
        kernel_y[radius - dy, radius - dx] += np.float32(weights_y[index])
    return kernel_x, kernel_y


def _is_smooth_fft_size(value: int) -> bool:
    if value < 2:
        return False
    for factor in (2, 3, 5, 7):
        while value % factor == 0:
            value //= factor
    return value == 1


def _next_smooth_fft_size(value: int) -> int:
    value = max(2, int(value))
    while not _is_smooth_fft_size(value):
        value += 1
    return value


def _validate_image(image: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("WVF CPU FFT expects a 2-D image")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("image width and height must be positive")
    return img


def wvf_fft_magnitude_angle_cpu(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute WVF components through a portable CPU FFT fallback."""
    img = _validate_image(image)
    radius = int(radius)
    degree = int(degree)
    if radius < 1:
        raise ValueError("radius must be positive")
    if degree < 1:
        raise ValueError("degree must be at least 1")

    height, width = img.shape
    padded_h = height + 2 * radius
    padded_w = width + 2 * radius
    fft_h = _next_smooth_fft_size(height + 4 * radius)
    fft_w = _next_smooth_fft_size(width + 4 * radius)

    kernel_x, kernel_y = dense_kernels(radius, degree)
    kernel_width = kernel_x.shape[0]

    image_plane = np.zeros((fft_h, fft_w), dtype=np.float32)
    image_plane[:padded_h, :padded_w] = np.pad(
        img,
        ((radius, radius), (radius, radius)),
        mode="symmetric",
    )

    kernel_plane_x = np.zeros((fft_h, fft_w), dtype=np.float32)
    kernel_plane_y = np.zeros((fft_h, fft_w), dtype=np.float32)
    kernel_plane_x[:kernel_width, :kernel_width] = kernel_x
    kernel_plane_y[:kernel_width, :kernel_width] = kernel_y

    image_spec = np.fft.rfft2(image_plane)
    gx_plane = np.fft.irfft2(image_spec * np.fft.rfft2(kernel_plane_x), s=image_plane.shape)
    gy_plane = np.fft.irfft2(image_spec * np.fft.rfft2(kernel_plane_y), s=image_plane.shape)

    crop = 2 * radius
    gx = np.ascontiguousarray(gx_plane[crop : crop + height, crop : crop + width], dtype=np.float32)
    gy = np.ascontiguousarray(gy_plane[crop : crop + height, crop : crop + width], dtype=np.float32)
    magnitude = np.ascontiguousarray(np.hypot(gx, gy), dtype=np.float32)
    angle = np.ascontiguousarray(np.mod(np.arctan2(gy, gx), np.pi), dtype=np.float32)
    return gx, gy, magnitude, angle
