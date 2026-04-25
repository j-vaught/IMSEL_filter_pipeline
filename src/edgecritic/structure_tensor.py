"""Disk- and Gaussian-weighted structure tensor orientation estimates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from edgecritic.core.taylor import (
    build_taylor_matrix,
    get_circular_neighbors,
    get_square_neighbors,
)


@dataclass(frozen=True)
class StructureTensorResult:
    """Closed-form orientation and magnitude readouts from a structure tensor."""

    orientation: np.ndarray
    """Dominant gradient orientation in radians, mapped to ``[0, pi)``."""

    magnitude_a: np.ndarray
    """Gradient-energy magnitude, ``sqrt(lambda_1)``."""

    magnitude_b: np.ndarray
    """Anisotropy-penalized magnitude, ``sqrt(lambda_1 - lambda_2)``."""

    coherence: np.ndarray
    """Anisotropy score, ``(lambda_1 - lambda_2) / (lambda_1 + lambda_2)``."""

    energy: np.ndarray
    """Total local gradient energy, ``lambda_1 + lambda_2``."""

    lambda1: np.ndarray
    """Larger structure-tensor eigenvalue."""

    lambda2: np.ndarray
    """Smaller structure-tensor eigenvalue."""

    gx: np.ndarray
    """Canonical x derivative produced by the WVF/Taylor fit."""

    gy: np.ndarray
    """Canonical y derivative produced by the WVF/Taylor fit."""

    def __iter__(self):
        """Allow ``orientation, mag_a, mag_b, coherence = result`` unpacking."""
        yield self.orientation
        yield self.magnitude_a
        yield self.magnitude_b
        yield self.coherence


@lru_cache(maxsize=64)
def _derivative_kernels(
    np_count: int,
    order: int,
    neighbor_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical WVF/Taylor derivative kernels for ``Gx`` and ``Gy``."""
    if neighbor_type == "circular":
        neighbors = get_circular_neighbors(np_count)
    elif neighbor_type == "square":
        neighbors = get_square_neighbors(np_count)
    else:
        raise ValueError("neighbor_type must be 'circular' or 'square'")

    design = build_taylor_matrix(neighbors, order=order)
    pinv = np.linalg.pinv(design)
    weights_x = pinv[1, :]
    weights_y = pinv[2, :]

    max_dx = int(np.max(np.abs(neighbors[:, 0])))
    max_dy = int(np.max(np.abs(neighbors[:, 1])))
    kernel_x = np.zeros((2 * max_dy + 1, 2 * max_dx + 1), dtype=np.float64)
    kernel_y = np.zeros_like(kernel_x)

    for index, (dx, dy) in enumerate(neighbors.astype(np.int64)):
        kernel_x[dy + max_dy, dx + max_dx] += weights_x[index]
        kernel_y[dy + max_dy, dx + max_dx] += weights_y[index]

    return kernel_x, kernel_y


@lru_cache(maxsize=64)
def disk_kernel(radius: int) -> np.ndarray:
    """Return a normalized hard disk kernel with integer radius ``radius``."""
    if radius < 1:
        raise ValueError("radius must be at least 1")
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = (xx * xx + yy * yy) <= float(radius * radius)
    weights = kernel.astype(np.float64)
    weights /= np.sum(weights)
    return weights


def wvf_component_gradients(
    image: np.ndarray,
    np_count: int = 15,
    order: int = 4,
    neighbor_type: str = "circular",
    mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute canonical ``Gx`` and ``Gy`` from the WVF/Taylor derivative rows."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("wvf_component_gradients expects a 2-D image")

    kernel_x, kernel_y = _derivative_kernels(
        int(np_count),
        int(order),
        str(neighbor_type),
    )
    gx = ndimage.correlate(img, kernel_x, mode=mode)
    gy = ndimage.correlate(img, kernel_y, mode=mode)
    return gx, gy


def _integrate(
    values: np.ndarray,
    radius: int,
    weight_type: str,
    mode: str,
    gaussian_sigma: float | None,
) -> np.ndarray:
    if weight_type == "disk":
        return ndimage.convolve(values, disk_kernel(int(radius)), mode=mode)
    if weight_type == "gaussian":
        sigma = float(gaussian_sigma) if gaussian_sigma is not None else max(float(radius) / 2.0, 1e-12)
        return ndimage.gaussian_filter(values, sigma=sigma, mode=mode)
    raise ValueError("weight_type must be 'disk' or 'gaussian'")


def structure_tensor_from_gradients(
    gx: np.ndarray,
    gy: np.ndarray,
    radius: int,
    weight_type: str = "disk",
    mode: str = "reflect",
    gaussian_sigma: float | None = None,
    eps: float = 1e-10,
) -> StructureTensorResult:
    """Compute structure tensor orientation and readouts from derivative fields."""
    gx_arr = np.asarray(gx, dtype=np.float64)
    gy_arr = np.asarray(gy, dtype=np.float64)
    if gx_arr.shape != gy_arr.shape:
        raise ValueError("gx and gy must have matching shapes")

    jxx = _integrate(gx_arr * gx_arr, radius, weight_type, mode, gaussian_sigma)
    jyy = _integrate(gy_arr * gy_arr, radius, weight_type, mode, gaussian_sigma)
    jxy = _integrate(gx_arr * gy_arr, radius, weight_type, mode, gaussian_sigma)

    trace = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    lambda1 = np.maximum(0.5 * (trace + disc), 0.0)
    lambda2 = np.maximum(0.5 * (trace - disc), 0.0)

    orientation = (0.5 * np.arctan2(2.0 * jxy, jxx - jyy)) % np.pi
    magnitude_a = np.sqrt(lambda1)
    magnitude_b = np.sqrt(np.maximum(lambda1 - lambda2, 0.0))
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + float(eps))

    return StructureTensorResult(
        orientation=orientation,
        magnitude_a=magnitude_a,
        magnitude_b=magnitude_b,
        coherence=coherence,
        energy=trace,
        lambda1=lambda1,
        lambda2=lambda2,
        gx=gx_arr,
        gy=gy_arr,
    )


def structure_tensor_orientation(
    image: np.ndarray,
    radius: int,
    weight_type: str = "disk",
    np_count: int = 15,
    order: int = 4,
    neighbor_type: str = "circular",
    mode: str = "reflect",
    gaussian_sigma: float | None = None,
) -> StructureTensorResult:
    """Compute WVF-component structure tensor orientation and magnitudes.

    The returned object supports tuple unpacking as
    ``orientation, magnitude_a, magnitude_b, coherence = result``.
    """
    gx, gy = wvf_component_gradients(
        image,
        np_count=np_count,
        order=order,
        neighbor_type=neighbor_type,
        mode=mode,
    )
    return structure_tensor_from_gradients(
        gx,
        gy,
        radius=radius,
        weight_type=weight_type,
        mode=mode,
        gaussian_sigma=gaussian_sigma,
    )
