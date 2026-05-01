"""Wide View Filter CPU reference and Metal component helpers."""

from __future__ import annotations

import numpy as np

from core.types import EdgeResult
from wvf.components import wvf_component_backend, wvf_component_gradients
from wvf.reference import wvf_image as _wvf_cpu
from wvf.reference import wvf_single_pixel
from wvf.radius import (
    WVFRadiusKernels,
    build_wvf_radius_kernels,
    disk_offsets,
    wvf_radius_gradients_cpu,
)
from wvf.metal import wvf_gradients_metal, wvf_magnitude_angle_metal


def _select_backend(backend: str) -> str:
    if backend not in {"auto", "cpu"}:
        raise ValueError("wvf_image supports backend='auto' or backend='cpu'")
    return "cpu"


def wvf_image(
    image: np.ndarray,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    backend: str = "auto",
    device: str = "cpu",
    pixel_batch_size: int | None = None,
) -> EdgeResult:
    """Apply the Wide View Filter to an image.

    This high-level orientation sweep is the raw Python reference path.
    Metal support for radius-defined gradient components is exposed through
    ``wvf_component_gradients``.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    np_count : int
        Number of neighbor pixels.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sweep.
    backend : str
        'auto' or 'cpu'.
    device : str
        Reserved for API compatibility.
    pixel_batch_size : int, optional
        Reserved for API compatibility.

    Returns
    -------
    EdgeResult
        Supports tuple unpacking: ``mag, angle, cond = wvf_image(...)``.
    """
    _select_backend(backend)
    _ = device, pixel_batch_size
    mag, angle, cond = _wvf_cpu(
        image, np_count=np_count, order=order,
        n_orientations=n_orientations,
    )
    return EdgeResult(
        gradient_mag=mag,
        gradient_angle=angle,
        condition_numbers=cond,
        backend="cpu",
    )


__all__ = [
    "build_wvf_radius_kernels",
    "disk_offsets",
    "WVFRadiusKernels",
    "wvf_component_backend",
    "wvf_component_gradients",
    "wvf_gradients_metal",
    "wvf_image",
    "wvf_magnitude_angle_metal",
    "wvf_radius_gradients_cpu",
    "wvf_single_pixel",
]
