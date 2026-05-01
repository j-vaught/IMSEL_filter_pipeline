"""Public-facing API for the standalone WVF package."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from .metal import (
    MetalBackendError,
    backend_info,
    metal_backend_available,
    wvf_gradients_metal,
    wvf_magnitude_angle_metal,
    wvf_magnitude_metal,
    wvf_magnitude_orientation_metal,
)

__all__ = [
    "Components",
    "FftBackend",
    "Gradients",
    "MagnitudeOrientation",
    "MetalBackendError",
    "Variant",
    "backend_info",
    "components",
    "gradients",
    "magnitude",
    "magnitude_orientation",
    "metal_backend_available",
]

Variant = Literal[
    "direct",
    "antipodal",
    "split",
    "fft",
]
FftBackend = Literal["auto", "cpu", "vkfft"]


class Gradients(NamedTuple):
    gx: np.ndarray
    gy: np.ndarray


class MagnitudeOrientation(NamedTuple):
    magnitude: np.ndarray
    angle: np.ndarray


class Components(NamedTuple):
    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray
    angle: np.ndarray


def gradients(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: Variant = "split",
    fft_backend: FftBackend | None = "auto",
    device_index: int | None = None,
) -> Gradients:
    """Return WVF ``gx`` and ``gy``."""
    return Gradients(
        *wvf_gradients_metal(
            image,
            radius=radius,
            degree=degree,
            variant=variant,
            fft_backend=fft_backend,
            device_index=device_index,
        )
    )


def magnitude(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: Variant = "split",
    fft_backend: FftBackend | None = "auto",
    device_index: int | None = None,
) -> np.ndarray:
    """Return WVF magnitude."""
    return wvf_magnitude_metal(
        image,
        radius=radius,
        degree=degree,
        variant=variant,
        fft_backend=fft_backend,
        device_index=device_index,
    )


def magnitude_orientation(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: Variant = "split",
    fft_backend: FftBackend | None = "auto",
    device_index: int | None = None,
) -> MagnitudeOrientation:
    """Return WVF magnitude and angle."""
    return MagnitudeOrientation(
        *wvf_magnitude_orientation_metal(
            image,
            radius=radius,
            degree=degree,
            variant=variant,
            fft_backend=fft_backend,
            device_index=device_index,
        )
    )


def components(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: Variant = "split",
    fft_backend: FftBackend | None = "auto",
    device_index: int | None = None,
) -> Components:
    """Return WVF ``gx``, ``gy``, magnitude, and angle."""
    return Components(
        *wvf_magnitude_angle_metal(
            image,
            radius=radius,
            degree=degree,
            variant=variant,
            fft_backend=fft_backend,
            device_index=device_index,
        )
    )
