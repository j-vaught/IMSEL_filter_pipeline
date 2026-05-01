"""Standalone radius-defined WVF implementation with an Apple Metal backend."""

from __future__ import annotations

from .metal import (
    MetalBackendError,
    metal_backend_available,
    wvf_gradients_metal,
    wvf_magnitude_angle_metal,
    wvf_radius_gradients_metal,
)
from .radius import (
    WVFAntipodalKernels,
    WVFRadiusKernels,
    build_wvf_antipodal_kernels,
    build_wvf_radius_kernels,
    disk_offsets,
    wvf_radius_gradients_cpu,
)

__all__ = [
    "MetalBackendError",
    "WVFAntipodalKernels",
    "WVFRadiusKernels",
    "build_wvf_antipodal_kernels",
    "build_wvf_radius_kernels",
    "disk_offsets",
    "metal_backend_available",
    "wvf_gradients_metal",
    "wvf_magnitude_angle_metal",
    "wvf_radius_gradients_cpu",
    "wvf_radius_gradients_metal",
]
