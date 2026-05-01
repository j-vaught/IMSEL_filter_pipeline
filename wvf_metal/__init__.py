"""Standalone radius-defined WVF with native GPU and CPU FFT backends."""

from __future__ import annotations

from .api import (
    Components,
    FftBackend,
    Gradients,
    MagnitudeOrientation,
    Variant,
    backend_info,
    components,
    gradients,
    magnitude,
    magnitude_orientation,
)
from .metal import (
    FFT_BACKEND_NAMES,
    VARIANT_NAMES,
    MetalBackendError,
    metal_backend_available,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Components",
    "FFT_BACKEND_NAMES",
    "FftBackend",
    "Gradients",
    "MagnitudeOrientation",
    "MetalBackendError",
    "VARIANT_NAMES",
    "Variant",
    "backend_info",
    "components",
    "gradients",
    "magnitude",
    "magnitude_orientation",
    "metal_backend_available",
]
