"""Radius-only WVF derivative component API."""

from __future__ import annotations

import platform

import numpy as np

from edgecritic.wvf.radius import build_wvf_radius_kernels, wvf_radius_gradients_cpu


def _select_component_backend(backend: str) -> str:
    requested = str(backend).lower()
    if requested not in {"auto", "cpu", "metal"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
    if requested != "auto":
        return requested
    if platform.system() == "Darwin":
        try:
            from edgecritic.wvf.metal import metal_backend_available

            if metal_backend_available():
                return "metal"
        except Exception:
            pass
    return "cpu"


def wvf_component_gradients(
    image: np.ndarray,
    radius: int,
    order: int = 4,
    backend: str = "auto",
    mode: str = "reflect",
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic WVF ``Gx`` and ``Gy`` from an integer disk radius.

    This is the radius-only API intended for the cubic-spline LF pipeline.
    ``radius`` defines the complete integer disk support. No ``Np`` value is
    accepted or inferred.
    """
    kernels = build_wvf_radius_kernels(radius=radius, order=order)
    chosen = _select_component_backend(backend)

    if chosen == "metal":
        if mode != "reflect":
            raise ValueError("Metal backend currently supports mode='reflect' only")
        from edgecritic.wvf.metal import wvf_radius_gradients_metal

        return wvf_radius_gradients_metal(image, kernels, output_dtype=output_dtype)

    return wvf_radius_gradients_cpu(
        image,
        radius=radius,
        order=order,
        mode=mode,
        output_dtype=output_dtype,
    )


def wvf_component_backend(backend: str = "auto") -> str:
    """Return the concrete backend that would be used for WVF components."""
    return _select_component_backend(backend)
