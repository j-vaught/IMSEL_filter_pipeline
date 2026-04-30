"""Line Filter CPU reference and Metal response-stack helpers."""

from __future__ import annotations

import numpy as np

from edgecritic._types import EdgeResult
from edgecritic.lf._cpu import lf_image as _lf_cpu
from edgecritic.lf._cpu import line_filter_single_pixel
from edgecritic.lf._metal import lf_length_stack, lf_response, lf_response_batch, lf_stack


def _select_backend(backend: str) -> str:
    if backend not in {"auto", "cpu"}:
        raise ValueError("lf_image supports backend='auto' or backend='cpu'")
    return "cpu"


def lf_image(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    use_weights: bool = True,
    subsample: int = 1,
    backend: str = "auto",
    device: str = "cpu",
    max_vram_gb: float | None = None,
) -> EdgeResult:
    """Apply the Line Filter to an image.

    This high-level orientation sweep is the raw Python reference path.
    Metal LF response and stack helpers are exposed as separate functions.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    half_width : int
        Half-width m of the line filter. Total length = 2m+1.
    np_count : int
        Number of neighbor pixels per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sweep.
    use_weights : bool
        Gaussian weighting (CPU only).
    subsample : int
        Process every Nth pixel (CPU only).
    backend : str
        'auto' or 'cpu'.
    device : str
        Reserved for API compatibility.
    max_vram_gb : float, optional
        Reserved for API compatibility.

    Returns
    -------
    EdgeResult
        Supports tuple unpacking: ``mag, angle, extra = lf_image(...)``.
    """
    _select_backend(backend)
    _ = device, max_vram_gb
    mag, angle, all_grads = _lf_cpu(
        image, half_width=half_width, np_count=np_count,
        order=order, n_orientations=n_orientations,
        use_weights=use_weights, subsample=subsample,
    )
    return EdgeResult(
        gradient_mag=mag,
        gradient_angle=angle,
        all_gradients=all_grads,
        backend="cpu",
    )


__all__ = [
    "lf_image",
    "lf_length_stack",
    "lf_response",
    "lf_response_batch",
    "lf_stack",
    "line_filter_single_pixel",
]
