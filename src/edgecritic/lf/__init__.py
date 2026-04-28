"""Line Filter with automatic backend selection."""

from __future__ import annotations

import numpy as np

from edgecritic._types import EdgeResult
from edgecritic.lf._cpu import lf_image as _lf_cpu
from edgecritic.lf._cpu import line_filter_single_pixel
from edgecritic.lf._metal import lf_orientation_stack_metal


def _select_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
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
    device: str = "cuda",
    max_vram_gb: float | None = None,
) -> EdgeResult:
    """Apply the Line Filter to an image.

    Automatically selects GPU or CPU backend based on availability.

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
        'auto', 'cpu', or 'cuda'.
    device : str
        PyTorch device (only used when backend='cuda').
    max_vram_gb : float, optional
        Maximum VRAM budget in GB for the CUDA backend. If None,
        auto-detects from free device memory. Pixels are processed
        in batches sized to fit within this budget.

    Returns
    -------
    EdgeResult
        Supports tuple unpacking: ``mag, angle, extra = lf_image(...)``.
    """
    chosen = _select_backend(backend)

    if chosen == "cuda":
        from edgecritic.lf._cuda import lf_image_cuda

        mag, angle, cond = lf_image_cuda(
            image, half_width=half_width, np_count=np_count,
            order=order, n_orientations=n_orientations, device=device,
            max_vram_gb=max_vram_gb,
        )
        return EdgeResult(
            gradient_mag=mag,
            gradient_angle=angle,
            condition_numbers=cond,
            backend="cuda",
        )
    else:
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


__all__ = ["lf_image", "lf_orientation_stack_metal", "line_filter_single_pixel"]
