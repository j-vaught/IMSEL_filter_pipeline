"""Wide View Filter with automatic backend selection."""

from __future__ import annotations

import numpy as np

from edgecritic._types import EdgeResult
from edgecritic.wvf._components import wvf_component_backend, wvf_component_gradients
from edgecritic.wvf._cpu import wvf_image as _wvf_cpu
from edgecritic.wvf._cpu import wvf_single_pixel
from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels, disk_offsets


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


def wvf_image(
    image: np.ndarray,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    backend: str = "auto",
    device: str = "cuda",
    pixel_batch_size: int | None = None,
) -> EdgeResult:
    """Apply the Wide View Filter to an image.

    Automatically selects GPU or CPU backend based on availability.

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
        'auto', 'cpu', or 'cuda'.
    device : str
        PyTorch device (only used when backend='cuda').
    pixel_batch_size : int, optional
        If set and using CUDA, process pixels in batches to limit VRAM.

    Returns
    -------
    EdgeResult
        Supports tuple unpacking: ``mag, angle, cond = wvf_image(...)``.
    """
    chosen = _select_backend(backend)

    if chosen == "cuda":
        from edgecritic.wvf._cuda import wvf_image_cuda, wvf_image_cuda_batched

        if pixel_batch_size is not None:
            mag, angle, cond = wvf_image_cuda_batched(
                image, np_count=np_count, order=order,
                n_orientations=n_orientations, device=device,
                pixel_batch_size=pixel_batch_size,
            )
        else:
            mag, angle, cond = wvf_image_cuda(
                image, np_count=np_count, order=order,
                n_orientations=n_orientations, device=device,
            )
        return EdgeResult(
            gradient_mag=mag,
            gradient_angle=angle,
            condition_numbers=cond,
            backend="cuda",
        )
    else:
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
    "wvf_component_backend",
    "wvf_component_gradients",
    "wvf_image",
    "wvf_single_pixel",
]
