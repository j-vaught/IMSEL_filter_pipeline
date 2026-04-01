"""Anisotropic LF via PyTorch F.conv2d (cuDNN backend).

Embeds each orientation's fused stencil into a dense 2D kernel, then
applies all orientations as a batched convolution. Simple, leverages
cuDNN's optimized implementation, good baseline for comparison.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from edgecritic.aniso_lf._stencil import (
    build_anisotropic_stencils,
    stencils_to_dense_kernels,
)

logger = logging.getLogger("edgecritic")


def aniso_lf_conv2d(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Anisotropic LF using dense conv2d kernels.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    half_width : int
        LF half-width m.
    np_count : int
        Neighbor count per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations.
    neighbor_type : str
        "circular" or "square".
    device : str
        PyTorch device.
    dtype : torch.dtype
        Computation dtype (float32 or float16 for FP16 experiment).

    Returns
    -------
    gradient_mag : ndarray, (H, W)
    gradient_angle : ndarray, (H, W)
    timing : dict with timing breakdown
    """
    device = torch.device(device)
    timing = {}

    t0 = time.perf_counter()

    # Phase 1: Build stencils on CPU
    stencils = build_anisotropic_stencils(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )
    kernels_np, border = stencils_to_dense_kernels(stencils)

    t1 = time.perf_counter()
    timing["stencil_build"] = t1 - t0

    # Phase 2: Transfer to GPU
    # kernels: (N_s, kh, kw) -> (N_s, 1, kh, kw) for grouped conv
    kernels_t = torch.tensor(kernels_np, dtype=dtype, device=device).unsqueeze(1)
    img_t = torch.tensor(image, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    # (1, 1, H, W)

    t2 = time.perf_counter()
    timing["transfer"] = t2 - t1

    # Phase 3: Apply all orientations via conv2d
    # Expand image to N_s channels for grouped convolution
    img_expanded = img_t.expand(-1, n_orientations, -1, -1)
    # grouped conv: each of the N_s kernels convolves with its own channel
    # But all channels are the same image, so this is equivalent to
    # applying each kernel independently
    responses = F.conv2d(
        img_expanded,
        kernels_t,
        padding=(kernels_np.shape[1] // 2, kernels_np.shape[2] // 2),
        groups=n_orientations,
    )
    # responses: (1, N_s, H, W)

    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    timing["conv2d"] = t3 - t2

    # Phase 4: Find best orientation per pixel
    abs_responses = torch.abs(responses.squeeze(0))  # (N_s, H, W)
    best_mag, best_idx = abs_responses.max(dim=0)    # (H, W) each

    t4 = time.perf_counter()
    timing["reduce"] = t4 - t3

    # Phase 5: Transfer back
    gradient_mag = best_mag.float().cpu().numpy().astype(np.float64)
    gradient_angle_idx = best_idx.cpu().numpy()
    angles = stencils["angles"]
    gradient_angle = angles[gradient_angle_idx]

    t5 = time.perf_counter()
    timing["transfer_back"] = t5 - t4
    timing["total"] = t5 - t0
    timing["stencil_stats"] = stencils["stats"]
    timing["kernel_shape"] = list(kernels_np.shape)

    logger.debug(
        "aniso_lf_conv2d: stencil=%.3fs conv2d=%.3fs reduce=%.3fs "
        "total=%.3fs kernel=%s dedup=%.1f%%",
        timing["stencil_build"], timing["conv2d"], timing["reduce"],
        timing["total"], timing["kernel_shape"],
        timing["stencil_stats"]["dedup_ratio"] * 100,
    )

    return gradient_mag, gradient_angle, timing
