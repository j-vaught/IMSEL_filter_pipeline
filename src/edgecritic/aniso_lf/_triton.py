"""Anisotropic LF via custom Triton kernel.

Uses sparse stencil representation: for each orientation, a list of
(dy, dx) offsets and corresponding weights. The kernel loads image
values at the stencil positions, multiplies by weights, sums, and
tracks the best orientation per pixel.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from edgecritic.aniso_lf._stencil import (
    build_anisotropic_stencils,
    stencils_to_sparse_tensors,
)

logger = logging.getLogger("edgecritic")

_triton_available = False
try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    pass


if _triton_available:
    @triton.jit
    def _aniso_lf_kernel(
        image_ptr,
        offsets_ptr,
        weights_ptr,
        n_unique_ptr,
        out_mag_ptr,
        out_idx_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        Ns: tl.constexpr,
        N_max: tl.constexpr,
        border: tl.constexpr,
        BLOCK_X: tl.constexpr,
    ):
        """One program instance handles BLOCK_X pixels in a row."""
        row = tl.program_id(0) + border
        col_base = tl.program_id(1) * BLOCK_X + border
        cols = col_base + tl.arange(0, BLOCK_X)
        mask = (cols < W - border) & (row < H - border)

        best_mag = tl.zeros([BLOCK_X], dtype=tl.float32)
        best_idx = tl.zeros([BLOCK_X], dtype=tl.int32)

        for k in range(Ns):
            nk = tl.load(n_unique_ptr + k)
            response = tl.zeros([BLOCK_X], dtype=tl.float32)

            for i in range(N_max):
                active = i < nk
                # Load offset (dy, dx) for this orientation's stencil point
                off_base = k * N_max * 2 + i * 2
                dy = tl.load(offsets_ptr + off_base, mask=active, other=0)
                dx = tl.load(offsets_ptr + off_base + 1, mask=active, other=0)
                w = tl.load(weights_ptr + k * N_max + i, mask=active, other=0.0)

                # Gather image values at (row + dy, cols + dx)
                img_idx = (row + dy) * W + (cols + dx)
                vals = tl.load(image_ptr + img_idx, mask=mask & active, other=0.0)
                response += w * vals

            abs_r = tl.abs(response)
            improved = abs_r > best_mag
            best_mag = tl.where(improved, abs_r, best_mag)
            best_idx = tl.where(improved, k, best_idx)

        # Write results
        out_base = row * W + cols
        tl.store(out_mag_ptr + out_base, best_mag, mask=mask)
        tl.store(out_idx_ptr + out_base, best_idx, mask=mask)


def aniso_lf_triton(
    image: np.ndarray,
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Anisotropic LF using a custom Triton kernel.

    Parameters
    ----------
    image : ndarray, shape (H, W)
    half_width, np_count, order, n_orientations : int
    neighbor_type : str
    device : str

    Returns
    -------
    gradient_mag : ndarray, (H, W)
    gradient_angle : ndarray, (H, W)
    timing : dict
    """
    if not _triton_available:
        raise RuntimeError("Triton is not installed. pip install triton")

    device = torch.device(device)
    timing = {}
    H, W = image.shape

    t0 = time.perf_counter()

    # Phase 1: Build stencils
    stencils = build_anisotropic_stencils(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )
    offsets_padded, weights_padded, n_unique, border = stencils_to_sparse_tensors(
        stencils
    )

    t1 = time.perf_counter()
    timing["stencil_build"] = t1 - t0

    # Phase 2: Transfer to GPU
    img_t = torch.tensor(image, dtype=torch.float32, device=device).contiguous()
    offsets_t = torch.tensor(offsets_padded, dtype=torch.int32, device=device).contiguous()
    weights_t = torch.tensor(weights_padded, dtype=torch.float32, device=device).contiguous()
    n_unique_t = torch.tensor(n_unique, dtype=torch.int32, device=device).contiguous()

    out_mag = torch.zeros(H, W, dtype=torch.float32, device=device)
    out_idx = torch.zeros(H, W, dtype=torch.int32, device=device)

    t2 = time.perf_counter()
    timing["transfer"] = t2 - t1

    # Phase 3: Launch kernel
    N_max = int(offsets_padded.shape[1])
    BLOCK_X = 128
    n_rows = H - 2 * border
    n_col_blocks = (W - 2 * border + BLOCK_X - 1) // BLOCK_X

    if n_rows <= 0 or n_col_blocks <= 0:
        timing["kernel"] = 0.0
        timing["total"] = time.perf_counter() - t0
        return (
            np.zeros((H, W), dtype=np.float64),
            np.zeros((H, W), dtype=np.float64),
            timing,
        )

    grid = (n_rows, n_col_blocks)

    _aniso_lf_kernel[grid](
        img_t,
        offsets_t,
        weights_t,
        n_unique_t,
        out_mag,
        out_idx,
        H=H,
        W=W,
        Ns=n_orientations,
        N_max=N_max,
        border=border,
        BLOCK_X=BLOCK_X,
    )
    torch.cuda.synchronize(device)

    t3 = time.perf_counter()
    timing["kernel"] = t3 - t2

    # Phase 4: Transfer back
    gradient_mag = out_mag.cpu().numpy().astype(np.float64)
    gradient_angle_idx = out_idx.cpu().numpy()
    angles = stencils["angles"]
    gradient_angle = np.zeros((H, W), dtype=np.float64)
    valid = gradient_mag > 0
    gradient_angle[valid] = angles[gradient_angle_idx[valid]]

    t4 = time.perf_counter()
    timing["transfer_back"] = t4 - t3
    timing["total"] = t4 - t0
    timing["stencil_stats"] = stencils["stats"]
    timing["N_max"] = N_max
    timing["border"] = border

    logger.debug(
        "aniso_lf_triton: stencil=%.3fs kernel=%.3fs total=%.3fs "
        "N_max=%d border=%d dedup=%.1f%%",
        timing["stencil_build"], timing["kernel"], timing["total"],
        N_max, border, timing["stencil_stats"]["dedup_ratio"] * 100,
    )

    return gradient_mag, gradient_angle, timing
