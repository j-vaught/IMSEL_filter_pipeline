"""Precompute fused anisotropic stencils for the Line Filter.

For each orientation theta_k, the LF response is:
    R_k = sum_{j=-m}^{m} w_j * p_fx^{(k)} . f_j

where f_j are neighbor intensities around virtual pixel j. By expanding
the pixel positions, this becomes a single weighted sum over a fixed
stencil of (2m+1)*Np positions (with deduplication of overlapping pixels).

The stencil weights alpha_{k,ell} = sum of w_j * p_i^{(k)} for all (j,i)
pairs that map to the same pixel offset after rounding.
"""

import numpy as np

from edgecritic.core.taylor import (
    build_taylor_matrix,
    get_circular_neighbors,
    get_square_neighbors,
    rotate_coordinates,
)


def _compute_pfx_row(neighbors, theta, order):
    """Compute row 1 of the pseudoinverse (the f_x extraction row)."""
    local = rotate_coordinates(neighbors, theta)
    A = build_taylor_matrix(local, order=order)
    P = np.linalg.pinv(A)
    return P[1, :]  # f_x row


def build_anisotropic_stencils(
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
) -> dict:
    """Build deduplicated stencils for all orientations.

    Parameters
    ----------
    half_width : int
        Half-width m of the line filter.
    np_count : int
        Number of neighbor pixels per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sweep over [0, 2*pi).
    neighbor_type : str
        "circular" or "square" neighbor selection.

    Returns
    -------
    dict with keys:
        offsets : list of ndarray, each (N'_k, 2) int offsets
        weights : list of ndarray, each (N'_k,) float weights
        angles : ndarray, (n_orientations,) angles in radians
        stats : dict with deduplication statistics
    """
    if neighbor_type == "square":
        neighbors = get_square_neighbors(np_count)
    else:
        neighbors = get_circular_neighbors(np_count)

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    sigma = half_width / 2.0
    j_range = np.arange(-half_width, half_width + 1, dtype=np.float64)
    gauss_weights = np.exp(-0.5 * (j_range / sigma) ** 2)
    gauss_weights /= gauss_weights.sum()

    L = 2 * half_width + 1
    raw_count = L * np_count

    all_offsets = []
    all_weights = []
    unique_counts = []

    for theta in angles:
        pfx = _compute_pfx_row(neighbors, theta, order)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Build all (j, i) -> (dy, dx) mappings and their weights
        stencil_map = {}
        for ji, j in enumerate(range(-half_width, half_width + 1)):
            vdx = int(round(j * cos_t))
            vdy = int(round(j * sin_t))
            w_j = gauss_weights[ji]

            for ni in range(np_count):
                ndx = int(neighbors[ni, 0])
                ndy = int(neighbors[ni, 1])
                key = (vdy + ndy, vdx + ndx)  # (row_offset, col_offset)
                alpha = w_j * pfx[ni]
                if key in stencil_map:
                    stencil_map[key] += alpha
                else:
                    stencil_map[key] = alpha

        # Convert to arrays
        offsets_k = np.array(list(stencil_map.keys()), dtype=np.int32)
        weights_k = np.array(list(stencil_map.values()), dtype=np.float64)

        all_offsets.append(offsets_k)
        all_weights.append(weights_k)
        unique_counts.append(len(stencil_map))

    stats = {
        "raw_stencil_size": raw_count,
        "unique_counts": unique_counts,
        "mean_unique": float(np.mean(unique_counts)),
        "min_unique": int(np.min(unique_counts)),
        "max_unique": int(np.max(unique_counts)),
        "dedup_ratio": float(np.mean(unique_counts) / raw_count),
    }

    return {
        "offsets": all_offsets,
        "weights": all_weights,
        "angles": angles,
        "stats": stats,
    }


def stencils_to_dense_kernels(stencils: dict) -> tuple:
    """Convert sparse stencils to dense 2D kernels for conv2d.

    Returns
    -------
    kernels : ndarray, (n_orientations, kernel_h, kernel_w)
        Dense kernel array (zero-padded to common size).
    border : int
        Required image border (max offset magnitude).
    """
    all_offsets = stencils["offsets"]
    all_weights = stencils["weights"]
    n_orient = len(all_offsets)

    # Find global bounding box
    max_dy = max_dx = 0
    for offsets in all_offsets:
        if len(offsets) > 0:
            max_dy = max(max_dy, int(np.max(np.abs(offsets[:, 0]))))
            max_dx = max(max_dx, int(np.max(np.abs(offsets[:, 1]))))

    kh = 2 * max_dy + 1
    kw = 2 * max_dx + 1
    border = max(max_dy, max_dx)

    kernels = np.zeros((n_orient, kh, kw), dtype=np.float64)
    for ki in range(n_orient):
        for idx in range(len(all_offsets[ki])):
            dy, dx = all_offsets[ki][idx]
            kernels[ki, dy + max_dy, dx + max_dx] = all_weights[ki][idx]

    return kernels, border


def stencils_to_sparse_tensors(stencils: dict) -> tuple:
    """Convert sparse stencils to padded arrays for the Triton kernel.

    Returns
    -------
    offsets_padded : ndarray, (n_orientations, N_max, 2) int32
    weights_padded : ndarray, (n_orientations, N_max) float32
    n_unique : ndarray, (n_orientations,) int32
    border : int
    """
    all_offsets = stencils["offsets"]
    all_weights = stencils["weights"]
    n_orient = len(all_offsets)
    N_max = max(len(o) for o in all_offsets)

    offsets_padded = np.zeros((n_orient, N_max, 2), dtype=np.int32)
    weights_padded = np.zeros((n_orient, N_max), dtype=np.float32)
    n_unique = np.zeros(n_orient, dtype=np.int32)

    max_abs = 0
    for ki in range(n_orient):
        nk = len(all_offsets[ki])
        n_unique[ki] = nk
        offsets_padded[ki, :nk, :] = all_offsets[ki]
        weights_padded[ki, :nk] = all_weights[ki].astype(np.float32)
        if nk > 0:
            max_abs = max(max_abs, int(np.max(np.abs(all_offsets[ki]))))

    return offsets_padded, weights_padded, n_unique, max_abs
