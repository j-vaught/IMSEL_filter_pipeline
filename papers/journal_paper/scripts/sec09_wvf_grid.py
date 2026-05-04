#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.taylor import build_taylor_matrix, default_pinv_rcond
from wvf.radius import disk_offsets


WVF_GRID_RADII: tuple[int, ...] = (2, 3, 5, 7, 9, 11, 13, 15, 18, 21, 25, 30)
WVF_GRID_DEGREES: tuple[int, ...] = (1, 3, 5, 7, 9, 11)


def wvf_conditioning_diagnostics(radius: int, degree: int, normalize_coords: bool = True) -> dict[str, object]:
    offsets = disk_offsets(int(radius), include_center=False)
    design = build_taylor_matrix(
        offsets,
        order=int(degree),
        normalize_radius=int(radius) if normalize_coords else None,
    )
    singular_values = np.linalg.svd(design, compute_uv=False, hermitian=False)
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    cutoff = float(default_pinv_rcond(design.shape, dtype=np.float64)) * sigma_max
    numerical_rank = int(np.count_nonzero(singular_values > cutoff))
    n_samples = int(design.shape[0])
    n_coeffs = int(design.shape[1])
    rank_deficient_count = int(max(0, n_coeffs - numerical_rank))
    return {
        "radius": int(radius),
        "degree": int(degree),
        "normalize_coords": bool(normalize_coords),
        "support_cardinality": int(offsets.shape[0]),
        "coefficient_count": int(n_coeffs),
        "sigma_max": float(sigma_max),
        "sigma_min": float(sigma_min),
        "kappa_design_matrix": float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf"),
        "rank_deficient_count": int(rank_deficient_count),
        "status": "ok" if (n_samples >= n_coeffs and rank_deficient_count == 0) else (
            "rank_deficient" if rank_deficient_count > 0 else "underdetermined"
        ),
    }


def feasible_wvf_grid(
    radii: tuple[int, ...] = WVF_GRID_RADII,
    degrees: tuple[int, ...] = WVF_GRID_DEGREES,
    normalize_coords: bool = True,
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for radius in radii:
        for degree in degrees:
            cell = wvf_conditioning_diagnostics(int(radius), int(degree), normalize_coords=bool(normalize_coords))
            if str(cell["status"]) == "ok":
                cells.append(cell)
    return cells
