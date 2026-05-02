#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from square_support_sg import build_square_support_kernels
from wvf.radius import build_wvf_radius_kernels


RECOMMENDED_WVF_DEGREES: dict[int, int] = {
    3: 5,
    5: 9,
    9: 11,
    15: 11,
    25: 11,
    50: 11,
}

FARID_SMOOTH_7 = np.asarray(
    [0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711],
    dtype=np.float64,
)
FARID_DERIV_7 = np.asarray(
    [-0.018708, -0.125376, -0.193091, 0.0, 0.193091, 0.125376, 0.018708],
    dtype=np.float64,
)


@dataclass(frozen=True)
class KernelSpec:
    method: str
    label: str
    config: dict[str, Any]
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    support_half_extent: int
    white_noise_gain: float
    support_cardinality: int | None = None
    kappa_design_matrix: float | None = None

    @property
    def config_label(self) -> str:
        parts: list[str] = []
        for key, value in self.config.items():
            if isinstance(value, float):
                text = f"{value:g}"
            else:
                text = str(value)
            parts.append(f"{key}={text}")
        return ", ".join(parts)


def _as_contiguous(kernel: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(kernel, dtype=np.float64))


def _half_extent_from_shape(shape: tuple[int, int]) -> int:
    return int(max(shape[0] - 1, shape[1] - 1) // 2 + max(shape[0] % 2 == 0, shape[1] % 2 == 0))


def _white_noise_gain(kernel_x: np.ndarray) -> float:
    return float(np.sum(np.asarray(kernel_x, dtype=np.float64) ** 2))


def recommended_wvf_degree(radius: int) -> int:
    radius_i = int(radius)
    if radius_i in RECOMMENDED_WVF_DEGREES:
        return int(RECOMMENDED_WVF_DEGREES[radius_i])
    if radius_i <= 3:
        return 5
    if radius_i <= 5:
        return 9
    return 11


def build_roberts() -> KernelSpec:
    kernel_x = 0.5 * np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    kernel_y = 0.5 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    return KernelSpec(
        method="roberts",
        label="Roberts",
        config={"window": "2x2"},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=1,
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_prewitt() -> KernelSpec:
    kernel_x = np.asarray(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    ) / 6.0
    kernel_y = kernel_x.T.copy()
    return KernelSpec(
        method="prewitt",
        label="Prewitt",
        config={"window": "3x3"},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=1,
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_sobel() -> KernelSpec:
    kernel_x = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    ) / 8.0
    kernel_y = kernel_x.T.copy()
    return KernelSpec(
        method="sobel",
        label="Sobel",
        config={"window": "3x3"},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=1,
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_scharr() -> KernelSpec:
    kernel_x = np.asarray(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        dtype=np.float64,
    ) / 32.0
    kernel_y = kernel_x.T.copy()
    return KernelSpec(
        method="scharr",
        label="Scharr",
        config={"window": "3x3"},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=1,
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_dog(sigma: float) -> KernelSpec:
    sigma_f = float(sigma)
    if sigma_f <= 0.0:
        raise ValueError("sigma must be positive")
    half_extent = int(math.ceil(4.0 * sigma_f))
    coords = np.arange(-half_extent, half_extent + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rho2 = xx**2 + yy**2
    gaussian = np.exp(-rho2 / (2.0 * sigma_f * sigma_f)) / (2.0 * math.pi * sigma_f * sigma_f)
    kernel_x = (xx / (sigma_f * sigma_f)) * gaussian
    kernel_y = (yy / (sigma_f * sigma_f)) * gaussian
    return KernelSpec(
        method="dog",
        label="DoG",
        config={"sigma": sigma_f},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=int(half_extent),
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_farid_simoncelli() -> KernelSpec:
    kernel_x = np.outer(FARID_SMOOTH_7, FARID_DERIV_7)
    kernel_y = np.outer(FARID_DERIV_7, FARID_SMOOTH_7)
    return KernelSpec(
        method="farid_simoncelli",
        label="Farid-Simoncelli",
        config={"window": "7tap"},
        kernel_x=_as_contiguous(kernel_x),
        kernel_y=_as_contiguous(kernel_y),
        support_half_extent=3,
        white_noise_gain=_white_noise_gain(kernel_x),
    )


def build_square_sg(window_size: int, degree: int, normalize_coords: bool = True) -> KernelSpec:
    window_i = int(window_size)
    if window_i <= 0 or window_i % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    half_side = window_i // 2
    kernels = build_square_support_kernels(
        half_side=float(half_side),
        degree=int(degree),
        normalize_coords=bool(normalize_coords),
    )
    return KernelSpec(
        method="square_sg",
        label="Square SG",
        config={"N": int(window_i), "d": int(degree), "normalize_coords": bool(normalize_coords)},
        kernel_x=_as_contiguous(kernels.kernel_x),
        kernel_y=_as_contiguous(kernels.kernel_y),
        support_half_extent=int(half_side),
        white_noise_gain=_white_noise_gain(kernels.kernel_x),
        support_cardinality=int(kernels.support_cardinality),
        kappa_design_matrix=float(kernels.kappa_design_matrix),
    )


def build_wvf(radius: int, degree: int | None = None, normalize_coords: bool = True) -> KernelSpec:
    radius_i = int(radius)
    degree_i = recommended_wvf_degree(radius_i) if degree is None else int(degree)
    kernels = build_wvf_radius_kernels(
        radius=radius_i,
        order=degree_i,
        normalize_coords=bool(normalize_coords),
    )
    return KernelSpec(
        method="wvf",
        label="WVF",
        config={"r": int(radius_i), "d": int(degree_i), "normalize_coords": bool(normalize_coords)},
        kernel_x=_as_contiguous(kernels.kernel_x),
        kernel_y=_as_contiguous(kernels.kernel_y),
        support_half_extent=int(math.ceil(radius_i)),
        white_noise_gain=_white_noise_gain(kernels.kernel_x),
        support_cardinality=int(np.asarray(kernels.offsets_xy).shape[0]),
    )


def fixed_method_order() -> tuple[str, ...]:
    return (
        "roberts",
        "prewitt",
        "sobel",
        "scharr",
        "dog",
        "farid_simoncelli",
        "square_sg",
        "wvf",
    )


def build_method(method: str, **config: Any) -> KernelSpec:
    name = str(method)
    if name == "roberts":
        return build_roberts()
    if name == "prewitt":
        return build_prewitt()
    if name == "sobel":
        return build_sobel()
    if name == "scharr":
        return build_scharr()
    if name == "dog":
        return build_dog(float(config["sigma"]))
    if name == "farid_simoncelli":
        return build_farid_simoncelli()
    if name == "square_sg":
        return build_square_sg(
            window_size=int(config["N"]),
            degree=int(config["d"]),
            normalize_coords=bool(config.get("normalize_coords", True)),
        )
    if name == "wvf":
        return build_wvf(
            radius=int(config["r"]),
            degree=None if config.get("d") is None else int(config["d"]),
            normalize_coords=bool(config.get("normalize_coords", True)),
        )
    raise ValueError(f"unsupported method {method!r}")
