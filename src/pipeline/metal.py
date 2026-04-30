"""ctypes binding for the fused Rust/Metal WVF, LF, and recovery backend."""

from __future__ import annotations

import ctypes
import math
from functools import lru_cache

import numpy as np

from lf.metal import _as_int32, _as_uint32
from wvf.metal import (
    MetalBackendError,
    _load_library as _load_wvf_library,
)
from wvf.radius import build_wvf_radius_kernels


@lru_cache(maxsize=1)
def _load_pipeline_library() -> ctypes.CDLL:
    try:
        lib = _load_wvf_library()
        lib.edgecritic_metal_wvf_lf_recover.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_wvf_lf_recover.restype = ctypes.c_int
    except AttributeError as exc:
        raise MetalBackendError("Rust/Metal fused pipeline symbols are not available") from exc
    except OSError as exc:
        raise MetalBackendError(str(exc)) from exc
    return lib


def pipeline_backend_available() -> bool:
    """Return whether the local machine can build and load the fused backend."""
    try:
        _load_pipeline_library()
    except MetalBackendError:
        return False
    except OSError:
        return False
    return True


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def wvf_lf_recover_metal(
    image: np.ndarray,
    radius: int,
    degree: int,
    lf_half_length: int,
    n_orientations: int = 64,
    tau_sec_floor: float = 0.40,
    tau_validity: float = 0.10,
    dense_n: int = 500,
    min_sep_frac: float = 0.125,
    method: str = "box",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run WVF -> LF -> orientation recovery as a single fused Metal pipeline."""
    method_name = str(method).lower()
    if method_name != "box":
        raise ValueError("method must be 'box'")

    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("image must be a 2-D array")
    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError("image width and height must be positive")

    radius_value = int(radius)
    degree_value = int(degree)
    lf_value = int(lf_half_length)
    n = int(n_orientations)
    if n <= 0:
        raise ValueError("n_orientations must be positive")
    dense_count = int(dense_n)
    if dense_count <= 0:
        raise ValueError("dense_n must be positive")

    tau_sec = _finite_float(tau_sec_floor, "tau_sec_floor")
    tau_valid = _finite_float(tau_validity, "tau_validity")
    sep_frac = _finite_float(min_sep_frac, "min_sep_frac")

    kernels = build_wvf_radius_kernels(radius=radius_value, order=degree_value)
    offsets = kernels.offsets_xy.astype(np.int32, copy=False)
    dx = np.ascontiguousarray(offsets[:, 0], dtype=np.int32)
    dy = np.ascontiguousarray(offsets[:, 1], dtype=np.int32)
    wx = np.ascontiguousarray(kernels.weights_x, dtype=np.float32)
    wy = np.ascontiguousarray(kernels.weights_y, dtype=np.float32)

    theta_p_f = np.empty(img.size, dtype=np.float32)
    m_p_f = np.empty(img.size, dtype=np.float32)
    theta_s_f = np.empty(img.size, dtype=np.float32)
    m_s_f = np.empty(img.size, dtype=np.float32)
    v = np.empty(img.size, dtype=np.uint8)
    error_buffer = ctypes.create_string_buffer(4096)

    status = _load_pipeline_library().edgecritic_metal_wvf_lf_recover(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(w, "image width"),
        _as_uint32(h, "image height"),
        dx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        wx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(kernels.support_size, "WVF support size"),
        _as_int32(lf_value, "LF half-length"),
        _as_uint32(n, "orientation count"),
        ctypes.c_uint(1),
        ctypes.c_int(-1),
        ctypes.c_float(tau_sec),
        ctypes.c_float(tau_valid),
        _as_uint32(dense_count, "dense_n"),
        ctypes.c_float(sep_frac),
        theta_p_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m_p_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        theta_s_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m_s_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))

    shape = img.shape
    return (
        theta_p_f.reshape(shape).astype(np.float64),
        m_p_f.reshape(shape).astype(np.float64),
        theta_s_f.reshape(shape).astype(np.float64),
        m_s_f.reshape(shape).astype(np.float64),
        v.reshape(shape),
    )


__all__ = [
    "pipeline_backend_available",
    "wvf_lf_recover_metal",
]
