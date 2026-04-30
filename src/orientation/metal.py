"""ctypes binding for the Rust/Metal orientation recovery backend."""

from __future__ import annotations

import ctypes
import math
from functools import lru_cache

import numpy as np

from lf.metal import _as_uint32
from wvf.metal import (
    MetalBackendError,
    _load_library as _load_wvf_library,
)


@lru_cache(maxsize=1)
def _load_recovery_library() -> ctypes.CDLL:
    try:
        lib = _load_wvf_library()
        lib.edgecritic_metal_recover_two_peaks.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
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
        lib.edgecritic_metal_recover_two_peaks.restype = ctypes.c_int
    except AttributeError as exc:
        raise MetalBackendError("Rust/Metal recovery symbols are not available") from exc
    except OSError as exc:
        raise MetalBackendError(str(exc)) from exc
    return lib


def recovery_backend_available() -> bool:
    """Return whether the local machine can build and load the recovery backend."""
    try:
        _load_recovery_library()
    except MetalBackendError:
        return False
    except OSError:
        return False
    return True


def _validate_angles(angles_rad: np.ndarray) -> np.ndarray:
    angles = np.ascontiguousarray(angles_rad, dtype=np.float64)
    if angles.ndim != 1:
        raise ValueError("angles_rad must be a 1-D array")
    if angles.size == 0:
        raise ValueError("angles_rad must be non-empty")
    if not np.all(np.isfinite(angles)):
        raise ValueError("angles_rad must be finite")

    expected = np.linspace(0.0, math.pi, angles.size, endpoint=False, dtype=np.float64)
    if not np.allclose(angles, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("angles_rad must be evenly spaced over [0, pi)")
    return angles


def _validate_tunable_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def recover_two_peaks_metal(
    angles_rad: np.ndarray,
    response_2d: np.ndarray,
    tau_sec_floor: float = 0.40,
    tau_validity: float = 0.10,
    dense_n: int = 500,
    min_sep_frac: float = 0.125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover primary and secondary orientation peaks with the Metal backend."""
    angles = _validate_angles(angles_rad)
    response = np.ascontiguousarray(response_2d, dtype=np.float32)
    if response.ndim != 2:
        raise ValueError("response_2d must be a 2-D array")
    if response.shape[1] != angles.size:
        raise ValueError("response_2d must have one column per angle")

    n_rows, k = response.shape
    _as_uint32(n_rows, "row count")
    _as_uint32(k, "angle count")
    dense_count = int(dense_n)
    if dense_count <= 0:
        raise ValueError("dense_n must be positive")
    _as_uint32(dense_count, "dense_n")

    tau_sec = _validate_tunable_float(tau_sec_floor, "tau_sec_floor")
    tau_valid = _validate_tunable_float(tau_validity, "tau_validity")
    sep_frac = _validate_tunable_float(min_sep_frac, "min_sep_frac")

    theta_p_f = np.empty(n_rows, dtype=np.float32)
    m_p_f = np.empty(n_rows, dtype=np.float32)
    theta_s_f = np.empty(n_rows, dtype=np.float32)
    m_s_f = np.empty(n_rows, dtype=np.float32)
    v = np.empty(n_rows, dtype=np.uint8)
    if n_rows == 0:
        return (
            theta_p_f.astype(np.float64),
            m_p_f.astype(np.float64),
            theta_s_f.astype(np.float64),
            m_s_f.astype(np.float64),
            v,
        )

    error_buffer = ctypes.create_string_buffer(4096)
    status = _load_recovery_library().edgecritic_metal_recover_two_peaks(
        angles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        response.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(n_rows, "row count"),
        _as_uint32(k, "angle count"),
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

    return (
        theta_p_f.astype(np.float64),
        m_p_f.astype(np.float64),
        theta_s_f.astype(np.float64),
        m_s_f.astype(np.float64),
        v,
    )


__all__ = [
    "recover_two_peaks_metal",
    "recovery_backend_available",
]
