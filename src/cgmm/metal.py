"""ctypes binding for the Rust/Metal c-GMM fusion backend."""

from __future__ import annotations

import ctypes
import math
from functools import lru_cache
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from wvf.metal import (
    MetalBackendError,
    _load_library as _load_wvf_library,
)

_MAX_UINT32 = np.iinfo(np.uint32).max
_K_SUPPORTED = 3
_N_ITERS_SUPPORTED = 30
_N_MAX = 64


class CGMMResult(TypedDict):
    theta_primary: npt.NDArray[np.float64]
    M_primary: npt.NDArray[np.float64]
    theta_sec: npt.NDArray[np.float64]
    M_sec: npt.NDArray[np.float64]
    v_fused: npt.NDArray[np.uint8]
    primary_pi: npt.NDArray[np.float64]
    primary_mu: npt.NDArray[np.float64]
    primary_kappa: npt.NDArray[np.float64]
    secondary_pi: npt.NDArray[np.float64]
    secondary_mu: npt.NDArray[np.float64]
    secondary_kappa: npt.NDArray[np.float64]
    keep_secondary_mask: npt.NDArray[np.uint8]


@lru_cache(maxsize=1)
def _load_cgmm_library() -> ctypes.CDLL:
    try:
        lib = _load_wvf_library()
        lib.edgecritic_metal_cgmm_fuse_two_pass.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_cgmm_fuse_two_pass.restype = ctypes.c_int
    except AttributeError as exc:
        raise MetalBackendError("Rust/Metal c-GMM symbols are not available") from exc
    except OSError as exc:
        raise MetalBackendError(str(exc)) from exc
    return lib


def cgmm_backend_available() -> bool:
    """Return whether the local machine can build and load the c-GMM backend."""
    try:
        _load_cgmm_library()
    except MetalBackendError:
        return False
    except OSError:
        return False
    return True


def _validate_inputs(
    phi_p: np.ndarray,
    w_p: np.ndarray,
    phi_s: np.ndarray,
    w_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        np.ascontiguousarray(arr, dtype=np.float32)
        for arr in (phi_p, w_p, phi_s, w_s)
    )
    names = ("phi_p", "w_p", "phi_s", "w_s")
    for name, arr in zip(names, arrays, strict=True):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array")

    shape = arrays[0].shape
    for name, arr in zip(names[1:], arrays[1:], strict=True):
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    p_count, n_count = shape
    if n_count <= 0 or n_count > _N_MAX:
        raise ValueError(f"N must satisfy 0 < N <= {_N_MAX}")
    if p_count > _MAX_UINT32:
        raise ValueError("P must fit in uint32")

    return arrays


def _validate_finite_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _empty_result(p_count: int) -> CGMMResult:
    diag_shape = (p_count, _K_SUPPORTED)
    return {
        "theta_primary": np.empty(p_count, dtype=np.float64),
        "M_primary": np.empty(p_count, dtype=np.float64),
        "theta_sec": np.empty(p_count, dtype=np.float64),
        "M_sec": np.empty(p_count, dtype=np.float64),
        "v_fused": np.empty(p_count, dtype=np.uint8),
        "primary_pi": np.empty(diag_shape, dtype=np.float64),
        "primary_mu": np.empty(diag_shape, dtype=np.float64),
        "primary_kappa": np.empty(diag_shape, dtype=np.float64),
        "secondary_pi": np.empty(diag_shape, dtype=np.float64),
        "secondary_mu": np.empty(diag_shape, dtype=np.float64),
        "secondary_kappa": np.empty(diag_shape, dtype=np.float64),
        "keep_secondary_mask": np.empty(p_count, dtype=np.uint8),
    }


def cgmm_fuse_two_pass_metal(
    phi_p: np.ndarray,
    w_p: np.ndarray,
    phi_s: np.ndarray,
    w_s: np.ndarray,
    K: int = 3,
    n_iters: int = 30,
    init_kappa: float = 4.0,
    hard_em: bool = True,
    tau_M_rel: float = 0.05,
    theta_min_deg: float = 10.0,
) -> CGMMResult:
    """Run two-pass K=3 hard-EM circular GMM fusion with the Metal backend."""
    if not bool(hard_em):
        raise NotImplementedError("soft EM is not implemented by the Metal c-GMM backend")
    if int(K) != _K_SUPPORTED:
        raise ValueError("Metal c-GMM supports K=3 only")
    if int(n_iters) != _N_ITERS_SUPPORTED:
        raise ValueError("Metal c-GMM supports n_iters=30 only")

    phi_p_arr, w_p_arr, phi_s_arr, w_s_arr = _validate_inputs(phi_p, w_p, phi_s, w_s)
    p_count, n_count = phi_p_arr.shape

    init_kappa_value = _validate_finite_float(init_kappa, "init_kappa")
    tau_m_rel_value = _validate_finite_float(tau_M_rel, "tau_M_rel")
    theta_min_value = _validate_finite_float(theta_min_deg, "theta_min_deg")

    if p_count == 0:
        return _empty_result(p_count)

    theta_primary_f = np.empty(p_count, dtype=np.float32)
    m_primary_f = np.empty(p_count, dtype=np.float32)
    theta_sec_f = np.empty(p_count, dtype=np.float32)
    m_sec_f = np.empty(p_count, dtype=np.float32)
    v_fused = np.empty(p_count, dtype=np.uint8)
    primary_pi_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    primary_mu_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    primary_kappa_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    secondary_pi_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    secondary_mu_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    secondary_kappa_f = np.empty((p_count, _K_SUPPORTED), dtype=np.float32)
    keep_secondary_mask = np.empty(p_count, dtype=np.uint8)

    error_buffer = ctypes.create_string_buffer(4096)
    status = _load_cgmm_library().edgecritic_metal_cgmm_fuse_two_pass(
        phi_p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_p_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        phi_s_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_s_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint(p_count),
        ctypes.c_uint(n_count),
        ctypes.c_uint(_N_ITERS_SUPPORTED),
        ctypes.c_float(init_kappa_value),
        ctypes.c_float(tau_m_rel_value),
        ctypes.c_float(2.0 * math.radians(theta_min_value)),
        theta_primary_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m_primary_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        theta_sec_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m_sec_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        primary_pi_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        primary_mu_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        primary_kappa_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        secondary_pi_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        secondary_mu_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        secondary_kappa_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        keep_secondary_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))

    return {
        "theta_primary": theta_primary_f.astype(np.float64),
        "M_primary": m_primary_f.astype(np.float64),
        "theta_sec": theta_sec_f.astype(np.float64),
        "M_sec": m_sec_f.astype(np.float64),
        "v_fused": v_fused,
        "primary_pi": primary_pi_f.astype(np.float64),
        "primary_mu": primary_mu_f.astype(np.float64),
        "primary_kappa": primary_kappa_f.astype(np.float64),
        "secondary_pi": secondary_pi_f.astype(np.float64),
        "secondary_mu": secondary_mu_f.astype(np.float64),
        "secondary_kappa": secondary_kappa_f.astype(np.float64),
        "keep_secondary_mask": keep_secondary_mask,
    }


__all__ = [
    "CGMMResult",
    "cgmm_backend_available",
    "cgmm_fuse_two_pass_metal",
]
