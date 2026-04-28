"""ctypes binding for the Rust/Metal LF response backend."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np

from edgecritic.wvf._metal import (
    MetalBackendError,
    _load_library as _load_wvf_library,
    metal_backend_available,
)

_MAX_UINT32 = np.iinfo(np.uint32).max
_MIN_INT32 = np.iinfo(np.int32).min
_MAX_INT32 = np.iinfo(np.int32).max
_MAX_BATCH_MS = 32
_MAX_BOX_PASSES = 32


@lru_cache(maxsize=1)
def _load_lf_library() -> ctypes.CDLL:
    try:
        lib = _load_wvf_library()
        lib.edgecritic_metal_lf_response.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_response.restype = ctypes.c_int
        lib.edgecritic_metal_lf_response_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_response_batch.restype = ctypes.c_int
        lib.edgecritic_metal_lf_orientation_stack.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_orientation_stack.restype = ctypes.c_int
        lib.edgecritic_metal_lf_orientation_stack_box.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_orientation_stack_box.restype = ctypes.c_int
        lib.edgecritic_metal_lf_orientation_length_stack_box.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_orientation_length_stack_box.restype = ctypes.c_int
        lib.edgecritic_metal_lf_orientation_stack_scanline.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        lib.edgecritic_metal_lf_orientation_stack_scanline.restype = ctypes.c_int
    except AttributeError as exc:
        raise MetalBackendError("Rust/Metal LF symbols are not available") from exc
    except OSError as exc:
        raise MetalBackendError(str(exc)) from exc
    return lib


def _as_uint32(value: int, name: str) -> ctypes.c_uint:
    if value < 0 or value > _MAX_UINT32:
        raise ValueError(f"{name} must fit in uint32")
    return ctypes.c_uint(value)


def _as_int32(value: int, name: str) -> ctypes.c_int:
    if value < _MIN_INT32 or value > _MAX_INT32:
        raise ValueError(f"{name} must fit in int32")
    return ctypes.c_int(value)


def _components(g_x: np.ndarray, g_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.ascontiguousarray(g_x, dtype=np.float32)
    gy = np.ascontiguousarray(g_y, dtype=np.float32)
    if gx.ndim != 2 or gy.ndim != 2:
        raise ValueError("g_x and g_y must be 2-D arrays")
    if gx.shape != gy.shape:
        raise ValueError("g_x and g_y must have the same shape")
    h, w = gx.shape
    _as_uint32(h, "image height")
    _as_uint32(w, "image width")
    if h == 0 or w == 0:
        raise ValueError("g_x and g_y must be non-empty")
    return gx, gy


def _pixel_coords(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(px, dtype=np.int32)
    y = np.ascontiguousarray(py, dtype=np.int32)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("px and py must be 1-D arrays")
    if x.shape != y.shape:
        raise ValueError("px and py must have the same shape")
    _as_uint32(x.size, "pixel count")
    return x, y


def _raise_native_error(error_buffer: ctypes.Array[ctypes.c_char]) -> None:
    raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))


def lf_response_metal(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    theta: float,
    m: int,
) -> np.ndarray:
    """Compute LF response for one ``(theta, m)`` pair using Rust/Metal."""
    gx, gy = _components(g_x, g_y)
    x, y = _pixel_coords(px, py)
    if x.size == 0:
        return np.empty(0, dtype=np.float64)

    h, w = gx.shape
    out = np.empty(x.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    status = _load_lf_library().edgecritic_metal_lf_response(
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(w, "image width"),
        _as_uint32(h, "image height"),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_uint32(x.size, "pixel count"),
        ctypes.c_double(float(theta)),
        _as_int32(int(m), "m"),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        _raise_native_error(error_buffer)
    return out.astype(np.float64)


def lf_response_metal_batch(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    thetas: np.ndarray,
    ms: np.ndarray,
) -> np.ndarray:
    """Compute a ``(theta, m, pixel)`` LF response grid using Rust/Metal."""
    gx, gy = _components(g_x, g_y)
    x, y = _pixel_coords(px, py)
    theta_arr = np.ascontiguousarray(thetas, dtype=np.float64)
    m_arr = np.ascontiguousarray(ms, dtype=np.int32)
    if theta_arr.ndim != 1 or m_arr.ndim != 1:
        raise ValueError("thetas and ms must be 1-D arrays")
    if m_arr.size > _MAX_BATCH_MS:
        raise ValueError(f"batched LF supports at most {_MAX_BATCH_MS} m values")
    _as_uint32(theta_arr.size, "theta count")
    _as_uint32(m_arr.size, "m count")

    shape = (theta_arr.size, m_arr.size, x.size)
    if theta_arr.size == 0 or m_arr.size == 0 or x.size == 0:
        return np.empty(shape, dtype=np.float64)

    h, w = gx.shape
    out = np.empty(shape, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    status = _load_lf_library().edgecritic_metal_lf_response_batch(
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(w, "image width"),
        _as_uint32(h, "image height"),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_uint32(x.size, "pixel count"),
        theta_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        _as_uint32(theta_arr.size, "theta count"),
        m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_uint32(m_arr.size, "m count"),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        _raise_native_error(error_buffer)
    return out.astype(np.float64)


def lf_orientation_length_stack_metal(
    g_x: np.ndarray,
    g_y: np.ndarray,
    ms: np.ndarray,
    n_orientations: int = 16,
    output_dtype: np.dtype | type = np.float32,
    method: str = "box",
    out: np.ndarray | None = None,
    box_passes: int = 1,
) -> np.ndarray:
    """Compute full-frame LF responses for multiple lengths.

    Returns an array of shape ``(n_orientations, n_ms, H, W)``. The current
    multi-length full-frame backend supports the one-pass box method.
    """
    method_name = str(method).lower()
    if method_name != "box":
        raise ValueError("multi-length full-frame LF currently supports method='box'")
    if int(box_passes) != 1:
        raise ValueError("multi-length full-frame LF currently supports box_passes=1")
    if int(n_orientations) <= 0:
        raise ValueError("n_orientations must be positive")

    gx, gy = _components(g_x, g_y)
    m_arr = np.ascontiguousarray(ms, dtype=np.int32)
    if m_arr.ndim != 1:
        raise ValueError("ms must be a 1-D array")
    if m_arr.size > _MAX_BATCH_MS:
        raise ValueError(f"full-image batched LF supports at most {_MAX_BATCH_MS} m values")
    _as_uint32(m_arr.size, "m count")

    h, w = gx.shape
    n = int(n_orientations)
    shape = (n, m_arr.size, h, w)
    if out is None:
        out_arr = np.empty(shape, dtype=np.float32)
    else:
        out_arr = np.asarray(out)
        if out_arr.shape != shape:
            raise ValueError("out must have shape (n_orientations, n_ms, H, W)")
        if out_arr.dtype != np.dtype(np.float32):
            raise ValueError("out must have dtype float32")
        if not out_arr.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
    if m_arr.size == 0:
        dtype = np.dtype(output_dtype)
        if dtype == np.dtype(np.float32):
            return out_arr
        return out_arr.astype(dtype)

    error_buffer = ctypes.create_string_buffer(4096)
    status = _load_lf_library().edgecritic_metal_lf_orientation_length_stack_box(
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _as_uint32(w, "image width"),
        _as_uint32(h, "image height"),
        _as_uint32(n, "orientation count"),
        m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_uint32(m_arr.size, "m count"),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        _raise_native_error(error_buffer)

    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float32):
        return out_arr
    return out_arr.astype(dtype)


def lf_orientation_stack_metal(
    g_x: np.ndarray,
    g_y: np.ndarray,
    m: int,
    n_orientations: int = 16,
    output_dtype: np.dtype | type = np.float32,
    method: str = "box",
    execution: str = "auto",
    out: np.ndarray | None = None,
    box_passes: int = 1,
    box_radius: int | None = None,
) -> np.ndarray:
    """Compute full-frame LF responses for equally spaced orientations.

    Returns an array of shape ``(n_orientations, H, W)``. Orientations are
    equally spaced over ``[0, pi)``.
    """
    method_name = str(method).lower()
    if method_name not in {"exact", "box", "scanline"}:
        raise ValueError("method must be 'exact', 'box', or 'scanline'")
    execution_mode = {"auto": 0, "direct": 1, "projected": 2}.get(str(execution).lower())
    if execution_mode is None:
        raise ValueError("execution must be 'auto', 'direct', or 'projected'")
    if method_name in {"box", "scanline"} and execution_mode != 0:
        raise ValueError("execution must be 'auto' when method is 'box' or 'scanline'")
    if int(n_orientations) <= 0:
        raise ValueError("n_orientations must be positive")
    box_pass_count = int(box_passes)
    if box_pass_count < 1 or box_pass_count > _MAX_BOX_PASSES:
        raise ValueError(f"box_passes must be between 1 and {_MAX_BOX_PASSES}")
    if box_radius is None:
        box_radius_value = -1
    else:
        box_radius_value = int(box_radius)
        if box_radius_value < 0:
            raise ValueError("box_radius must be non-negative or None")

    gx, gy = _components(g_x, g_y)
    h, w = gx.shape
    n = int(n_orientations)
    if out is None:
        out_arr = np.empty((n, h, w), dtype=np.float32)
    else:
        out_arr = np.asarray(out)
        if out_arr.shape != (n, h, w):
            raise ValueError("out must have shape (n_orientations, H, W)")
        if out_arr.dtype != np.dtype(np.float32):
            raise ValueError("out must have dtype float32")
        if not out_arr.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
    error_buffer = ctypes.create_string_buffer(4096)
    if method_name == "exact":
        status = _load_lf_library().edgecritic_metal_lf_orientation_stack(
            gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _as_uint32(w, "image width"),
            _as_uint32(h, "image height"),
            _as_uint32(n, "orientation count"),
            _as_int32(int(m), "m"),
            _as_uint32(execution_mode, "execution mode"),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    elif method_name == "box":
        status = _load_lf_library().edgecritic_metal_lf_orientation_stack_box(
            gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _as_uint32(w, "image width"),
            _as_uint32(h, "image height"),
            _as_uint32(n, "orientation count"),
            _as_int32(int(m), "m"),
            _as_uint32(box_pass_count, "box pass count"),
            _as_int32(box_radius_value, "box radius"),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    else:
        status = _load_lf_library().edgecritic_metal_lf_orientation_stack_scanline(
            gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _as_uint32(w, "image width"),
            _as_uint32(h, "image height"),
            _as_uint32(n, "orientation count"),
            _as_int32(int(m), "m"),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    if status != 0:
        _raise_native_error(error_buffer)

    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float32):
        return out_arr
    return out_arr.astype(dtype)


__all__ = [
    "MetalBackendError",
    "lf_orientation_length_stack_metal",
    "lf_orientation_stack_metal",
    "lf_response_metal",
    "lf_response_metal_batch",
    "metal_backend_available",
]
