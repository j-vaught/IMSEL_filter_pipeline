"""ctypes binding for the Rust/Metal LF response backend."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np

from wvf.metal import (
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
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
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


def _raise_metal_error(error_buffer: ctypes.Array[ctypes.c_char]) -> None:
    raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))


def lf_response(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    theta: float,
    lf_half_length: int,
) -> np.ndarray:
    """Compute LF response for one ``(theta, lf_half_length)`` pair."""
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
        _as_int32(int(lf_half_length), "LF half-length"),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        _raise_metal_error(error_buffer)
    return out.astype(np.float64)


def lf_response_batch(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    thetas: np.ndarray,
    lf_half_lengths: np.ndarray,
) -> np.ndarray:
    """Compute a ``(theta, LF half-length, pixel)`` response grid."""
    gx, gy = _components(g_x, g_y)
    x, y = _pixel_coords(px, py)
    theta_arr = np.ascontiguousarray(thetas, dtype=np.float64)
    length_arr = np.ascontiguousarray(lf_half_lengths, dtype=np.int32)
    if theta_arr.ndim != 1 or length_arr.ndim != 1:
        raise ValueError("thetas and lf_half_lengths must be 1-D arrays")
    if length_arr.size > _MAX_BATCH_MS:
        raise ValueError(f"batched LF supports at most {_MAX_BATCH_MS} LF half-lengths")
    _as_uint32(theta_arr.size, "theta count")
    _as_uint32(length_arr.size, "LF half-length count")

    shape = (theta_arr.size, length_arr.size, x.size)
    if theta_arr.size == 0 or length_arr.size == 0 or x.size == 0:
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
        length_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _as_uint32(length_arr.size, "LF half-length count"),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        _raise_metal_error(error_buffer)
    return out.astype(np.float64)


def lf_length_stack(
    g_x: np.ndarray,
    g_y: np.ndarray,
    lf_half_lengths: np.ndarray,
    n_orientations: int = 16,
    output_dtype: np.dtype | type = np.float32,
    method: str = "box",
    out: np.ndarray | None = None,
    box_passes: int = 1,
    output_layout: str = "theta_yx_m",
    max_chunk_bytes: int | None = 2 * 1024 * 1024 * 1024,
    chunk_pause_s: float = 0.0,
) -> np.ndarray:
    """Compute full-frame LF responses for multiple lengths.

    The preferred output layout is ``theta_yx_m``, with shape
    ``(n_orientations, H, W, n_lengths)``. The legacy ``theta_m_yx`` layout
    returns ``(n_orientations, n_lengths, H, W)``. The current multi-length
    full-frame backend supports the one-pass box method.
    """
    method_name = str(method).lower()
    if method_name != "box":
        raise ValueError("multi-length full-frame LF currently supports method='box'")
    if int(box_passes) != 1:
        raise ValueError("multi-length full-frame LF currently supports box_passes=1")
    if int(n_orientations) <= 0:
        raise ValueError("n_orientations must be positive")
    layout_name = str(output_layout).lower()
    layout_codes = {"theta_m_yx": 0, "theta_yx_m": 1}
    if layout_name not in layout_codes:
        raise ValueError("output_layout must be 'theta_yx_m' or 'theta_m_yx'")
    pause_seconds = float(chunk_pause_s)
    if pause_seconds < 0.0:
        raise ValueError("chunk_pause_s must be non-negative")

    gx, gy = _components(g_x, g_y)
    length_arr = np.ascontiguousarray(lf_half_lengths, dtype=np.int32)
    if length_arr.ndim != 1:
        raise ValueError("lf_half_lengths must be a 1-D array")
    if length_arr.size > _MAX_BATCH_MS:
        raise ValueError(
            f"full-image batched LF supports at most {_MAX_BATCH_MS} LF half-lengths"
        )
    _as_uint32(length_arr.size, "LF half-length count")

    h, w = gx.shape
    n = int(n_orientations)
    shape = (
        (n, length_arr.size, h, w)
        if layout_name == "theta_m_yx"
        else (n, h, w, length_arr.size)
    )
    if out is None:
        out_arr = np.empty(shape, dtype=np.float32)
    else:
        out_arr = np.asanyarray(out)
        if out_arr.shape != shape:
            if layout_name == "theta_m_yx":
                expected = "(n_orientations, n_lengths, H, W)"
            else:
                expected = "(n_orientations, H, W, n_lengths)"
            raise ValueError(f"out must have shape {expected}")
        if out_arr.dtype != np.dtype(np.float32):
            raise ValueError("out must have dtype float32")
        if not out_arr.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
    if length_arr.size == 0:
        dtype = np.dtype(output_dtype)
        if dtype == np.dtype(np.float32):
            return out_arr
        return out_arr.astype(dtype)

    if max_chunk_bytes is None:
        chunk_size = n
    else:
        chunk_byte_limit = int(max_chunk_bytes)
        if chunk_byte_limit <= 0:
            raise ValueError("max_chunk_bytes must be positive or None")
        bytes_per_orientation = h * w * length_arr.size * np.dtype(np.float32).itemsize
        chunk_size = max(1, min(n, chunk_byte_limit // bytes_per_orientation))

    error_buffer = ctypes.create_string_buffer(4096)
    lib = _load_lf_library()
    for theta_start in range(0, n, chunk_size):
        theta_count = min(chunk_size, n - theta_start)
        out_chunk = out_arr[theta_start : theta_start + theta_count]
        if not out_chunk.flags.c_contiguous:
            raise ValueError("orientation chunks must be C-contiguous")
        status = lib.edgecritic_metal_lf_orientation_length_stack_box(
            gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            _as_uint32(w, "image width"),
            _as_uint32(h, "image height"),
            _as_uint32(theta_start, "orientation start"),
            _as_uint32(theta_count, "orientation chunk count"),
            _as_uint32(n, "orientation count"),
            length_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            _as_uint32(length_arr.size, "LF half-length count"),
            _as_uint32(layout_codes[layout_name], "output layout"),
            out_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            _raise_metal_error(error_buffer)
        if pause_seconds > 0.0 and theta_start + theta_count < n:
            import time

            time.sleep(pause_seconds)

    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float32):
        return out_arr
    return out_arr.astype(dtype)


def lf_stack(
    g_x: np.ndarray,
    g_y: np.ndarray,
    lf_half_length: int,
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
            _as_int32(int(lf_half_length), "LF half-length"),
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
            _as_int32(int(lf_half_length), "LF half-length"),
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
            _as_int32(int(lf_half_length), "LF half-length"),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        )
    if status != 0:
        _raise_metal_error(error_buffer)

    dtype = np.dtype(output_dtype)
    if dtype == np.dtype(np.float32):
        return out_arr
    return out_arr.astype(dtype)


def lf_response_metal(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    theta: float,
    m: int,
) -> np.ndarray:
    return lf_response(g_x, g_y, px, py, theta, lf_half_length=m)


def lf_response_metal_batch(
    g_x: np.ndarray,
    g_y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    thetas: np.ndarray,
    ms: np.ndarray,
) -> np.ndarray:
    return lf_response_batch(g_x, g_y, px, py, thetas, lf_half_lengths=ms)


def lf_orientation_length_stack_metal(
    g_x: np.ndarray,
    g_y: np.ndarray,
    lf_half_lengths: np.ndarray,
    n_orientations: int = 16,
    output_dtype: np.dtype | type = np.float32,
    method: str = "box",
    out: np.ndarray | None = None,
    box_passes: int = 1,
    output_layout: str = "theta_yx_m",
    max_chunk_bytes: int | None = 2 * 1024 * 1024 * 1024,
    chunk_pause_s: float = 0.0,
) -> np.ndarray:
    return lf_length_stack(
        g_x,
        g_y,
        lf_half_lengths=lf_half_lengths,
        n_orientations=n_orientations,
        output_dtype=output_dtype,
        method=method,
        out=out,
        box_passes=box_passes,
        output_layout=output_layout,
        max_chunk_bytes=max_chunk_bytes,
        chunk_pause_s=chunk_pause_s,
    )


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
    return lf_stack(
        g_x,
        g_y,
        lf_half_length=m,
        n_orientations=n_orientations,
        output_dtype=output_dtype,
        method=method,
        execution=execution,
        out=out,
        box_passes=box_passes,
        box_radius=box_radius,
    )


__all__ = [
    "MetalBackendError",
    "lf_length_stack",
    "lf_response",
    "lf_response_batch",
    "lf_stack",
    "metal_backend_available",
]
