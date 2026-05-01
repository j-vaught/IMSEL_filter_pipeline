"""Python bindings for the standalone WVF Metal backend."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .radius import (
    WVFAntipodalKernels,
    WVFRadiusKernels,
    build_wvf_antipodal_kernels,
    build_wvf_radius_kernels,
)


class MetalBackendError(RuntimeError):
    """Raised when the local Metal backend cannot run."""


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _crate_manifest() -> Path:
    return _package_root() / "Cargo.toml"


def _target_dir() -> Path:
    return _package_root() / "build" / "target"


def _library_path() -> Path:
    if platform.system() != "Darwin":
        raise MetalBackendError("Metal backend is only available on macOS")
    if shutil.which("cargo") is None:
        raise MetalBackendError("cargo is required to build the Metal backend")

    manifest = _crate_manifest()
    if not manifest.exists():
        raise MetalBackendError(f"Cargo manifest not found at {manifest}")

    target_dir = _target_dir()
    dylib = target_dir / "release" / "libwvf_metal_backend.dylib"
    if dylib.exists() and not _needs_rebuild(dylib):
        return dylib

    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    result = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        cwd=str(_package_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise MetalBackendError(f"failed to build Metal backend: {message}")
    if not dylib.exists():
        raise MetalBackendError(f"Metal build succeeded but {dylib} was not produced")
    return dylib


def _needs_rebuild(dylib: Path) -> bool:
    dylib_mtime = dylib.stat().st_mtime
    root = _package_root()
    build_inputs = [root / "Cargo.toml", root / "Cargo.lock"]
    build_inputs.extend((root / "rust").rglob("*.rs"))
    build_inputs.extend((root / "rust").rglob("*.metal"))
    return any(path.exists() and path.stat().st_mtime > dylib_mtime for path in build_inputs)


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_library_path()))
    direct_args = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.wvf_metal_convolve_direct.argtypes = direct_args
    lib.wvf_metal_convolve_direct.restype = ctypes.c_int
    lib.wvf_metal_convolve_antipodal.argtypes = direct_args
    lib.wvf_metal_convolve_antipodal.restype = ctypes.c_int
    lib.wvf_metal_convolve_split.argtypes = direct_args[:8] + [ctypes.c_uint] + direct_args[8:]
    lib.wvf_metal_convolve_split.restype = ctypes.c_int
    return lib


def metal_backend_available() -> bool:
    """Return whether this machine can build and load the Metal backend."""
    try:
        _load_library()
    except (MetalBackendError, OSError):
        return False
    return True


def _as_float_image(image: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("WVF Metal expects a 2-D image")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("image width and height must be positive")
    return img


def _call_backend(
    function_name: str,
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    n_offsets: int,
    output_dtype: np.dtype | type,
    radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    img = _as_float_image(image)
    out_x = np.empty(img.size, dtype=np.float32)
    out_y = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    args = [
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint(w),
        ctypes.c_uint(h),
        dx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        wx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint(n_offsets),
    ]
    if radius is not None:
        args.append(ctypes.c_uint(radius))
    args.extend(
        [
            out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error_buffer,
            ctypes.c_size_t(len(error_buffer)),
        ]
    )

    status = getattr(_load_library(), function_name)(*args)
    if status != 0:
        raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))

    dtype = np.dtype(output_dtype)
    gx = out_x.reshape(img.shape)
    gy = out_y.reshape(img.shape)
    if dtype == np.dtype(np.float32):
        return gx, gy
    return gx.astype(dtype), gy.astype(dtype)


def wvf_direct_gradients_metal(
    image: np.ndarray,
    kernels: WVFRadiusKernels,
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF gradients with the direct sparse Metal kernel."""
    offsets = kernels.offsets_xy.astype(np.int32, copy=False)
    return _call_backend(
        "wvf_metal_convolve_direct",
        image,
        np.ascontiguousarray(offsets[:, 0], dtype=np.int32),
        np.ascontiguousarray(offsets[:, 1], dtype=np.int32),
        np.ascontiguousarray(kernels.weights_x, dtype=np.float32),
        np.ascontiguousarray(kernels.weights_y, dtype=np.float32),
        kernels.support_size,
        output_dtype,
    )


def wvf_antipodal_gradients_metal(
    image: np.ndarray,
    kernels: WVFAntipodalKernels,
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF gradients with antipodal Metal pairs."""
    offsets = kernels.offsets_xy.astype(np.int32, copy=False)
    return _call_backend(
        "wvf_metal_convolve_antipodal",
        image,
        np.ascontiguousarray(offsets[:, 0], dtype=np.int32),
        np.ascontiguousarray(offsets[:, 1], dtype=np.int32),
        np.ascontiguousarray(kernels.weights_x, dtype=np.float32),
        np.ascontiguousarray(kernels.weights_y, dtype=np.float32),
        kernels.pair_count,
        output_dtype,
    )


def wvf_split_gradients_metal(
    image: np.ndarray,
    kernels: WVFAntipodalKernels,
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF gradients with split interior and boundary Metal kernels."""
    offsets = kernels.offsets_xy.astype(np.int32, copy=False)
    return _call_backend(
        "wvf_metal_convolve_split",
        image,
        np.ascontiguousarray(offsets[:, 0], dtype=np.int32),
        np.ascontiguousarray(offsets[:, 1], dtype=np.int32),
        np.ascontiguousarray(kernels.weights_x, dtype=np.float32),
        np.ascontiguousarray(kernels.weights_y, dtype=np.float32),
        kernels.pair_count,
        output_dtype,
        radius=kernels.radius,
    )


def wvf_radius_gradients_metal(
    image: np.ndarray,
    kernels: WVFRadiusKernels,
    output_dtype: np.dtype | type = np.float64,
    variant: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF gradients for prebuilt radius kernels."""
    mode = str(variant).lower()
    if mode in {"direct", "baseline"}:
        return wvf_direct_gradients_metal(image, kernels, output_dtype)
    if mode not in {"split", "antipodal", "optimized", "auto"}:
        raise ValueError("variant must be 'split', 'antipodal', or 'direct'")

    antipodal = build_wvf_antipodal_kernels(kernels.radius, kernels.degree)
    if mode == "antipodal":
        return wvf_antipodal_gradients_metal(image, antipodal, output_dtype)
    return wvf_split_gradients_metal(image, antipodal, output_dtype)


def wvf_gradients_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    output_dtype: np.dtype | type = np.float32,
    variant: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standalone WVF ``Gx`` and ``Gy`` on Metal."""
    kernels = build_wvf_radius_kernels(radius=radius, degree=degree)
    return wvf_radius_gradients_metal(
        image,
        kernels,
        output_dtype=output_dtype,
        variant=variant,
    )


def wvf_magnitude_angle_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    output_dtype: np.dtype | type = np.float32,
    variant: str = "split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute WVF components, magnitude, and unsigned orientation angle."""
    gx, gy = wvf_gradients_metal(
        image,
        radius=radius,
        degree=degree,
        output_dtype=output_dtype,
        variant=variant,
    )
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    return (
        gx,
        gy,
        mag.astype(output_dtype, copy=False),
        angle.astype(output_dtype, copy=False),
    )
