"""Python binding for the Rust/Metal WVF convolution backend."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from edgecritic.wvf._radius_kernels import WVFRadiusKernels


class MetalBackendError(RuntimeError):
    """Raised when the Rust/Metal WVF backend cannot run."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _crate_manifest() -> Path:
    return _repo_root() / "native" / "edgecritic_metal" / "Cargo.toml"


def _target_dir() -> Path:
    return _repo_root() / "build" / "edgecritic_metal_target"


def _library_path() -> Path:
    if platform.system() != "Darwin":
        raise MetalBackendError("Metal backend is only available on macOS")
    if shutil.which("cargo") is None:
        raise MetalBackendError("cargo is required to build the Rust/Metal backend")

    manifest = _crate_manifest()
    if not manifest.exists():
        raise MetalBackendError(f"Rust/Metal manifest not found at {manifest}")

    target_dir = _target_dir()
    dylib = target_dir / "release" / "libedgecritic_metal.dylib"
    if dylib.exists() and not _needs_rebuild(dylib):
        return dylib

    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    result = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        cwd=str(_repo_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise MetalBackendError(f"failed to build Rust/Metal backend: {message}")
    if not dylib.exists():
        raise MetalBackendError(f"Rust/Metal build succeeded but {dylib} was not produced")
    return dylib


def _needs_rebuild(dylib: Path) -> bool:
    dylib_mtime = dylib.stat().st_mtime
    manifest = _crate_manifest()
    crate_root = manifest.parent
    build_inputs = [manifest, crate_root / "Cargo.lock"]
    build_inputs.extend((crate_root / "src").rglob("*.rs"))
    return any(path.exists() and path.stat().st_mtime > dylib_mtime for path in build_inputs)


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_library_path()))
    lib.edgecritic_metal_wvf_convolve_pair.argtypes = [
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
    lib.edgecritic_metal_wvf_convolve_pair.restype = ctypes.c_int
    return lib


def metal_backend_available() -> bool:
    """Return whether the local machine can build and load the Metal backend."""
    try:
        _load_library()
    except MetalBackendError:
        return False
    except OSError:
        return False
    return True


def wvf_radius_gradients_metal(
    image: np.ndarray,
    kernels: WVFRadiusKernels,
    output_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic WVF ``Gx`` and ``Gy`` using the Rust/Metal backend."""
    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("wvf_radius_gradients_metal expects a 2-D image")

    offsets = kernels.offsets_xy.astype(np.int32, copy=False)
    dx = np.ascontiguousarray(offsets[:, 0], dtype=np.int32)
    dy = np.ascontiguousarray(offsets[:, 1], dtype=np.int32)
    wx = np.ascontiguousarray(kernels.weights_x, dtype=np.float32)
    wy = np.ascontiguousarray(kernels.weights_y, dtype=np.float32)
    out_x = np.empty(img.size, dtype=np.float32)
    out_y = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)

    h, w = img.shape
    status = _load_library().edgecritic_metal_wvf_convolve_pair(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint(w),
        ctypes.c_uint(h),
        dx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dy.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        wx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint(kernels.support_size),
        out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    if status != 0:
        raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))

    dtype = np.dtype(output_dtype)
    gx = out_x.reshape(img.shape)
    gy = out_y.reshape(img.shape)
    if dtype == np.dtype(np.float32):
        return gx, gy
    return gx.astype(dtype), gy.astype(dtype)
