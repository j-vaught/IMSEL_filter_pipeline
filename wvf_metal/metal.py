"""Thin Python bindings for the standalone WVF Metal backend."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


class MetalBackendError(RuntimeError):
    """Raised when the local Metal backend cannot run."""


_VARIANTS = {
    "direct": 0,
    "baseline": 0,
    "antipodal": 1,
    "split": 2,
    "optimized": 2,
    "auto": 2,
}


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
    gradient_args = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.wvf_metal_gradients.argtypes = gradient_args
    lib.wvf_metal_gradients.restype = ctypes.c_int

    magnitude_args = gradient_args[:8] + [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.wvf_metal_magnitude_angle.argtypes = magnitude_args
    lib.wvf_metal_magnitude_angle.restype = ctypes.c_int
    return lib


def metal_backend_available() -> bool:
    """Return whether this machine can build and load the Metal backend."""
    try:
        _load_library()
    except (MetalBackendError, OSError):
        return False
    return True


def _variant_id(variant: str) -> int:
    try:
        return _VARIANTS[str(variant).lower()]
    except KeyError as exc:
        raise ValueError("variant must be 'split', 'antipodal', or 'direct'") from exc


def _as_float_image(image: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("WVF Metal expects a 2-D image")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("image width and height must be positive")
    return img


def _checked_uint(value: int, name: str) -> ctypes.c_uint:
    intval = int(value)
    if intval < 0 or intval > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in uint32")
    return ctypes.c_uint(intval)


def _raise_if_failed(status: int, error_buffer: ctypes.Array[ctypes.c_char]) -> None:
    if status != 0:
        raise MetalBackendError(error_buffer.value.decode("utf-8", errors="replace"))


def wvf_gradients_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standalone WVF ``Gx`` and ``Gy`` on Metal."""
    img = _as_float_image(image)
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    status = _load_library().wvf_metal_gradients(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _checked_uint(w, "image width"),
        _checked_uint(h, "image height"),
        _checked_uint(radius, "radius"),
        _checked_uint(degree, "degree"),
        ctypes.c_uint(_variant_id(variant)),
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    _raise_if_failed(status, error_buffer)
    return gx.reshape(img.shape), gy.reshape(img.shape)


def wvf_magnitude_angle_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: str = "split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute WVF components, magnitude, and unsigned orientation angle on Metal."""
    img = _as_float_image(image)
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    magnitude = np.empty(img.size, dtype=np.float32)
    angle = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    status = _load_library().wvf_metal_magnitude_angle(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        _checked_uint(w, "image width"),
        _checked_uint(h, "image height"),
        _checked_uint(radius, "radius"),
        _checked_uint(degree, "degree"),
        ctypes.c_uint(_variant_id(variant)),
        gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        magnitude.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    _raise_if_failed(status, error_buffer)
    shape = img.shape
    return (
        gx.reshape(shape),
        gy.reshape(shape),
        magnitude.reshape(shape),
        angle.reshape(shape),
    )
