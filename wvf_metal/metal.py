"""Thin Python bindings for the standalone WVF native backends."""

from __future__ import annotations

import ctypes
import contextlib
import json
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
    "vkfft": 3,
    "fft": 3,
}

_FFT_VARIANT_IDS = {3}
_FFT_BACKENDS = {"auto", "cpu", "vkfft", "metal", "gpu"}


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _crate_manifest() -> Path:
    return _package_root() / "Cargo.toml"


def _target_dir() -> Path:
    return _package_root() / "build" / "target"


def _build_fingerprint_path() -> Path:
    return _target_dir() / "release" / ".wvf_build_fingerprint.json"


def _build_fingerprint() -> dict[str, object]:
    env_keys = (
        "WVF_CUDA_HOME",
        "CUDA_HOME",
        "CUDA_PATH",
        "WVF_CUDA_HOST_CXX",
        "CUDAHOSTCXX",
        "CXX",
    )
    return {
        "system": platform.system(),
        "env": {key: os.environ.get(key) for key in env_keys},
    }


def _stored_build_fingerprint(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _candidate_linux_cuda_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    direct = os.environ.get("WVF_CUDA_LIB_DIR")
    if direct:
        candidates.append(Path(direct))

    for key in ("WVF_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"):
        raw_root = os.environ.get(key)
        if not raw_root:
            continue
        root = Path(raw_root)
        candidates.append(root / "lib64")
        candidates.append(root / "targets" / "x86_64-linux" / "lib")

    for root in (
        Path("/usr/local/cuda"),
        Path("/usr/local/MATLAB/R2024b/sys/cuda/glnxa64/cuda"),
        Path("/usr/local/MATLAB/R2025a/sys/cuda/glnxa64/cuda"),
    ):
        candidates.append(root / "lib64")
        candidates.append(root / "targets" / "x86_64-linux" / "lib")

    for matlab_root in (
        Path("/usr/local/MATLAB/R2024b"),
        Path("/usr/local/MATLAB/R2025a"),
    ):
        candidates.append(matlab_root / "bin" / "glnxa64")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


@lru_cache(maxsize=1)
def _linux_cuda_runtime_lib_dir() -> Path | None:
    if platform.system() != "Linux":
        return None

    for candidate in _candidate_linux_cuda_lib_dirs():
        if not candidate.is_dir():
            continue
        has_cudart = any((candidate / name).exists() for name in ("libcudart.so", "libcudart.so.12"))
        has_nvrtc = any((candidate / name).exists() for name in ("libnvrtc.so", "libnvrtc.so.12"))
        if has_cudart and has_nvrtc:
            return candidate
    return None


def _resolve_versioned_cuda_library(directory: Path, base_name: str) -> Path | None:
    exact = directory / base_name
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"{base_name}.*"))
    return matches[0] if matches else None


def _prepend_env_path(name: str, entry: Path) -> None:
    if not entry:
        return
    current = os.environ.get(name, "")
    parts = [str(entry)]
    if current:
        parts.append(current)
    os.environ[name] = os.pathsep.join(parts)


@lru_cache(maxsize=1)
def _prepare_linux_cuda_runtime() -> None:
    lib_dir = _linux_cuda_runtime_lib_dir()
    if lib_dir is None:
        return

    _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
    for library in ("libcudart.so", "libnvrtc.so", "libnvrtc-builtins.so"):
        resolved = _resolve_versioned_cuda_library(lib_dir, library)
        if resolved is None:
            continue
        try:
            ctypes.CDLL(str(resolved), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError:
            continue


def _library_path() -> Path:
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        raise MetalBackendError("native WVF backend is only available on macOS or Linux")
    if shutil.which("cargo") is None:
        raise MetalBackendError("cargo is required to build the native WVF backend")

    manifest = _crate_manifest()
    if not manifest.exists():
        raise MetalBackendError(f"Cargo manifest not found at {manifest}")

    target_dir = _target_dir()
    suffix = ".dylib" if system == "Darwin" else ".so"
    library = target_dir / "release" / f"libwvf_metal_backend{suffix}"
    fingerprint = _build_fingerprint()
    if (
        library.exists()
        and not _needs_rebuild(library)
        and _stored_build_fingerprint(_build_fingerprint_path()) == fingerprint
    ):
        return library

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
        raise MetalBackendError(f"failed to build native backend: {message}")
    if not library.exists():
        raise MetalBackendError(f"native build succeeded but {library} was not produced")
    fingerprint_path = _build_fingerprint_path()
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
    return library


def _needs_rebuild(library: Path) -> bool:
    dylib_mtime = library.stat().st_mtime
    root = _package_root()
    build_inputs = [root / "Cargo.toml", root / "Cargo.lock"]
    build_inputs.extend((root / "rust").glob("build.rs"))
    build_inputs.extend((root / "rust").rglob("*.rs"))
    build_inputs.extend((root / "rust").rglob("*.metal"))
    build_inputs.extend((root / "rust").rglob("*.cpp"))
    build_inputs.extend((root / "rust").rglob("*.cu"))
    build_inputs.extend((root / "rust" / "third_party").rglob("*.h"))
    build_inputs.extend((root / "rust" / "third_party").rglob("*.hpp"))
    return any(path.exists() and path.stat().st_mtime > dylib_mtime for path in build_inputs)


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    _prepare_linux_cuda_runtime()
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
    """Return whether this machine can build and load the native backend."""
    try:
        _load_library()
    except (MetalBackendError, OSError):
        return False
    return True


def _normalize_fft_backend(fft_backend: str | None) -> str:
    normalized = "auto" if fft_backend is None else str(fft_backend).strip().lower()
    if normalized not in _FFT_BACKENDS:
        raise ValueError("fft_backend must be 'auto', 'cpu', or 'vkfft'")
    if normalized in {"metal", "gpu"}:
        return "vkfft"
    return normalized


def _variant_is_fft(variant: str) -> bool:
    return _variant_id(variant) in _FFT_VARIANT_IDS


def _resolve_fft_backend(variant: str, fft_backend: str | None) -> str | None:
    if not _variant_is_fft(variant):
        chosen = _normalize_fft_backend(fft_backend)
        if chosen != "auto":
            raise ValueError("fft_backend only applies to variant='fft' or variant='vkfft'")
        return None

    return _normalize_fft_backend(fft_backend)


def _checked_device_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None
    checked = int(device_index)
    if checked < 0:
        raise ValueError("device_index must be non-negative")
    return checked


@contextlib.contextmanager
def _temporary_env_var(name: str, value: str | None):
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _variant_id(variant: str) -> int:
    try:
        return _VARIANTS[str(variant).lower()]
    except KeyError as exc:
        raise ValueError(
            "variant must be 'split', 'antipodal', 'direct', or 'fft'/'vkfft'"
        ) from exc


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


def _run_native_gradients(
    img: np.ndarray,
    radius: int,
    degree: int,
    variant: str,
    fft_backend: str | None,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    with _temporary_env_var("WVF_METAL_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "WVF_METAL_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
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


def _run_native_magnitude_angle(
    img: np.ndarray,
    radius: int,
    degree: int,
    variant: str,
    fft_backend: str | None,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    magnitude = np.empty(img.size, dtype=np.float32)
    angle = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    with _temporary_env_var("WVF_METAL_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "WVF_METAL_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
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


def wvf_gradients_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: str = "split",
    fft_backend: str | None = "auto",
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standalone WVF ``Gx`` and ``Gy`` through the native backends."""
    img = _as_float_image(image)
    chosen_fft_backend = _resolve_fft_backend(variant, fft_backend)
    checked_device_index = _checked_device_index(device_index)

    if platform.system() not in {"Darwin", "Linux"}:
        raise MetalBackendError(
            "wvf_metal requires macOS or Linux for the native extension."
        )
    return _run_native_gradients(
        img,
        radius=radius,
        degree=degree,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )


def wvf_magnitude_angle_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    variant: str = "split",
    fft_backend: str | None = "auto",
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute WVF components, magnitude, and angle through the native backends."""
    img = _as_float_image(image)
    chosen_fft_backend = _resolve_fft_backend(variant, fft_backend)
    checked_device_index = _checked_device_index(device_index)

    if platform.system() not in {"Darwin", "Linux"}:
        raise MetalBackendError(
            "wvf_metal requires macOS or Linux for the native extension."
        )
    return _run_native_magnitude_angle(
        img,
        radius=radius,
        degree=degree,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )
