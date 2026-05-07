"""Thin Python bindings for the standalone WVF native backends."""

from __future__ import annotations

import ctypes
import contextlib
import json
import os
import platform
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


class MetalBackendError(RuntimeError):
    """Raised when the local Metal backend cannot run."""


_VARIANTS = {
    "direct": 0,
    "antipodal": 1,
    "split": 2,
    "fft": 3,
}

_FFT_VARIANT_IDS = {3}
_FFT_BACKENDS = {"auto", "cpu", "vkfft"}
_AUTO_FFT_CACHE_VERSION = 3
_AUTO_FFT_CACHE: dict[str, str] | None = None

VARIANT_NAMES = tuple(dict.fromkeys(_VARIANTS))
FFT_BACKEND_NAMES = ("auto", "cpu", "vkfft")


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


def _user_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "fast_wvf"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "fast_wvf"


def _auto_fft_cache_path() -> Path:
    return _user_cache_dir() / "fft_backend_auto_v2.json"


def _load_auto_fft_cache() -> dict[str, str]:
    global _AUTO_FFT_CACHE
    if _AUTO_FFT_CACHE is not None:
        return _AUTO_FFT_CACHE

    path = _auto_fft_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {}

    if (
        isinstance(payload, dict)
        and payload.get("version") == _AUTO_FFT_CACHE_VERSION
        and isinstance(payload.get("entries"), dict)
    ):
        entries = {
            str(key): str(value)
            for key, value in payload["entries"].items()
            if value in {"cpu", "vkfft"}
        }
    else:
        entries = {}
    _AUTO_FFT_CACHE = entries
    return _AUTO_FFT_CACHE


def _store_auto_fft_cache() -> None:
    if _AUTO_FFT_CACHE is None:
        return
    path = _auto_fft_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _AUTO_FFT_CACHE_VERSION,
        "entries": _AUTO_FFT_CACHE,
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _auto_fft_cache_key(
    image: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    device_index: int | None,
    output_mode: str,
) -> str:
    key = {
        "build": _build_fingerprint(),
        "degree": int(degree),
        "device_index": 0 if device_index is None else int(device_index),
        "height": int(image.shape[0]),
        "host": platform.node(),
        "machine": platform.machine(),
        "normalize_coords": bool(normalize_coords),
        "output_mode": output_mode,
        "radius": int(radius),
        "system": platform.system(),
        "width": int(image.shape[1]),
    }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _cached_auto_fft_backend(
    image: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    device_index: int | None,
    output_mode: str,
) -> str | None:
    cache = _load_auto_fft_cache()
    return cache.get(
        _auto_fft_cache_key(
            image,
            radius,
            degree,
            normalize_coords,
            device_index,
            output_mode,
        )
    )


def _cache_auto_fft_backend(
    image: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    device_index: int | None,
    output_mode: str,
    backend: str,
) -> None:
    if backend not in {"cpu", "vkfft"}:
        return
    cache = _load_auto_fft_cache()
    cache[
        _auto_fft_cache_key(
            image,
            radius,
            degree,
            normalize_coords,
            device_index,
            output_mode,
        )
    ] = backend
    _store_auto_fft_cache()


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
    library = target_dir / "release" / f"libfast_wvf_backend{suffix}"
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
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.fast_wvf_gradients.argtypes = gradient_args
    lib.fast_wvf_gradients.restype = ctypes.c_int

    gradient_with_kernel_args = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.fast_wvf_fft_gradients_with_kernel.argtypes = gradient_with_kernel_args
    lib.fast_wvf_fft_gradients_with_kernel.restype = ctypes.c_int

    magnitude_angle_args = gradient_args[:9] + [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.fast_wvf_magnitude_angle.argtypes = magnitude_angle_args
    lib.fast_wvf_magnitude_angle.restype = ctypes.c_int

    magnitude_args = gradient_args[:7] + [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.fast_wvf_magnitude.argtypes = magnitude_args
    lib.fast_wvf_magnitude.restype = ctypes.c_int

    magnitude_orientation_args = gradient_args[:7] + [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.fast_wvf_magnitude_orientation.argtypes = magnitude_orientation_args
    lib.fast_wvf_magnitude_orientation.restype = ctypes.c_int
    return lib


def metal_backend_available() -> bool:
    """Return whether this machine can build and load the native backend."""
    try:
        _load_library()
    except (MetalBackendError, OSError):
        return False
    return True


def backend_info() -> dict[str, object]:
    """Return basic installation and backend diagnostics."""
    native_error: str | None = None
    native_available = True
    try:
        _load_library()
    except (MetalBackendError, OSError) as exc:
        native_available = False
        native_error = str(exc)

    cuda_runtime_dir = _linux_cuda_runtime_lib_dir()
    return {
        "package": "fast-wvf",
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "package_root": str(_package_root()),
        "target_dir": str(_target_dir()),
        "cache_dir": str(_user_cache_dir()),
        "cargo_in_path": shutil.which("cargo") is not None,
        "native_backend_available": native_available,
        "native_backend_error": native_error,
        "spatial_variants_supported": platform.system() == "Darwin",
        "fft_variants_supported": platform.system() in {"Darwin", "Linux"},
        "available_variants": VARIANT_NAMES,
        "available_fft_backends": FFT_BACKEND_NAMES,
        "detected_cuda_runtime_lib_dir": (
            None if cuda_runtime_dir is None else str(cuda_runtime_dir)
        ),
    }


def _normalize_fft_backend(fft_backend: str | None) -> str:
    normalized = "auto" if fft_backend is None else str(fft_backend).strip().lower()
    if normalized not in _FFT_BACKENDS:
        raise ValueError("fft_backend must be 'auto', 'cpu', or 'vkfft'")
    return normalized


def _variant_is_fft(variant: str) -> bool:
    return _variant_id(variant) in _FFT_VARIANT_IDS


def _resolve_fft_backend(variant: str, fft_backend: str | None) -> str | None:
    if not _variant_is_fft(variant):
        chosen = _normalize_fft_backend(fft_backend)
        if chosen != "auto":
            raise ValueError("fft_backend only applies to variant='fft'")
        return None

    return _normalize_fft_backend(fft_backend)


def _checked_device_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None
    checked = int(device_index)
    if checked < 0:
        raise ValueError("device_index must be non-negative")
    return checked


def _checked_normalize_coords(normalize_coords: bool) -> ctypes.c_uint:
    return ctypes.c_uint(1 if bool(normalize_coords) else 0)


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
            "variant must be 'split', 'antipodal', 'direct', or 'fft'"
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
    normalize_coords: bool,
    variant: str,
    fft_backend: str | None,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    with _temporary_env_var("FAST_WVF_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "FAST_WVF_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
                status = _load_library().fast_wvf_gradients(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(w, "image width"),
                    _checked_uint(h, "image height"),
                    _checked_uint(radius, "radius"),
                    _checked_uint(degree, "degree"),
                    _checked_normalize_coords(normalize_coords),
                    ctypes.c_uint(_variant_id(variant)),
                    gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    error_buffer,
                    ctypes.c_size_t(len(error_buffer)),
                )
    _raise_if_failed(status, error_buffer)
    return gx.reshape(img.shape), gy.reshape(img.shape)


def _run_native_fft_gradients_with_kernel(
    img: np.ndarray,
    radius: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str | None,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.empty(img.size, dtype=np.float32)
    gy = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape
    kernel_width = int(kernel_x.shape[0])

    with _temporary_env_var("FAST_WVF_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "FAST_WVF_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
                status = _load_library().fast_wvf_fft_gradients_with_kernel(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(w, "image width"),
                    _checked_uint(h, "image height"),
                    _checked_uint(radius, "radius"),
                    kernel_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    kernel_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(kernel_width, "kernel width"),
                    gx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    gy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    error_buffer,
                    ctypes.c_size_t(len(error_buffer)),
                )
    _raise_if_failed(status, error_buffer)
    return gx.reshape(img.shape), gy.reshape(img.shape)


def _run_native_magnitude(
    img: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    variant: str,
    fft_backend: str | None,
    device_index: int | None,
) -> np.ndarray:
    magnitude = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    with _temporary_env_var("FAST_WVF_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "FAST_WVF_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
                status = _load_library().fast_wvf_magnitude(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(w, "image width"),
                    _checked_uint(h, "image height"),
                    _checked_uint(radius, "radius"),
                    _checked_uint(degree, "degree"),
                    _checked_normalize_coords(normalize_coords),
                    ctypes.c_uint(_variant_id(variant)),
                    magnitude.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    error_buffer,
                    ctypes.c_size_t(len(error_buffer)),
                )
    _raise_if_failed(status, error_buffer)
    return magnitude.reshape(img.shape)


def _run_native_magnitude_orientation(
    img: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    variant: str,
    fft_backend: str | None,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.empty(img.size, dtype=np.float32)
    angle = np.empty(img.size, dtype=np.float32)
    error_buffer = ctypes.create_string_buffer(4096)
    h, w = img.shape

    with _temporary_env_var("FAST_WVF_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "FAST_WVF_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
                status = _load_library().fast_wvf_magnitude_orientation(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(w, "image width"),
                    _checked_uint(h, "image height"),
                    _checked_uint(radius, "radius"),
                    _checked_uint(degree, "degree"),
                    _checked_normalize_coords(normalize_coords),
                    ctypes.c_uint(_variant_id(variant)),
                    magnitude.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    error_buffer,
                    ctypes.c_size_t(len(error_buffer)),
                )
    _raise_if_failed(status, error_buffer)
    return magnitude.reshape(img.shape), angle.reshape(img.shape)


def _run_native_magnitude_angle(
    img: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
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

    with _temporary_env_var("FAST_WVF_FFT_BACKEND", fft_backend):
        with _temporary_env_var(
            "WVF_GPU_DEVICE_INDEX",
            None if device_index is None else str(device_index),
        ):
            with _temporary_env_var(
                "FAST_WVF_DEVICE_INDEX",
                None if device_index is None else str(device_index),
            ):
                status = _load_library().fast_wvf_magnitude_angle(
                    img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    _checked_uint(w, "image width"),
                    _checked_uint(h, "image height"),
                    _checked_uint(radius, "radius"),
                    _checked_uint(degree, "degree"),
                    _checked_normalize_coords(normalize_coords),
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


def _benchmark_auto_fft_backend(
    img: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    variant: str,
    device_index: int | None,
    output_mode: str,
    runner,
):
    timings: dict[str, float] = {}
    results: dict[str, object] = {}
    errors: dict[str, Exception] = {}
    _load_library()

    for backend in ("vkfft", "cpu"):
        try:
            runner(
                img,
                radius=radius,
                degree=degree,
                normalize_coords=normalize_coords,
                variant=variant,
                fft_backend=backend,
                device_index=device_index,
            )
            started = time.perf_counter()
            result = runner(
                img,
                radius=radius,
                degree=degree,
                normalize_coords=normalize_coords,
                variant=variant,
                fft_backend=backend,
                device_index=device_index,
            )
        except (MetalBackendError, OSError, RuntimeError, ValueError) as exc:
            errors[backend] = exc
            continue
        timings[backend] = time.perf_counter() - started
        results[backend] = result

    if not timings:
        messages = ", ".join(f"{backend}: {error}" for backend, error in errors.items())
        raise MetalBackendError(f"no FFT backend succeeded for auto selection: {messages}")

    chosen_backend = min(timings, key=timings.__getitem__)
    _cache_auto_fft_backend(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        device_index=device_index,
        output_mode=output_mode,
        backend=chosen_backend,
    )
    return chosen_backend, results[chosen_backend]


def _run_auto_fft(
    img: np.ndarray,
    radius: int,
    degree: int,
    normalize_coords: bool,
    variant: str,
    device_index: int | None,
    output_mode: str,
    runner,
):
    cached_backend = _cached_auto_fft_backend(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        device_index=device_index,
        output_mode=output_mode,
    )
    if cached_backend is not None:
        try:
            return runner(
                img,
                radius=radius,
                degree=degree,
                normalize_coords=normalize_coords,
                variant=variant,
                fft_backend=cached_backend,
                device_index=device_index,
            )
        except (MetalBackendError, OSError, RuntimeError, ValueError):
            pass

    _, result = _benchmark_auto_fft_backend(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        variant=variant,
        device_index=device_index,
        output_mode=output_mode,
        runner=runner,
    )
    return result


def wvf_gradients_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    normalize_coords: bool = False,
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
            "fast_wvf requires macOS or Linux for the native extension."
        )
    if _variant_is_fft(variant) and chosen_fft_backend == "auto":
        return _run_auto_fft(
            img,
            radius=radius,
            degree=degree,
            normalize_coords=normalize_coords,
            variant=variant,
            device_index=checked_device_index,
            output_mode="gradients",
            runner=_run_native_gradients,
        )
    return _run_native_gradients(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )


def fft_gradients_with_kernel(
    image: np.ndarray,
    *,
    radius: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str | None = "auto",
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    img = _as_float_image(image)
    kernel_x_arr = np.asarray(kernel_x, dtype=np.float32)
    kernel_y_arr = np.asarray(kernel_y, dtype=np.float32)
    kernel_x32 = np.ascontiguousarray(np.flip(kernel_x_arr, axis=(0, 1)))
    kernel_y32 = np.ascontiguousarray(np.flip(kernel_y_arr, axis=(0, 1)))
    if kernel_x32.ndim != 2 or kernel_y32.ndim != 2:
        raise ValueError("kernel_x and kernel_y must be 2D arrays")
    if kernel_x32.shape != kernel_y32.shape:
        raise ValueError("kernel_x and kernel_y must have the same shape")
    if kernel_x32.shape[0] != kernel_x32.shape[1]:
        raise ValueError("kernel_x and kernel_y must be square")
    if kernel_x32.shape[0] % 2 != 1:
        raise ValueError("kernel_x and kernel_y must have odd width")
    checked_radius = int(radius)
    expected_width = 2 * checked_radius + 1
    if kernel_x32.shape != (expected_width, expected_width):
        raise ValueError(
            f"kernel_x and kernel_y must have shape {(expected_width, expected_width)} for radius={checked_radius}"
        )
    if platform.system() not in {"Darwin", "Linux"}:
        raise MetalBackendError(
            "fast_wvf requires macOS or Linux for the native extension."
        )
    return _run_native_fft_gradients_with_kernel(
        img,
        radius=checked_radius,
        kernel_x=kernel_x32,
        kernel_y=kernel_y32,
        fft_backend=_normalize_fft_backend(fft_backend),
        device_index=_checked_device_index(device_index),
    )


def wvf_magnitude_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    normalize_coords: bool = False,
    variant: str = "split",
    fft_backend: str | None = "auto",
    device_index: int | None = None,
) -> np.ndarray:
    """Compute WVF magnitude through the native backends."""
    img = _as_float_image(image)
    chosen_fft_backend = _resolve_fft_backend(variant, fft_backend)
    checked_device_index = _checked_device_index(device_index)

    if platform.system() not in {"Darwin", "Linux"}:
        raise MetalBackendError(
            "fast_wvf requires macOS or Linux for the native extension."
        )
    if _variant_is_fft(variant) and chosen_fft_backend == "auto":
        return _run_auto_fft(
            img,
            radius=radius,
            degree=degree,
            normalize_coords=normalize_coords,
            variant=variant,
            device_index=checked_device_index,
            output_mode="magnitude",
            runner=_run_native_magnitude,
        )
    return _run_native_magnitude(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )


def wvf_magnitude_orientation_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    normalize_coords: bool = False,
    variant: str = "split",
    fft_backend: str | None = "auto",
    device_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute WVF magnitude and angle through the native backends."""
    img = _as_float_image(image)
    chosen_fft_backend = _resolve_fft_backend(variant, fft_backend)
    checked_device_index = _checked_device_index(device_index)

    if platform.system() not in {"Darwin", "Linux"}:
        raise MetalBackendError(
            "fast_wvf requires macOS or Linux for the native extension."
        )
    if _variant_is_fft(variant) and chosen_fft_backend == "auto":
        return _run_auto_fft(
            img,
            radius=radius,
            degree=degree,
            normalize_coords=normalize_coords,
            variant=variant,
            device_index=checked_device_index,
            output_mode="magnitude_orientation",
            runner=_run_native_magnitude_orientation,
        )
    return _run_native_magnitude_orientation(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )


def wvf_magnitude_angle_metal(
    image: np.ndarray,
    radius: int,
    degree: int = 4,
    normalize_coords: bool = False,
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
            "fast_wvf requires macOS or Linux for the native extension."
        )
    if _variant_is_fft(variant) and chosen_fft_backend == "auto":
        return _run_auto_fft(
            img,
            radius=radius,
            degree=degree,
            normalize_coords=normalize_coords,
            variant=variant,
            device_index=checked_device_index,
            output_mode="magnitude_angle",
            runner=_run_native_magnitude_angle,
        )
    return _run_native_magnitude_angle(
        img,
        radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
        variant=variant,
        fft_backend=chosen_fft_backend,
        device_index=checked_device_index,
    )
