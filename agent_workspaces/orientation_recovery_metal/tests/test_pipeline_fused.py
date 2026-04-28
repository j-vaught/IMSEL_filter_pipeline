"""Acceptance tests for the fused Metal WVF, LF, and recovery pipeline."""

from __future__ import annotations

import gc
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from edgecritic.lf._metal import lf_stack
    from edgecritic.pipeline._metal import (
        pipeline_backend_available,
        wvf_lf_recover_metal,
    )
    from edgecritic.recovery._metal import recover_two_peaks_metal
    from edgecritic.wvf._metal import wvf_radius_gradients_metal
    from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels

    METAL_OK = pipeline_backend_available()
except Exception as ex:
    METAL_OK = False
    _IMPORT_ERR = repr(ex)

ABS_TOL = 1e-5
_IMPORT_ERR = "backend unavailable"


@lru_cache(maxsize=1)
def _full_image() -> np.ndarray:
    from PIL import Image

    img_path = (
        ROOT
        / "example_images/synthetic_nested_shapes/clean/4096"
        / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png"
    )
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    rgb_n = np.clip(
        rgb + rng.normal(0.0, 13.0, rgb.shape).astype(np.float32),
        0.0,
        255.0,
    )
    return (
        0.2126 * rgb_n[..., 0]
        + 0.7152 * rgb_n[..., 1]
        + 0.0722 * rgb_n[..., 2]
    ).astype(np.float32)


def _run_unfused(
    image: np.ndarray,
    radius: int,
    degree: int,
    lf_half_length: int,
    n_orientations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kernels = build_wvf_radius_kernels(radius=radius, order=degree)
    gx, gy = wvf_radius_gradients_metal(image, kernels, output_dtype=np.float32)
    stack = lf_stack(
        gx,
        gy,
        lf_half_length=lf_half_length,
        n_orientations=n_orientations,
        output_dtype=np.float32,
        method="box",
    )
    h, w = image.shape
    response = stack.transpose(1, 2, 0).reshape(h * w, n_orientations).copy()
    del gx, gy, stack
    gc.collect()

    angles = np.linspace(0.0, math.pi, n_orientations, endpoint=False)
    out = recover_two_peaks_metal(
        angles,
        response,
        tau_sec_floor=0.40,
        tau_validity=0.10,
        dense_n=500,
        min_sep_frac=0.125,
    )
    del response
    gc.collect()
    return tuple(arr.reshape(image.shape) for arr in out[:4]) + (out[4].reshape(image.shape),)


def _run_fused(
    image: np.ndarray,
    radius: int,
    degree: int,
    lf_half_length: int,
    n_orientations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return wvf_lf_recover_metal(
        image,
        radius=radius,
        degree=degree,
        lf_half_length=lf_half_length,
        n_orientations=n_orientations,
        tau_sec_floor=0.40,
        tau_validity=0.10,
        dense_n=500,
        min_sep_frac=0.125,
        method="box",
    )


def _assert_outputs_match(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tag: str,
) -> None:
    names = ("theta_primary", "M_primary", "theta_secondary", "M_secondary")
    for name, got, want in zip(names, actual[:4], expected[:4]):
        got_nan = np.isnan(got)
        want_nan = np.isnan(want)
        if not np.array_equal(got_nan, want_nan):
            n_bad = int((got_nan ^ want_nan).sum())
            raise AssertionError(f"{tag} {name} NaN mask differs on {n_bad} pixels")
        mask = ~got_nan
        max_abs = float(np.max(np.abs(got[mask] - want[mask]))) if mask.any() else 0.0
        print(f"  [{tag}] {name}: max abs {max_abs:.3e}")
        if max_abs > ABS_TOL:
            raise AssertionError(f"{tag} {name} max abs {max_abs:.3e} > {ABS_TOL}")

    if not np.array_equal(actual[4], expected[4]):
        n_bad = int((actual[4] != expected[4]).sum())
        raise AssertionError(f"{tag} validity differs on {n_bad} pixels")
    print(f"  [{tag}] v: exact")


def _check_config(radius: int, degree: int, lf_half_length: int) -> None:
    image = _full_image()
    tag = f"r={radius}, d={degree}, m={lf_half_length}"
    print(f"[pipeline] {tag}")
    expected = _run_unfused(image, radius, degree, lf_half_length, 64)
    actual = _run_fused(image, radius, degree, lf_half_length, 64)
    _assert_outputs_match(actual, expected, tag)
    del expected, actual
    gc.collect()
    print("  PASS")


def test_pipeline_correctness():
    if not METAL_OK:
        raise RuntimeError(f"Metal fused pipeline backend unavailable: {_IMPORT_ERR}")
    _check_config(radius=9, degree=3, lf_half_length=60)


def test_pipeline_correctness_param_sweep():
    if not METAL_OK:
        raise RuntimeError("Metal fused pipeline backend unavailable")
    for radius in (5, 9):
        for degree in (1, 3):
            for lf_half_length in (40, 60, 80, 100):
                _check_config(radius=radius, degree=degree, lf_half_length=lf_half_length)


def test_pipeline_speed():
    if not METAL_OK:
        raise RuntimeError("Metal fused pipeline backend unavailable")
    image = _full_image()
    _ = _run_fused(image, 9, 3, 60, 64)

    t0 = time.perf_counter()
    _ = _run_fused(image, 9, 3, 60, 64)
    elapsed = time.perf_counter() - t0

    print(f"  fused full-image pipeline: {elapsed*1000:.0f} ms")
    if elapsed > 1.5:
        raise AssertionError(f"fused pipeline {elapsed*1000:.0f} ms > 1500 ms target")
    print("  PASS")


if __name__ == "__main__":
    failed = 0
    for name, fn in [
        ("test_pipeline_correctness", test_pipeline_correctness),
        ("test_pipeline_correctness_param_sweep", test_pipeline_correctness_param_sweep),
        ("test_pipeline_speed", test_pipeline_speed),
    ]:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as ex:
            print(f"  FAIL: {ex}")
            failed += 1
    sys.exit(failed)
