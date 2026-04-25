"""Benchmark the radius-based WVF Metal backend against the CPU path."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from edgecritic.wvf import build_wvf_radius_kernels
from edgecritic.wvf._metal import metal_backend_available, wvf_radius_gradients_metal
from edgecritic.wvf._radius_kernels import wvf_radius_gradients_cpu


def _time_min_seconds(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one image size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("image sizes must be positive")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--sizes", type=_parse_sizes, default=[128, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--large-cpu-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not metal_backend_available():
        raise SystemExit("Metal backend is unavailable on this machine")

    kernels = build_wvf_radius_kernels(radius=args.radius, order=args.order)
    rng = np.random.default_rng(args.seed)
    print(
        "size, radius, order, support, metal_s, cpu_s, speedup, max_abs_error",
        flush=True,
    )
    for size in args.sizes:
        image = rng.random((size, size), dtype=np.float32)
        metal_result: tuple[np.ndarray, np.ndarray] | None = None
        cpu_result: tuple[np.ndarray, np.ndarray] | None = None

        def run_metal() -> None:
            nonlocal metal_result
            metal_result = wvf_radius_gradients_metal(
                image,
                kernels,
                output_dtype=np.float32,
            )

        def run_cpu() -> None:
            nonlocal cpu_result
            cpu_result = wvf_radius_gradients_cpu(
                image,
                radius=args.radius,
                order=args.order,
                output_dtype=np.float32,
            )

        metal_s = _time_min_seconds(run_metal, args.warmup, args.repeat)
        cpu_repeat = args.repeat if size <= 1024 else args.large_cpu_repeat
        cpu_s = _time_min_seconds(run_cpu, 1, cpu_repeat)
        assert metal_result is not None
        assert cpu_result is not None
        max_abs_error = max(
            float(np.max(np.abs(metal_result[0] - cpu_result[0]))),
            float(np.max(np.abs(metal_result[1] - cpu_result[1]))),
        )
        print(
            f"{size}, {args.radius}, {args.order}, {kernels.support_size}, "
            f"{metal_s:.6f}, {cpu_s:.6f}, {cpu_s / metal_s:.2f}, {max_abs_error:.3e}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
