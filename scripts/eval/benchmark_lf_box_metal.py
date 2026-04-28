"""Benchmark exact LF Metal against the scanline box approximation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from edgecritic.lf._metal import lf_orientation_stack_metal, metal_backend_available


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_radius_deltas(value: str) -> list[int | None]:
    if value.strip().lower() == "auto":
        return [None]
    return _parse_int_list(value)


def _auto_box_radius(m: int, passes: int) -> int:
    if m <= 0:
        return 0
    sigma = m / 2.0
    radius = ((1.0 + 12.0 * sigma * sigma / passes) ** 0.5 - 1.0) / 2.0
    return max(1, int(round(radius)))


def _make_inputs(size: int, rng: np.random.Generator, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "random":
        return (
            rng.normal(size=(size, size)).astype(np.float32),
            rng.normal(size=(size, size)).astype(np.float32),
        )
    y, x = np.indices((size, size), dtype=np.float32)
    scale = max(float(size), 1.0)
    smooth_x = (
        np.sin(2.0 * np.pi * x / scale * 3.0)
        + 0.55 * np.cos(2.0 * np.pi * y / scale * 2.0)
        + 0.20 * np.sin(2.0 * np.pi * (x + y) / scale * 5.0)
    )
    smooth_y = (
        np.cos(2.0 * np.pi * x / scale * 2.5)
        - 0.45 * np.sin(2.0 * np.pi * y / scale * 3.5)
        + 0.15 * np.cos(2.0 * np.pi * (x - y) / scale * 4.0)
    )
    if kind == "smooth":
        return smooth_x.astype(np.float32), smooth_y.astype(np.float32)
    if kind == "mixed":
        noise_x = rng.normal(scale=0.05, size=(size, size)).astype(np.float32)
        noise_y = rng.normal(scale=0.05, size=(size, size)).astype(np.float32)
        return (smooth_x + noise_x).astype(np.float32), (smooth_y + noise_y).astype(np.float32)
    raise ValueError(f"unknown input kind: {kind}")


def _time_min_seconds(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _sample_stack(stack: np.ndarray, max_pixels: int) -> np.ndarray:
    pixel_count = stack.shape[1] * stack.shape[2]
    flat = stack.reshape(stack.shape[0], pixel_count)
    if max_pixels <= 0 or pixel_count <= max_pixels:
        return flat
    step = max(pixel_count // max_pixels, 1)
    idx = np.arange(0, pixel_count, step, dtype=np.int64)[:max_pixels]
    return flat[:, idx]


def _metrics(exact: np.ndarray, approx: np.ndarray, max_pixels: int) -> dict[str, float]:
    exact64 = _sample_stack(exact, max_pixels).astype(np.float64, copy=False)
    approx64 = _sample_stack(approx, max_pixels).astype(np.float64, copy=False)
    diff = approx64 - exact64
    exact_abs_mean = float(np.mean(np.abs(exact64)))
    exact_rms = float(np.sqrt(np.mean(exact64 * exact64)))
    rel_mae = float(np.mean(np.abs(diff))) / max(exact_abs_mean, 1e-12)
    rel_rmse = float(np.sqrt(np.mean(diff * diff))) / max(exact_rms, 1e-12)
    max_abs = float(np.max(np.abs(diff)))

    exact_flat = exact64.ravel()
    approx_flat = approx64.ravel()
    exact_std = float(np.std(exact_flat))
    approx_std = float(np.std(approx_flat))
    if exact_std <= 1e-12 or approx_std <= 1e-12:
        corr = 1.0 if max_abs <= 1e-6 else 0.0
    else:
        corr = float(np.corrcoef(exact_flat, approx_flat)[0, 1])

    exact_top = np.argmax(exact64, axis=0)
    approx_top = np.argmax(approx64, axis=0)
    n_orientations = exact.shape[0]
    cyclic_delta = np.abs(exact_top - approx_top)
    cyclic_delta = np.minimum(cyclic_delta, n_orientations - cyclic_delta)
    angle_step = np.pi / n_orientations

    return {
        "rel_mae": rel_mae,
        "rel_rmse": rel_rmse,
        "max_abs": max_abs,
        "corr": corr,
        "top1": float(np.mean(exact_top == approx_top)),
        "mean_angle_error_rad": float(np.mean(cyclic_delta) * angle_step),
        "metric_pixels": float(exact64.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_parse_int_list, default=[1024, 2048, 4096])
    parser.add_argument("--orientations", type=_parse_int_list, default=[16, 32])
    parser.add_argument("--m-values", type=_parse_int_list, default=[50, 80, 150, 300])
    parser.add_argument("--passes", type=_parse_int_list, default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--radius-deltas", type=_parse_radius_deltas, default=[None])
    parser.add_argument("--exact-execution", default="auto", choices=["auto", "projected", "direct"])
    parser.add_argument("--input-kind", default="random", choices=["random", "smooth", "mixed", "all"])
    parser.add_argument("--metric-samples", type=int, default=262144)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()

    if not metal_backend_available():
        raise SystemExit("Metal backend is unavailable on this machine")

    print(
        "input_kind,size,n_orientations,m,passes,radius,exact_s,box_s,speedup,"
        "rel_mae,rel_rmse,max_abs,corr,top1,mean_angle_error_rad,metric_pixels,"
        "exact_checksum,box_checksum",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    input_kinds = ["random", "smooth", "mixed"] if args.input_kind == "all" else [args.input_kind]
    for input_kind in input_kinds:
        for size in args.sizes:
            g_x, g_y = _make_inputs(size, rng, input_kind)

            for n_orientations in args.orientations:
                exact_out = np.empty((n_orientations, size, size), dtype=np.float32)
                box_out = np.empty_like(exact_out)
                for m in args.m_values:

                    def run_exact() -> np.ndarray:
                        return lf_orientation_stack_metal(
                            g_x,
                            g_y,
                            m=m,
                            n_orientations=n_orientations,
                            execution=args.exact_execution,
                            out=exact_out,
                        )

                    exact_s = _time_min_seconds(run_exact, args.warmup, args.repeat)
                    exact_checksum = float(
                        exact_out.reshape(-1)[:: max(exact_out.size // 4096, 1)].sum()
                    )

                    for passes in args.passes:
                        auto_radius = _auto_box_radius(m, passes)
                        for delta in args.radius_deltas:
                            radius = None if delta is None else max(0, auto_radius + delta)
                            reported_radius = auto_radius if radius is None else radius

                            def run_box() -> np.ndarray:
                                return lf_orientation_stack_metal(
                                    g_x,
                                    g_y,
                                    m=m,
                                    n_orientations=n_orientations,
                                    method="box",
                                    out=box_out,
                                    box_passes=passes,
                                    box_radius=radius,
                                )

                            box_s = _time_min_seconds(run_box, args.warmup, args.repeat)
                            values = _metrics(exact_out, box_out, args.metric_samples)
                            box_checksum = float(
                                box_out.reshape(-1)[:: max(box_out.size // 4096, 1)].sum()
                            )
                            print(
                                f"{input_kind},{size},{n_orientations},{m},{passes},{reported_radius},"
                                f"{exact_s:.6f},{box_s:.6f},{exact_s / box_s:.3f},"
                                f"{values['rel_mae']:.6e},{values['rel_rmse']:.6e},"
                                f"{values['max_abs']:.6e},{values['corr']:.6f},"
                                f"{values['top1']:.6f},{values['mean_angle_error_rad']:.6e},"
                                f"{int(values['metric_pixels'])},"
                                f"{exact_checksum:.6f},{box_checksum:.6f}",
                                flush=True,
                            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
