#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sec07_support_shape_matching import _build_spec
from wvf.radius import build_wvf_radius_kernels
from wvf_metal.metal import fft_gradients_with_kernel

IMAGE_SIZE = 1024
CONTRAST = 1.0
RUNS = 100
WARMUP_RUNS = 5
DEFAULT_FFT_BACKEND = "vkfft"
MAIN_CONFIGS = (
    {"label": "r5_d9", "radius": 5, "degree": 9},
    {"label": "r15_d11", "radius": 15, "degree": 11},
    {"label": "r50_d11", "radius": 50, "degree": 11},
)
ALTERNATIVE_SHAPES = ("triangle", "diamond", "hexagon", "octagon", "square")


def _render_multiscale_scene() -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(IMAGE_SIZE, dtype=np.float64), np.arange(IMAGE_SIZE, dtype=np.float64), indexing="ij")
    scene = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    specs = (
        ((192.0, 180.0), 8.0),
        ((336.0, 180.0), 16.0),
        ((512.0, 180.0), 32.0),
        ((720.0, 180.0), 64.0),
        ((848.0, 180.0), 128.0),
        ((240.0, 420.0), 12.0),
        ((420.0, 420.0), 24.0),
        ((612.0, 420.0), 48.0),
        ((824.0, 420.0), 96.0),
        ((512.0, 700.0), 160.0),
    )
    for center_xy, diameter in specs:
        radius = 0.5 * float(diameter)
        mask = (xx - float(center_xy[0])) ** 2 + (yy - float(center_xy[1])) ** 2 <= radius * radius
        scene[mask] = float(CONTRAST)
    return scene


def _disk_kernel(radius: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    kernels = build_wvf_radius_kernels(int(radius), order=int(degree), normalize_coords=True)
    return np.asarray(kernels.kernel_x, dtype=np.float64), np.asarray(kernels.kernel_y, dtype=np.float64)


def _timed_scipy(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndimage.correlate(np.asarray(image, dtype=np.float64), np.asarray(kernel_x, dtype=np.float64), mode="reflect")
    gy = ndimage.correlate(np.asarray(image, dtype=np.float64), np.asarray(kernel_y, dtype=np.float64), mode="reflect")
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _timed_fft(image: np.ndarray, radius: int, kernel_x: np.ndarray, kernel_y: np.ndarray, fft_backend: str, device_index: int | None) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = fft_gradients_with_kernel(
        np.asarray(image, dtype=np.float32),
        radius=int(radius),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _bench(name: str, fn, runs: int, warmup_runs: int) -> tuple[float, float]:
    for _ in range(int(warmup_runs)):
        fn()
    samples = []
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    values = np.asarray(samples, dtype=np.float64)
    mean_s = float(np.mean(values))
    std_s = float(np.std(values, ddof=0))
    print(f"sec712 {name} mean={mean_s:.6e}s std={std_s:.6e}s")
    return mean_s, std_s


def _throughput(mean_s: float) -> float:
    return float(IMAGE_SIZE * IMAGE_SIZE) / max(float(mean_s), 1.0e-12) / 1.0e6


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.12 figure cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def run_experiment(
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    image = _render_multiscale_scene()
    rows = []

    for config in MAIN_CONFIGS:
        kernel_x, kernel_y = _disk_kernel(int(config["radius"]), int(config["degree"]))
        scipy_mean, scipy_std = _bench(
            f"{config['label']}_scipy",
            lambda: _timed_scipy(image, kernel_x, kernel_y),
            runs=RUNS,
            warmup_runs=WARMUP_RUNS,
        )
        rows.append(
            {
                "panel": "main",
                "config_label": str(config["label"]),
                "backend": "scipy_cpu",
                "shape": "disk",
                "radius": int(config["radius"]),
                "degree": int(config["degree"]),
                "mean_seconds": scipy_mean,
                "std_seconds": scipy_std,
                "throughput_mp_s": _throughput(scipy_mean),
            }
        )

        cpu_mean, cpu_std = _bench(
            f"{config['label']}_cpufft",
            lambda: _timed_fft(image, int(config["radius"]), kernel_x, kernel_y, "cpu", device_index),
            runs=RUNS,
            warmup_runs=WARMUP_RUNS,
        )
        rows.append(
            {
                "panel": "main",
                "config_label": str(config["label"]),
                "backend": "cpu_fft",
                "shape": "disk",
                "radius": int(config["radius"]),
                "degree": int(config["degree"]),
                "mean_seconds": cpu_mean,
                "std_seconds": cpu_std,
                "throughput_mp_s": _throughput(cpu_mean),
            }
        )

        vk_mean, vk_std = _bench(
            f"{config['label']}_vkfft",
            lambda: _timed_fft(image, int(config["radius"]), kernel_x, kernel_y, str(fft_backend), device_index),
            runs=RUNS,
            warmup_runs=WARMUP_RUNS,
        )
        rows.append(
            {
                "panel": "main",
                "config_label": str(config["label"]),
                "backend": "vkfft_cuda",
                "shape": "disk",
                "radius": int(config["radius"]),
                "degree": int(config["degree"]),
                "mean_seconds": vk_mean,
                "std_seconds": vk_std,
                "throughput_mp_s": _throughput(vk_mean),
            }
        )

    for shape_name in ALTERNATIVE_SHAPES:
        spec = _build_spec(shape_name, 15.0, 11, True)
        mean_s, std_s = _bench(
            f"{shape_name}_scipy",
            lambda: _timed_scipy(image, spec.kernel_x, spec.kernel_y),
            runs=RUNS,
            warmup_runs=WARMUP_RUNS,
        )
        rows.append(
            {
                "panel": "alt_shapes",
                "config_label": "r15_d11",
                "backend": "scipy_cpu",
                "shape": str(shape_name),
                "radius": 15,
                "degree": 11,
                "mean_seconds": mean_s,
                "std_seconds": std_s,
                "throughput_mp_s": _throughput(mean_s),
            }
        )

    payload = {
        "title": "Section 7.12 compute cost rollup",
        "subtitle": "Disk support main panel uses precomputed-kernel application on a fixed 1024^2 multi-scale composite scene. Alternative shapes use the paper-local CPU path only.",
        "config": {
            "image_size_px": int(IMAGE_SIZE),
            "runs": int(RUNS),
            "warmup_runs": int(WARMUP_RUNS),
            "backend_columns": ["scipy_cpu", "cpu_fft", "vkfft_cuda"],
            "main_configs": MAIN_CONFIGS,
            "alternative_shapes": list(ALTERNATIVE_SHAPES),
            "fft_backend_requested": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
        },
        "rows": rows,
    }
    _write_json(summary_json, payload)
    return {"summary_json": summary_json}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.12 compute-cost rollup.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_compute_cost_rollup" / "sec07_compute_cost_rollup_summary.json",
        help="Path for the Section 7.12 summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("vkfft",), help="GPU FFT backend to use for the CUDA column.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ bar chart.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_experiment(
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )
    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_compute_cost_rollup.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_compute_cost_rollup.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["plot_pdf"] = figure_pdf
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
