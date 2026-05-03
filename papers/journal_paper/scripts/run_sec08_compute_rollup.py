#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_method
from section8_common import CONTRAST, compile_plot
from wvf_metal.metal import fft_gradients_with_kernel


IMAGE_SIZE = 4096
RUNS = 10
WARMUP_RUNS = 1
DEFAULT_FFT_BACKEND = "vkfft"
VALIDATION_SUMMARY = (
    ROOT
    / "papers"
    / "journal_paper"
    / "figures"
    / "data"
    / "sec08_baseline_validation"
    / "sec08_baseline_validation_summary.json"
)
BACKENDS = ("scipy_cpu", "cpu_fft", "vkfft_cuda")


def _render_multiscale_scene(image_size: int = IMAGE_SIZE) -> np.ndarray:
    scale = float(image_size) / 1024.0
    yy, xx = np.meshgrid(
        np.arange(int(image_size), dtype=np.float64),
        np.arange(int(image_size), dtype=np.float64),
        indexing="ij",
    )
    scene = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
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
        cx = float(center_xy[0]) * scale
        cy = float(center_xy[1]) * scale
        radius = 0.5 * float(diameter) * scale
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        scene[mask] = float(CONTRAST)
    return scene


def _load_validation_roster(summary_json: Path) -> list[dict[str, object]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    roster = payload.get("method_roster")
    if not isinstance(roster, list):
        raise ValueError(f"{summary_json} is missing method_roster")
    return roster


def _timed_scipy(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndimage.correlate(np.asarray(image, dtype=np.float64), np.asarray(kernel_x, dtype=np.float64), mode="reflect")
    gy = ndimage.correlate(np.asarray(image, dtype=np.float64), np.asarray(kernel_y, dtype=np.float64), mode="reflect")
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _timed_fft(
    image: np.ndarray,
    support_half_extent: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = fft_gradients_with_kernel(
        np.asarray(image, dtype=np.float32),
        radius=int(support_half_extent),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=str(fft_backend),
        device_index=device_index,
    )
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _measure_backend(name: str, fn, runs: int, warmup_runs: int) -> dict[str, float]:
    for _ in range(int(warmup_runs)):
        fn()
    samples = []
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    values = np.asarray(samples, dtype=np.float64)
    median_s = float(np.median(values))
    p95_s = float(np.percentile(values, 95.0))
    print(f"sec85 {name} median={median_s:.6e}s p95={p95_s:.6e}s")
    return {
        "median_seconds": median_s,
        "p95_seconds": p95_s,
        "samples_seconds": [float(value) for value in values.tolist()],
    }


def _throughput_mp_s(seconds: float) -> float:
    return float(IMAGE_SIZE * IMAGE_SIZE) / max(float(seconds), 1.0e-12) / 1.0e6


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    summary_json: Path,
    validation_summary: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
) -> dict[str, Path]:
    image32 = _render_multiscale_scene(IMAGE_SIZE)
    image64 = np.asarray(image32, dtype=np.float64)
    roster = _load_validation_roster(validation_summary)
    rows = []

    for row in roster:
        method = str(row["method"])
        label = str(row["label"])
        config = dict(row["config"])
        kernel = build_method(method, **config)
        kernel_x = np.asarray(kernel.kernel_x, dtype=np.float64)
        kernel_y = np.asarray(kernel.kernel_y, dtype=np.float64)

        backend_specs = (
            ("scipy_cpu", lambda: _timed_scipy(image64, kernel_x, kernel_y)),
            ("cpu_fft", lambda: _timed_fft(image32, kernel.support_half_extent, kernel_x, kernel_y, "cpu", device_index)),
            ("vkfft_cuda", lambda: _timed_fft(image32, kernel.support_half_extent, kernel_x, kernel_y, str(fft_backend), device_index)),
        )
        for backend_name, backend_fn in backend_specs:
            metrics = _measure_backend(
                f"{method}_{backend_name}",
                backend_fn,
                runs=RUNS,
                warmup_runs=WARMUP_RUNS,
            )
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "config": config,
                    "config_label": str(kernel.config_label),
                    "support_half_extent": int(kernel.support_half_extent),
                    "white_noise_gain": float(kernel.white_noise_gain),
                    "backend": str(backend_name),
                    "median_seconds": float(metrics["median_seconds"]),
                    "p95_seconds": float(metrics["p95_seconds"]),
                    "median_throughput_mp_s": _throughput_mp_s(float(metrics["median_seconds"])),
                    "p95_throughput_mp_s": _throughput_mp_s(float(metrics["p95_seconds"])),
                    "samples_seconds": metrics["samples_seconds"],
                }
            )

    payload = {
        "title": "Section 8.5 compute and throughput",
        "subtitle": "Validation-selected operating points on a fixed 4096^2 multi-scale composite scene. CPU and CUDA timing only.",
        "config": {
            "image_size_px": int(IMAGE_SIZE),
            "runs": int(RUNS),
            "warmup_runs": int(WARMUP_RUNS),
            "backend_order": list(BACKENDS),
            "validation_summary": str(validation_summary.relative_to(ROOT)),
            "fft_backend_requested": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
        },
        "method_order": [str(row["method"]) for row in roster],
        "rows": rows,
    }
    _write_json(summary_json, payload)

    outputs = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec08_compute_rollup.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_compute_rollup.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 8.5 compute and throughput comparison.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_compute_rollup" / "sec08_compute_rollup_summary.json",
        help="Path for the Section 8.5 summary JSON.",
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=VALIDATION_SUMMARY,
        help="Path to the Section 8.1 validation summary used to load the selected operating points.",
    )
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("vkfft",), help="CUDA FFT backend to use for the GPU column.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_experiment(
        summary_json=args.summary_json.resolve(),
        validation_summary=args.validation_summary.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
