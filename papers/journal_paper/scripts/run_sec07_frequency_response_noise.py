#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from weighted_disk_sg import build_weighted_disk_kernels
from wvf_metal.metal import fft_gradients_with_kernel


RADIUS = 15
DEGREE = 3
NORMALIZE_COORDS = True
IMAGE_SIZE = 1024
CONTRAST = 1.0
FREQUENCIES = (1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0)
ORIENTATION_STEP_DEG = 5.0
AWGN_SNR_DB = (30.0, 20.0, 10.0)
POISSON_COUNTS = (1000.0, 100.0, 10.0)
SPECKLE_SIGMAS = (0.1, 0.4, 0.8)
VARIANCE_DRAWS = 100
BATCH_CASES = 16
INTERIOR_MARGIN = RADIUS
DEFAULT_FFT_BACKEND = "vkfft"
NOISE_TYPE_SEED_BASE = {
    "awgn": 1100,
    "poisson": 2200,
    "speckle": 3300,
}


@dataclass(frozen=True)
class GratingCase:
    orientation_deg: float
    frequency_cyc_px: float
    image: np.ndarray
    sin_basis: np.ndarray
    cos_basis: np.ndarray


def _orientation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(180.0 / float(step_deg)))
    return tuple(float(step_deg) * i for i in range(count))


def _local_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64), indexing="ij")
    return xx, yy


def _render_grating(phase_field: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * float(CONTRAST) * np.sin(np.asarray(phase_field, dtype=np.float64))


def _awgn_sigma(snr_db: float) -> float:
    return float(CONTRAST) / (10.0 ** (float(snr_db) / 20.0))


def _apply_noise(image: np.ndarray, noise_type: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    if noise_type == "awgn":
        return src + _awgn_sigma(float(severity)) * rng.normal(size=src.shape)
    if noise_type == "poisson":
        counts = float(severity)
        return rng.poisson(np.clip(src, 0.0, None) * counts).astype(np.float64) / counts
    if noise_type == "speckle":
        return src * (1.0 + float(severity) * rng.normal(size=src.shape))
    raise ValueError(f"unsupported noise_type {noise_type!r}")


def _tile_cases(cases: list[GratingCase]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    if not cases:
        raise ValueError("cannot tile an empty case batch")
    tile_h, tile_w = cases[0].image.shape
    cols = int(math.ceil(math.sqrt(len(cases))))
    rows = int(math.ceil(len(cases) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    placements: list[tuple[slice, slice]] = []
    for index, case in enumerate(cases):
        row = index // cols
        col = index % cols
        row_slice = slice(row * tile_h, (row + 1) * tile_h)
        col_slice = slice(col * tile_w, (col + 1) * tile_w)
        canvas[row_slice, col_slice] = np.asarray(case.image, dtype=np.float32)
        placements.append((row_slice, col_slice))
    return canvas, placements


def _apply_cases_batched(
    cases: list[GratingCase],
    radius: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    canvas, placements = _tile_cases(cases)
    gx_canvas, gy_canvas = fft_gradients_with_kernel(
        canvas,
        radius=int(radius),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    outputs = []
    for row_slice, col_slice in placements:
        outputs.append(
            (
                np.asarray(gx_canvas[row_slice, col_slice], dtype=np.float64).copy(),
                np.asarray(gy_canvas[row_slice, col_slice], dtype=np.float64).copy(),
            )
        )
    return outputs


def _fit_directional_response(
    directional: np.ndarray,
    sin_basis: np.ndarray,
    cos_basis: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    response = np.asarray(directional, dtype=np.float64)[mask] * (2.0 / float(CONTRAST))
    sin_vals = np.asarray(sin_basis, dtype=np.float64)[mask]
    cos_vals = np.asarray(cos_basis, dtype=np.float64)[mask]
    a_sin = 2.0 * float(np.mean(response * sin_vals))
    b_cos = 2.0 * float(np.mean(response * cos_vals))
    amplitude = math.hypot(a_sin, b_cos)
    phase = math.atan2(b_cos, a_sin)
    return amplitude, phase


def _wrap_phase(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _severity_label(noise_type: str, severity: float) -> str:
    if noise_type == "awgn":
        return f"{float(severity):g} dB"
    if noise_type == "poisson":
        return f"{int(severity)}"
    return f"{float(severity):g}"


def _severity_slug(noise_type: str, severity: float) -> str:
    if noise_type == "awgn":
        text = f"{float(severity):g}".replace(".", "p")
        return f"snr_{text}"
    if noise_type == "poisson":
        return f"counts_{int(severity)}"
    return "sigma_" + f"{float(severity):g}".replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ("frequency_cyc_px", "orientation_deg", "mag_error", "phase_error", "snr_or_severity")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst_bin = shutil.which("typst") or str(Path.home() / "bin" / "typst")
    subprocess.run(
        [
            typst_bin,
            "compile",
            "--root",
            str(ROOT),
            str(figure_src),
            str(figure_pdf),
        ],
        check=True,
        cwd=str(ROOT),
    )


def _build_clean_cases(xx: np.ndarray, yy: np.ndarray) -> list[GratingCase]:
    cases = []
    for orientation_deg in _orientation_values(ORIENTATION_STEP_DEG):
        theta = math.radians(float(orientation_deg))
        projection = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta)
        for frequency in FREQUENCIES:
            phase_field = 2.0 * math.pi * float(frequency) * projection
            cases.append(
                GratingCase(
                    orientation_deg=float(orientation_deg),
                    frequency_cyc_px=float(frequency),
                    image=np.asarray(_render_grating(phase_field), dtype=np.float32),
                    sin_basis=np.asarray(np.sin(phase_field), dtype=np.float64),
                    cos_basis=np.asarray(np.cos(phase_field), dtype=np.float64),
                )
            )
    return cases


def _evaluate_cases(
    cases: list[GratingCase],
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
    mask: np.ndarray,
) -> list[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    for batch_start in range(0, len(cases), int(BATCH_CASES)):
        batch = cases[batch_start : batch_start + int(BATCH_CASES)]
        responses = _apply_cases_batched(
            batch,
            radius=RADIUS,
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        for case, (gx, gy) in zip(batch, responses, strict=True):
            theta = math.radians(float(case.orientation_deg))
            directional = np.asarray(gx, dtype=np.float64) * math.cos(theta) + np.asarray(gy, dtype=np.float64) * math.sin(theta)
            amplitude, phase = _fit_directional_response(
                directional,
                case.sin_basis,
                case.cos_basis,
                mask,
            )
            omega = 2.0 * math.pi * float(case.frequency_cyc_px)
            outputs.append(
                {
                    "frequency_cyc_px": float(case.frequency_cyc_px),
                    "orientation_deg": float(case.orientation_deg),
                    "measured_magnitude": float(amplitude),
                    "measured_phase_rad": float(phase),
                    "mag_error": float(amplitude - omega),
                    "phase_error": float(_wrap_phase(phase - 0.5 * math.pi)),
                }
            )
    return outputs


def _noise_variance_records(
    noise_type: str,
    severities: tuple[float, ...],
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
    white_noise_gain: float,
) -> list[dict[str, object]]:
    interior = np.s_[INTERIOR_MARGIN:-INTERIOR_MARGIN, INTERIOR_MARGIN:-INTERIOR_MARGIN]
    base = np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    records = []
    for severity_index, severity in enumerate(severities):
        rng = np.random.default_rng(20260502 + NOISE_TYPE_SEED_BASE[noise_type] + severity_index)
        variances = []
        flat_cases: list[GratingCase] = []
        for _ in range(VARIANCE_DRAWS):
            noisy = _apply_noise(base, noise_type, float(severity), rng)
            flat_cases.append(
                GratingCase(
                    orientation_deg=0.0,
                    frequency_cyc_px=0.0,
                    image=np.asarray(noisy, dtype=np.float32),
                    sin_basis=np.zeros_like(noisy, dtype=np.float64),
                    cos_basis=np.zeros_like(noisy, dtype=np.float64),
                )
            )
        for batch_start in range(0, len(flat_cases), int(BATCH_CASES)):
            batch = flat_cases[batch_start : batch_start + int(BATCH_CASES)]
            responses = _apply_cases_batched(
                batch,
                radius=RADIUS,
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
            )
            for gx, _ in responses:
                variances.append(float(np.var(np.asarray(gx, dtype=np.float64)[interior])))

        analytical = None
        if noise_type == "awgn":
            analytical = float(_awgn_sigma(float(severity)) ** 2 * white_noise_gain)
        records.append(
            {
                "noise_type": noise_type,
                "severity_value": float(severity),
                "severity_label": _severity_label(noise_type, float(severity)),
                "empirical_variance": float(np.mean(np.asarray(variances, dtype=np.float64))),
                "analytical_variance": analytical,
            }
        )
    return records


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    kernels = build_weighted_disk_kernels(
        radius=RADIUS,
        degree=DEGREE,
        normalize_coords=NORMALIZE_COORDS,
        sigma_w=None,
    )
    kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
    kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    white_noise_gain = float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2))

    xx, yy = _local_grid(IMAGE_SIZE)
    interior_mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    interior_mask[:INTERIOR_MARGIN, :] = False
    interior_mask[-INTERIOR_MARGIN:, :] = False
    interior_mask[:, :INTERIOR_MARGIN] = False
    interior_mask[:, -INTERIOR_MARGIN:] = False

    clean_cases = _build_clean_cases(xx, yy)
    clean_records = _evaluate_cases(
        clean_cases,
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        fft_backend=fft_backend,
        device_index=device_index,
        mask=interior_mask,
    )

    noise_specs = {
        "awgn": AWGN_SNR_DB,
        "poisson": POISSON_COUNTS,
        "speckle": SPECKLE_SIGMAS,
    }
    noise_records: list[dict[str, object]] = []

    for noise_type, severities in noise_specs.items():
        csv_rows = []
        for severity_index, severity in enumerate(severities):
            rng = np.random.default_rng(20260502 + 10 * NOISE_TYPE_SEED_BASE[noise_type] + severity_index)
            noisy_cases = []
            for base_case in clean_cases:
                noisy_image = _apply_noise(np.asarray(base_case.image, dtype=np.float64), noise_type, float(severity), rng)
                noisy_cases.append(
                    GratingCase(
                        orientation_deg=base_case.orientation_deg,
                        frequency_cyc_px=base_case.frequency_cyc_px,
                        image=np.asarray(noisy_image, dtype=np.float32),
                        sin_basis=base_case.sin_basis,
                        cos_basis=base_case.cos_basis,
                    )
                )
            evaluated = _evaluate_cases(
                noisy_cases,
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
                mask=interior_mask,
            )
            for record in evaluated:
                row = {
                    "noise_type": noise_type,
                    "severity_value": float(severity),
                    "severity_label": _severity_label(noise_type, float(severity)),
                    **record,
                }
                noise_records.append(row)
                csv_rows.append(
                    {
                        "frequency_cyc_px": f"{record['frequency_cyc_px']:.17e}",
                        "orientation_deg": f"{record['orientation_deg']:.6f}",
                        "mag_error": f"{record['mag_error']:.17e}",
                        "phase_error": f"{record['phase_error']:.17e}",
                        "snr_or_severity": _severity_label(noise_type, float(severity)),
                    }
                )
            print(
                f"{noise_type} severity={_severity_label(noise_type, float(severity))} "
                f"mean_mag_error={np.mean([r['mag_error'] for r in evaluated]):.6e}"
            )
        _write_csv(
            output_dir / f"sec07_frequency_response_{noise_type}_r{RADIUS}_d{DEGREE}_normalized.csv",
            csv_rows,
        )

    variance_records = []
    for noise_type, severities in noise_specs.items():
        variance_records.extend(
            _noise_variance_records(
                noise_type=noise_type,
                severities=tuple(float(v) for v in severities),
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
                white_noise_gain=white_noise_gain,
            )
        )

    summary = {
        "title": "Section 7.8 frequency response and noise characterization",
        "subtitle": "Disk support, r = 15, d = 3, normalize_coords = True",
        "config": {
            "radius": RADIUS,
            "degree": DEGREE,
            "normalize_coords": NORMALIZE_COORDS,
            "image_size": IMAGE_SIZE,
            "contrast": CONTRAST,
            "frequencies_cyc_px": list(FREQUENCIES),
            "orientation_step_deg": ORIENTATION_STEP_DEG,
            "variance_draws": VARIANCE_DRAWS,
            "fft_backend": fft_backend,
            "device_index": device_index,
        },
        "clean_records": clean_records,
        "noise_records": noise_records,
        "variance_records": variance_records,
        "white_noise_gain": white_noise_gain,
    }
    _write_json(summary_json, summary)
    return {"summary_json": summary_json}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_frequency_response",
        help="Directory for per-noise CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_frequency_response" / "sec07_frequency_response_summary_r15_d3_normalized.json",
        help="Path for the combined frequency-response summary JSON.",
    )
    parser.add_argument(
        "--fft-backend",
        default=DEFAULT_FFT_BACKEND,
        choices=("cpu", "vkfft", "auto"),
        help="FFT backend for applying the precomputed kernel.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional device index for the GPU FFT backend.",
    )
    parser.add_argument(
        "--compile-plots",
        action="store_true",
        help="Compile the magnitude-error and variance plots after writing the data files.",
    )
    args = parser.parse_args(argv)

    run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )

    if args.compile_plots:
        plot_specs = (
            ("fig_sec07_frequency_awgn_mag_error.typ", "fig_sec07_frequency_awgn_mag_error_r15_d3_normalized.pdf"),
            ("fig_sec07_frequency_poisson_mag_error.typ", "fig_sec07_frequency_poisson_mag_error_r15_d3_normalized.pdf"),
            ("fig_sec07_frequency_speckle_mag_error.typ", "fig_sec07_frequency_speckle_mag_error_r15_d3_normalized.pdf"),
            ("fig_sec07_frequency_variance.typ", "fig_sec07_frequency_variance_r15_d3_normalized.pdf"),
        )
        for src_name, pdf_name in plot_specs:
            compile_plot(
                ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / src_name,
                ROOT / "papers" / "journal_paper" / "figures" / pdf_name,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
