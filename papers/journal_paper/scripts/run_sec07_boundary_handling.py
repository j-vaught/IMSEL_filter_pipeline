#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weighted_disk_sg import build_weighted_disk_kernels
from wvf_metal.metal import fft_gradients_with_kernel


RADIUS = 15
DEGREE = 3
NORMALIZE_COORDS = True
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
PATCH_SIZE = 257
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
SNR_DB_VALUES: tuple[float | str, ...] = ("inf", 20.0, 10.0)
NOISE_DRAWS = 4
NORMAL_BAND_HALF_PX = 6.0
INTERIOR_OFFSET_PX = 120.0
OFFSET_SPECS: tuple[tuple[str, float], ...] = (
    ("0", 0.0),
    ("15", 15.0),
    ("30", 30.0),
    ("60", 60.0),
    ("interior", INTERIOR_OFFSET_PX),
)
PADDING_MODES = ("reflect", "zero", "constant_value", "edge")
PADDING_LABELS = {
    "reflect": "Reflection",
    "zero": "Zero",
    "constant_value": "Border-constant",
    "edge": "Clamp",
}
DEFAULT_FFT_BACKEND = "vkfft"


def _orientation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(180.0 / float(step_deg)))
    return tuple(float(step_deg) * i for i in range(count))


def _phase_values(count: int, step_px: float) -> tuple[float, ...]:
    return tuple(float(step_px) * i for i in range(int(count)))


def _snr_label(value: float | str) -> str:
    if value == "inf":
        return "clean"
    return f"{float(value):g} dB"


def _snr_slug(value: float | str) -> str:
    if value == "inf":
        return "inf"
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _render_smoothed_step(
    projection: np.ndarray,
    contrast: float,
    width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = 0.5 * float(contrast) * (1.0 + np.tanh(np.asarray(projection, dtype=np.float64) / float(width_px)))
    factor = 0.5 * float(contrast) / float(width_px) * (1.0 - np.tanh(np.asarray(projection, dtype=np.float64) / float(width_px)) ** 2)
    return np.asarray(image, dtype=np.float64), np.asarray(factor, dtype=np.float64)


def _pad_constant_value(image: np.ndarray, pad: int) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    top = float(np.mean(src[0, :]))
    bottom = float(np.mean(src[-1, :]))
    left = float(np.mean(src[:, 0]))
    right = float(np.mean(src[:, -1]))
    padded = np.empty((src.shape[0] + 2 * pad, src.shape[1] + 2 * pad), dtype=np.float64)
    padded[pad:-pad, pad:-pad] = src
    padded[:pad, pad:-pad] = top
    padded[-pad:, pad:-pad] = bottom
    padded[pad:-pad, :pad] = left
    padded[pad:-pad, -pad:] = right
    padded[:pad, :pad] = 0.5 * (top + left)
    padded[:pad, -pad:] = 0.5 * (top + right)
    padded[-pad:, :pad] = 0.5 * (bottom + left)
    padded[-pad:, -pad:] = 0.5 * (bottom + right)
    return padded


def _prepad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    if mode == "reflect":
        return np.pad(src, ((pad, pad), (pad, pad)), mode="reflect")
    if mode == "zero":
        return np.pad(src, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    if mode == "edge":
        return np.pad(src, ((pad, pad), (pad, pad)), mode="edge")
    if mode == "constant_value":
        return _pad_constant_value(src, pad)
    raise ValueError(f"unsupported padding mode {mode!r}")


def _apply_gradients(
    image: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    pad: int,
    mode: str,
    fft_backend: str,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    padded = _prepad_image(image, pad, mode)
    gx_padded, gy_padded = fft_gradients_with_kernel(
        np.asarray(padded, dtype=np.float32),
        radius=int(RADIUS),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    return (
        np.asarray(gx_padded[pad:-pad, pad:-pad], dtype=np.float64),
        np.asarray(gy_padded[pad:-pad, pad:-pad], dtype=np.float64),
    )


def _signal_noise_sigma(snr_db: float | str) -> float:
    if snr_db == "inf":
        return 0.0
    return float(CONTRAST) / (10.0 ** (float(snr_db) / 20.0))


def _local_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64), indexing="ij")
    return xx, yy


def _eval_mask(
    xx: np.ndarray,
    projection: np.ndarray,
    offset_label: str,
) -> np.ndarray:
    band_mask = (
        np.abs(np.asarray(xx, dtype=np.float64) - float(INTERIOR_OFFSET_PX)) <= 4.0 * float(RADIUS)
        if offset_label == "interior"
        else np.asarray(xx, dtype=np.float64) <= 4.0 * float(RADIUS)
    )
    edge_mask = np.abs(np.asarray(projection, dtype=np.float64)) <= float(NORMAL_BAND_HALF_PX)
    return np.asarray(band_mask & edge_mask, dtype=bool)


def _response_mask(xx: np.ndarray, offset_label: str) -> np.ndarray:
    if offset_label == "interior":
        return np.asarray(np.abs(np.asarray(xx, dtype=np.float64) - float(INTERIOR_OFFSET_PX)) <= 4.0 * float(RADIUS), dtype=bool)
    return np.asarray(np.asarray(xx, dtype=np.float64) <= 4.0 * float(RADIUS), dtype=bool)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ("edge_offset_px", "snr_db", "grad_rmse", "anisotropy_ratio")
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
    pad = int(kernels.kernel_half_extent)
    xx, yy = _local_grid(PATCH_SIZE)
    center_y = 0.5 * float(PATCH_SIZE - 1)
    rng = np.random.default_rng(20260502)
    orientation_values = _orientation_values(ORIENTATION_STEP_DEG)
    phase_values = _phase_values(PHASE_COUNT, PHASE_STEP_PX)

    cell_records: dict[tuple[str, str, str], dict[str, object]] = {}
    rows_by_padding: dict[str, list[dict[str, object]]] = {mode: [] for mode in PADDING_MODES}

    for snr_db in SNR_DB_VALUES:
        sigma_noise = _signal_noise_sigma(snr_db)
        for offset_label, offset_px in OFFSET_SPECS:
            for mode in PADDING_MODES:
                cell_records[(mode, offset_label, _snr_slug(snr_db))] = {
                    "padding_mode": mode,
                    "padding_label": PADDING_LABELS[mode],
                    "edge_offset_px": offset_label,
                    "edge_offset_numeric_px": float(offset_px),
                    "snr_db": _snr_label(snr_db),
                    "grad_sq_sum": 0.0,
                    "grad_count": 0,
                    "orientation_peaks": {f"{theta_deg:g}": [] for theta_deg in orientation_values},
                }

            draw_count = NOISE_DRAWS if sigma_noise > 0.0 else 1
            for orientation_deg in orientation_values:
                theta = math.radians(float(orientation_deg))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                base_projection = (xx - float(offset_px)) * cos_t + (yy - center_y) * sin_t
                response_mask_base = _response_mask(xx, offset_label)

                for phase_px in phase_values:
                    projection = base_projection - float(phase_px)
                    clean_image, factor = _render_smoothed_step(projection, CONTRAST, EDGE_WIDTH_PX)
                    true_gx = factor * cos_t
                    true_gy = factor * sin_t
                    eval_mask = _eval_mask(xx, projection, offset_label)

                    for _ in range(draw_count):
                        noisy_image = np.asarray(clean_image, dtype=np.float64)
                        if sigma_noise > 0.0:
                            noisy_image = noisy_image + sigma_noise * rng.normal(size=noisy_image.shape)
                        for mode in PADDING_MODES:
                            gx, gy = _apply_gradients(
                                noisy_image,
                                kernels.kernel_x,
                                kernels.kernel_y,
                                pad,
                                mode,
                                fft_backend,
                                device_index,
                            )
                            diff_sq = (gx - true_gx) ** 2 + (gy - true_gy) ** 2
                            directional = gx * cos_t + gy * sin_t
                            record = cell_records[(mode, offset_label, _snr_slug(snr_db))]
                            record["grad_sq_sum"] += float(np.sum(diff_sq[eval_mask]))
                            record["grad_count"] += int(np.count_nonzero(eval_mask))
                            masked = np.asarray(directional[response_mask_base], dtype=np.float64)
                            record["orientation_peaks"][f"{orientation_deg:g}"].append(float(np.max(np.abs(masked))))
            print(f"offset={offset_label} snr={_snr_label(snr_db)} status=done")

    summary_records = []
    for mode in PADDING_MODES:
        csv_rows = []
        for offset_label, _ in OFFSET_SPECS:
            for snr_db in SNR_DB_VALUES:
                key = (mode, offset_label, _snr_slug(snr_db))
                record = cell_records[key]
                orientation_means = [
                    float(np.mean(np.asarray(values, dtype=np.float64)))
                    for values in record["orientation_peaks"].values()
                ]
                grad_rmse = math.sqrt(record["grad_sq_sum"] / max(1, int(record["grad_count"])))
                anisotropy = float(np.max(orientation_means) / np.min(orientation_means))
                row = {
                    "edge_offset_px": offset_label,
                    "snr_db": _snr_label(snr_db),
                    "grad_rmse": f"{grad_rmse:.17e}",
                    "anisotropy_ratio": f"{anisotropy:.17e}",
                }
                csv_rows.append(row)
                summary_records.append(
                    {
                        "padding_mode": mode,
                        "padding_label": PADDING_LABELS[mode],
                        "edge_offset_px": offset_label,
                        "edge_offset_numeric_px": record["edge_offset_numeric_px"],
                        "snr_db": _snr_label(snr_db),
                        "grad_rmse": grad_rmse,
                        "anisotropy_ratio": anisotropy,
                    }
                )
        rows_by_padding[mode] = csv_rows
        _write_csv(
            output_dir / f"sec07_boundary_handling_{mode}_r{RADIUS}_d{DEGREE}_normalized.csv",
            csv_rows,
        )

    summary = {
        "title": "Section 7.7 boundary handling",
        "subtitle": "Disk support, r = 15, d = 3, normalize_coords = True",
        "config": {
            "radius": RADIUS,
            "degree": DEGREE,
            "normalize_coords": NORMALIZE_COORDS,
            "patch_size": PATCH_SIZE,
            "contrast": CONTRAST,
            "edge_width_px": EDGE_WIDTH_PX,
            "orientation_step_deg": ORIENTATION_STEP_DEG,
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "snr_db_values": [_snr_label(v) for v in SNR_DB_VALUES],
            "noise_draws": NOISE_DRAWS,
            "constant_value_definition": "Each padded side is filled with the mean value of the adjacent border row or column.",
        },
        "records": summary_records,
    }
    _write_json(summary_json, summary)

    for mode in PADDING_MODES:
        print(
            f"{mode}: "
            + ", ".join(
                f"{row['edge_offset_px']}@{row['snr_db']} rmse={float(row['grad_rmse']):.4e}"
                for row in rows_by_padding[mode]
                if row["snr_db"] == "10 dB"
            )
        )

    return {"summary_json": summary_json}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_boundary_handling",
        help="Directory for per-padding CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_boundary_handling" / "sec07_boundary_handling_summary_r15_d3_normalized.json",
        help="Path for the combined boundary-handling summary JSON.",
    )
    parser.add_argument(
        "--compile-plot",
        action="store_true",
        help="Compile the RMSE-vs-offset plot after writing the data files.",
    )
    parser.add_argument(
        "--fft-backend",
        default=DEFAULT_FFT_BACKEND,
        choices=("cpu", "vkfft", "auto"),
        help="FFT backend for the padded-kernel application path.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional device index for the GPU FFT backend.",
    )
    args = parser.parse_args(argv)

    run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )

    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_boundary_handling_rmse.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_boundary_handling_rmse_r15_d3_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
