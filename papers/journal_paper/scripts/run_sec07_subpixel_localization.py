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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wvf.radius import build_wvf_radius_kernels
from wvf_metal.metal import fft_gradients_with_kernel

RADIUS = 9
DEGREE = 3
NORMALIZE_COORDS = True
PATCH_HALF_SIZE = 128
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 16
PHASE_STEP_PX = 0.0625
SNR_VALUES: tuple[float | str, ...] = ("inf", 30.0, 20.0, 10.0)
NOISE_DRAWS = 100
PROFILE_STEP_PX = 0.25
PROFILE_MARGIN_PX = 8.0
BATCH_CASES = 64
DEFAULT_FFT_BACKEND = "vkfft"
SEED = 72010


@dataclass(frozen=True)
class StepCase:
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    t_coords: np.ndarray


def _orientation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(180.0 / float(step_deg)))
    return tuple(float(step_deg) * idx for idx in range(count))


def _phase_values(count: int, step_px: float) -> tuple[float, ...]:
    return tuple(float(step_px) * idx for idx in range(int(count)))


def _snr_label(value: float | str) -> str:
    if value == "inf":
        return "clean"
    return f"{float(value):g} dB"


def _snr_slug(value: float | str) -> str:
    if value == "inf":
        return "clean"
    return "snr_" + f"{float(value):g}".replace(".", "p")


def _local_grid() -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _render_smoothed_step(xx: np.ndarray, yy: np.ndarray, orientation_deg: float, phase_px: float) -> np.ndarray:
    theta = math.radians(float(orientation_deg))
    normal = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta) - float(phase_px)
    return np.asarray(0.5 * float(CONTRAST) * (1.0 + np.tanh(normal / float(EDGE_WIDTH_PX))), dtype=np.float32)


def _profile_geometry(orientation_deg: float, phase_px: float, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(float(orientation_deg))
    search_half = float(radius) + 12.0 * float(EDGE_WIDTH_PX) + float(PROFILE_MARGIN_PX)
    t_coords = np.arange(-search_half, search_half + 0.5 * float(PROFILE_STEP_PX), float(PROFILE_STEP_PX), dtype=np.float64)
    center = float(PATCH_HALF_SIZE)
    center_x = center + float(phase_px) * math.cos(theta)
    center_y = center + float(phase_px) * math.sin(theta)
    sample_x = center_x + t_coords * math.cos(theta)
    sample_y = center_y + t_coords * math.sin(theta)
    return t_coords, sample_x, sample_y


def _build_cases() -> list[StepCase]:
    xx, yy = _local_grid()
    cases: list[StepCase] = []
    for orientation_deg in _orientation_values(ORIENTATION_STEP_DEG):
        for phase_px in _phase_values(PHASE_COUNT, PHASE_STEP_PX):
            t_coords, sample_x, sample_y = _profile_geometry(float(orientation_deg), float(phase_px), int(RADIUS))
            cases.append(
                StepCase(
                    orientation_deg=float(orientation_deg),
                    phase_px=float(phase_px),
                    image=_render_smoothed_step(xx, yy, float(orientation_deg), float(phase_px)),
                    sample_x=np.asarray(sample_x, dtype=np.float64),
                    sample_y=np.asarray(sample_y, dtype=np.float64),
                    t_coords=np.asarray(t_coords, dtype=np.float64),
                )
            )
    return cases


def _build_kernel() -> tuple[np.ndarray, np.ndarray]:
    kernels = build_wvf_radius_kernels(int(RADIUS), order=int(DEGREE), normalize_coords=bool(NORMALIZE_COORDS))
    return np.asarray(kernels.kernel_x, dtype=np.float64), np.asarray(kernels.kernel_y, dtype=np.float64)


def _tile_cases(images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    tile_h, tile_w = images[0].shape
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    placements: list[tuple[slice, slice]] = []
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        row_slice = slice(row * tile_h, (row + 1) * tile_h)
        col_slice = slice(col * tile_w, (col + 1) * tile_w)
        canvas[row_slice, col_slice] = np.asarray(image, dtype=np.float32)
        placements.append((row_slice, col_slice))
    return canvas, placements


def _apply_batched_magnitude(
    images: list[np.ndarray],
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
) -> list[np.ndarray]:
    canvas, placements = _tile_cases(images)
    gx_canvas, gy_canvas = fft_gradients_with_kernel(
        canvas,
        radius=int(RADIUS),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    magnitude = np.hypot(np.asarray(gx_canvas, dtype=np.float64), np.asarray(gy_canvas, dtype=np.float64))
    outputs = []
    for row_slice, col_slice in placements:
        outputs.append(np.asarray(magnitude[row_slice, col_slice], dtype=np.float64).copy())
    return outputs


def _sample_profile(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    return np.asarray(
        ndimage.map_coordinates(
            np.asarray(image, dtype=np.float64),
            np.vstack((np.asarray(sample_y, dtype=np.float64), np.asarray(sample_x, dtype=np.float64))),
            order=1,
            mode="nearest",
        ),
        dtype=np.float64,
    )


def _quadratic_peak_refinement(x_coords: np.ndarray, profile: np.ndarray, peak_index: int) -> tuple[float, float]:
    idx = int(peak_index)
    if idx <= 0 or idx >= profile.shape[0] - 1:
        return float(x_coords[idx]), float(profile[idx])
    y_prev = float(profile[idx - 1])
    y_mid = float(profile[idx])
    y_next = float(profile[idx + 1])
    denom = y_prev - 2.0 * y_mid + y_next
    if abs(denom) <= 1.0e-15:
        return float(x_coords[idx]), y_mid
    delta = 0.5 * (y_prev - y_next) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    peak_x = float(x_coords[idx]) + delta * float(x_coords[idx + 1] - x_coords[idx])
    peak_y = y_mid - 0.25 * (y_prev - y_next) * delta
    return float(peak_x), float(peak_y)


def _localization_offset(profile: np.ndarray, t_coords: np.ndarray) -> float:
    values = np.asarray(profile, dtype=np.float64)
    t = np.asarray(t_coords, dtype=np.float64)
    peak_index = int(np.argmax(values))
    peak_t, _ = _quadratic_peak_refinement(t, values, peak_index)
    return float(peak_t)


def _signal_std_reference(cases: list[StepCase]) -> float:
    clean = np.concatenate([np.asarray(case.image, dtype=np.float64).ravel() for case in cases])
    return float(np.std(clean))


def _sigma_from_snr_db(snr_db: float | str, signal_std: float) -> float:
    if snr_db == "inf":
        return 0.0
    return float(signal_std) / (10.0 ** (float(snr_db) / 20.0))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("snr_db", "phase_px", "orientation_deg", "localisation_offset", "draw_index"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.10 figures cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    cases = _build_cases()
    kernel_x, kernel_y = _build_kernel()
    signal_std = _signal_std_reference(cases)
    rng = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    clean_phase_offsets: dict[float, list[float]] = {float(phase): [] for phase in _phase_values(PHASE_COUNT, PHASE_STEP_PX)}
    rms_records = []

    clean_images = [case.image for case in cases]
    clean_magnitudes: list[np.ndarray] = []
    for batch_start in range(0, len(clean_images), int(BATCH_CASES)):
        clean_magnitudes.extend(
            _apply_batched_magnitude(
                clean_images[batch_start : batch_start + int(BATCH_CASES)],
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
            )
        )

    for case, mag in zip(cases, clean_magnitudes, strict=True):
        offset = _localization_offset(_sample_profile(mag, case.sample_x, case.sample_y), case.t_coords)
        rows.append(
            {
                "snr_db": "clean",
                "phase_px": f"{case.phase_px:.4f}",
                "orientation_deg": f"{case.orientation_deg:.1f}",
                "localisation_offset": f"{offset:.17e}",
                "draw_index": 0,
            }
        )
        clean_phase_offsets[float(case.phase_px)].append(float(offset))

    clean_offsets = [float(row["localisation_offset"]) for row in rows if row["snr_db"] == "clean"]
    rms_records.append({"snr_db": "clean", "localisation_rms": float(np.sqrt(np.mean(np.asarray(clean_offsets, dtype=np.float64) ** 2)))})
    print(f"sec710 snr=clean rms={rms_records[-1]['localisation_rms']:.6e}")

    for snr_db in SNR_VALUES:
        if snr_db == "inf":
            continue
        sigma = _sigma_from_snr_db(snr_db, signal_std)
        offsets = []
        for draw_index in range(int(NOISE_DRAWS)):
            noisy_images = [np.asarray(case.image, dtype=np.float64) + sigma * rng.normal(size=case.image.shape) for case in cases]
            noisy_magnitudes: list[np.ndarray] = []
            for batch_start in range(0, len(noisy_images), int(BATCH_CASES)):
                noisy_magnitudes.extend(
                    _apply_batched_magnitude(
                        [np.asarray(image, dtype=np.float32) for image in noisy_images[batch_start : batch_start + int(BATCH_CASES)]],
                        kernel_x=kernel_x,
                        kernel_y=kernel_y,
                        fft_backend=fft_backend,
                        device_index=device_index,
                    )
                )
            for case, mag in zip(cases, noisy_magnitudes, strict=True):
                offset = _localization_offset(_sample_profile(mag, case.sample_x, case.sample_y), case.t_coords)
                offsets.append(float(offset))
                rows.append(
                    {
                        "snr_db": f"{float(snr_db):g}",
                        "phase_px": f"{case.phase_px:.4f}",
                        "orientation_deg": f"{case.orientation_deg:.1f}",
                        "localisation_offset": f"{offset:.17e}",
                        "draw_index": int(draw_index),
                    }
                )
            if (draw_index + 1) % 20 == 0:
                print(f"sec710 snr={float(snr_db):g}dB draw={draw_index + 1}/{int(NOISE_DRAWS)}")
        rms = float(np.sqrt(np.mean(np.asarray(offsets, dtype=np.float64) ** 2)))
        rms_records.append({"snr_db": f"{float(snr_db):g}", "localisation_rms": rms})
        print(f"sec710 snr={float(snr_db):g}dB rms={rms:.6e}")

    phase_curve = [
        {
            "phase_px": float(phase_px),
            "mean_localisation_offset": float(np.mean(np.asarray(values, dtype=np.float64))),
        }
        for phase_px, values in clean_phase_offsets.items()
    ]

    csv_path = output_dir / "sec07_subpixel_localization_r9_d3_normalized.csv"
    _write_csv(csv_path, rows)

    payload = {
        "title": "Section 7.10 sub-pixel localisation accuracy",
        "subtitle": "Disk support, (r, d) = (9, 3), normalize_coords = True, batched local-patch FFT evaluation",
        "config": {
            "radius": int(RADIUS),
            "degree": int(DEGREE),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "patch_size_px": int(PATCH_SIZE),
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "snr_db_values": [_snr_label(value) for value in SNR_VALUES],
            "noise_draws": int(NOISE_DRAWS),
            "batch_cases": int(BATCH_CASES),
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "apparatus_reduction": "The subsection is evaluated on centered local patches rather than full 1024^2 frames because the localisation metric depends only on the normal profile near the target edge. Reflection padding is preserved on each patch.",
        },
        "csv_path": str(csv_path),
        "rms_by_snr": rms_records,
        "clean_phase_curve": phase_curve,
    }
    _write_json(summary_json, payload)
    return {"csv": csv_path, "summary_json": summary_json}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.10 sub-pixel localisation experiment.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_subpixel_localization",
        help="Directory for the Section 7.10 CSV output.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_subpixel_localization" / "sec07_subpixel_localization_summary_r9_d3_normalized.json",
        help="Path for the Section 7.10 summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figures after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )
    if args.compile_plots:
        rms_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_subpixel_localization_rms.typ"
        rms_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_subpixel_localization_rms.pdf"
        compile_plot(rms_src, rms_pdf)
        outputs["rms_plot_pdf"] = rms_pdf

        phase_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_subpixel_localization_phase_bias.typ"
        phase_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_subpixel_localization_phase_bias.pdf"
        compile_plot(phase_src, phase_pdf)
        outputs["phase_plot_pdf"] = phase_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
