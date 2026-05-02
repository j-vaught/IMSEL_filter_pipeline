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
from scipy import fft, ndimage


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wvf.radius import build_wvf_radius_kernels


IMAGE_SIZE = 1024
DEGREE = 3
NORMALIZE_COORDS = True
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
ANGLE_STEP_DEG = 0.5
PHASE_STEP_PX = 0.25
PHASE_COUNT = 4
NOISE_DRAWS = 100
RADII = (2, 3, 4, 5, 7, 9, 11, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128)
SNR_DB_VALUES: tuple[float | str, ...] = ("inf", 30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 7.5, 5.0, 2.5, 1.0, 0.5, 0.0)


@dataclass(frozen=True)
class NoiseSpectrumPlan:
    profile_len: int
    real_zero_factor: np.ndarray
    real_nyquist_factor: np.ndarray | None
    complex_factors: np.ndarray


@dataclass(frozen=True)
class RadiusPlan:
    radius: int
    support_cardinality: int
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    kernel_max: float
    white_noise_gain: float
    x_coords: np.ndarray
    clean_profiles: tuple[dict[str, np.ndarray | float], ...]
    noise_plan: NoiseSpectrumPlan


def _parse_value_list(text: str | None, cast) -> tuple:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.lower() in {"inf", "clean"}:
            values.append("inf")
        else:
            values.append(cast(item))
    return tuple(values)


def _snr_slug(value: float | str) -> str:
    if value == "inf":
        return "inf"
    text = f"{float(value):.1f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _snr_label(value: float | str) -> str:
    if value == "inf":
        return "clean"
    return f"{float(value):g} dB"


def _phases_px(count: int, step_px: float) -> np.ndarray:
    return np.arange(int(count), dtype=np.float64) * float(step_px)


def _canonical_x_coords(length: int) -> np.ndarray:
    center = int(length) // 2
    return np.arange(int(length), dtype=np.float64) - float(center)


def _render_smoothed_step_1d(
    x_coords: np.ndarray,
    phase_px: float,
    contrast: float,
    width_px: float,
) -> np.ndarray:
    return 0.5 * float(contrast) * (
        1.0 + np.tanh((np.asarray(x_coords, dtype=np.float64) - float(phase_px)) / float(width_px))
    )


def _analytic_step_derivative_1d(
    x_coords: np.ndarray,
    phase_px: float,
    contrast: float,
    width_px: float,
) -> np.ndarray:
    normalized = (np.asarray(x_coords, dtype=np.float64) - float(phase_px)) / float(width_px)
    return 0.5 * float(contrast) / float(width_px) / np.cosh(normalized) ** 2


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))


def _build_noise_plan(kernel_x: np.ndarray, kernel_y: np.ndarray, profile_len: int) -> NoiseSpectrumPlan:
    response_x = fft.rfft(np.asarray(kernel_x, dtype=np.float64), n=int(profile_len), axis=1)
    response_y = fft.rfft(np.asarray(kernel_y, dtype=np.float64), n=int(profile_len), axis=1)
    sxx = np.sum(response_x * np.conj(response_x), axis=0).real
    syy = np.sum(response_y * np.conj(response_y), axis=0).real
    sxy = np.sum(response_x * np.conj(response_y), axis=0)
    spectrum_len = sxx.shape[0]

    real_zero = _matrix_sqrt_psd(
        np.array(
            [[sxx[0], sxy[0].real], [sxy[0].real, syy[0]]],
            dtype=np.float64,
        )
    )
    real_nyquist = None
    complex_factors = []

    last_complex = spectrum_len
    if profile_len % 2 == 0:
        nyquist = spectrum_len - 1
        real_nyquist = _matrix_sqrt_psd(
            np.array(
                [[sxx[nyquist], sxy[nyquist].real], [sxy[nyquist].real, syy[nyquist]]],
                dtype=np.float64,
            )
        )
        last_complex = nyquist

    for index in range(1, last_complex):
        complex_factors.append(
            _matrix_sqrt_psd(
                np.array(
                    [[sxx[index], sxy[index]], [np.conj(sxy[index]), syy[index]]],
                    dtype=np.complex128,
                )
            )
        )

    return NoiseSpectrumPlan(
        profile_len=int(profile_len),
        real_zero_factor=np.asarray(real_zero, dtype=np.float64),
        real_nyquist_factor=None if real_nyquist is None else np.asarray(real_nyquist, dtype=np.float64),
        complex_factors=np.asarray(complex_factors, dtype=np.complex128),
    )


def _sample_joint_noise_profiles(
    plan: NoiseSpectrumPlan,
    sigma_noise: float,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(draws)
    profile_len = int(plan.profile_len)
    if count <= 0 or float(sigma_noise) <= 0.0:
        zeros = np.zeros((count, profile_len), dtype=np.float64)
        return zeros, zeros.copy()

    spectrum_len = profile_len // 2 + 1
    zx = np.zeros((count, spectrum_len), dtype=np.complex128)
    zy = np.zeros((count, spectrum_len), dtype=np.complex128)
    scale = float(sigma_noise) * math.sqrt(float(profile_len))

    real_zero = scale * (plan.real_zero_factor @ rng.normal(size=(2, count)))
    zx[:, 0] = real_zero[0]
    zy[:, 0] = real_zero[1]

    if plan.complex_factors.size:
        for index, factor in enumerate(plan.complex_factors, start=1):
            gaussian = (
                rng.normal(size=(2, count)) + 1.0j * rng.normal(size=(2, count))
            ) / math.sqrt(2.0)
            mapped = scale * (factor @ gaussian)
            zx[:, index] = mapped[0]
            zy[:, index] = mapped[1]

    if plan.real_nyquist_factor is not None:
        mapped = scale * (plan.real_nyquist_factor @ rng.normal(size=(2, count)))
        zx[:, -1] = mapped[0]
        zy[:, -1] = mapped[1]

    gx_noise = fft.irfft(zx, n=profile_len, axis=1)
    gy_noise = fft.irfft(zy, n=profile_len, axis=1)
    return np.asarray(gx_noise, dtype=np.float64), np.asarray(gy_noise, dtype=np.float64)


def _quadratic_peak_refinement(x_coords: np.ndarray, profile: np.ndarray, peak_index: int) -> tuple[float, float]:
    index = int(peak_index)
    if index <= 0 or index >= profile.shape[0] - 1:
        return float(x_coords[index]), float(profile[index])
    y_prev = float(profile[index - 1])
    y_mid = float(profile[index])
    y_next = float(profile[index + 1])
    denom = y_prev - 2.0 * y_mid + y_next
    if abs(denom) <= 1.0e-15:
        return float(x_coords[index]), y_mid
    delta = 0.5 * (y_prev - y_next) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    peak_x = float(x_coords[index]) + delta
    peak_y = y_mid - 0.25 * (y_prev - y_next) * delta
    return peak_x, float(peak_y)


def _crossing_x(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
) -> float:
    if abs(y1 - y0) <= 1.0e-15:
        return 0.5 * (float(x0) + float(x1))
    alpha = (float(target) - float(y0)) / (float(y1) - float(y0))
    return float(x0) + float(np.clip(alpha, 0.0, 1.0)) * (float(x1) - float(x0))


def _fwhm_from_profile(
    profile: np.ndarray,
    x_coords: np.ndarray,
    peak_index: int,
    peak_height: float,
    baseline: float,
    search_start: int,
    search_stop: int,
) -> float:
    if peak_height <= baseline:
        return 0.0
    target = float(baseline) + 0.5 * (float(peak_height) - float(baseline))
    left = int(peak_index)
    while left > int(search_start) and float(profile[left]) >= target:
        left -= 1
    if float(profile[left]) >= target:
        left_x = float(x_coords[left])
    else:
        left_x = _crossing_x(
            float(x_coords[left]),
            float(profile[left]),
            float(x_coords[left + 1]),
            float(profile[left + 1]),
            target,
        )

    right = int(peak_index)
    while right < int(search_stop) - 1 and float(profile[right]) >= target:
        right += 1
    if float(profile[right]) >= target:
        right_x = float(x_coords[right])
    else:
        right_x = _crossing_x(
            float(x_coords[right - 1]),
            float(profile[right - 1]),
            float(x_coords[right]),
            float(profile[right]),
            target,
        )
    return max(0.0, float(right_x - left_x))


def _peak_localization_and_fwhm(
    magnitude: np.ndarray,
    x_coords: np.ndarray,
    phase_px: float,
    radius: int,
    width_px: float,
) -> tuple[float, float]:
    search_half = int(max(16, int(radius), int(math.ceil(12.0 * float(width_px)))))
    center_pos = float(phase_px)
    center_index = int(round(center_pos + float(x_coords.shape[0] // 2)))
    search_start = max(1, center_index - search_half)
    search_stop = min(x_coords.shape[0] - 1, center_index + search_half + 1)
    local = magnitude[search_start:search_stop]
    peak_index = int(search_start + int(np.argmax(local)))
    peak_x, peak_height = _quadratic_peak_refinement(x_coords, magnitude, peak_index)

    far_mask = np.abs(x_coords - center_pos) > float(search_half)
    baseline = float(np.median(magnitude[far_mask])) if np.any(far_mask) else 0.0
    fwhm = _fwhm_from_profile(magnitude, x_coords, peak_index, peak_height, baseline, search_start, search_stop)
    localisation_error = float(peak_x - center_pos)
    return fwhm, localisation_error


def _build_radius_plan(radius: int, degree: int, normalize_coords: bool, image_size: int, contrast: float, width_px: float) -> RadiusPlan:
    kernels = build_wvf_radius_kernels(radius=int(radius), order=int(degree), normalize_coords=bool(normalize_coords))
    kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
    kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    kx_profile = np.sum(kernel_x, axis=0)
    ky_profile = np.sum(kernel_y, axis=0)
    x_coords = _canonical_x_coords(int(image_size))

    clean_profiles = []
    for phase_px in _phases_px(PHASE_COUNT, PHASE_STEP_PX):
        signal = _render_smoothed_step_1d(x_coords, float(phase_px), contrast, width_px)
        true_gx = _analytic_step_derivative_1d(x_coords, float(phase_px), contrast, width_px)
        gx_clean = ndimage.correlate1d(signal, kx_profile, mode="reflect")
        gy_clean = ndimage.correlate1d(signal, ky_profile, mode="reflect")
        clean_profiles.append(
            {
                "phase_px": float(phase_px),
                "signal": np.asarray(signal, dtype=np.float64),
                "true_gx": np.asarray(true_gx, dtype=np.float64),
                "gx_clean": np.asarray(gx_clean, dtype=np.float64),
                "gy_clean": np.asarray(gy_clean, dtype=np.float64),
            }
        )

    noise_plan = _build_noise_plan(kernel_x, kernel_y, profile_len=int(image_size))
    return RadiusPlan(
        radius=int(radius),
        support_cardinality=int(kernels.support_size),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        kernel_max=float(np.max(np.abs(kernel_x))),
        white_noise_gain=float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2)),
        x_coords=np.asarray(x_coords, dtype=np.float64),
        clean_profiles=tuple(clean_profiles),
        noise_plan=noise_plan,
    )


def _signal_std_reference(image_size: int, contrast: float, width_px: float) -> float:
    x_coords = _canonical_x_coords(int(image_size))
    clean = _render_smoothed_step_1d(x_coords, 0.0, contrast, width_px)
    return float(np.std(clean))


def _sigma_from_snr_db(snr_db: float | str, signal_std: float) -> float:
    if snr_db == "inf":
        return 0.0
    return float(signal_std) / (10.0 ** (float(snr_db) / 20.0))


def _evaluate_radius_snr_cell(
    plan: RadiusPlan,
    snr_db: float | str,
    signal_std: float,
    noise_draws: int,
    rng: np.random.Generator,
    width_px: float,
) -> dict[str, float]:
    sigma_noise = _sigma_from_snr_db(snr_db, signal_std)
    if sigma_noise > 0.0:
        gx_noise, gy_noise = _sample_joint_noise_profiles(plan.noise_plan, sigma_noise, int(noise_draws), rng)
    else:
        gx_noise = np.zeros((1, plan.x_coords.shape[0]), dtype=np.float64)
        gy_noise = np.zeros((1, plan.x_coords.shape[0]), dtype=np.float64)

    fwhm_values: list[float] = []
    localisation_sq: list[float] = []
    grad_rmse_values: list[float] = []
    draw_count = gx_noise.shape[0]

    for clean in plan.clean_profiles:
        phase_px = float(clean["phase_px"])
        gx_clean = np.asarray(clean["gx_clean"], dtype=np.float64)
        gy_clean = np.asarray(clean["gy_clean"], dtype=np.float64)
        true_gx = np.asarray(clean["true_gx"], dtype=np.float64)
        mean_magnitude = np.zeros_like(plan.x_coords, dtype=np.float64)
        rmse_half_width = max(12.0, 6.0 * float(width_px))
        rmse_mask = np.abs(plan.x_coords - phase_px) <= rmse_half_width
        for draw_index in range(draw_count):
            gx_profile = gx_clean + gx_noise[draw_index]
            gy_profile = gy_clean + gy_noise[draw_index]
            magnitude = np.hypot(gx_profile, gy_profile)
            mean_magnitude += magnitude
            _, localisation_error = _peak_localization_and_fwhm(
                magnitude=magnitude,
                x_coords=plan.x_coords,
                phase_px=phase_px,
                radius=plan.radius,
                width_px=width_px,
            )
            localisation_sq.append(float(localisation_error) ** 2)
            grad_sq_error = (gx_profile[rmse_mask] - true_gx[rmse_mask]) ** 2 + gy_profile[rmse_mask] ** 2
            grad_rmse_values.append(float(np.sqrt(np.mean(grad_sq_error))))
        mean_magnitude /= float(draw_count)
        phase_fwhm, _ = _peak_localization_and_fwhm(
            magnitude=mean_magnitude,
            x_coords=plan.x_coords,
            phase_px=phase_px,
            radius=plan.radius,
            width_px=width_px,
        )
        fwhm_values.append(float(phase_fwhm))

    return {
        "radius": float(plan.radius),
        "kernel_max": float(plan.kernel_max),
        "white_noise_gain": float(plan.white_noise_gain),
        "fwhm": float(np.mean(np.asarray(fwhm_values, dtype=np.float64))),
        "localisation_rms": float(np.sqrt(np.mean(np.asarray(localisation_sq, dtype=np.float64)))),
        "grad_rmse": float(np.mean(np.asarray(grad_rmse_values, dtype=np.float64))),
        "support_cardinality": float(plan.support_cardinality),
        "sigma_noise": float(sigma_noise),
    }


def _write_csv(csv_path: Path, records: list[dict[str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("radius", "kernel_max", "white_noise_gain", "fwhm", "localisation_rms", "grad_rmse"),
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "radius": f"{row['radius']:.0f}",
                    "kernel_max": f"{row['kernel_max']:.17e}",
                    "white_noise_gain": f"{row['white_noise_gain']:.17e}",
                    "fwhm": f"{row['fwhm']:.17e}",
                    "localisation_rms": f"{row['localisation_rms']:.17e}",
                    "grad_rmse": f"{row['grad_rmse']:.17e}",
                }
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _plot_series_payload(snr_values: tuple[float | str, ...], grid_records: dict[str, list[dict[str, float]]]) -> list[dict[str, object]]:
    series = []
    for snr_db in snr_values:
        key = _snr_slug(snr_db)
        records = grid_records[key]
        series.append(
            {
                "snr_db": snr_db,
                "label": _snr_label(snr_db),
                "points": [
                    {
                        "radius": row["radius"],
                        "kernel_max": row["kernel_max"],
                        "white_noise_gain": row["white_noise_gain"],
                        "fwhm": row["fwhm"],
                        "localisation_rms": row["localisation_rms"],
                        "grad_rmse": row["grad_rmse"],
                    }
                    for row in records
                ],
            }
        )
    return series


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    image_size: int,
    radii: tuple[int, ...],
    snr_values: tuple[float | str, ...],
    noise_draws: int,
    contrast: float,
    width_px: float,
    seed: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    signal_std = _signal_std_reference(int(image_size), float(contrast), float(width_px))
    plans = {
        int(radius): _build_radius_plan(
            radius=int(radius),
            degree=DEGREE,
            normalize_coords=NORMALIZE_COORDS,
            image_size=int(image_size),
            contrast=float(contrast),
            width_px=float(width_px),
        )
        for radius in radii
    }

    grid_records: dict[str, list[dict[str, float]]] = {}
    outputs: dict[str, Path] = {}

    for snr_db in snr_values:
        key = _snr_slug(snr_db)
        records = []
        for radius in radii:
            cell = _evaluate_radius_snr_cell(
                plan=plans[int(radius)],
                snr_db=snr_db,
                signal_std=signal_std,
                noise_draws=int(noise_draws),
                rng=rng,
                width_px=float(width_px),
            )
            records.append(cell)
        csv_path = output_dir / f"sec07_support_radius_noise_snr_{key}_d{DEGREE}_normalized.csv"
        _write_csv(csv_path, records)
        grid_records[key] = records
        outputs[f"csv_{key}"] = csv_path
        print(
            f"{_snr_label(snr_db)}: best_fwhm_radius="
            f"{min(records, key=lambda row: row['fwhm'])['radius']:.0f}, "
            f"best_rmse_radius={min(records, key=lambda row: row['grad_rmse'])['radius']:.0f}"
        )

    summary_payload = {
        "title": "Section 7.4 support radius first pass",
        "subtitle": "Disk support, smoothed step edge, $d = 3$, normalize_coords = True, AWGN sweep",
        "config": {
            "image_size": int(image_size),
            "degree": int(DEGREE),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "contrast": float(contrast),
            "edge_width_px": float(width_px),
            "radius_schedule": [int(radius) for radius in radii],
            "snr_db_schedule": [snr for snr in snr_values],
            "angle_count": int(round(180.0 / ANGLE_STEP_DEG)),
            "angle_step_deg": float(ANGLE_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "noise_draws_per_cell": int(noise_draws),
            "signal_std_reference": float(signal_std),
            "snr_definition": "sigma_signal / sigma_noise with signal_std taken from the clean 1D smoothed step profile",
            "apparatus_reduction": "Rotational averaging is collapsed to a canonical horizontal edge because the disk support is isotropic; phases are sampled explicitly and the joint gx/gy noise process is sampled exactly from the filtered white-noise spectrum.",
        },
        "per_snr": [
            {
                "snr_db": snr_db,
                "label": _snr_label(snr_db),
                "csv_path": str(output_dir / f"sec07_support_radius_noise_snr_{_snr_slug(snr_db)}_d{DEGREE}_normalized.csv"),
                "rows": grid_records[_snr_slug(snr_db)],
            }
            for snr_db in snr_values
        ],
        "pareto_series": _plot_series_payload(snr_values, grid_records),
        "rmse_series": _plot_series_payload(snr_values, grid_records),
    }
    _write_json(summary_json, summary_payload)
    outputs["summary_json"] = summary_json
    return outputs


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.4 figures cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.4 support-radius AWGN sweep on smoothed step edges.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_noise",
        help="Directory for per-SNR CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_noise" / "sec07_support_radius_noise_summary_d3_normalized.json",
        help="Path for the combined Section 7.4 summary JSON.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Canonical profile length, matching the image width.")
    parser.add_argument("--noise-draws", type=int, default=NOISE_DRAWS, help="Number of AWGN draws per (radius, SNR) cell.")
    parser.add_argument("--contrast", type=float, default=CONTRAST, help="Smoothed step contrast amplitude.")
    parser.add_argument("--edge-width-px", type=float, default=EDGE_WIDTH_PX, help="Width parameter for the tanh-smoothed step.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for the AWGN sampler.")
    parser.add_argument("--radius-list", type=str, default=None, help="Optional comma-separated radius subset for smoke tests.")
    parser.add_argument("--snr-list", type=str, default=None, help="Optional comma-separated SNR subset, with 'inf' for clean.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ plots after writing data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_value_list(args.radius_list, int) or RADII))
    snr_values = _parse_value_list(args.snr_list, float) or SNR_DB_VALUES

    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        image_size=int(args.image_size),
        radii=radii,
        snr_values=snr_values,
        noise_draws=int(args.noise_draws),
        contrast=float(args.contrast),
        width_px=float(args.edge_width_px),
        seed=int(args.seed),
    )

    if args.compile_plot:
        pareto_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_radius_noise_pareto_d3_normalized.typ"
        pareto_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_radius_noise_pareto_d3_normalized.pdf"
        compile_plot(pareto_src, pareto_pdf)
        outputs["pareto_pdf"] = pareto_pdf

        rmse_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_radius_noise_grad_rmse_d3_normalized.typ"
        rmse_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_radius_noise_grad_rmse_d3_normalized.pdf"
        compile_plot(rmse_src, rmse_pdf)
        outputs["rmse_pdf"] = rmse_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
