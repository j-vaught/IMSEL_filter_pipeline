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

from weighted_disk_sg import WeightedDiskKernels, build_weighted_disk_kernels


RADIUS = 15
DEGREE = 3
NORMALIZE_COORDS = True
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
PATCH_HALF_SIZE = 128
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
SIGMA_VALUES: tuple[float | str, ...] = ("inf", 30.0, 15.0, 7.5, 5.0)


def _orientation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(180.0 / float(step_deg)))
    return tuple(float(step_deg) * i for i in range(count))


def _phase_values(count: int, step_px: float) -> tuple[float, ...]:
    return tuple(float(step_px) * i for i in range(int(count)))


def _sigma_slug(value: float | str) -> str:
    if value == "inf":
        return "uniform"
    text = f"{float(value):g}"
    return "sigma_" + text.replace(".", "p")


def _sigma_label(value: float | str) -> str:
    if value == "inf":
        return "uniform"
    return f"sigma_w = {float(value):g} px"


def _render_smoothed_step(
    projection: np.ndarray,
    phase_px: float,
    contrast: float,
    width_px: float,
) -> np.ndarray:
    return 0.5 * float(contrast) * (1.0 + np.tanh((projection - float(phase_px)) / float(width_px)))


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


def _crossing_x(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
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


def _peak_metrics_from_profile(
    profile: np.ndarray,
    x_coords: np.ndarray,
    phase_px: float,
    radius: int,
    width_px: float,
) -> tuple[float, float, float]:
    search_half = int(max(16, int(radius), int(math.ceil(12.0 * float(width_px)))))
    center_pos = float(phase_px)
    center_index = int(round(center_pos + float(x_coords.shape[0] // 2)))
    search_start = max(1, center_index - search_half)
    search_stop = min(x_coords.shape[0] - 1, center_index + search_half + 1)
    local = np.asarray(profile[search_start:search_stop], dtype=np.float64)
    peak_index = int(search_start + int(np.argmax(local)))
    peak_x, peak_height = _quadratic_peak_refinement(x_coords, profile, peak_index)
    far_mask = np.abs(x_coords - center_pos) > float(search_half)
    baseline = float(np.median(profile[far_mask])) if np.any(far_mask) else 0.0
    fwhm = _fwhm_from_profile(profile, x_coords, peak_index, peak_height, baseline, search_start, search_stop)
    return float(peak_height), float(peak_x - center_pos), float(fwhm)


def _local_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return axis, xx, yy


def _write_csv(path: Path, row: dict[str, object]) -> None:
    fieldnames = (
        "sigma_label",
        "sigma_w_px",
        "localisation_offset",
        "fwhm",
        "white_noise_gain",
        "condition_number",
        "anisotropy_ratio",
        "support_cardinality",
        "rank_deficient_count",
        "sigma_min",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    line_coords, xx, yy = _local_grid()
    center = float(PATCH_HALF_SIZE)
    orientation_values = _orientation_values(ORIENTATION_STEP_DEG)
    phase_values = _phase_values(PHASE_COUNT, PHASE_STEP_PX)
    bandwidth_records = []

    for sigma_value in SIGMA_VALUES:
        sigma_w = None if sigma_value == "inf" else float(sigma_value)
        kernels = build_weighted_disk_kernels(
            radius=RADIUS,
            degree=DEGREE,
            normalize_coords=NORMALIZE_COORDS,
            sigma_w=sigma_w,
        )
        localisation_errors = []
        fwhm_values = []
        orientation_means = []

        for orientation_deg in orientation_values:
            theta = math.radians(float(orientation_deg))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            projection = xx * cos_t + yy * sin_t
            xs = center + line_coords * cos_t
            ys = center + line_coords * sin_t
            phase_peaks = []
            for phase_px in phase_values:
                image = _render_smoothed_step(projection, float(phase_px), CONTRAST, EDGE_WIDTH_PX)
                gx = ndimage.correlate(
                    np.asarray(image, dtype=np.float64),
                    np.asarray(kernels.kernel_x, dtype=np.float64),
                    mode="reflect",
                )
                gy = ndimage.correlate(
                    np.asarray(image, dtype=np.float64),
                    np.asarray(kernels.kernel_y, dtype=np.float64),
                    mode="reflect",
                )
                directional = gx * cos_t + gy * sin_t
                profile = ndimage.map_coordinates(
                    np.asarray(directional, dtype=np.float64),
                    np.vstack((ys, xs)),
                    order=1,
                    mode="reflect",
                )
                peak_height, localisation_error, fwhm = _peak_metrics_from_profile(
                    profile=np.asarray(profile, dtype=np.float64),
                    x_coords=line_coords,
                    phase_px=float(phase_px),
                    radius=RADIUS,
                    width_px=EDGE_WIDTH_PX,
                )
                phase_peaks.append(float(peak_height))
                localisation_errors.append(float(localisation_error))
                fwhm_values.append(float(fwhm))
            orientation_means.append(float(np.mean(np.asarray(phase_peaks, dtype=np.float64))))

        response_values = np.asarray(orientation_means, dtype=np.float64)
        record = {
            "sigma_label": _sigma_label(sigma_value),
            "sigma_w_px": None if sigma_value == "inf" else float(sigma_value),
            "localisation_offset": float(
                np.sqrt(np.mean(np.asarray(localisation_errors, dtype=np.float64) ** 2))
            ),
            "fwhm": float(np.mean(np.asarray(fwhm_values, dtype=np.float64))),
            "white_noise_gain": float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2)),
            "condition_number": float(kernels.kappa_design_matrix),
            "anisotropy_ratio": float(np.max(response_values) / np.min(response_values)),
            "support_cardinality": int(kernels.support_cardinality),
            "rank_deficient_count": int(kernels.rank_deficient_count),
            "sigma_min": float(kernels.sigma_min),
        }
        bandwidth_records.append(record)

        slug = _sigma_slug(sigma_value)
        _write_csv(
            output_dir / f"sec07_weighting_{slug}_r{RADIUS}_d{DEGREE}_normalized.csv",
            record,
        )
        _write_json(
            output_dir / f"sec07_weighting_{slug}_r{RADIUS}_d{DEGREE}_normalized.json",
            {
                "title": "Section 7.6 weighting",
                "subtitle": "Gaussian pixel weighting on disk support, r = 15, d = 3, normalize_coords = True",
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
                },
                "record": record,
            },
        )
        print(
            f"sigma={record['sigma_label']} fwhm={record['fwhm']:.6f} "
            f"wng={record['white_noise_gain']:.6e} anisotropy={record['anisotropy_ratio']:.6f}"
        )

    summary = {
        "title": "Section 7.6 weighting",
        "subtitle": "Uniform versus Gaussian-weighted local polynomial fit on disk support",
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
        },
        "records": bandwidth_records,
    }
    _write_json(summary_json, summary)
    return {"summary_json": summary_json}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_weighting",
        help="Directory for per-bandwidth CSV/JSON outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_weighting" / "sec07_weighting_summary_r15_d3_normalized.json",
        help="Path for the combined weighting summary JSON.",
    )
    parser.add_argument(
        "--compile-plot",
        action="store_true",
        help="Compile the Pareto plot after writing the data files.",
    )
    args = parser.parse_args(argv)

    run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
    )

    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_weighting_pareto.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_weighting_pareto_r15_d3_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
