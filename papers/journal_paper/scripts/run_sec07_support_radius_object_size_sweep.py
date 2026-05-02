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
RADII = (2, 3, 4, 5, 7, 9, 11, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128)
FEATURE_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
BAR_ANGLE_STEP_DEG = 5.0
BAR_PHASE_COUNT = 8
DISK_PHASE_COUNT = 8
PATCH_HALF_SIZE = 192
PROFILE_STEP_PX = 0.25
PROFILE_MARGIN_PX = 8.0
FFT_BATCH_SIZE = 16
CENTER_WINDOW_HALF_PX = 4
NOISE_FLOOR_SIGMA = 1.0 / 255.0


@dataclass(frozen=True)
class KernelPlan:
    radius: int
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    kernel_shape: tuple[int, int]
    kernel_max: float
    white_noise_gain: float
    noise_floor: float
    detection_threshold: float
    fft_x: np.ndarray
    fft_y: np.ndarray


@dataclass(frozen=True)
class StimulusCase:
    subclass: str
    feature_width: int
    angle_deg: float
    phase_px: float
    image: np.ndarray
    center_x: float
    center_y: float
    t_coords: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray


def _parse_value_list(text: str | None, cast) -> tuple:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(cast(item))
    return tuple(values)


def _phase_offsets(count: int) -> np.ndarray:
    return (np.arange(int(count), dtype=np.float64) + 0.5) / float(count) - 0.5


def _bar_angles_deg(step_deg: float) -> np.ndarray:
    count = int(round(180.0 / float(step_deg)))
    return np.arange(count, dtype=np.float64) * float(step_deg)


def _patch_coords(half_size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-int(half_size), int(half_size) + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _render_bar_patch(
    xx: np.ndarray,
    yy: np.ndarray,
    width_px: float,
    angle_deg: float,
    phase_px: float,
    contrast: float,
) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    u = xx * math.cos(theta) + yy * math.sin(theta) - float(phase_px)
    return np.where(np.abs(u) <= 0.5 * float(width_px), float(contrast), 0.0)


def _render_disk_patch(
    xx: np.ndarray,
    yy: np.ndarray,
    width_px: float,
    phase_px: float,
    contrast: float,
) -> np.ndarray:
    radius = 0.5 * float(width_px)
    shifted_x = xx - float(phase_px)
    return np.where(shifted_x * shifted_x + yy * yy <= radius * radius, float(contrast), 0.0)


def _profile_sample_geometry(
    subclass: str,
    width_px: float,
    angle_deg: float,
    phase_px: float,
    max_radius: int,
    patch_half_size: int,
    step_px: float,
    margin_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    search_half = 0.5 * float(width_px) + float(max_radius) + float(margin_px)
    t_coords = np.arange(-search_half, search_half + 0.5 * float(step_px), float(step_px), dtype=np.float64)
    center = float(patch_half_size)
    if subclass == "bars":
        theta = math.radians(float(angle_deg))
        center_x = center + float(phase_px) * math.cos(theta)
        center_y = center + float(phase_px) * math.sin(theta)
        sample_x = center_x + t_coords * math.cos(theta)
        sample_y = center_y + t_coords * math.sin(theta)
    else:
        center_x = center + float(phase_px)
        center_y = center
        sample_x = center_x + t_coords
        sample_y = np.full_like(sample_x, center_y)
    return t_coords, sample_x, sample_y, float(center_x), float(center_y)


def _build_bar_cases(
    widths: tuple[int, ...],
    patch_half_size: int,
    max_radius: int,
    contrast: float,
    step_deg: float,
    phase_count: int,
) -> list[StimulusCase]:
    xx, yy = _patch_coords(patch_half_size)
    phases = _phase_offsets(int(phase_count))
    cases: list[StimulusCase] = []
    for width in widths:
        for angle_deg in _bar_angles_deg(step_deg):
            for phase_px in phases:
                t_coords, sample_x, sample_y, center_x, center_y = _profile_sample_geometry(
                    subclass="bars",
                    width_px=float(width),
                    angle_deg=float(angle_deg),
                    phase_px=float(phase_px),
                    max_radius=int(max_radius),
                    patch_half_size=int(patch_half_size),
                    step_px=PROFILE_STEP_PX,
                    margin_px=PROFILE_MARGIN_PX,
                )
                cases.append(
                    StimulusCase(
                        subclass="bars",
                        feature_width=int(width),
                        angle_deg=float(angle_deg),
                        phase_px=float(phase_px),
                        image=np.asarray(
                            _render_bar_patch(
                                xx=xx,
                                yy=yy,
                                width_px=float(width),
                                angle_deg=float(angle_deg),
                                phase_px=float(phase_px),
                                contrast=float(contrast),
                            ),
                            dtype=np.float64,
                        ),
                        center_x=float(center_x),
                        center_y=float(center_y),
                        t_coords=t_coords,
                        sample_x=sample_x,
                        sample_y=sample_y,
                    )
                )
    return cases


def _build_disk_cases(
    widths: tuple[int, ...],
    patch_half_size: int,
    max_radius: int,
    contrast: float,
    phase_count: int,
) -> list[StimulusCase]:
    xx, yy = _patch_coords(patch_half_size)
    phases = _phase_offsets(int(phase_count))
    cases: list[StimulusCase] = []
    for width in widths:
        for phase_px in phases:
            t_coords, sample_x, sample_y, center_x, center_y = _profile_sample_geometry(
                subclass="disks",
                width_px=float(width),
                angle_deg=0.0,
                phase_px=float(phase_px),
                max_radius=int(max_radius),
                patch_half_size=int(patch_half_size),
                step_px=PROFILE_STEP_PX,
                margin_px=PROFILE_MARGIN_PX,
            )
            cases.append(
                StimulusCase(
                    subclass="disks",
                    feature_width=int(width),
                    angle_deg=0.0,
                    phase_px=float(phase_px),
                    image=np.asarray(
                        _render_disk_patch(
                            xx=xx,
                            yy=yy,
                            width_px=float(width),
                            phase_px=float(phase_px),
                            contrast=float(contrast),
                        ),
                        dtype=np.float64,
                    ),
                    center_x=float(center_x),
                    center_y=float(center_y),
                    t_coords=t_coords,
                    sample_x=sample_x,
                    sample_y=sample_y,
                )
            )
    return cases


def _common_fft_shape(patch_size: int, max_radius: int) -> tuple[int, int]:
    kernel_size = 2 * int(max_radius) + 1
    full_size = int(patch_size) + kernel_size - 1
    fast = fft.next_fast_len(full_size)
    return fast, fast


def _build_kernel_plan(radius: int, fft_shape: tuple[int, int]) -> KernelPlan:
    kernels = build_wvf_radius_kernels(int(radius), order=DEGREE, normalize_coords=NORMALIZE_COORDS)
    kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
    kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    white_noise_gain = float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2))
    noise_floor = math.sqrt(2.0 * white_noise_gain)
    detection_threshold = 5.0 * float(NOISE_FLOOR_SIGMA) * noise_floor
    return KernelPlan(
        radius=int(radius),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        kernel_shape=(int(kernel_x.shape[0]), int(kernel_x.shape[1])),
        kernel_max=float(np.max(np.abs(kernel_x))),
        white_noise_gain=white_noise_gain,
        noise_floor=float(noise_floor),
        detection_threshold=float(detection_threshold),
        fft_x=fft.rfft2(kernel_x[::-1, ::-1], s=fft_shape),
        fft_y=fft.rfft2(kernel_y[::-1, ::-1], s=fft_shape),
    )


def _valid_response(
    image_fft: np.ndarray,
    kernel_fft: np.ndarray,
    fft_shape: tuple[int, int],
    patch_size: int,
    kernel_shape: tuple[int, int],
    workers: int,
) -> np.ndarray:
    full = fft.irfft2(image_fft * kernel_fft[None, :, :], s=fft_shape, axes=(-2, -1), workers=workers)
    kh, kw = kernel_shape
    return np.asarray(full[:, kh - 1:kh - 1 + patch_size, kw - 1:kw - 1 + patch_size], dtype=np.float64)


def _sample_profile(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.vstack((np.asarray(sample_y, dtype=np.float64), np.asarray(sample_x, dtype=np.float64))),
        order=1,
        mode="nearest",
    )


def _crossing_x(t0: float, y0: float, t1: float, y1: float, target: float) -> float:
    if abs(float(y1) - float(y0)) <= 1.0e-15:
        return 0.5 * (float(t0) + float(t1))
    alpha = (float(target) - float(y0)) / (float(y1) - float(y0))
    return float(t0) + float(np.clip(alpha, 0.0, 1.0)) * (float(t1) - float(t0))


def _profile_metrics(
    profile: np.ndarray,
    t_coords: np.ndarray,
    search_half: float,
    threshold: float,
) -> tuple[float, float, float]:
    mask = np.abs(np.asarray(t_coords, dtype=np.float64)) <= float(search_half)
    local_t = np.asarray(t_coords, dtype=np.float64)[mask]
    local_profile = np.asarray(profile, dtype=np.float64)[mask]
    peak_index = int(np.argmax(local_profile))
    peak_value = float(local_profile[peak_index])
    detected = 1.0 if peak_value > float(threshold) else 0.0
    if peak_value <= 0.0:
        return 0.0, 0.0, detected

    half_value = 0.5 * peak_value
    above = np.flatnonzero(local_profile >= half_value)
    left = int(above[0])
    right = int(above[-1])
    if left == 0:
        left_t = float(local_t[left])
    else:
        left_t = _crossing_x(local_t[left - 1], local_profile[left - 1], local_t[left], local_profile[left], half_value)
    if right == local_profile.shape[0] - 1:
        right_t = float(local_t[right])
    else:
        right_t = _crossing_x(local_t[right], local_profile[right], local_t[right + 1], local_profile[right + 1], half_value)
    return peak_value, max(0.0, float(right_t - left_t)), detected


def _center_window_peak(
    magnitude: np.ndarray,
    center_x: float,
    center_y: float,
    half_size_px: int,
) -> float:
    x0 = max(0, int(math.floor(float(center_x) - int(half_size_px))))
    x1 = min(magnitude.shape[1], int(math.ceil(float(center_x) + int(half_size_px) + 1)))
    y0 = max(0, int(math.floor(float(center_y) - int(half_size_px))))
    y1 = min(magnitude.shape[0], int(math.ceil(float(center_y) + int(half_size_px) + 1)))
    return float(np.max(np.asarray(magnitude[y0:y1, x0:x1], dtype=np.float64)))


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("radius", "feature_width", "detection_magnitude", "fwhm", "detection_rate"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "radius": f"{int(row['radius'])}",
                    "feature_width": f"{int(row['feature_width'])}",
                    "detection_magnitude": f"{row['detection_magnitude']:.17e}",
                    "fwhm": f"{row['fwhm']:.17e}",
                    "detection_rate": f"{row['detection_rate']:.17e}",
                }
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _accumulator_template(radii: tuple[int, ...], widths: tuple[int, ...]) -> dict[tuple[int, int], dict[str, float]]:
    template: dict[tuple[int, int], dict[str, float]] = {}
    for radius in radii:
        for width in widths:
            template[(int(radius), int(width))] = {"count": 0.0, "mag_sum": 0.0, "fwhm_sum": 0.0, "detect_sum": 0.0}
    return template


def _run_subclass(
    subclass: str,
    cases: list[StimulusCase],
    kernel_plans: dict[int, KernelPlan],
    fft_shape: tuple[int, int],
    patch_size: int,
    batch_size: int,
    workers: int,
    widths: tuple[int, ...],
) -> list[dict[str, float]]:
    radii = tuple(sorted(kernel_plans.keys()))
    accum = _accumulator_template(radii, widths)
    for start in range(0, len(cases), int(batch_size)):
        batch_cases = cases[start:start + int(batch_size)]
        images = np.stack([case.image for case in batch_cases], axis=0)
        image_fft = fft.rfft2(images, s=fft_shape, axes=(-2, -1), workers=workers)
        for radius in radii:
            plan = kernel_plans[int(radius)]
            gx = _valid_response(image_fft, plan.fft_x, fft_shape, patch_size, plan.kernel_shape, workers)
            gy = _valid_response(image_fft, plan.fft_y, fft_shape, patch_size, plan.kernel_shape, workers)
            magnitude = np.hypot(gx, gy)
            for batch_index, case in enumerate(batch_cases):
                profile = _sample_profile(magnitude[batch_index], case.sample_x, case.sample_y)
                search_half = 0.5 * float(case.feature_width) + float(radius) + PROFILE_MARGIN_PX
                _, fwhm, _ = _profile_metrics(
                    profile=profile,
                    t_coords=case.t_coords,
                    search_half=search_half,
                    threshold=plan.detection_threshold,
                )
                detection_mag = _center_window_peak(
                    magnitude=magnitude[batch_index],
                    center_x=case.center_x,
                    center_y=case.center_y,
                    half_size_px=int(CENTER_WINDOW_HALF_PX),
                )
                detected = 1.0 if detection_mag > float(plan.detection_threshold) else 0.0
                bucket = accum[(int(radius), int(case.feature_width))]
                bucket["count"] += 1.0
                bucket["mag_sum"] += float(detection_mag)
                bucket["fwhm_sum"] += float(fwhm)
                bucket["detect_sum"] += float(detected)

    rows = []
    for radius in radii:
        for width in widths:
            bucket = accum[(int(radius), int(width))]
            count = max(bucket["count"], 1.0)
            rows.append(
                {
                    "radius": float(radius),
                    "feature_width": float(width),
                    "detection_magnitude": float(bucket["mag_sum"] / count),
                    "fwhm": float(bucket["fwhm_sum"] / count),
                    "detection_rate": float(bucket["detect_sum"] / count),
                    "threshold": float(kernel_plans[int(radius)].detection_threshold),
                    "noise_floor": float(kernel_plans[int(radius)].noise_floor),
                    "white_noise_gain": float(kernel_plans[int(radius)].white_noise_gain),
                    "kernel_max": float(kernel_plans[int(radius)].kernel_max),
                }
            )
    return rows


def _heatmap_matrix(rows: list[dict[str, float]], radii: tuple[int, ...], widths: tuple[int, ...], field: str) -> list[list[float]]:
    lookup = {(int(row["radius"]), int(row["feature_width"])): float(row[field]) for row in rows}
    return [[lookup[(int(radius), int(width))] for width in widths] for radius in radii]


def _summary_payload(
    bars_rows: list[dict[str, float]],
    disk_rows: list[dict[str, float]],
    output_dir: Path,
    widths: tuple[int, ...],
    radii: tuple[int, ...],
    patch_size: int,
) -> dict[str, object]:
    return {
        "title": "Section 7.4 support radius vs object size",
        "subtitle": "Disk support, bars and disks, $d = 3$, normalize_coords = True, clean binary stimulus",
        "config": {
            "image_size": int(IMAGE_SIZE),
            "local_patch_size": int(patch_size),
            "degree": int(DEGREE),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "contrast": float(CONTRAST),
            "radius_schedule": [int(radius) for radius in radii],
            "feature_width_schedule": [int(width) for width in widths],
            "bar_angle_count": int(round(180.0 / BAR_ANGLE_STEP_DEG)),
            "bar_angle_step_deg": float(BAR_ANGLE_STEP_DEG),
            "bar_phase_count": int(BAR_PHASE_COUNT),
            "disk_phase_count": int(DISK_PHASE_COUNT),
            "phase_offsets_px": list(_phase_offsets(BAR_PHASE_COUNT)),
            "profile_step_px": float(PROFILE_STEP_PX),
            "profile_margin_px": float(PROFILE_MARGIN_PX),
            "center_window_half_px": int(CENTER_WINDOW_HALF_PX),
            "detection_threshold_definition": "5 * (1/255) * sqrt(2 * white_noise_gain)",
            "apparatus_reduction": "Metrics are measured from a centered local patch rather than the full 1024^2 frame because the feature is isolated and the largest support radius is 128 px; the patch fully contains the feature and every neighborhood that can influence the central measurement window.",
        },
        "subclasses": {
            "bars": {
                "label": "Bars",
                "csv_path": str(output_dir / "sec07_support_radius_object_size_bars_d3_normalized.csv"),
                "rows": bars_rows,
                "heatmap": {
                    "radii": [int(radius) for radius in radii],
                    "feature_widths": [int(width) for width in widths],
                    "detection_magnitude": _heatmap_matrix(bars_rows, radii, widths, "detection_magnitude"),
                    "fwhm": _heatmap_matrix(bars_rows, radii, widths, "fwhm"),
                    "detection_rate": _heatmap_matrix(bars_rows, radii, widths, "detection_rate"),
                },
            },
            "disks": {
                "label": "Disks",
                "csv_path": str(output_dir / "sec07_support_radius_object_size_disks_d3_normalized.csv"),
                "rows": disk_rows,
                "heatmap": {
                    "radii": [int(radius) for radius in radii],
                    "feature_widths": [int(width) for width in widths],
                    "detection_magnitude": _heatmap_matrix(disk_rows, radii, widths, "detection_magnitude"),
                    "fwhm": _heatmap_matrix(disk_rows, radii, widths, "fwhm"),
                    "detection_rate": _heatmap_matrix(disk_rows, radii, widths, "detection_rate"),
                },
            },
        },
    }


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    radii: tuple[int, ...],
    widths: tuple[int, ...],
    batch_size: int,
    workers: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    patch_size = 2 * int(PATCH_HALF_SIZE) + 1
    fft_shape = _common_fft_shape(patch_size=patch_size, max_radius=max(radii))
    kernel_plans = {int(radius): _build_kernel_plan(int(radius), fft_shape) for radius in radii}
    bars = _build_bar_cases(
        widths=widths,
        patch_half_size=int(PATCH_HALF_SIZE),
        max_radius=max(radii),
        contrast=float(CONTRAST),
        step_deg=float(BAR_ANGLE_STEP_DEG),
        phase_count=int(BAR_PHASE_COUNT),
    )
    disks = _build_disk_cases(
        widths=widths,
        patch_half_size=int(PATCH_HALF_SIZE),
        max_radius=max(radii),
        contrast=float(CONTRAST),
        phase_count=int(DISK_PHASE_COUNT),
    )

    bars_rows = _run_subclass(
        subclass="bars",
        cases=bars,
        kernel_plans=kernel_plans,
        fft_shape=fft_shape,
        patch_size=patch_size,
        batch_size=int(batch_size),
        workers=int(workers),
        widths=widths,
    )
    disk_rows = _run_subclass(
        subclass="disks",
        cases=disks,
        kernel_plans=kernel_plans,
        fft_shape=fft_shape,
        patch_size=patch_size,
        batch_size=int(batch_size),
        workers=int(workers),
        widths=widths,
    )

    bars_csv = output_dir / "sec07_support_radius_object_size_bars_d3_normalized.csv"
    disks_csv = output_dir / "sec07_support_radius_object_size_disks_d3_normalized.csv"
    _write_csv(bars_csv, bars_rows)
    _write_csv(disks_csv, disk_rows)
    _write_json(
        summary_json,
        _summary_payload(
            bars_rows=bars_rows,
            disk_rows=disk_rows,
            output_dir=output_dir,
            widths=widths,
            radii=radii,
            patch_size=patch_size,
        ),
    )

    for label, rows in (("bars", bars_rows), ("disks", disk_rows)):
        best = max(rows, key=lambda row: row["detection_magnitude"])
        print(
            f"{label}: best_mag_radius={int(best['radius'])}, "
            f"best_mag_width={int(best['feature_width'])}, "
            f"detection={best['detection_magnitude']:.6f}"
        )

    return {
        "bars_csv": bars_csv,
        "disks_csv": disks_csv,
        "summary_json": summary_json,
    }


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.4 object-size figures cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.4 radius-vs-object-size sweep on bars and disks.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_object_size",
        help="Directory for subclass CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_object_size" / "sec07_support_radius_object_size_summary_d3_normalized.json",
        help="Path for the combined Section 7.4 object-size summary JSON.",
    )
    parser.add_argument("--radius-list", type=str, default=None, help="Optional comma-separated radius subset for smoke tests.")
    parser.add_argument("--width-list", type=str, default=None, help="Optional comma-separated feature-width subset for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=FFT_BATCH_SIZE, help="Batch size for FFT-based kernel application.")
    parser.add_argument("--fft-workers", type=int, default=-1, help="Worker count for scipy.fft. Use -1 for all available workers.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ heatmaps after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_value_list(args.radius_list, int) or RADII))
    widths = tuple(int(value) for value in (_parse_value_list(args.width_list, int) or FEATURE_WIDTHS))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        radii=radii,
        widths=widths,
        batch_size=int(args.batch_size),
        workers=int(args.fft_workers),
    )

    if args.compile_plot:
        bars_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_radius_object_size_bars_d3_normalized.typ"
        bars_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_radius_object_size_bars_d3_normalized.pdf"
        compile_plot(bars_src, bars_pdf)
        outputs["bars_pdf"] = bars_pdf

        disks_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_radius_object_size_disks_d3_normalized.typ"
        disks_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_radius_object_size_disks_d3_normalized.pdf"
        compile_plot(disks_src, disks_pdf)
        outputs["disks_pdf"] = disks_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
