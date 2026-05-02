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

from wvf.radius import build_wvf_radius_kernels, disk_offsets
from wvf_metal.metal import fft_gradients_with_kernel

IMAGE_SIZE = 1024
MULTISCALE_FEATURE_SCALES = (2, 4, 8, 16, 32, 64, 128, 256)
RADIUS_SCHEDULE = (2, 3, 4, 5, 7, 9, 11, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128)
CLOSE_EDGE_SEPARATIONS = (1, 2, 4, 8, 16, 32)
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
NOISE_FLOOR_SIGMA = 1.0 / 255.0
BATCH_CASES = 32
PATCH_HALF_SIZE = 192
PROFILE_STEP_PX = 0.25
PROFILE_MARGIN_PX = 8.0
WINDOW_SCALE = 0.125
EDGE_WINDOW_HALF_PX = 3.0
NORMALIZE_COORDS = True
INSTANCE_OFFSETS = ((-0.25, 0.20), (0.30, -0.15))
DEFAULT_FFT_BACKEND = "vkfft"


@dataclass(frozen=True)
class FeatureInstance:
    baseline_scale_px: int
    feature_scale_px: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class KernelPlan:
    radius: int
    degree: int
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    white_noise_gain: float
    threshold: float


@dataclass(frozen=True)
class BarCase:
    separation_px: int
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    t_coords: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray


@dataclass(frozen=True)
class EdgeCase:
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    t_coords: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray


def _parse_int_list(text: str | None) -> tuple[int, ...]:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(int(item))
    return tuple(values)


def _phase_values() -> tuple[float, ...]:
    return tuple(float(PHASE_STEP_PX) * idx for idx in range(int(PHASE_COUNT)))


def _orientation_values() -> tuple[float, ...]:
    count = int(round(180.0 / float(ORIENTATION_STEP_DEG)))
    return tuple(float(ORIENTATION_STEP_DEG) * idx for idx in range(count))


def _recommended_target_degree(radius: int) -> int:
    if int(radius) < 5:
        return 5
    if int(radius) < 9:
        return 9
    return 11


def _coeff_count(degree: int) -> int:
    return (int(degree) + 1) * (int(degree) + 2) // 2


def _recommended_degree(radius: int) -> int:
    target = int(_recommended_target_degree(int(radius)))
    support = int(disk_offsets(int(radius), include_center=False).shape[0])
    degree = int(target)
    if degree % 2 == 0:
        degree -= 1
    while degree > 1 and _coeff_count(degree) > support:
        degree -= 2
    return max(1, int(degree))


def _build_kernel_plan(radius: int) -> KernelPlan:
    degree = _recommended_degree(int(radius))
    kernels = build_wvf_radius_kernels(int(radius), order=int(degree), normalize_coords=bool(NORMALIZE_COORDS))
    kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
    kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    white_noise_gain = float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2))
    threshold = 5.0 * float(NOISE_FLOOR_SIGMA) * math.sqrt(2.0 * white_noise_gain)
    return KernelPlan(
        radius=int(radius),
        degree=int(degree),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        white_noise_gain=float(white_noise_gain),
        threshold=float(threshold),
    )


def _scene_layout(image_size: int, baseline_scales: tuple[int, ...]) -> list[FeatureInstance]:
    cell_size = float(image_size) / 4.0
    instances: list[FeatureInstance] = []
    for pass_index, offset_xy in enumerate(INSTANCE_OFFSETS):
        for scale_index, baseline_scale in enumerate(baseline_scales):
            row = scale_index // 2
            col = scale_index % 2 + 2 * pass_index
            center_x = (float(col) + 0.5) * cell_size + float(offset_xy[0])
            center_y = (float(row) + 0.5) * cell_size + float(offset_xy[1])
            instances.append(
                FeatureInstance(
                    baseline_scale_px=int(baseline_scale),
                    feature_scale_px=int(baseline_scale),
                    center_x=float(center_x),
                    center_y=float(center_y),
                )
            )
    return instances


def _render_multiscale_scene() -> tuple[np.ndarray, list[FeatureInstance]]:
    yy, xx = np.meshgrid(
        np.arange(IMAGE_SIZE, dtype=np.float64),
        np.arange(IMAGE_SIZE, dtype=np.float64),
        indexing="ij",
    )
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    instances = _scene_layout(IMAGE_SIZE, MULTISCALE_FEATURE_SCALES)
    for instance in instances:
        radius = 0.5 * float(instance.feature_scale_px)
        mask = (xx - float(instance.center_x)) ** 2 + (yy - float(instance.center_y)) ** 2 <= radius * radius
        image[mask] = float(CONTRAST)
    return image, instances


def _window_peak(image: np.ndarray, center_x: float, center_y: float, feature_scale_px: int) -> float:
    half = max(2, int(math.ceil(float(feature_scale_px) * float(WINDOW_SCALE))))
    x0 = max(0, int(math.floor(float(center_x) - half)))
    x1 = min(image.shape[1], int(math.ceil(float(center_x) + half + 1)))
    y0 = max(0, int(math.floor(float(center_y) - half)))
    y1 = min(image.shape[0], int(math.ceil(float(center_y) + half + 1)))
    return float(np.max(np.asarray(image[y0:y1, x0:x1], dtype=np.float64)))


def _tile_cases(images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    if not images:
        raise ValueError("cannot tile an empty batch")
    tile_h, tile_w = images[0].shape
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    placements: list[tuple[slice, slice]] = []
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        row_slice = slice(row * tile_h, (row + 1) * tile_h)
        col_slice = slice(col * tile_w, (col + 1) * tile_w)
        canvas[row_slice, col_slice] = np.asarray(image, dtype=np.float32)
        placements.append((row_slice, col_slice))
    return canvas, placements


def _apply_batched_magnitude(
    images: list[np.ndarray],
    radius: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
) -> list[np.ndarray]:
    canvas, placements = _tile_cases(images)
    gx_canvas, gy_canvas = fft_gradients_with_kernel(
        canvas,
        radius=int(radius),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    mag_canvas = np.hypot(np.asarray(gx_canvas, dtype=np.float64), np.asarray(gy_canvas, dtype=np.float64))
    outputs = []
    for row_slice, col_slice in placements:
        outputs.append(np.asarray(mag_canvas[row_slice, col_slice], dtype=np.float64).copy())
    return outputs


def _local_patch_coords() -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _render_bar_patch(xx: np.ndarray, yy: np.ndarray, width_px: float, angle_deg: float, phase_px: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    normal = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta) - float(phase_px)
    return 0.5 * float(CONTRAST) * (
        np.tanh((normal + 0.5 * float(width_px)) / float(EDGE_WIDTH_PX))
        - np.tanh((normal - 0.5 * float(width_px)) / float(EDGE_WIDTH_PX))
    )


def _render_step_patch(xx: np.ndarray, yy: np.ndarray, angle_deg: float, phase_px: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    normal = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta) - float(phase_px)
    return 0.5 * float(CONTRAST) * (1.0 + np.tanh(normal / float(EDGE_WIDTH_PX)))


def _bar_profile_geometry(width_px: float, angle_deg: float, phase_px: float, max_radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(float(angle_deg))
    search_half = 0.5 * float(width_px) + float(max_radius) + float(PROFILE_MARGIN_PX)
    t_coords = np.arange(-search_half, search_half + 0.5 * float(PROFILE_STEP_PX), float(PROFILE_STEP_PX), dtype=np.float64)
    center = float(PATCH_HALF_SIZE)
    center_x = center + float(phase_px) * math.cos(theta)
    center_y = center + float(phase_px) * math.sin(theta)
    sample_x = center_x + t_coords * math.cos(theta)
    sample_y = center_y + t_coords * math.sin(theta)
    return t_coords, sample_x, sample_y


def _build_bar_cases(separations: tuple[int, ...]) -> list[BarCase]:
    xx, yy = _local_patch_coords()
    cases: list[BarCase] = []
    max_radius = max(int(radius) for radius in RADIUS_SCHEDULE)
    for separation in separations:
        for angle_deg in _orientation_values():
            for phase_px in _phase_values():
                t_coords, sample_x, sample_y = _bar_profile_geometry(
                    width_px=float(separation),
                    angle_deg=float(angle_deg),
                    phase_px=float(phase_px),
                    max_radius=int(max_radius),
                )
                cases.append(
                    BarCase(
                        separation_px=int(separation),
                        orientation_deg=float(angle_deg),
                        phase_px=float(phase_px),
                        image=np.asarray(
                            _render_bar_patch(
                                xx,
                                yy,
                                width_px=float(separation),
                                angle_deg=float(angle_deg),
                                phase_px=float(phase_px),
                            ),
                            dtype=np.float32,
                        ),
                        t_coords=np.asarray(t_coords, dtype=np.float64),
                        sample_x=np.asarray(sample_x, dtype=np.float64),
                        sample_y=np.asarray(sample_y, dtype=np.float64),
                    )
                )
    return cases


def _build_edge_reference_cases() -> list[EdgeCase]:
    xx, yy = _local_patch_coords()
    cases: list[EdgeCase] = []
    max_radius = max(int(radius) for radius in RADIUS_SCHEDULE)
    max_separation = max(int(value) for value in CLOSE_EDGE_SEPARATIONS)
    for angle_deg in _orientation_values():
        for phase_px in _phase_values():
            t_coords, sample_x, sample_y = _bar_profile_geometry(
                width_px=float(max_separation),
                angle_deg=float(angle_deg),
                phase_px=float(phase_px),
                max_radius=int(max_radius),
            )
            cases.append(
                EdgeCase(
                    orientation_deg=float(angle_deg),
                    phase_px=float(phase_px),
                    image=np.asarray(
                        _render_step_patch(
                            xx,
                            yy,
                            angle_deg=float(angle_deg),
                            phase_px=float(phase_px),
                        ),
                        dtype=np.float32,
                    ),
                    t_coords=np.asarray(t_coords, dtype=np.float64),
                    sample_x=np.asarray(sample_x, dtype=np.float64),
                    sample_y=np.asarray(sample_y, dtype=np.float64),
                )
            )
    return cases


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


def _windowed_profile_peak(t_coords: np.ndarray, profile: np.ndarray, center_pos: float, half_width: float) -> float:
    t = np.asarray(t_coords, dtype=np.float64)
    values = np.asarray(profile, dtype=np.float64)
    mask = np.abs(t - float(center_pos)) <= float(half_width)
    if not bool(np.any(mask)):
        return 0.0
    return float(np.max(values[mask]))


def _multiscale_rows(
    fft_backend: str,
    device_index: int | None,
    radii: tuple[int, ...],
) -> tuple[list[dict[str, float]], dict[str, object]]:
    scene, instances = _render_multiscale_scene()
    rows: list[dict[str, float]] = []
    heatmap_matrix = []
    full_coverage = []
    for radius in radii:
        plan = _build_kernel_plan(int(radius))
        gx, gy = fft_gradients_with_kernel(
            scene,
            radius=int(plan.radius),
            kernel_x=plan.kernel_x,
            kernel_y=plan.kernel_y,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        mag = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        row_values = []
        all_detected = True
        for scale in MULTISCALE_FEATURE_SCALES:
            matching = [item for item in instances if int(item.baseline_scale_px) == int(scale)]
            peaks = [_window_peak(mag, item.center_x, item.center_y, item.feature_scale_px) for item in matching]
            detections = [1.0 if float(peak) > float(plan.threshold) else 0.0 for peak in peaks]
            detection_rate = float(np.mean(np.asarray(detections, dtype=np.float64)))
            mean_peak = float(np.mean(np.asarray(peaks, dtype=np.float64)))
            rows.append(
                {
                    "radius": float(radius),
                    "degree": float(plan.degree),
                    "feature_scale_px": float(scale),
                    "detection_rate": float(detection_rate),
                    "detection_magnitude": float(mean_peak),
                    "white_noise_gain": float(plan.white_noise_gain),
                    "threshold": float(plan.threshold),
                }
            )
            row_values.append(float(detection_rate))
            if detection_rate < 1.0:
                all_detected = False
        heatmap_matrix.append(row_values)
        full_coverage.append(
            {
                "radius": int(radius),
                "degree": int(plan.degree),
                "all_feature_scales_detected": bool(all_detected),
            }
        )
        print(
            f"sec79a r={int(radius)} d={int(plan.degree)} "
            f"coverage={sum(1 for value in row_values if value >= 1.0)}/{len(row_values)}"
        )
    heatmap = {
        "radii": [int(radius) for radius in radii],
        "feature_scales_px": [int(scale) for scale in MULTISCALE_FEATURE_SCALES],
        "detection_rate": heatmap_matrix,
    }
    return rows, {"heatmap": heatmap, "full_coverage_by_radius": full_coverage}


def _clutter_rows(
    fft_backend: str,
    device_index: int | None,
    radii: tuple[int, ...],
    separations: tuple[int, ...],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    cases = _build_bar_cases(separations)
    ref_cases = _build_edge_reference_cases()
    rows: list[dict[str, float]] = []
    thresholds: list[dict[str, float]] = []
    images = [case.image for case in cases]
    ref_images = [case.image for case in ref_cases]
    for radius in radii:
        plan = _build_kernel_plan(int(radius))
        magnitudes: list[np.ndarray] = []
        for batch_start in range(0, len(images), int(BATCH_CASES)):
            batch_images = images[batch_start : batch_start + int(BATCH_CASES)]
            magnitudes.extend(
                _apply_batched_magnitude(
                    batch_images,
                    radius=int(plan.radius),
                    kernel_x=plan.kernel_x,
                    kernel_y=plan.kernel_y,
                    fft_backend=fft_backend,
                    device_index=device_index,
                )
            )
        reference_magnitudes: list[np.ndarray] = []
        for batch_start in range(0, len(ref_images), int(BATCH_CASES)):
            batch_images = ref_images[batch_start : batch_start + int(BATCH_CASES)]
            reference_magnitudes.extend(
                _apply_batched_magnitude(
                    batch_images,
                    radius=int(plan.radius),
                    kernel_x=plan.kernel_x,
                    kernel_y=plan.kernel_y,
                    fft_backend=fft_backend,
                    device_index=device_index,
                )
            )
        reference_lookup: dict[tuple[float, float], float] = {}
        for case, mag in zip(ref_cases, reference_magnitudes, strict=True):
            profile = _sample_profile(mag, case.sample_x, case.sample_y)
            reference_lookup[(float(case.orientation_deg), float(case.phase_px))] = _windowed_profile_peak(
                t_coords=case.t_coords,
                profile=profile,
                center_pos=0.0,
                half_width=float(EDGE_WINDOW_HALF_PX),
            )
        threshold_sep: int | None = None
        for separation in separations:
            retention_values = []
            for case, mag in zip(cases, magnitudes, strict=True):
                if int(case.separation_px) != int(separation):
                    continue
                profile = _sample_profile(mag, case.sample_x, case.sample_y)
                left_peak = _windowed_profile_peak(
                    t_coords=case.t_coords,
                    profile=profile,
                    center_pos=-0.5 * float(separation),
                    half_width=float(EDGE_WINDOW_HALF_PX),
                )
                right_peak = _windowed_profile_peak(
                    t_coords=case.t_coords,
                    profile=profile,
                    center_pos=0.5 * float(separation),
                    half_width=float(EDGE_WINDOW_HALF_PX),
                )
                ref_peak = float(reference_lookup[(float(case.orientation_deg), float(case.phase_px))])
                bar_peak = 0.5 * (float(left_peak) + float(right_peak))
                retention = 0.0 if ref_peak <= 0.0 else float(bar_peak) / float(ref_peak)
                retention_values.append(float(retention))
            retention_ratio = float(np.mean(np.asarray(retention_values, dtype=np.float64)))
            rows.append(
                {
                    "radius": float(radius),
                    "degree": float(plan.degree),
                    "separation_px": float(separation),
                    "retention_ratio": float(retention_ratio),
                }
            )
            if retention_ratio > 0.9 and threshold_sep is None:
                threshold_sep = int(separation)
        thresholds.append(
            {
                "radius": int(radius),
                "degree": int(plan.degree),
                "resolution_threshold_px": None if threshold_sep is None else int(threshold_sep),
            }
        )
        shown = "none" if threshold_sep is None else str(int(threshold_sep))
        print(f"sec79b r={int(radius)} d={int(plan.degree)} threshold={shown}")
    return rows, thresholds


def _write_multiscale_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("radius", "degree", "feature_scale_px", "detection_rate", "detection_magnitude", "white_noise_gain", "threshold"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "radius": f"{int(round(float(row['radius'])))}",
                    "degree": f"{int(round(float(row['degree'])))}",
                    "feature_scale_px": f"{int(round(float(row['feature_scale_px'])))}",
                    "detection_rate": f"{float(row['detection_rate']):.17e}",
                    "detection_magnitude": f"{float(row['detection_magnitude']):.17e}",
                    "white_noise_gain": f"{float(row['white_noise_gain']):.17e}",
                    "threshold": f"{float(row['threshold']):.17e}",
                }
            )


def _write_clutter_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("radius", "separation", "retention_ratio"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "radius": f"{int(round(float(row['radius'])))}",
                    "separation": f"{int(round(float(row['separation_px'])))}",
                    "retention_ratio": f"{float(row['retention_ratio']):.17e}",
                }
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    radii: tuple[int, ...],
    separations: tuple[int, ...],
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    multiscale_rows, multiscale_extra = _multiscale_rows(
        fft_backend=fft_backend,
        device_index=device_index,
        radii=radii,
    )
    clutter_rows, thresholds = _clutter_rows(
        fft_backend=fft_backend,
        device_index=device_index,
        radii=radii,
        separations=separations,
    )

    multiscale_csv = output_dir / "sec07_multiscale_stress_detection_rate.csv"
    clutter_csv = output_dir / "sec07_close_edge_clutter_retention.csv"
    _write_multiscale_csv(multiscale_csv, multiscale_rows)
    _write_clutter_csv(clutter_csv, clutter_rows)

    payload = {
        "title": "Section 7.9 multi-scale stress and close-edge clutter",
        "subtitle": "Disk support, normalize_coords = True, recommended degree defaults, precomputed-kernel FFT application",
        "config": {
            "image_size_px": int(IMAGE_SIZE),
            "radii": [int(radius) for radius in radii],
            "feature_scales_px": [int(scale) for scale in MULTISCALE_FEATURE_SCALES],
            "close_edge_separations_px": [int(value) for value in separations],
            "degree_rule": "r < 5 -> target d=5, 5 <= r < 9 -> target d=9, r >= 9 -> target d=11, capped to the largest feasible odd degree for the disk support",
            "normalize_coords": bool(NORMALIZE_COORDS),
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "noise_floor_sigma": float(NOISE_FLOOR_SIGMA),
            "multiscale_threshold_definition": "5 * (1/255) * sqrt(2 * white_noise_gain)",
            "clutter_retention_definition": "Mean of the two bar-edge peak magnitudes divided by the isolated single-edge peak magnitude for the same orientation and phase.",
            "clutter_threshold_rule": "Smallest separation with retention_ratio > 0.9",
        },
        "multiscale_stress": {
            "rows": multiscale_rows,
            "csv_path": str(multiscale_csv),
            "heatmap": multiscale_extra["heatmap"],
            "full_coverage_by_radius": multiscale_extra["full_coverage_by_radius"],
        },
        "close_edge_clutter": {
            "rows": clutter_rows,
            "csv_path": str(clutter_csv),
            "threshold_by_radius": thresholds,
        },
    }
    _write_json(summary_json, payload)
    return {
        "multiscale_csv": multiscale_csv,
        "clutter_csv": clutter_csv,
        "summary_json": summary_json,
    }


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.9 figures cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.9 multi-scale stress and close-edge clutter experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_multiscale_and_clutter",
        help="Directory for the Section 7.9 CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_multiscale_and_clutter" / "sec07_multiscale_and_clutter_summary_normalized.json",
        help="Path for the combined Section 7.9 summary JSON.",
    )
    parser.add_argument("--radius-list", type=str, default=None, help="Optional comma-separated radius subset.")
    parser.add_argument("--separation-list", type=str, default=None, help="Optional comma-separated close-edge separation subset.")
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figures after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_int_list(args.radius_list) or RADIUS_SCHEDULE))
    separations = tuple(int(value) for value in (_parse_int_list(args.separation_list) or CLOSE_EDGE_SEPARATIONS))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        radii=radii,
        separations=separations,
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )

    if args.compile_plots:
        detection_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_multiscale_stress_detection_rate.typ"
        detection_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_multiscale_stress_detection_rate.pdf"
        compile_plot(detection_src, detection_pdf)
        outputs["multiscale_plot_pdf"] = detection_pdf

        clutter_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_close_edge_clutter_threshold.typ"
        clutter_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_close_edge_clutter_threshold.pdf"
        compile_plot(clutter_src, clutter_pdf)
        outputs["clutter_plot_pdf"] = clutter_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
