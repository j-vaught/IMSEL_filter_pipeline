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

from wvf_metal import magnitude
from wvf.radius import build_wvf_radius_kernels

IMAGE_SIZES = (512, 1024, 2048, 4096)
BASELINE_FEATURE_SCALES = (2, 4, 8, 16, 32, 64, 128, 256)
RADIUS_SCHEDULE = (2, 3, 4, 5, 7, 9, 11, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512)
DEGREE = 3
NORMALIZE_COORDS = True
CONTRAST = 1.0
VARIANT = "fft"
DEFAULT_FFT_BACKEND = "auto"
NOISE_FLOOR_SIGMA = 1.0 / 255.0
PROFILE_STEP_PX = 0.25
PROFILE_MARGIN_PX = 8.0
WINDOW_SCALE = 0.125
INSTANCE_OFFSETS = ((-0.25, 0.20), (0.30, -0.15))


@dataclass(frozen=True)
class FeatureInstance:
    baseline_scale_px: int
    feature_scale_px: int
    center_x: float
    center_y: float


def _parse_int_list(text: str | None) -> tuple[int, ...]:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(int(item))
    return tuple(values)


def _scaled_feature_scales(image_size: int, baseline_scales: tuple[int, ...]) -> tuple[int, ...]:
    factor = float(image_size) / 1024.0
    return tuple(int(round(float(scale) * factor)) for scale in baseline_scales)


def _radii_for_image_size(image_size: int, full_schedule: tuple[int, ...]) -> tuple[int, ...]:
    cap = int(image_size) // 8
    return tuple(int(radius) for radius in full_schedule if int(radius) <= cap)


def _scene_layout(image_size: int, baseline_scales: tuple[int, ...]) -> list[FeatureInstance]:
    cell_size = float(image_size) / 4.0
    scaled = _scaled_feature_scales(int(image_size), baseline_scales)
    instances: list[FeatureInstance] = []
    for pass_index, offset_xy in enumerate(INSTANCE_OFFSETS):
        for scale_index, (baseline_scale, feature_scale) in enumerate(zip(baseline_scales, scaled, strict=True)):
            row = scale_index // 2
            col = scale_index % 2 + 2 * pass_index
            center_x = (float(col) + 0.5) * cell_size + float(offset_xy[0])
            center_y = (float(row) + 0.5) * cell_size + float(offset_xy[1])
            if not (0.0 <= center_x < float(image_size) and 0.0 <= center_y < float(image_size)):
                raise ValueError(
                    f"feature center escaped the image frame for size={image_size}, "
                    f"baseline_scale={baseline_scale}, pass={pass_index}, row={row}, col={col}"
                )
            instances.append(
                FeatureInstance(
                    baseline_scale_px=int(baseline_scale),
                    feature_scale_px=int(feature_scale),
                    center_x=float(center_x),
                    center_y=float(center_y),
                )
            )
    return instances


def _render_scene(image_size: int, baseline_scales: tuple[int, ...]) -> tuple[np.ndarray, list[FeatureInstance]]:
    yy, xx = np.meshgrid(np.arange(int(image_size), dtype=np.float64), np.arange(int(image_size), dtype=np.float64), indexing="ij")
    scene = np.zeros((int(image_size), int(image_size)), dtype=np.float32)
    instances = _scene_layout(int(image_size), baseline_scales)
    for instance in instances:
        radius = 0.5 * float(instance.feature_scale_px)
        mask = (xx - float(instance.center_x)) ** 2 + (yy - float(instance.center_y)) ** 2 <= radius * radius
        scene[mask] = float(CONTRAST)
    return scene, instances


def _sample_profile(image: np.ndarray, center_x: float, center_y: float, feature_scale_px: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    search_half = 0.5 * float(feature_scale_px) + float(radius) + float(PROFILE_MARGIN_PX)
    t_coords = np.arange(-search_half, search_half + 0.5 * PROFILE_STEP_PX, PROFILE_STEP_PX, dtype=np.float64)
    sample_x = float(center_x) + t_coords
    sample_y = np.full_like(sample_x, float(center_y))
    profile = ndimage.map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.vstack((sample_y, sample_x)),
        order=1,
        mode="nearest",
    )
    return t_coords, np.asarray(profile, dtype=np.float64)


def _crossing_x(t0: float, y0: float, t1: float, y1: float, target: float) -> float:
    if abs(float(y1) - float(y0)) <= 1.0e-15:
        return 0.5 * (float(t0) + float(t1))
    alpha = (float(target) - float(y0)) / (float(y1) - float(y0))
    return float(t0) + float(np.clip(alpha, 0.0, 1.0)) * (float(t1) - float(t0))


def _profile_fwhm(t_coords: np.ndarray, profile: np.ndarray, feature_scale_px: int, radius: int) -> float:
    search_half = 0.5 * float(feature_scale_px) + float(radius) + float(PROFILE_MARGIN_PX)
    mask = np.abs(np.asarray(t_coords, dtype=np.float64)) <= float(search_half)
    local_t = np.asarray(t_coords, dtype=np.float64)[mask]
    local_profile = np.asarray(profile, dtype=np.float64)[mask]
    peak_value = float(np.max(local_profile))
    if peak_value <= 0.0:
        return 0.0
    half_value = 0.5 * peak_value
    above = np.flatnonzero(local_profile >= half_value)
    if above.size == 0:
        return 0.0
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
    return max(0.0, float(right_t - left_t))


def _window_peak(image: np.ndarray, center_x: float, center_y: float, feature_scale_px: int) -> float:
    half = max(2, int(math.ceil(float(feature_scale_px) * float(WINDOW_SCALE))))
    x0 = max(0, int(math.floor(float(center_x) - half)))
    x1 = min(image.shape[1], int(math.ceil(float(center_x) + half + 1)))
    y0 = max(0, int(math.floor(float(center_y) - half)))
    y1 = min(image.shape[0], int(math.ceil(float(center_y) + half + 1)))
    return float(np.max(np.asarray(image[y0:y1, x0:x1], dtype=np.float64)))


def _kernel_threshold(radius: int) -> tuple[float, float, float]:
    kernels = build_wvf_radius_kernels(int(radius), order=DEGREE, normalize_coords=NORMALIZE_COORDS)
    white_noise_gain = float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2))
    noise_floor = float(NOISE_FLOOR_SIGMA) * math.sqrt(2.0 * white_noise_gain)
    threshold = 5.0 * noise_floor
    return white_noise_gain, noise_floor, threshold


def _rows_for_image_size(
    image_size: int,
    baseline_scales: tuple[int, ...],
    radii: tuple[int, ...],
    fft_backend: str,
    device_index: int | None,
) -> tuple[list[dict[str, float]], dict[int, dict[str, float]]]:
    scene, instances = _render_scene(int(image_size), baseline_scales)
    rows: list[dict[str, float]] = []
    grouped = {int(scale): [] for scale in baseline_scales}

    for radius in radii:
        mag = magnitude(
            scene,
            radius=int(radius),
            degree=int(DEGREE),
            normalize_coords=bool(NORMALIZE_COORDS),
            variant=VARIANT,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        white_noise_gain, noise_floor, threshold = _kernel_threshold(int(radius))
        for baseline_scale in baseline_scales:
            values = [instance for instance in instances if int(instance.baseline_scale_px) == int(baseline_scale)]
            detection_peaks = []
            fwhm_values = []
            detections = []
            for instance in values:
                peak = _window_peak(mag, instance.center_x, instance.center_y, instance.feature_scale_px)
                t_coords, profile = _sample_profile(mag, instance.center_x, instance.center_y, instance.feature_scale_px, int(radius))
                fwhm = _profile_fwhm(t_coords, profile, instance.feature_scale_px, int(radius))
                detection_peaks.append(float(peak))
                fwhm_values.append(float(fwhm))
                detections.append(1.0 if float(peak) > float(threshold) else 0.0)
            row = {
                "image_size_px": float(image_size),
                "feature_scale_px": float(_scaled_feature_scales(int(image_size), (int(baseline_scale),))[0]),
                "baseline_feature_scale_px": float(baseline_scale),
                "radius": float(radius),
                "detection_magnitude": float(np.mean(np.asarray(detection_peaks, dtype=np.float64))),
                "fwhm": float(np.mean(np.asarray(fwhm_values, dtype=np.float64))),
                "detection_rate": float(np.mean(np.asarray(detections, dtype=np.float64))),
                "white_noise_gain": float(white_noise_gain),
                "noise_floor": float(noise_floor),
                "threshold": float(threshold),
            }
            rows.append(row)
            grouped[int(baseline_scale)].append(row)

    optimal: dict[int, dict[str, float]] = {}
    for baseline_scale, group_rows in grouped.items():
        best = max(group_rows, key=lambda row: float(row["detection_magnitude"]))
        optimal[int(baseline_scale)] = {
            "feature_scale_px": float(best["feature_scale_px"]),
            "optimal_radius": float(best["radius"]),
            "detection_magnitude": float(best["detection_magnitude"]),
            "fwhm": float(best["fwhm"]),
            "detection_rate": float(best["detection_rate"]),
        }
    return rows, optimal


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("feature_scale_px", "radius", "detection_magnitude", "fwhm", "detection_rate"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "feature_scale_px": f"{int(round(float(row['feature_scale_px'])))}",
                    "radius": f"{int(round(float(row['radius'])))}",
                    "detection_magnitude": f"{float(row['detection_magnitude']):.17e}",
                    "fwhm": f"{float(row['fwhm']):.17e}",
                    "detection_rate": f"{float(row['detection_rate']):.17e}",
                }
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    image_sizes: tuple[int, ...],
    baseline_scales: tuple[int, ...],
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    per_size: dict[str, dict[str, object]] = {}
    plot_series = []

    for image_size in image_sizes:
        radii = _radii_for_image_size(int(image_size), RADIUS_SCHEDULE)
        rows, optimal = _rows_for_image_size(
            image_size=int(image_size),
            baseline_scales=baseline_scales,
            radii=radii,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        csv_path = output_dir / f"sec07_support_radius_image_size_{int(image_size)}_d3_normalized.csv"
        _write_csv(csv_path, rows)
        outputs[f"csv_{int(image_size)}"] = csv_path
        per_size[str(int(image_size))] = {
            "image_size_px": int(image_size),
            "radii": [int(radius) for radius in radii],
            "rows": rows,
            "optimal_by_baseline_scale": optimal,
            "csv_path": str(csv_path),
        }
        plot_points = [
            {
                "baseline_feature_scale_px": int(baseline_scale),
                "feature_scale_px": float(optimal[int(baseline_scale)]["feature_scale_px"]),
                "optimal_radius": float(optimal[int(baseline_scale)]["optimal_radius"]),
            }
            for baseline_scale in baseline_scales
        ]
        plot_series.append(
            {
                "image_size_px": int(image_size),
                "label": f"{int(image_size)}^2",
                "points": plot_points,
            }
        )
        best = max(rows, key=lambda row: float(row["detection_magnitude"]))
        print(
            f"{int(image_size)}^2: best_scale={int(round(float(best['feature_scale_px'])))} px, "
            f"best_radius={int(round(float(best['radius'])))} px, "
            f"mag={float(best['detection_magnitude']):.6f}"
        )

    payload = {
        "title": "Section 7.4 radius vs image size on the multi-scale composite scene",
        "subtitle": "Disk support, $d = 3$, normalize_coords = True, clean multi-scale disk composite",
        "config": {
            "image_sizes_px": [int(size) for size in image_sizes],
            "baseline_feature_scales_px": [int(scale) for scale in baseline_scales],
            "radius_schedule_full": [int(radius) for radius in RADIUS_SCHEDULE],
            "degree": int(DEGREE),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "contrast": float(CONTRAST),
            "variant": VARIANT,
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "composite_scene_layout": "A 4x4 grid of filled disks with two instances per baseline scale. The same scene layout is rendered at each image size, and each feature diameter scales proportionally with image size so the feature occupies the same fraction of the frame at every resolution.",
            "instance_offsets_px": [list(offset) for offset in INSTANCE_OFFSETS],
            "window_scale": float(WINDOW_SCALE),
            "profile_step_px": float(PROFILE_STEP_PX),
            "profile_margin_px": float(PROFILE_MARGIN_PX),
            "detection_threshold_definition": "5 * (1/255) * sqrt(2 * white_noise_gain)",
        },
        "per_image_size": per_size,
        "optimal_radius_plot": plot_series,
    }
    _write_json(summary_json, payload)
    outputs["summary_json"] = summary_json
    return outputs


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.4 image-size figure cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.4 radius-vs-image-size sweep on the multi-scale composite scene.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_image_size",
        help="Directory for per-image-size CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_radius_image_size" / "sec07_support_radius_image_size_summary_d3_normalized.json",
        help="Path for the combined Section 7.4 image-size summary JSON.",
    )
    parser.add_argument("--image-size-list", type=str, default=None, help="Optional comma-separated image-size subset for smoke tests.")
    parser.add_argument("--baseline-scale-list", type=str, default=None, help="Optional comma-separated baseline feature-scale subset.")
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for the FFT backend.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ headline plot after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_sizes = tuple(int(value) for value in (_parse_int_list(args.image_size_list) or IMAGE_SIZES))
    baseline_scales = tuple(int(value) for value in (_parse_int_list(args.baseline_scale_list) or BASELINE_FEATURE_SCALES))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        image_sizes=image_sizes,
        baseline_scales=baseline_scales,
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )

    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_radius_image_size_d3_normalized.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_radius_image_size_d3_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["plot_pdf"] = figure_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
