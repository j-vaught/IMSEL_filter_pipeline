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
from scipy import fft


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from square_support_sg import (
    SupportKernels,
    build_design_matrix,
    build_polygon_support_kernels,
    build_square_support_kernels,
    design_condition_number,
    regular_polygon_vertices,
)
from wvf.radius import build_wvf_radius_kernels


CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
IMAGE_SIZE = 1024
ANGLE_STEP_DEG = 0.5
PHASE_STEP_PX = 0.25
PHASE_COUNT = 4
DEGREE = 3
NORMALIZE_COORDS = True
DISK_RADIUS = 15.0
SHAPE_ORDER = ("triangle", "square", "diamond", "hexagon", "octagon", "disk")
MATCH_MODES = ("support-cardinality", "white-noise-gain", "second-moment")

CARDINALITY_SEARCH_RANGES = {
    "square": (11.5, 14.5, 121),
    "diamond": (16.5, 20.5, 161),
    "triangle": (21.0, 25.5, 181),
    "hexagon": (15.0, 18.5, 141),
    "octagon": (14.5, 17.0, 101),
}

COARSE_MATCH_RANGES = {
    "square": (7.5, 26.0),
    "diamond": (7.5, 30.0),
    "triangle": (7.5, 30.0),
    "hexagon": (7.5, 26.0),
    "octagon": (7.5, 24.0),
}


@dataclass(frozen=True)
class ShapeDefinition:
    name: str
    label: str
    support_key: str
    symmetry_order: int
    polygon_sides: int | None = None
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    label: str
    support_key: str
    support_value: float
    symmetry_order: int
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    offsets_xy: np.ndarray
    support_cardinality: int
    white_noise_gain: float
    effective_second_moment: float
    design_matrix_shape: tuple[int, int]
    kappa_design_matrix: float

    @property
    def slug(self) -> str:
        return f"{self.name}_{self.support_key}{_format_param_tag(self.support_value)}_d{DEGREE}_normalized"


SHAPES: dict[str, ShapeDefinition] = {
    "triangle": ShapeDefinition("triangle", "Triangle", "r", 3, polygon_sides=3, rotation_deg=90.0),
    "square": ShapeDefinition("square", "Square", "h", 4),
    "diamond": ShapeDefinition("diamond", "Diamond", "r", 4, polygon_sides=4, rotation_deg=0.0),
    "hexagon": ShapeDefinition("hexagon", "Hexagon", "r", 6, polygon_sides=6, rotation_deg=0.0),
    "octagon": ShapeDefinition("octagon", "Octagon", "r", 8, polygon_sides=8, rotation_deg=0.0),
    "disk": ShapeDefinition("disk", "Disk", "r", 999),
}


def _format_param_tag(value: float) -> str:
    rounded_int = int(round(float(value)))
    if math.isclose(float(value), float(rounded_int), rel_tol=0.0, abs_tol=1.0e-9):
        return str(rounded_int)
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _angles_deg(step_deg: float) -> np.ndarray:
    count = int(round(180.0 / float(step_deg)))
    return np.arange(count, dtype=np.float64) * float(step_deg)


def _phases_px(count: int, step_px: float) -> np.ndarray:
    return np.arange(int(count), dtype=np.float64) * float(step_px)


def _render_smoothed_step(
    projection: np.ndarray,
    phase_px: float,
    contrast: float,
    width_px: float,
) -> np.ndarray:
    return 0.5 * float(contrast) * (1.0 + np.tanh((projection - float(phase_px)) / float(width_px)))


def _build_spec(shape_name: str, support_value: float, degree: int, normalize_coords: bool) -> ShapeSpec:
    shape = SHAPES[shape_name]
    if shape_name == "disk":
        kernels = build_wvf_radius_kernels(
            radius=int(round(float(support_value))),
            order=int(degree),
            normalize_coords=bool(normalize_coords),
        )
        offsets_xy = np.asarray(kernels.offsets_xy, dtype=np.float64)
        weights_x = np.asarray(kernels.weights_x, dtype=np.float64)
        weights_y = np.asarray(kernels.weights_y, dtype=np.float64)
        kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
        kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    else:
        if shape_name == "square":
            built: SupportKernels = build_square_support_kernels(
                half_side=float(support_value),
                degree=int(degree),
                normalize_coords=bool(normalize_coords),
            )
        else:
            assert shape.polygon_sides is not None
            built = build_polygon_support_kernels(
                name=shape.name,
                vertices_xy=regular_polygon_vertices(
                    shape.polygon_sides,
                    float(support_value),
                    rotation_deg=float(shape.rotation_deg),
                ),
                bounding_radius=float(support_value),
                degree=int(degree),
                normalize_coords=bool(normalize_coords),
            )
        offsets_xy = np.asarray(built.offsets_xy, dtype=np.float64)
        weights_x = np.asarray(built.weights_x, dtype=np.float64)
        weights_y = np.asarray(built.weights_y, dtype=np.float64)
        kernel_x = np.asarray(built.kernel_x, dtype=np.float64)
        kernel_y = np.asarray(built.kernel_y, dtype=np.float64)

    white_noise_gain = float(np.sum(weights_x**2))
    radial_sq = offsets_xy[:, 0] ** 2 + offsets_xy[:, 1] ** 2
    effective_second_moment = float(np.sum(radial_sq * (weights_x**2)) / white_noise_gain)
    design = build_design_matrix(
        offsets_xy,
        degree=int(degree),
        normalize_radius=float(support_value) if normalize_coords else None,
    )
    return ShapeSpec(
        name=shape.name,
        label=shape.label,
        support_key=shape.support_key,
        support_value=float(support_value),
        symmetry_order=int(shape.symmetry_order),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        offsets_xy=offsets_xy,
        support_cardinality=int(offsets_xy.shape[0]),
        white_noise_gain=white_noise_gain,
        effective_second_moment=effective_second_moment,
        design_matrix_shape=(int(design.shape[0]), int(design.shape[1])),
        kappa_design_matrix=float(design_condition_number(design)),
    )


def _metric_value(spec: ShapeSpec, match_mode: str) -> float:
    if match_mode == "support-cardinality":
        return float(spec.support_cardinality)
    if match_mode == "white-noise-gain":
        return float(spec.white_noise_gain)
    if match_mode == "second-moment":
        return float(spec.effective_second_moment)
    raise ValueError(f"unsupported match mode {match_mode}")


def _metric_label(match_mode: str) -> str:
    return {
        "support-cardinality": "support_cardinality",
        "white-noise-gain": "white_noise_gain",
        "second-moment": "effective_second_moment",
    }[match_mode]


def _match_mode_title(match_mode: str) -> tuple[str, str, str]:
    if match_mode == "support-cardinality":
        return (
            "Section 7.3 support-shape six-shape sweep",
            "Matched support cardinality, $d = 3$, normalize_coords = True, clean smoothed step edge",
            "sec07_support_shape_cardinality_match",
        )
    if match_mode == "white-noise-gain":
        return (
            "Section 7.3 support-shape six-shape sweep",
            "Matched white-noise gain, $d = 3$, normalize_coords = True, clean smoothed step edge",
            "sec07_support_shape_noise_gain_match",
        )
    if match_mode == "second-moment":
        return (
            "Section 7.3 support-shape six-shape sweep",
            "Matched effective second moment, $d = 3$, normalize_coords = True, clean smoothed step edge",
            "sec07_support_shape_second_moment_match",
        )
    raise ValueError(f"unsupported match mode {match_mode}")


def _coarse_values(shape_name: str) -> np.ndarray:
    lo, hi = COARSE_MATCH_RANGES[shape_name]
    return np.linspace(float(lo), float(hi), 11, dtype=np.float64)


def _cardinality_values(shape_name: str) -> np.ndarray:
    lo, hi, count = CARDINALITY_SEARCH_RANGES[shape_name]
    return np.linspace(float(lo), float(hi), int(count), dtype=np.float64)


def _calibration_records(
    shape_name: str,
    candidate_values: np.ndarray,
    degree: int,
    normalize_coords: bool,
) -> tuple[list[dict[str, float]], list[ShapeSpec]]:
    records: list[dict[str, float]] = []
    specs: list[ShapeSpec] = []
    for value in candidate_values:
        spec = _build_spec(shape_name, float(value), degree, normalize_coords)
        specs.append(spec)
        records.append(
            {
                "support_value": float(value),
                "support_cardinality": float(spec.support_cardinality),
                "white_noise_gain": float(spec.white_noise_gain),
                "effective_second_moment": float(spec.effective_second_moment),
                "kappa_design_matrix": float(spec.kappa_design_matrix),
            }
        )
    return records, specs


def _pick_best_spec(specs: list[ShapeSpec], target_value: float, match_mode: str) -> tuple[ShapeSpec, float]:
    best_spec = min(specs, key=lambda spec: abs(_metric_value(spec, match_mode) - target_value))
    best_error = abs(_metric_value(best_spec, match_mode) - target_value)
    return best_spec, float(best_error)


def _calibrate_non_disk(
    shape_name: str,
    match_mode: str,
    target_value: float,
    degree: int,
    normalize_coords: bool,
) -> tuple[ShapeSpec, list[dict[str, float]]]:
    if match_mode == "support-cardinality":
        values = _cardinality_values(shape_name)
        records, specs = _calibration_records(shape_name, values, degree, normalize_coords)
        chosen, error = _pick_best_spec(specs, target_value, match_mode)
        if error > 0.05 * float(target_value):
            raise RuntimeError(
                f"{shape_name} support-cardinality match missed 5% tolerance: "
                f"target={target_value:.3f}, realized={_metric_value(chosen, match_mode):.3f}"
            )
        return chosen, records

    coarse_values = _coarse_values(shape_name)
    coarse_records, coarse_specs = _calibration_records(shape_name, coarse_values, degree, normalize_coords)
    coarse_best_spec, _ = _pick_best_spec(coarse_specs, target_value, match_mode)
    coarse_best_index = min(
        range(len(coarse_specs)),
        key=lambda index: abs(_metric_value(coarse_specs[index], match_mode) - target_value),
    )
    if len(coarse_values) == 1:
        fine_lo = fine_hi = float(coarse_values[0])
    else:
        left = float(coarse_values[max(0, coarse_best_index - 1)])
        right = float(coarse_values[min(len(coarse_values) - 1, coarse_best_index + 1)])
        fine_lo = left
        fine_hi = right
    fine_values = np.linspace(fine_lo, fine_hi, 81, dtype=np.float64)
    fine_records, fine_specs = _calibration_records(shape_name, fine_values, degree, normalize_coords)
    chosen, _ = _pick_best_spec(fine_specs, target_value, match_mode)
    return chosen, coarse_records + fine_records


def _calibrate_specs(match_mode: str, degree: int, normalize_coords: bool) -> tuple[dict[str, ShapeSpec], dict[str, list[dict[str, float]]], dict[str, float]]:
    disk = _build_spec("disk", DISK_RADIUS, degree, normalize_coords)
    target_value = _metric_value(disk, match_mode)
    specs = {"disk": disk}
    calibrations = {
        "disk": [
            {
                "support_value": float(DISK_RADIUS),
                "support_cardinality": float(disk.support_cardinality),
                "white_noise_gain": float(disk.white_noise_gain),
                "effective_second_moment": float(disk.effective_second_moment),
                "kappa_design_matrix": float(disk.kappa_design_matrix),
            }
        ]
    }
    for name in SHAPE_ORDER:
        if name == "disk":
            continue
        chosen, records = _calibrate_non_disk(name, match_mode, target_value, degree, normalize_coords)
        specs[name] = chosen
        calibrations[name] = records
    return specs, calibrations, {"target_value": float(target_value)}


def _fft_support(
    kernel_shapes: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    max_kh = max(shape[0] for shape in kernel_shapes)
    max_kw = max(shape[1] for shape in kernel_shapes)
    pad_y = max_kh // 2
    pad_x = max_kw // 2
    padded_h = image_shape[0] + 2 * pad_y
    padded_w = image_shape[1] + 2 * pad_x
    full_h = padded_h + max_kh - 1
    full_w = padded_w + max_kw - 1
    return (pad_y, pad_x), (full_h, full_w), (fft.next_fast_len(full_h), fft.next_fast_len(full_w))


def _directional_kernel_fft(
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    theta_rad: float,
    fft_shape: tuple[int, int],
    workers: int,
) -> np.ndarray:
    directional = kernel_x * math.cos(theta_rad) + kernel_y * math.sin(theta_rad)
    return fft.rfft2(directional[::-1, ::-1], s=fft_shape, workers=workers)


def _response_peak_from_fft(
    image_fft: np.ndarray,
    kernel_fft: np.ndarray,
    fft_shape: tuple[int, int],
    image_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    workers: int,
) -> float:
    response = fft.irfft2(image_fft * kernel_fft, s=fft_shape, workers=workers)
    kh, kw = kernel_shape
    height, width = image_shape
    valid = response[kh - 1:kh - 1 + height, kw - 1:kw - 1 + width]
    return float(np.max(np.abs(valid)))


def _write_curve_csv(csv_path: Path, records: list[dict[str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("theta_deg", "response_magnitude", "response_magnitude_std"),
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "theta_deg": f"{row['theta_deg']:.6f}",
                    "response_magnitude": f"{row['response_magnitude']:.17e}",
                    "response_magnitude_std": f"{row['response_magnitude_std']:.17e}",
                }
            )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    match_mode: str,
    output_dir: Path,
    figure_data_path: Path,
    image_size: int,
    angle_step_deg: float,
    phase_count: int,
    phase_step_px: float,
    contrast: float,
    width_px: float,
    workers: int,
) -> dict[str, Path]:
    title, subtitle, _ = _match_mode_title(match_mode)
    specs_by_name, calibrations, target = _calibrate_specs(match_mode, DEGREE, NORMALIZE_COORDS)
    specs = [specs_by_name[name] for name in SHAPE_ORDER]

    image_shape = (int(image_size), int(image_size))
    kernel_shapes = [spec.kernel_x.shape for spec in specs]
    pad, _, fft_shape = _fft_support(kernel_shapes, image_shape)
    pad_y, pad_x = pad

    coords = np.arange(image_size, dtype=np.float64) - (image_size - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    angles_deg = _angles_deg(angle_step_deg)
    phases_px = _phases_px(phase_count, phase_step_px)
    metric_name = _metric_label(match_mode)
    target_value = float(target["target_value"])

    per_shape_records: dict[str, list[dict[str, float]]] = {spec.name: [] for spec in specs}

    for theta_deg in angles_deg:
        theta_rad = math.radians(float(theta_deg))
        projection = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)
        kernel_ffts = {
            spec.name: _directional_kernel_fft(spec.kernel_x, spec.kernel_y, theta_rad, fft_shape, workers)
            for spec in specs
        }
        phase_peaks: dict[str, list[float]] = {spec.name: [] for spec in specs}

        for phase_px in phases_px:
            image = _render_smoothed_step(projection, phase_px, contrast, width_px)
            padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
            image_fft = fft.rfft2(padded, s=fft_shape, workers=workers)
            for spec in specs:
                phase_peaks[spec.name].append(
                    _response_peak_from_fft(
                        image_fft,
                        kernel_ffts[spec.name],
                        fft_shape,
                        image_shape,
                        spec.kernel_x.shape,
                        workers,
                    )
                )

        for spec in specs:
            peaks = np.asarray(phase_peaks[spec.name], dtype=np.float64)
            per_shape_records[spec.name].append(
                {
                    "theta_deg": float(theta_deg),
                    "response_magnitude": float(np.mean(peaks)),
                    "response_magnitude_std": float(np.std(peaks, ddof=0)),
                }
            )

    outputs: dict[str, Path] = {}
    overlay_payload = {
        "title": title,
        "subtitle": subtitle,
        "shape_order": list(SHAPE_ORDER),
        "match_mode": match_mode,
        "shapes": {},
    }

    for spec in specs:
        records = per_shape_records[spec.name]
        response_values = np.asarray([row["response_magnitude"] for row in records], dtype=np.float64)
        theta_values = np.asarray([row["theta_deg"] for row in records], dtype=np.float64)
        max_index = int(np.argmax(response_values))
        min_index = int(np.argmin(response_values))
        anisotropy_ratio = float(response_values[max_index] / response_values[min_index])
        realized_metric = _metric_value(spec, match_mode)
        calibration_records = calibrations[spec.name]

        csv_path = output_dir / f"sec07_support_shape_{match_mode}_{spec.slug}.csv"
        json_path = output_dir / f"sec07_support_shape_{match_mode}_{spec.slug}.json"
        _write_curve_csv(csv_path, records)
        _write_json(
            json_path,
            {
                "shape": spec.name,
                "shape_label": spec.label,
                "degree": DEGREE,
                "normalize_coords": NORMALIZE_COORDS,
                "match_mode": match_mode,
                "matching_metric": metric_name,
                "matching_target_value": target_value,
                "support_key": spec.support_key,
                "support_value": spec.support_value,
                "support_cardinality": spec.support_cardinality,
                "white_noise_gain": spec.white_noise_gain,
                "effective_second_moment": spec.effective_second_moment,
                "symmetry_order": spec.symmetry_order,
                "anisotropy_ratio": anisotropy_ratio,
                "theta_max_deg": float(theta_values[max_index]),
                "theta_min_deg": float(theta_values[min_index]),
                "kappa_design_matrix": float(spec.kappa_design_matrix),
                "design_matrix_shape": [int(spec.design_matrix_shape[0]), int(spec.design_matrix_shape[1])],
                "matching_error_abs": float(abs(realized_metric - target_value)),
                "matching_error_rel": float(abs(realized_metric - target_value) / target_value),
                "calibration_curve": calibration_records,
                "render_config": {
                    "contrast": float(contrast),
                    "edge_width_px": float(width_px),
                    "image_size": int(image_size),
                    "phase_count": int(phase_count),
                    "phase_step_px": float(phase_step_px),
                    "angle_step_deg": float(angle_step_deg),
                },
                "curve": records,
            },
        )
        overlay_payload["shapes"][spec.name] = {
            "label": spec.label,
            "symmetry_order": spec.symmetry_order,
            "anisotropy_ratio": anisotropy_ratio,
            "support_key": spec.support_key,
            "support_value": spec.support_value,
            "support_cardinality": spec.support_cardinality,
            "white_noise_gain": spec.white_noise_gain,
            "effective_second_moment": spec.effective_second_moment,
            "curve": records,
        }
        outputs[f"{spec.name}_csv"] = csv_path
        outputs[f"{spec.name}_json"] = json_path
        print(
            f"{spec.label}: support_{spec.support_key}={spec.support_value:.3f}, "
            f"cardinality={spec.support_cardinality}, "
            f"gain={spec.white_noise_gain:.9e}, "
            f"m2={spec.effective_second_moment:.6f}, "
            f"anisotropy={anisotropy_ratio:.6f}, "
            f"kappa={spec.kappa_design_matrix:.6e}"
        )

    _write_json(figure_data_path, overlay_payload)
    outputs["overlay_json"] = figure_data_path
    return outputs


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the overlay plot cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        check=True,
        cwd=ROOT,
    )


def _default_paths(match_mode: str) -> tuple[Path, Path, Path, Path]:
    _, _, stem = _match_mode_title(match_mode)
    data_dir = ROOT / "papers" / "journal_paper" / "figures" / "data" / stem
    overlay_json = data_dir / f"{stem}_overlay_r15_d3_normalized.json"
    figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / f"fig_{stem}_r15_d3_normalized.typ"
    figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / f"fig_{stem}_r15_d3_normalized.pdf"
    return data_dir, overlay_json, figure_src, figure_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.3 matched support-shape sweep.")
    parser.add_argument(
        "--match-mode",
        choices=MATCH_MODES,
        required=True,
        help="Matching rule for the six-shape sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and JSON outputs. Defaults to the match-mode specific directory.",
    )
    parser.add_argument(
        "--overlay-json",
        type=Path,
        default=None,
        help="Path for the combined overlay-plot JSON payload.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Square image size for the step-edge render.")
    parser.add_argument("--angle-step-deg", type=float, default=ANGLE_STEP_DEG, help="Orientation sampling step in degrees on [0, 180).")
    parser.add_argument("--phase-count", type=int, default=PHASE_COUNT, help="Number of sub-pixel phases to average per orientation.")
    parser.add_argument("--phase-step-px", type=float, default=PHASE_STEP_PX, help="Phase step size in pixels.")
    parser.add_argument("--edge-width-px", type=float, default=EDGE_WIDTH_PX, help="Width parameter for the tanh-smoothed step edge.")
    parser.add_argument("--contrast", type=float, default=CONTRAST, help="Step-edge contrast amplitude.")
    parser.add_argument("--fft-workers", type=int, default=-1, help="Worker count for scipy.fft operations. Use -1 for all available workers.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ overlay plot after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_output_dir, default_overlay_json, figure_src, figure_pdf = _default_paths(args.match_mode)
    output_dir = (args.output_dir or default_output_dir).resolve()
    overlay_json = (args.overlay_json or default_overlay_json).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_json.parent.mkdir(parents=True, exist_ok=True)

    outputs = run_experiment(
        match_mode=str(args.match_mode),
        output_dir=output_dir,
        figure_data_path=overlay_json,
        image_size=int(args.image_size),
        angle_step_deg=float(args.angle_step_deg),
        phase_count=int(args.phase_count),
        phase_step_px=float(args.phase_step_px),
        contrast=float(args.contrast),
        width_px=float(args.edge_width_px),
        workers=int(args.fft_workers),
    )

    if args.compile_plot:
        compile_plot(figure_src, figure_pdf)
        outputs["overlay_pdf"] = figure_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
