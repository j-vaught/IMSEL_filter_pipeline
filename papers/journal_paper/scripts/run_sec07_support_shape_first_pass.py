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
    build_polygon_support_kernels,
    build_square_support_kernels,
    build_design_matrix,
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
BOUNDING_RADIUS = 15
SQUARE_HALF_SIDE = 15
DEGREE = 3
NORMALIZE_COORDS = True
SHAPE_ORDER = ("triangle", "square", "diamond", "hexagon", "octagon", "disk")


@dataclass(frozen=True)
class ShapeResponseSpec:
    name: str
    label: str
    support_value: float
    support_key: str
    symmetry_order: int
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    offsets_xy: np.ndarray
    support_cardinality: int
    design_matrix_shape: tuple[int, int]
    kappa_design_matrix: float

    @property
    def slug(self) -> str:
        value_label = str(int(self.support_value))
        return f"{self.name}_{self.support_key}{value_label}_d{DEGREE}_normalized"


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


def _spec_from_generic(
    kernels: SupportKernels,
    label: str,
    support_key: str,
    symmetry_order: int,
) -> ShapeResponseSpec:
    return ShapeResponseSpec(
        name=kernels.support_name,
        label=label,
        support_value=float(kernels.support_value),
        support_key=support_key,
        symmetry_order=int(symmetry_order),
        kernel_x=np.asarray(kernels.kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernels.kernel_y, dtype=np.float64),
        offsets_xy=np.asarray(kernels.offsets_xy, dtype=np.float64),
        support_cardinality=int(kernels.support_cardinality),
        design_matrix_shape=(
            int(kernels.design_matrix_shape[0]),
            int(kernels.design_matrix_shape[1]),
        ),
        kappa_design_matrix=float(kernels.kappa_design_matrix),
    )


def _disk_spec(radius: int, degree: int, normalize_coords: bool) -> ShapeResponseSpec:
    disk = build_wvf_radius_kernels(radius=radius, order=degree, normalize_coords=normalize_coords)
    disk_design = build_design_matrix(
        disk.offsets_xy,
        degree=degree,
        normalize_radius=float(radius) if normalize_coords else None,
    )
    return ShapeResponseSpec(
        name="disk",
        label="Disk",
        support_value=float(radius),
        support_key="r",
        symmetry_order=999,
        kernel_x=np.asarray(disk.kernel_x, dtype=np.float64),
        kernel_y=np.asarray(disk.kernel_y, dtype=np.float64),
        offsets_xy=np.asarray(disk.offsets_xy, dtype=np.float64),
        support_cardinality=int(disk.offsets_xy.shape[0]),
        design_matrix_shape=(int(disk_design.shape[0]), int(disk_design.shape[1])),
        kappa_design_matrix=design_condition_number(disk_design),
    )


def _shape_specs(radius: int, half_side: int, degree: int, normalize_coords: bool) -> tuple[ShapeResponseSpec, ...]:
    square = build_square_support_kernels(
        half_side=half_side,
        degree=degree,
        normalize_coords=normalize_coords,
    )
    triangle = build_polygon_support_kernels(
        name="triangle",
        vertices_xy=regular_polygon_vertices(3, radius, rotation_deg=90.0),
        bounding_radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
    )
    diamond = build_polygon_support_kernels(
        name="diamond",
        vertices_xy=regular_polygon_vertices(4, radius, rotation_deg=0.0),
        bounding_radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
    )
    hexagon = build_polygon_support_kernels(
        name="hexagon",
        vertices_xy=regular_polygon_vertices(6, radius, rotation_deg=0.0),
        bounding_radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
    )
    octagon = build_polygon_support_kernels(
        name="octagon",
        vertices_xy=regular_polygon_vertices(8, radius, rotation_deg=0.0),
        bounding_radius=radius,
        degree=degree,
        normalize_coords=normalize_coords,
    )
    specs = (
        _spec_from_generic(triangle, label="Triangle", support_key="r", symmetry_order=3),
        _spec_from_generic(square, label="Square", support_key="h", symmetry_order=4),
        _spec_from_generic(diamond, label="Diamond", support_key="r", symmetry_order=4),
        _spec_from_generic(hexagon, label="Hexagon", support_key="r", symmetry_order=6),
        _spec_from_generic(octagon, label="Octagon", support_key="r", symmetry_order=8),
        _disk_spec(radius=radius, degree=degree, normalize_coords=normalize_coords),
    )
    return tuple(sorted(specs, key=lambda spec: SHAPE_ORDER.index(spec.name)))


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
    specs = _shape_specs(
        radius=BOUNDING_RADIUS,
        half_side=SQUARE_HALF_SIDE,
        degree=DEGREE,
        normalize_coords=NORMALIZE_COORDS,
    )

    image_shape = (int(image_size), int(image_size))
    kernel_shapes = [spec.kernel_x.shape for spec in specs]
    pad, _, fft_shape = _fft_support(kernel_shapes, image_shape)
    pad_y, pad_x = pad

    coords = np.arange(image_size, dtype=np.float64) - (image_size - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    angles_deg = _angles_deg(angle_step_deg)
    phases_px = _phases_px(phase_count, phase_step_px)

    per_shape_records: dict[str, list[dict[str, float]]] = {spec.name: [] for spec in specs}

    for theta_deg in angles_deg:
        theta_rad = math.radians(float(theta_deg))
        projection = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)
        kernel_ffts = {
            spec.name: _directional_kernel_fft(
                spec.kernel_x,
                spec.kernel_y,
                theta_rad,
                fft_shape,
                workers,
            )
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
        "title": "Section 7.3 support-shape six-shape sweep",
        "shape_order": list(SHAPE_ORDER),
        "shapes": {},
    }

    for spec in specs:
        records = per_shape_records[spec.name]
        response_values = np.asarray([row["response_magnitude"] for row in records], dtype=np.float64)
        theta_values = np.asarray([row["theta_deg"] for row in records], dtype=np.float64)
        max_index = int(np.argmax(response_values))
        min_index = int(np.argmin(response_values))
        anisotropy_ratio = float(response_values[max_index] / response_values[min_index])

        csv_path = output_dir / f"sec07_support_shape_{spec.slug}.csv"
        json_path = output_dir / f"sec07_support_shape_{spec.slug}.json"
        _write_curve_csv(csv_path, records)
        _write_json(
            json_path,
            {
                "shape": spec.name,
                "shape_label": spec.label,
                "degree": DEGREE,
                "normalize_coords": NORMALIZE_COORDS,
                "support_key": spec.support_key,
                "support_value": spec.support_value,
                "support_cardinality": spec.support_cardinality,
                "symmetry_order": spec.symmetry_order,
                "anisotropy_ratio": anisotropy_ratio,
                "theta_max_deg": float(theta_values[max_index]),
                "theta_min_deg": float(theta_values[min_index]),
                "kappa_design_matrix": float(spec.kappa_design_matrix),
                "design_matrix_shape": [
                    int(spec.design_matrix_shape[0]),
                    int(spec.design_matrix_shape[1]),
                ],
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
            "support_cardinality": spec.support_cardinality,
            "curve": records,
        }
        outputs[f"{spec.name}_csv"] = csv_path
        outputs[f"{spec.name}_json"] = json_path
        print(
            f"{spec.label}: anisotropy={anisotropy_ratio:.6f}, "
            f"theta_max={theta_values[max_index]:.1f} deg, "
            f"theta_min={theta_values[min_index]:.1f} deg, "
            f"kappa={spec.kappa_design_matrix:.6e}, "
            f"cardinality={spec.support_cardinality}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.3 six-shape support comparison.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_shape_six_shape",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--overlay-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_support_shape_six_shape" / "sec07_support_shape_overlay_six_shape_r15_d3_normalized.json",
        help="Path for the combined overlay-plot JSON payload.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square image size for the clean step-edge render.",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=ANGLE_STEP_DEG,
        help="Orientation sampling step in degrees on [0, 180).",
    )
    parser.add_argument(
        "--phase-count",
        type=int,
        default=PHASE_COUNT,
        help="Number of sub-pixel phases to average per orientation.",
    )
    parser.add_argument(
        "--phase-step-px",
        type=float,
        default=PHASE_STEP_PX,
        help="Phase step size in pixels.",
    )
    parser.add_argument(
        "--edge-width-px",
        type=float,
        default=EDGE_WIDTH_PX,
        help="Width parameter for the tanh-smoothed step edge.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=CONTRAST,
        help="Step-edge contrast amplitude.",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=-1,
        help="Worker count for scipy.fft operations. Use -1 for all available workers.",
    )
    parser.add_argument(
        "--compile-plot",
        action="store_true",
        help="Compile the checked-in Typst/CeTZ overlay plot after writing JSON outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.overlay_json.parent.mkdir(parents=True, exist_ok=True)

    outputs = run_experiment(
        output_dir=output_dir,
        figure_data_path=args.overlay_json.resolve(),
        image_size=int(args.image_size),
        angle_step_deg=float(args.angle_step_deg),
        phase_count=int(args.phase_count),
        phase_step_px=float(args.phase_step_px),
        contrast=float(args.contrast),
        width_px=float(args.edge_width_px),
        workers=int(args.fft_workers),
    )

    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_support_shape_six_shape_r15_d3_normalized.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_support_shape_six_shape_r15_d3_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["overlay_pdf"] = figure_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
