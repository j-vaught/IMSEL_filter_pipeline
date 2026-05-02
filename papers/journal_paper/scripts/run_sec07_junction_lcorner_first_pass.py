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

from square_support_sg import build_design_matrix, build_square_support_kernels, design_condition_number
from wvf.radius import build_wvf_radius_kernels


CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
IMAGE_SIZE = 1024
ANGLE_STEP_DEG = 10.0
PHASE_STEP_PX = 0.25
PHASE_COUNT = 4
DEGREE = 3
NORMALIZE_COORDS = True
DISK_RADIUS = 15.0
SQUARE_HALF_SIDE = 15.0
BRANCH_SAMPLE_DISTANCES = (4.0, 6.0, 8.0)


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    label: str
    support_key: str
    support_value: float
    kernel_x: np.ndarray
    kernel_y: np.ndarray
    support_cardinality: int
    kappa_design_matrix: float
    design_matrix_shape: tuple[int, int]

    @property
    def slug(self) -> str:
        value = int(round(self.support_value))
        return f"{self.name}_{self.support_key}{value}_d{DEGREE}_normalized"


def _angles_deg(step_deg: float) -> np.ndarray:
    count = int(round(360.0 / float(step_deg)))
    return np.arange(count, dtype=np.float64) * float(step_deg)


def _phases_px(count: int, step_px: float) -> np.ndarray:
    return np.arange(int(count), dtype=np.float64) * float(step_px)


def _sigmoid(value: np.ndarray, width_px: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(value / float(width_px)))


def _render_smoothed_l_corner(
    xx: np.ndarray,
    yy: np.ndarray,
    theta_rad: float,
    phase_px: float,
    contrast: float,
    width_px: float,
) -> tuple[np.ndarray, tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    u = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)
    v = -xx * math.sin(theta_rad) + yy * math.cos(theta_rad)
    image = float(contrast) * _sigmoid(u - float(phase_px), width_px) * _sigmoid(v - float(phase_px), width_px)
    corner_xy = (
        float(phase_px * (math.cos(theta_rad) - math.sin(theta_rad))),
        float(phase_px * (math.sin(theta_rad) + math.cos(theta_rad))),
    )
    branch_dirs = (
        np.asarray((math.cos(theta_rad), math.sin(theta_rad)), dtype=np.float64),
        np.asarray((-math.sin(theta_rad), math.cos(theta_rad)), dtype=np.float64),
    )
    return image, corner_xy, branch_dirs


def _build_spec_disk() -> ShapeSpec:
    kernels = build_wvf_radius_kernels(
        radius=int(round(DISK_RADIUS)),
        order=DEGREE,
        normalize_coords=NORMALIZE_COORDS,
    )
    offsets_xy = np.asarray(kernels.offsets_xy, dtype=np.float64)
    design = build_design_matrix(offsets_xy, degree=DEGREE, normalize_radius=DISK_RADIUS)
    return ShapeSpec(
        name="disk",
        label="Disk",
        support_key="r",
        support_value=DISK_RADIUS,
        kernel_x=np.asarray(kernels.kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernels.kernel_y, dtype=np.float64),
        support_cardinality=int(offsets_xy.shape[0]),
        kappa_design_matrix=float(design_condition_number(design)),
        design_matrix_shape=(int(design.shape[0]), int(design.shape[1])),
    )


def _build_spec_square() -> ShapeSpec:
    kernels = build_square_support_kernels(
        half_side=SQUARE_HALF_SIDE,
        degree=DEGREE,
        normalize_coords=NORMALIZE_COORDS,
    )
    return ShapeSpec(
        name="square",
        label="Square",
        support_key="h",
        support_value=SQUARE_HALF_SIDE,
        kernel_x=np.asarray(kernels.kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernels.kernel_y, dtype=np.float64),
        support_cardinality=int(kernels.support_cardinality),
        kappa_design_matrix=float(kernels.kappa_design_matrix),
        design_matrix_shape=(
            int(kernels.design_matrix_shape[0]),
            int(kernels.design_matrix_shape[1]),
        ),
    )


def _fft_support(
    kernel_shapes: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    max_kh = max(shape[0] for shape in kernel_shapes)
    max_kw = max(shape[1] for shape in kernel_shapes)
    pad_y = max_kh // 2
    pad_x = max_kw // 2
    padded_h = image_shape[0] + 2 * pad_y
    padded_w = image_shape[1] + 2 * pad_x
    full_h = padded_h + max_kh - 1
    full_w = padded_w + max_kw - 1
    return (pad_y, pad_x), (fft.next_fast_len(full_h), fft.next_fast_len(full_w))


def _kernel_fft(kernel: np.ndarray, fft_shape: tuple[int, int], workers: int) -> np.ndarray:
    return fft.rfft2(kernel[::-1, ::-1], s=fft_shape, workers=workers)


def _valid_response(
    image_fft: np.ndarray,
    kernel_fft: np.ndarray,
    fft_shape: tuple[int, int],
    image_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
    workers: int,
) -> np.ndarray:
    response = fft.irfft2(image_fft * kernel_fft, s=fft_shape, workers=workers)
    kh, kw = kernel_shape
    height, width = image_shape
    return response[kh - 1:kh - 1 + height, kw - 1:kw - 1 + width]


def _bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
    height, width = image.shape
    col = float(x) + (width - 1) / 2.0
    row = float(y) + (height - 1) / 2.0
    col = min(max(col, 0.0), width - 1.0)
    row = min(max(row, 0.0), height - 1.0)
    x0 = int(math.floor(col))
    x1 = min(x0 + 1, width - 1)
    y0 = int(math.floor(row))
    y1 = min(y0 + 1, height - 1)
    wx = col - x0
    wy = row - y0
    top = (1.0 - wx) * float(image[y0, x0]) + wx * float(image[y0, x1])
    bottom = (1.0 - wx) * float(image[y1, x0]) + wx * float(image[y1, x1])
    return (1.0 - wy) * top + wy * bottom


def _write_curve_csv(csv_path: Path, records: list[dict[str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("theta_deg", "junction_magnitude", "branch_isotropy_ratio"),
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "theta_deg": f"{row['theta_deg']:.6f}",
                    "junction_magnitude": f"{row['junction_magnitude']:.17e}",
                    "branch_isotropy_ratio": f"{row['branch_isotropy_ratio']:.17e}",
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
    specs = (_build_spec_disk(), _build_spec_square())
    image_shape = (int(image_size), int(image_size))
    pad, fft_shape = _fft_support([spec.kernel_x.shape for spec in specs], image_shape)
    pad_y, pad_x = pad

    coords = np.arange(image_size, dtype=np.float64) - (image_size - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    angles_deg = _angles_deg(angle_step_deg)
    phases_px = _phases_px(phase_count, phase_step_px)

    kernel_ffts = {
        spec.name: (
            _kernel_fft(spec.kernel_x, fft_shape, workers),
            _kernel_fft(spec.kernel_y, fft_shape, workers),
        )
        for spec in specs
    }
    per_shape_records: dict[str, list[dict[str, float]]] = {spec.name: [] for spec in specs}

    for theta_deg in angles_deg:
        theta_rad = math.radians(float(theta_deg))
        phase_metrics = {
            spec.name: {"junction": [], "branch_iso": []}
            for spec in specs
        }

        for phase_px in phases_px:
            image, corner_xy, branch_dirs = _render_smoothed_l_corner(xx, yy, theta_rad, float(phase_px), contrast, width_px)
            padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
            image_fft = fft.rfft2(padded, s=fft_shape, workers=workers)

            for spec in specs:
                gx_fft, gy_fft = kernel_ffts[spec.name]
                gx = _valid_response(image_fft, gx_fft, fft_shape, image_shape, spec.kernel_x.shape, workers)
                gy = _valid_response(image_fft, gy_fft, fft_shape, image_shape, spec.kernel_y.shape, workers)
                magnitude = np.hypot(gx, gy)

                junction_mag = _bilinear_sample(magnitude, corner_xy[0], corner_xy[1])
                branch_values = []
                for direction in branch_dirs:
                    samples = [
                        _bilinear_sample(
                            magnitude,
                            corner_xy[0] + float(distance) * float(direction[0]),
                            corner_xy[1] + float(distance) * float(direction[1]),
                        )
                        for distance in BRANCH_SAMPLE_DISTANCES
                    ]
                    branch_values.append(float(np.mean(samples)))

                branch_iso = max(branch_values) / max(min(branch_values), 1.0e-15)
                phase_metrics[spec.name]["junction"].append(float(junction_mag))
                phase_metrics[spec.name]["branch_iso"].append(float(branch_iso))

        for spec in specs:
            junctions = np.asarray(phase_metrics[spec.name]["junction"], dtype=np.float64)
            branch_iso = np.asarray(phase_metrics[spec.name]["branch_iso"], dtype=np.float64)
            per_shape_records[spec.name].append(
                {
                    "theta_deg": float(theta_deg),
                    "junction_magnitude": float(np.mean(junctions)),
                    "branch_isotropy_ratio": float(np.mean(branch_iso)),
                }
            )

    outputs: dict[str, Path] = {}
    overlay_payload = {
        "title": "Section 7.3 L-corner junction first pass",
        "subtitle": "Disk vs square, matched bounding rule, $d = 3$, normalize_coords = True",
        "shape_order": ["disk", "square"],
        "shapes": {},
    }

    for spec in specs:
        records = per_shape_records[spec.name]
        junction_values = np.asarray([row["junction_magnitude"] for row in records], dtype=np.float64)
        branch_iso_values = np.asarray([row["branch_isotropy_ratio"] for row in records], dtype=np.float64)
        theta_values = np.asarray([row["theta_deg"] for row in records], dtype=np.float64)
        max_index = int(np.argmax(junction_values))
        min_index = int(np.argmin(junction_values))
        anisotropy_ratio = float(junction_values[max_index] / junction_values[min_index])
        branch_isotropy_mean = float(np.mean(branch_iso_values))

        csv_path = output_dir / f"sec07_junction_lcorner_{spec.slug}.csv"
        json_path = output_dir / f"sec07_junction_lcorner_{spec.slug}.json"
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
                "kappa_design_matrix": spec.kappa_design_matrix,
                "design_matrix_shape": [int(spec.design_matrix_shape[0]), int(spec.design_matrix_shape[1])],
                "junction_center_anisotropy_ratio": anisotropy_ratio,
                "theta_max_deg": float(theta_values[max_index]),
                "theta_min_deg": float(theta_values[min_index]),
                "branch_isotropy_ratio_mean": branch_isotropy_mean,
                "branch_sample_distances_px": list(BRANCH_SAMPLE_DISTANCES),
                "render_config": {
                    "contrast": float(contrast),
                    "edge_width_px": float(width_px),
                    "image_size": int(image_size),
                    "phase_count": int(phase_count),
                    "phase_step_px": float(phase_step_px),
                    "angle_step_deg": float(angle_step_deg),
                    "interior_angle_deg": 90.0,
                },
                "curve": records,
            },
        )
        overlay_payload["shapes"][spec.name] = {
            "label": spec.label,
            "junction_center_anisotropy_ratio": anisotropy_ratio,
            "branch_isotropy_ratio_mean": branch_isotropy_mean,
            "curve": records,
        }
        outputs[f"{spec.name}_csv"] = csv_path
        outputs[f"{spec.name}_json"] = json_path
        print(
            f"{spec.label}: junction_anisotropy={anisotropy_ratio:.6f}, "
            f"branch_iso_mean={branch_isotropy_mean:.6f}, "
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
    parser = argparse.ArgumentParser(description="Run the Section 7.3 L-corner junction first pass.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_junction_lcorner",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--overlay-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_junction_lcorner" / "sec07_junction_lcorner_overlay_disk_r15_square_h15_d3_normalized.json",
        help="Path for the combined overlay-plot JSON payload.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Square image size for the corner render.")
    parser.add_argument("--angle-step-deg", type=float, default=ANGLE_STEP_DEG, help="Orientation sampling step in degrees on [0, 360).")
    parser.add_argument("--phase-count", type=int, default=PHASE_COUNT, help="Number of sub-pixel phases to average per orientation.")
    parser.add_argument("--phase-step-px", type=float, default=PHASE_STEP_PX, help="Phase step size in pixels.")
    parser.add_argument("--edge-width-px", type=float, default=EDGE_WIDTH_PX, help="Width parameter for the tanh-smoothed corner edges.")
    parser.add_argument("--contrast", type=float, default=CONTRAST, help="Corner contrast amplitude.")
    parser.add_argument("--fft-workers", type=int, default=-1, help="Worker count for scipy.fft operations. Use -1 for all available workers.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ overlay plot after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    overlay_json = args.overlay_json.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_json.parent.mkdir(parents=True, exist_ok=True)

    outputs = run_experiment(
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
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_junction_lcorner_disk_r15_square_h15_d3_normalized.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_junction_lcorner_disk_r15_square_h15_d3_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["overlay_pdf"] = figure_pdf

    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
