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

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.taylor import build_taylor_matrix, default_pinv_rcond
from wvf.radius import disk_offsets
from wvf_metal import gradients

IMAGE_SIZE = 1024
PATCH_HALF_SIZE = 128
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
RADII = (5, 9, 15, 25)
DEGREES = (1, 3, 5, 7, 9)
CURVATURE_RADII = (20, 50, 100, 200)
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
NORMALIZE_COORDS = True
VARIANT = "fft"
DEFAULT_FFT_BACKEND = "auto"
NORMAL_BAND_HALF_PX = 6.0
TANGENTIAL_SPAN_FACTOR = 2.0


@dataclass(frozen=True)
class RadiusDegreeInfo:
    radius: int
    degree: int
    kappa_design: float
    rank_deficient_count: int
    support_cardinality: int


@dataclass(frozen=True)
class StimulusCase:
    stimulus_class: str
    curvature_radius: int
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    true_gx: np.ndarray
    true_gy: np.ndarray
    eval_mask: np.ndarray


def _parse_int_list(text: str | None) -> tuple[int, ...]:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(int(item))
    return tuple(values)


def _orientation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(180.0 / float(step_deg)))
    return tuple(float(step_deg) * i for i in range(count))


def _phase_values(count: int, step_px: float) -> tuple[float, ...]:
    return tuple(float(step_px) * i for i in range(int(count)))


def _local_coords() -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _rotate_to_local(xx: np.ndarray, yy: np.ndarray, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(theta_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    u = xx * cos_t + yy * sin_t
    v = -xx * sin_t + yy * cos_t
    return u, v


def _tanh_factor(phi: np.ndarray) -> np.ndarray:
    normalized = np.tanh(np.asarray(phi, dtype=np.float64) / float(EDGE_WIDTH_PX))
    return 0.5 * float(CONTRAST) / float(EDGE_WIDTH_PX) * (1.0 - normalized * normalized)


def _arc_level_set(u: np.ndarray, v: np.ndarray, rho: float, phase_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(u, dtype=np.float64) - float(phase_px)
    denom = np.sqrt((up - float(rho)) ** 2 + np.asarray(v, dtype=np.float64) ** 2)
    phi = float(rho) - denom
    dphi_du = (float(rho) - up) / np.maximum(denom, 1.0e-12)
    dphi_dv = -np.asarray(v, dtype=np.float64) / np.maximum(denom, 1.0e-12)
    return phi, dphi_du, dphi_dv


def _s_curve_level_set(u: np.ndarray, v: np.ndarray, rho: float, phase_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(u, dtype=np.float64) - float(phase_px)
    vv = np.asarray(v, dtype=np.float64)
    phi = up - (vv**3) / (3.0 * float(rho) ** 2)
    dphi_du = np.ones_like(phi, dtype=np.float64)
    dphi_dv = -(vv**2) / (float(rho) ** 2)
    return phi, dphi_du, dphi_dv


def _global_gradients(
    dphi_du: np.ndarray,
    dphi_dv: np.ndarray,
    factor: np.ndarray,
    theta_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(theta_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    gx = factor * (np.asarray(dphi_du, dtype=np.float64) * cos_t - np.asarray(dphi_dv, dtype=np.float64) * sin_t)
    gy = factor * (np.asarray(dphi_du, dtype=np.float64) * sin_t + np.asarray(dphi_dv, dtype=np.float64) * cos_t)
    return gx, gy


def _render_case(
    stimulus_class: str,
    curvature_radius: int,
    orientation_deg: float,
    phase_px: float,
    radius: int,
    xx: np.ndarray,
    yy: np.ndarray,
) -> StimulusCase:
    u, v = _rotate_to_local(xx, yy, float(orientation_deg))
    if stimulus_class == "arc":
        phi, dphi_du, dphi_dv = _arc_level_set(u, v, float(curvature_radius), float(phase_px))
    elif stimulus_class == "s_curve":
        phi, dphi_du, dphi_dv = _s_curve_level_set(u, v, float(curvature_radius), float(phase_px))
    else:
        raise ValueError(f"unsupported stimulus class {stimulus_class!r}")

    factor = _tanh_factor(phi)
    image = 0.5 * float(CONTRAST) * (1.0 + np.tanh(phi / float(EDGE_WIDTH_PX)))
    true_gx, true_gy = _global_gradients(dphi_du, dphi_dv, factor, float(orientation_deg))
    eval_mask = (
        (np.abs(phi) <= float(NORMAL_BAND_HALF_PX))
        & (np.abs(v) <= float(TANGENTIAL_SPAN_FACTOR) * float(radius))
    )
    return StimulusCase(
        stimulus_class=str(stimulus_class),
        curvature_radius=int(curvature_radius),
        orientation_deg=float(orientation_deg),
        phase_px=float(phase_px),
        image=np.asarray(image, dtype=np.float32),
        true_gx=np.asarray(true_gx, dtype=np.float64),
        true_gy=np.asarray(true_gy, dtype=np.float64),
        eval_mask=np.asarray(eval_mask, dtype=bool),
    )


def _radius_degree_info(radius: int, degree: int) -> RadiusDegreeInfo:
    offsets = disk_offsets(int(radius), include_center=False)
    design = build_taylor_matrix(
        offsets,
        order=int(degree),
        normalize_radius=int(radius) if NORMALIZE_COORDS else None,
    )
    singular_values = np.linalg.svd(design, compute_uv=False, hermitian=False)
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    kappa = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    cutoff = float(default_pinv_rcond(design.shape, dtype=np.float64)) * sigma_max
    rank_deficient_count = int(np.count_nonzero(singular_values <= cutoff))
    return RadiusDegreeInfo(
        radius=int(radius),
        degree=int(degree),
        kappa_design=kappa,
        rank_deficient_count=rank_deficient_count,
        support_cardinality=int(offsets.shape[0]),
    )


def _orientation_mae_deg(true_gx: np.ndarray, true_gy: np.ndarray, est_gx: np.ndarray, est_gy: np.ndarray) -> float:
    true_angle = np.mod(np.arctan2(np.asarray(true_gy, dtype=np.float64), np.asarray(true_gx, dtype=np.float64)), np.pi)
    est_angle = np.mod(np.arctan2(np.asarray(est_gy, dtype=np.float64), np.asarray(est_gx, dtype=np.float64)), np.pi)
    diff = np.abs((est_angle - true_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(np.degrees(np.mean(diff)))


def _case_metrics(case: StimulusCase, est_gx: np.ndarray, est_gy: np.ndarray) -> dict[str, float]:
    mask = np.asarray(case.eval_mask, dtype=bool)
    true_gx = np.asarray(case.true_gx, dtype=np.float64)[mask]
    true_gy = np.asarray(case.true_gy, dtype=np.float64)[mask]
    gx = np.asarray(est_gx, dtype=np.float64)[mask]
    gy = np.asarray(est_gy, dtype=np.float64)[mask]
    true_mag = np.sqrt(true_gx**2 + true_gy**2)
    est_mag = np.sqrt(gx**2 + gy**2)
    grad_rmse = float(np.sqrt(np.mean((gx - true_gx) ** 2 + (gy - true_gy) ** 2)))
    ang_mae = _orientation_mae_deg(true_gx, true_gy, gx, gy)
    mag_bias = float(np.mean(est_mag - true_mag))
    return {
        "grad_rmse": grad_rmse,
        "ang_mae_deg": ang_mae,
        "mag_bias": mag_bias,
    }


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    radii: tuple[int, ...],
    degrees: tuple[int, ...],
    curvature_radii: tuple[int, ...],
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    xx, yy = _local_coords()
    orientation_values = _orientation_values(ORIENTATION_STEP_DEG)
    phase_values = _phase_values(PHASE_COUNT, PHASE_STEP_PX)

    outputs: dict[str, Path] = {}
    per_radius_summary: dict[str, dict[str, object]] = {}
    arc_plot_series = []

    for radius in radii:
        radius_rows = []
        cell_records = []
        arc_curve_points = []
        for degree in degrees:
            rd_info = _radius_degree_info(int(radius), int(degree))
            degree_records = []
            for curvature_radius in curvature_radii:
                combined_case_metrics = []
                arc_case_metrics = []
                s_case_metrics = []
                for orientation_deg in orientation_values:
                    for phase_px in phase_values:
                        for stimulus_class in ("arc", "s_curve"):
                            case = _render_case(
                                stimulus_class=str(stimulus_class),
                                curvature_radius=int(curvature_radius),
                                orientation_deg=float(orientation_deg),
                                phase_px=float(phase_px),
                                radius=int(radius),
                                xx=xx,
                                yy=yy,
                            )
                            gx, gy = gradients(
                                case.image,
                                radius=int(radius),
                                degree=int(degree),
                                normalize_coords=bool(NORMALIZE_COORDS),
                                variant=VARIANT,
                                fft_backend=fft_backend,
                                device_index=device_index,
                            )
                            metrics = _case_metrics(case, gx, gy)
                            combined_case_metrics.append(metrics)
                            if stimulus_class == "arc":
                                arc_case_metrics.append(metrics)
                            else:
                                s_case_metrics.append(metrics)

                row = {
                    "degree": int(degree),
                    "curvature_radius": int(curvature_radius),
                    "grad_rmse": _mean_metric(combined_case_metrics, "grad_rmse"),
                    "ang_mae": _mean_metric(combined_case_metrics, "ang_mae_deg"),
                    "mag_bias": _mean_metric(combined_case_metrics, "mag_bias"),
                    "kappa": float(rd_info.kappa_design),
                    "rank_deficient_count": int(rd_info.rank_deficient_count),
                }
                radius_rows.append(row)
                record = {
                    "degree": int(degree),
                    "curvature_radius": int(curvature_radius),
                    "combined_metrics": row,
                    "arc_metrics": {
                        "grad_rmse": _mean_metric(arc_case_metrics, "grad_rmse"),
                        "ang_mae": _mean_metric(arc_case_metrics, "ang_mae_deg"),
                        "mag_bias": _mean_metric(arc_case_metrics, "mag_bias"),
                    },
                    "s_curve_metrics": {
                        "grad_rmse": _mean_metric(s_case_metrics, "grad_rmse"),
                        "ang_mae": _mean_metric(s_case_metrics, "ang_mae_deg"),
                        "mag_bias": _mean_metric(s_case_metrics, "mag_bias"),
                    },
                    "kappa_design_matrix": float(rd_info.kappa_design),
                    "rank_deficient_count": int(rd_info.rank_deficient_count),
                    "support_cardinality": int(rd_info.support_cardinality),
                }
                degree_records.append(record)
                print(
                    f"r={int(radius)} d={int(degree)} rho={int(curvature_radius)} "
                    f"grad_rmse={row['grad_rmse']:.6e} arc_rmse={record['arc_metrics']['grad_rmse']:.6e}"
                )

            csv_path = output_dir / f"sec07_degree_sweep_curvature_r{int(radius)}_d1_d9_normalized.csv"
            arc_curve_points.append(
                {
                    "degree": int(degree),
                    "grad_rmse": float(
                        np.mean(
                            np.asarray(
                                [float(entry["arc_metrics"]["grad_rmse"]) for entry in degree_records],
                                dtype=np.float64,
                            )
                        )
                    ),
                }
            )
            cell_records.extend(degree_records)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("degree", "curvature_radius", "grad_rmse", "ang_mae", "mag_bias", "kappa", "rank_deficient_count"),
            )
            writer.writeheader()
            for row in radius_rows:
                writer.writerow(
                    {
                        "degree": f"{int(row['degree'])}",
                        "curvature_radius": f"{int(row['curvature_radius'])}",
                        "grad_rmse": f"{float(row['grad_rmse']):.17e}",
                        "ang_mae": f"{float(row['ang_mae']):.17e}",
                        "mag_bias": f"{float(row['mag_bias']):.17e}",
                        "kappa": f"{float(row['kappa']):.17e}",
                        "rank_deficient_count": f"{int(row['rank_deficient_count'])}",
                    }
                )
        outputs[f"csv_r{int(radius)}"] = csv_path
        per_radius_summary[str(int(radius))] = {
            "radius": int(radius),
            "rows": radius_rows,
            "records": cell_records,
            "csv_path": str(csv_path),
        }
        arc_plot_series.append(
            {
                "radius": int(radius),
                "label": f"r = {int(radius)}",
                "points": arc_curve_points,
            }
        )

    payload = {
        "title": "Section 7.5 polynomial degree first pass",
        "subtitle": "Disk support, arcs and S-curves, normalize_coords = True, clean local-patch evaluation",
        "config": {
            "image_size_px": int(IMAGE_SIZE),
            "patch_size_px": int(PATCH_SIZE),
            "apparatus_reduction": "Metrics are measured on centered local patches rather than full 1024^2 frames. The features are isolated, the largest support radius is 25 px, and the evaluation mask is confined to a local tangent-normal neighborhood around the curved edge, so the reduced patch fully contains every neighborhood that can influence the reported errors.",
            "radii": [int(value) for value in radii],
            "degrees": [int(value) for value in degrees],
            "curvature_radii_px": [int(value) for value in curvature_radii],
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "variant": VARIANT,
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "stimuli": {
                "arc": "Circular-arc level set rendered as 0.5 * (1 + tanh(phi / w)) with exact analytical gradient.",
                "s_curve": "Odd cubic inflection curve rendered as 0.5 * (1 + tanh(phi / w)) with exact analytical gradient.",
            },
            "evaluation_mask_definition": "|phi| <= 6 px and |v| <= 2r in the local tangent-normal coordinates.",
            "csv_metric_average": "CSV rows average metrics over orientations, sub-pixel phases, and both stimulus classes.",
        },
        "per_radius": per_radius_summary,
        "arc_grad_rmse_plot": arc_plot_series,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    outputs["summary_json"] = summary_json
    return outputs


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.5 degree-sweep figure cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.5 degree sweep on smoothed arcs and S-curves.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_sweep_curvature",
        help="Directory for per-radius CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_sweep_curvature" / "sec07_degree_sweep_curvature_summary_normalized.json",
        help="Path for the combined degree-sweep summary JSON.",
    )
    parser.add_argument("--radii", type=str, default=None, help="Optional comma-separated radius subset.")
    parser.add_argument("--degrees", type=str, default=None, help="Optional comma-separated degree subset.")
    parser.add_argument("--curvature-radii", type=str, default=None, help="Optional comma-separated curvature-radius subset.")
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for the FFT backend.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ headline plot after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_int_list(args.radii) or RADII))
    degrees = tuple(int(value) for value in (_parse_int_list(args.degrees) or DEGREES))
    curvature_radii = tuple(int(value) for value in (_parse_int_list(args.curvature_radii) or CURVATURE_RADII))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        radii=radii,
        degrees=degrees,
        curvature_radii=curvature_radii,
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )
    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_degree_sweep_curvature_normalized.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_degree_sweep_curvature_normalized.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["plot_pdf"] = figure_pdf
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
