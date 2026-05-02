#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import traceback
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

from core.taylor import build_taylor_matrix, default_pinv_rcond
from wvf.radius import build_wvf_radius_kernels, disk_offsets
from wvf_metal import gradients

RADII = (5, 9, 15, 25, 50)
CURVATURE_RADII = (20, 50, 100, 200)
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
PATCH_HALF_SIZE = 192
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1
BATCH_CASES = 64
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
NORMALIZE_COORDS = True
VARIANT = "fft"
DEFAULT_FFT_BACKEND = "auto"
NORMAL_BAND_HALF_PX = 6.0
TANGENTIAL_SPAN_FACTOR = 2.0


@dataclass(frozen=True)
class StimulusCase:
    stimulus_class: str
    curvature_radius: int | None
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


def _step_case(xx: np.ndarray, yy: np.ndarray, orientation_deg: float, phase_px: float, radius: int) -> StimulusCase:
    u, v = _rotate_to_local(xx, yy, float(orientation_deg))
    phi = np.asarray(u, dtype=np.float64) - float(phase_px)
    factor = _tanh_factor(phi)
    image = 0.5 * float(CONTRAST) * (1.0 + np.tanh(phi / float(EDGE_WIDTH_PX)))
    true_gx, true_gy = _global_gradients(np.ones_like(phi), np.zeros_like(phi), factor, float(orientation_deg))
    eval_mask = (
        (np.abs(phi) <= float(NORMAL_BAND_HALF_PX))
        & (np.abs(v) <= float(TANGENTIAL_SPAN_FACTOR) * float(radius))
    )
    return StimulusCase(
        stimulus_class="step",
        curvature_radius=None,
        orientation_deg=float(orientation_deg),
        phase_px=float(phase_px),
        image=np.asarray(image, dtype=np.float32),
        true_gx=np.asarray(true_gx, dtype=np.float64),
        true_gy=np.asarray(true_gy, dtype=np.float64),
        eval_mask=np.asarray(eval_mask, dtype=bool),
    )


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


def _curved_case(
    stimulus_class: str,
    xx: np.ndarray,
    yy: np.ndarray,
    curvature_radius: int,
    orientation_deg: float,
    phase_px: float,
    radius: int,
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


def _d_max_for_radius(radius: int) -> tuple[int, int, int]:
    support_cardinality = int(disk_offsets(int(radius), include_center=False).shape[0])
    degree = 0
    while (degree + 1) * (degree + 2) // 2 <= support_cardinality:
        degree += 1
    degree -= 1
    coefficient_count = int((degree + 1) * (degree + 2) // 2)
    return int(degree), int(support_cardinality), coefficient_count


def _kernel_diagnostics(radius: int, degree: int) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "radius": int(radius),
        "degree": int(degree),
        "status": "ok",
        "error": None,
    }
    try:
        offsets = disk_offsets(int(radius), include_center=False)
        design = build_taylor_matrix(
            offsets,
            order=int(degree),
            normalize_radius=int(radius) if NORMALIZE_COORDS else None,
        )
        singular_values = np.linalg.svd(design, compute_uv=False, hermitian=False)
        sigma_max = float(np.max(singular_values))
        sigma_min = float(np.min(singular_values))
        cutoff = float(default_pinv_rcond(design.shape, dtype=np.float64)) * sigma_max
        rank_deficient_count = int(np.count_nonzero(singular_values <= cutoff))
        kappa = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
        kernels = build_wvf_radius_kernels(int(radius), order=int(degree), normalize_coords=bool(NORMALIZE_COORDS))
        kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
        kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
        diagnostics.update(
            {
                "support_cardinality": int(offsets.shape[0]),
                "coefficient_count": int(design.shape[1]),
                "kappa_design_matrix": kappa,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rank_deficient_count": rank_deficient_count,
                "kernel_max": float(np.max(np.abs(kernel_x))),
                "kernel_min": float(np.min(kernel_x)),
                "kernel_has_nan": bool(np.isnan(kernel_x).any() or np.isnan(kernel_y).any()),
                "kernel_has_inf": bool(np.isinf(kernel_x).any() or np.isinf(kernel_y).any()),
                "kernel_x": kernel_x.tolist(),
                "kernel_y": kernel_y.tolist(),
            }
        )
    except Exception as exc:
        diagnostics["status"] = "construction_failed"
        diagnostics["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return diagnostics


def _orientation_mae_deg(true_gx: np.ndarray, true_gy: np.ndarray, est_gx: np.ndarray, est_gy: np.ndarray) -> float:
    true_angle = np.mod(np.arctan2(np.asarray(true_gy, dtype=np.float64), np.asarray(true_gx, dtype=np.float64)), np.pi)
    est_angle = np.mod(np.arctan2(np.asarray(est_gy, dtype=np.float64), np.asarray(est_gx, dtype=np.float64)), np.pi)
    diff = np.abs((est_angle - true_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(np.degrees(np.mean(diff)))


def _case_metrics(case: StimulusCase, gx: np.ndarray, gy: np.ndarray) -> dict[str, float]:
    mask = np.asarray(case.eval_mask, dtype=bool)
    true_gx = np.asarray(case.true_gx, dtype=np.float64)[mask]
    true_gy = np.asarray(case.true_gy, dtype=np.float64)[mask]
    est_gx = np.asarray(gx, dtype=np.float64)[mask]
    est_gy = np.asarray(gy, dtype=np.float64)[mask]
    true_mag = np.sqrt(true_gx**2 + true_gy**2)
    est_mag = np.sqrt(est_gx**2 + est_gy**2)
    grad_rmse = float(np.sqrt(np.mean((est_gx - true_gx) ** 2 + (est_gy - true_gy) ** 2)))
    mag_bias = float(np.mean(est_mag - true_mag))
    ang_mae = _orientation_mae_deg(true_gx, true_gy, est_gx, est_gy)
    return {
        "grad_rmse": grad_rmse,
        "mag_bias": mag_bias,
        "ang_mae": ang_mae,
    }


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def _apply_native(image: np.ndarray, radius: int, degree: int, fft_backend: str, device_index: int | None) -> tuple[np.ndarray, np.ndarray]:
    return gradients(
        image,
        radius=int(radius),
        degree=int(degree),
        normalize_coords=bool(NORMALIZE_COORDS),
        variant=VARIANT,
        fft_backend=fft_backend,
        device_index=device_index,
    )


def _apply_scipy(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(image, dtype=np.float64)
    gx = ndimage.correlate(img, np.asarray(kernel_x, dtype=np.float64), mode="reflect")
    gy = ndimage.correlate(img, np.asarray(kernel_y, dtype=np.float64), mode="reflect")
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _tile_cases(cases: list[StimulusCase]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    if not cases:
        raise ValueError("cannot tile an empty case batch")
    tile_h, tile_w = cases[0].image.shape
    cols = int(math.ceil(math.sqrt(len(cases))))
    rows = int(math.ceil(len(cases) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    placements: list[tuple[slice, slice]] = []
    for index, case in enumerate(cases):
        row = index // cols
        col = index % cols
        row_slice = slice(row * tile_h, (row + 1) * tile_h)
        col_slice = slice(col * tile_w, (col + 1) * tile_w)
        canvas[row_slice, col_slice] = np.asarray(case.image, dtype=np.float32)
        placements.append((row_slice, col_slice))
    return canvas, placements


def _apply_cases_batched(
    cases: list[StimulusCase],
    radius: int,
    degree: int,
    fft_backend: str,
    device_index: int | None,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    use_scipy_fallback: bool,
    result: dict[str, object],
) -> list[tuple[np.ndarray, np.ndarray]]:
    canvas, placements = _tile_cases(cases)
    try:
        gx_canvas, gy_canvas = _apply_native(canvas, int(radius), int(degree), fft_backend, device_index)
    except Exception as exc:
        if not use_scipy_fallback:
            raise
        result["application_method"] = "scipy_fallback"
        if result["application_error"] is None:
            result["application_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        gx_canvas, gy_canvas = _apply_scipy(canvas, kernel_x, kernel_y)
    outputs = []
    for row_slice, col_slice in placements:
        outputs.append(
            (
                np.asarray(gx_canvas[row_slice, col_slice], dtype=np.float64).copy(),
                np.asarray(gy_canvas[row_slice, col_slice], dtype=np.float64).copy(),
            )
        )
    return outputs


def _evaluate_radius(
    radius: int,
    degree: int,
    kernel_info: dict[str, object],
    fft_backend: str,
    device_index: int | None,
    use_scipy_fallback: bool,
    batch_cases: int,
) -> dict[str, object]:
    xx, yy = _local_coords()
    orientation_values = _orientation_values(ORIENTATION_STEP_DEG)
    phase_values = _phase_values(PHASE_COUNT, PHASE_STEP_PX)
    result: dict[str, object] = {
        "radius": int(radius),
        "degree": int(degree),
        "application_method": "native",
        "application_error": None,
        "step_metrics": [],
        "arc_metrics": [],
        "s_curve_metrics": [],
    }
    if kernel_info.get("status") != "ok":
        result["application_method"] = "not_run"
        result["application_error"] = "kernel construction failed"
        return result

    kernel_x = np.asarray(kernel_info["kernel_x"], dtype=np.float64)
    kernel_y = np.asarray(kernel_info["kernel_y"], dtype=np.float64)

    step_cases = [
        _step_case(xx, yy, float(orientation_deg), float(phase_px), int(radius))
        for orientation_deg in orientation_values
        for phase_px in phase_values
    ]
    for batch_start in range(0, len(step_cases), int(batch_cases)):
        batch = step_cases[batch_start : batch_start + int(batch_cases)]
        outputs = _apply_cases_batched(
            batch,
            radius=int(radius),
            degree=int(degree),
            fft_backend=fft_backend,
            device_index=device_index,
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            use_scipy_fallback=bool(use_scipy_fallback),
            result=result,
        )
        for case, (gx, gy) in zip(batch, outputs, strict=True):
            metrics = _case_metrics(case, gx, gy)
            metrics.update(
                {
                    "orientation_deg": float(case.orientation_deg),
                    "phase_px": float(case.phase_px),
                }
            )
            result["step_metrics"].append(metrics)
        print(
            f"r={int(radius)} stage=step batch_end={min(batch_start + int(batch_cases), len(step_cases))}/{len(step_cases)} "
            f"method={result['application_method']}"
        )

    for stimulus_class, dest in (("arc", "arc_metrics"), ("s_curve", "s_curve_metrics")):
        curved_cases = [
            _curved_case(
                stimulus_class=str(stimulus_class),
                xx=xx,
                yy=yy,
                curvature_radius=int(curvature_radius),
                orientation_deg=float(orientation_deg),
                phase_px=float(phase_px),
                radius=int(radius),
            )
            for curvature_radius in CURVATURE_RADII
            for orientation_deg in orientation_values
            for phase_px in phase_values
        ]
        for batch_start in range(0, len(curved_cases), int(batch_cases)):
            batch = curved_cases[batch_start : batch_start + int(batch_cases)]
            outputs = _apply_cases_batched(
                batch,
                radius=int(radius),
                degree=int(degree),
                fft_backend=fft_backend,
                device_index=device_index,
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                use_scipy_fallback=bool(use_scipy_fallback),
                result=result,
            )
            for case, (gx, gy) in zip(batch, outputs, strict=True):
                metrics = _case_metrics(case, gx, gy)
                metrics.update(
                    {
                        "curvature_radius": int(case.curvature_radius),
                        "orientation_deg": float(case.orientation_deg),
                        "phase_px": float(case.phase_px),
                    }
                )
                result[dest].append(metrics)
            print(
                f"r={int(radius)} stage={str(stimulus_class)} "
                f"batch_end={min(batch_start + int(batch_cases), len(curved_cases))}/{len(curved_cases)} "
                f"method={result['application_method']}"
            )
    return result


def run_experiment(
    output_dir: Path,
    summary_csv: Path,
    summary_json: Path,
    radii: tuple[int, ...],
    fft_backend: str,
    device_index: int | None,
    use_scipy_fallback: bool,
    batch_cases: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    summary_rows = []
    summary_records = []

    for radius in radii:
        degree, support_cardinality, coefficient_count = _d_max_for_radius(int(radius))
        print(
            f"r={int(radius)} start dmax={int(degree)} support={int(support_cardinality)} coeffs={int(coefficient_count)} "
            f"fft_backend={str(fft_backend)} batch_cases={int(batch_cases)}"
        )
        print(f"r={int(radius)} phase=kernel_diagnostics status=begin")
        kernel_info = _kernel_diagnostics(int(radius), int(degree))
        print(
            f"r={int(radius)} phase=kernel_diagnostics status={str(kernel_info.get('status'))} "
            f"kappa={kernel_info.get('kappa_design_matrix')} sigma_min={kernel_info.get('sigma_min')} "
            f"rank_deficient_count={kernel_info.get('rank_deficient_count')}"
        )
        print(f"r={int(radius)} phase=evaluate status=begin")
        eval_info = _evaluate_radius(
            radius=int(radius),
            degree=int(degree),
            kernel_info=kernel_info,
            fft_backend=fft_backend,
            device_index=device_index,
            use_scipy_fallback=bool(use_scipy_fallback),
            batch_cases=int(batch_cases),
        )
        print(f"r={int(radius)} phase=evaluate status=end method={str(eval_info['application_method'])}")

        step_rmse = None
        arc_rmse = None
        s_curve_rmse = None
        if eval_info["step_metrics"]:
            step_rmse = _mean_metric(eval_info["step_metrics"], "grad_rmse")
        if eval_info["arc_metrics"]:
            arc_rmse = _mean_metric(eval_info["arc_metrics"], "grad_rmse")
        if eval_info["s_curve_metrics"]:
            s_curve_rmse = _mean_metric(eval_info["s_curve_metrics"], "grad_rmse")

        radius_record = {
            "radius": int(radius),
            "degree_max": int(degree),
            "support_cardinality": int(support_cardinality),
            "coefficient_count": int(coefficient_count),
            "kernel_diagnostics": kernel_info,
            "evaluation": {
                "application_method": str(eval_info["application_method"]),
                "application_error": eval_info["application_error"],
                "step_metrics": eval_info["step_metrics"],
                "arc_metrics": eval_info["arc_metrics"],
                "s_curve_metrics": eval_info["s_curve_metrics"],
                "step_grad_rmse_mean": step_rmse,
                "arc_grad_rmse_mean": arc_rmse,
                "s_curve_grad_rmse_mean": s_curve_rmse,
            },
        }
        json_path = output_dir / f"sec07_degree_push_to_max_r{int(radius)}_normalized.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(radius_record, handle, indent=2)
            handle.write("\n")
        outputs[f"json_r{int(radius)}"] = json_path

        summary_row = {
            "radius": int(radius),
            "degree_max": int(degree),
            "support_cardinality": int(support_cardinality),
            "coefficient_count": int(coefficient_count),
            "kappa": None if kernel_info.get("status") != "ok" else float(kernel_info["kappa_design_matrix"]),
            "sigma_min": None if kernel_info.get("status") != "ok" else float(kernel_info["sigma_min"]),
            "rank_deficient_count": None if kernel_info.get("status") != "ok" else int(kernel_info["rank_deficient_count"]),
            "kernel_max": None if kernel_info.get("status") != "ok" else float(kernel_info["kernel_max"]),
            "kernel_has_nan": bool(kernel_info.get("kernel_has_nan", False)),
            "status": str(kernel_info.get("status", "unknown")),
            "application_method": str(eval_info["application_method"]),
            "step_grad_rmse": step_rmse,
            "arc_grad_rmse": arc_rmse,
            "s_curve_grad_rmse": s_curve_rmse,
        }
        summary_rows.append(summary_row)
        summary_records.append(
            {
                **summary_row,
                "kernel_x": kernel_info.get("kernel_x"),
                "kernel_shape": None if kernel_info.get("status") != "ok" else [len(kernel_info["kernel_x"]), len(kernel_info["kernel_x"][0])],
            }
        )
        print(
            f"r={int(radius)} dmax={int(degree)} support={int(support_cardinality)} "
            f"kappa={summary_row['kappa']} sigma_min={summary_row['sigma_min']} "
            f"step_rmse={summary_row['step_grad_rmse']} arc_rmse={summary_row['arc_grad_rmse']} "
            f"method={summary_row['application_method']}"
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "radius",
                "degree_max",
                "support_cardinality",
                "coefficient_count",
                "kappa",
                "sigma_min",
                "rank_deficient_count",
                "kernel_max",
                "kernel_has_nan",
                "status",
                "application_method",
                "step_grad_rmse",
                "arc_grad_rmse",
                "s_curve_grad_rmse",
            ),
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "radius": f"{int(row['radius'])}",
                    "degree_max": f"{int(row['degree_max'])}",
                    "support_cardinality": f"{int(row['support_cardinality'])}",
                    "coefficient_count": f"{int(row['coefficient_count'])}",
                    "kappa": "" if row["kappa"] is None else f"{float(row['kappa']):.17e}",
                    "sigma_min": "" if row["sigma_min"] is None else f"{float(row['sigma_min']):.17e}",
                    "rank_deficient_count": "" if row["rank_deficient_count"] is None else f"{int(row['rank_deficient_count'])}",
                    "kernel_max": "" if row["kernel_max"] is None else f"{float(row['kernel_max']):.17e}",
                    "kernel_has_nan": f"{bool(row['kernel_has_nan'])}",
                    "status": str(row["status"]),
                    "application_method": str(row["application_method"]),
                    "step_grad_rmse": "" if row["step_grad_rmse"] is None else f"{float(row['step_grad_rmse']):.17e}",
                    "arc_grad_rmse": "" if row["arc_grad_rmse"] is None else f"{float(row['arc_grad_rmse']):.17e}",
                    "s_curve_grad_rmse": "" if row["s_curve_grad_rmse"] is None else f"{float(row['s_curve_grad_rmse']):.17e}",
                }
            )
    outputs["summary_csv"] = summary_csv

    payload = {
        "title": "Section 7.5 push-to-maximum diagnostic",
        "subtitle": "Disk support, normalize_coords = True, d = d_max(r), clean step and curved-edge stimuli",
        "config": {
            "radii": [int(value) for value in radii],
            "curvature_radii_px": [int(value) for value in CURVATURE_RADII],
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "variant": VARIANT,
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "patch_size_px": int(PATCH_SIZE),
            "apparatus_reduction": "Clean diagnostic metrics are evaluated on centered local patches rather than full 1024^2 frames. The largest support radius is 50 px, the evaluation masks are confined to a local tangent-normal neighborhood around each feature, and the patch fully contains every neighborhood that can influence the reported RMSE values.",
            "d_max_definition": "Largest degree d with M(d) = (d + 1)(d + 2) / 2 <= |S_r|.",
            "scipy_fallback_enabled": bool(use_scipy_fallback),
            "batch_cases": int(batch_cases),
        },
        "records": summary_records,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    outputs["summary_json"] = summary_json
    return outputs


def compile_heatmaps(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the push-to-max heatmaps cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.5 push-to-maximum diagnostic.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_push_to_max",
        help="Directory for per-radius JSON outputs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_push_to_max" / "sec07_degree_push_to_max_summary.csv",
        help="Path for the summary table CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_push_to_max" / "sec07_degree_push_to_max_summary.json",
        help="Path for the combined push-to-max summary JSON.",
    )
    parser.add_argument("--radii", type=str, default=None, help="Optional comma-separated radius subset.")
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use when applying the operator.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for the FFT backend.")
    parser.add_argument("--no-scipy-fallback", action="store_true", help="Disable SciPy reflect-convolution fallback if the native application path fails.")
    parser.add_argument("--batch-cases", type=int, default=BATCH_CASES, help="Number of synthetic patches to tile into each FFT application call.")
    parser.add_argument("--compile-heatmaps", action="store_true", help="Compile the checked-in Typst/CeTZ heatmap figure after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_int_list(args.radii) or RADII))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_csv=args.summary_csv.resolve(),
        summary_json=args.summary_json.resolve(),
        radii=radii,
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        use_scipy_fallback=not bool(args.no_scipy_fallback),
        batch_cases=int(args.batch_cases),
    )
    if args.compile_heatmaps:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_degree_push_to_max_heatmaps.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_degree_push_to_max_heatmaps.pdf"
        compile_heatmaps(figure_src, figure_pdf)
        outputs["heatmaps_pdf"] = figure_pdf
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
