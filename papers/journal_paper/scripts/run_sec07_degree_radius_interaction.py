#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from core.taylor import build_taylor_matrix, default_pinv_rcond
from wvf.radius import build_wvf_radius_kernels, disk_offsets
from wvf_metal.metal import fft_gradients_with_kernel

IMAGE_SIZE = 1024
PATCH_HALF_SIZE = 192
PATCH_SIZE = 2 * PATCH_HALF_SIZE + 1
BATCH_CASES = 64
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
RADII = (3, 5, 9, 15, 25, 50)
DEGREES = (1, 3, 5, 7, 9, 11, 15)
NORMALIZE_STATES = (False, True)
CURVATURE_RADII = (20, 50, 100, 200)
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
DEFAULT_FFT_BACKEND = "vkfft"
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


def _generate_cases(radius: int) -> dict[str, list[StimulusCase]]:
    xx, yy = _local_coords()
    orientation_values = _orientation_values(ORIENTATION_STEP_DEG)
    phase_values = _phase_values(PHASE_COUNT, PHASE_STEP_PX)
    step_cases = [
        _step_case(xx, yy, float(orientation_deg), float(phase_px), int(radius))
        for orientation_deg in orientation_values
        for phase_px in phase_values
    ]
    arc_cases = [
        _curved_case("arc", xx, yy, int(curvature_radius), float(orientation_deg), float(phase_px), int(radius))
        for curvature_radius in CURVATURE_RADII
        for orientation_deg in orientation_values
        for phase_px in phase_values
    ]
    s_curve_cases = [
        _curved_case("s_curve", xx, yy, int(curvature_radius), float(orientation_deg), float(phase_px), int(radius))
        for curvature_radius in CURVATURE_RADII
        for orientation_deg in orientation_values
        for phase_px in phase_values
    ]
    return {"step": step_cases, "arc": arc_cases, "s_curve": s_curve_cases}


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


def _apply_scipy(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(image, dtype=np.float64)
    gx = ndimage.correlate(img, np.asarray(kernel_x, dtype=np.float64), mode="reflect")
    gy = ndimage.correlate(img, np.asarray(kernel_y, dtype=np.float64), mode="reflect")
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _apply_cases_batched(
    cases: list[StimulusCase],
    radius: int,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
    use_scipy_fallback: bool,
    result: dict[str, object],
) -> list[tuple[np.ndarray, np.ndarray]]:
    canvas, placements = _tile_cases(cases)
    try:
        gx_canvas, gy_canvas = fft_gradients_with_kernel(
            canvas,
            radius=int(radius),
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            fft_backend=fft_backend,
            device_index=device_index,
        )
    except Exception as exc:
        if not use_scipy_fallback:
            raise
        result["application_method"] = "scipy_fallback"
        if result["application_error"] is None:
            result["application_error"] = f"{type(exc).__name__}: {exc}"
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
    grad_rmse = float(np.sqrt(np.mean((est_gx - true_gx) ** 2 + (est_gy - true_gy) ** 2)))
    ang_mae = _orientation_mae_deg(true_gx, true_gy, est_gx, est_gy)
    return {"grad_rmse": grad_rmse, "ang_mae_deg": ang_mae}


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean(np.asarray([float(row[key]) for row in rows], dtype=np.float64)))


def _cell_diagnostics(radius: int, degree: int, normalize_coords: bool) -> dict[str, object]:
    offsets = disk_offsets(int(radius), include_center=False)
    design = build_taylor_matrix(
        offsets,
        order=int(degree),
        normalize_radius=int(radius) if normalize_coords else None,
    )
    singular_values = np.linalg.svd(design, compute_uv=False, hermitian=False)
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    cutoff = float(default_pinv_rcond(design.shape, dtype=np.float64)) * sigma_max
    rank_deficient_count = int(np.count_nonzero(singular_values <= cutoff))
    kappa = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    kernels = build_wvf_radius_kernels(int(radius), order=int(degree), normalize_coords=bool(normalize_coords))
    kernel_x = np.asarray(kernels.kernel_x, dtype=np.float64)
    kernel_y = np.asarray(kernels.kernel_y, dtype=np.float64)
    white_noise_gain = float(np.sum(np.asarray(kernels.weights_x, dtype=np.float64) ** 2))
    return {
        "radius": int(radius),
        "degree": int(degree),
        "normalize_coords": bool(normalize_coords),
        "support_cardinality": int(offsets.shape[0]),
        "kappa_design_matrix": float(kappa),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "rank_deficient_count": int(rank_deficient_count),
        "white_noise_gain": float(white_noise_gain),
        "kernel_x": kernel_x,
        "kernel_y": kernel_y,
        "kernel_max": float(np.max(np.abs(kernel_x))),
    }


def _evaluate_cell(
    radius: int,
    degree: int,
    normalize_coords: bool,
    cases_by_radius: dict[int, dict[str, list[StimulusCase]]],
    fft_backend: str,
    device_index: int | None,
    use_scipy_fallback: bool,
) -> dict[str, object]:
    info = _cell_diagnostics(int(radius), int(degree), bool(normalize_coords))
    kernel_x = np.asarray(info["kernel_x"], dtype=np.float64)
    kernel_y = np.asarray(info["kernel_y"], dtype=np.float64)
    result: dict[str, object] = {
        "application_method": "native",
        "application_error": None,
        "step_metrics": [],
        "arc_metrics": [],
        "s_curve_metrics": [],
    }
    for stage_name in ("step", "arc", "s_curve"):
        cases = cases_by_radius[int(radius)][stage_name]
        dest = result[f"{stage_name}_metrics"]
        for batch_start in range(0, len(cases), int(BATCH_CASES)):
            batch = cases[batch_start : batch_start + int(BATCH_CASES)]
            outputs = _apply_cases_batched(
                batch,
                radius=int(radius),
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
                use_scipy_fallback=bool(use_scipy_fallback),
                result=result,
            )
            for case, (gx, gy) in zip(batch, outputs, strict=True):
                dest.append(_case_metrics(case, gx, gy))
        print(
            f"normalize={int(bool(normalize_coords))} r={int(radius)} d={int(degree)} stage={stage_name} "
            f"method={result['application_method']}"
        )
    return {
        **{key: value for key, value in info.items() if key not in {"kernel_x", "kernel_y"}},
        "application_method": str(result["application_method"]),
        "application_error": result["application_error"],
        "step_grad_rmse": _mean_metric(result["step_metrics"], "grad_rmse"),
        "step_ang_mae_deg": _mean_metric(result["step_metrics"], "ang_mae_deg"),
        "arc_grad_rmse": _mean_metric(result["arc_metrics"], "grad_rmse"),
        "arc_ang_mae_deg": _mean_metric(result["arc_metrics"], "ang_mae_deg"),
        "s_curve_grad_rmse": _mean_metric(result["s_curve_metrics"], "grad_rmse"),
        "s_curve_ang_mae_deg": _mean_metric(result["s_curve_metrics"], "ang_mae_deg"),
    }


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    radii: tuple[int, ...],
    degrees: tuple[int, ...],
    use_scipy_fallback: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    state_payloads = []
    for normalize_coords in NORMALIZE_STATES:
        cells = []
        recommendation = []
        for radius in radii:
            print(f"normalize={int(bool(normalize_coords))} r={int(radius)} phase=generate_cases status=begin")
            cases_for_radius = _generate_cases(int(radius))
            print(f"normalize={int(bool(normalize_coords))} r={int(radius)} phase=generate_cases status=end")
            radius_cells = []
            for degree in degrees:
                cell = _evaluate_cell(
                    radius=int(radius),
                    degree=int(degree),
                    normalize_coords=bool(normalize_coords),
                    cases_by_radius={int(radius): cases_for_radius},
                    fft_backend=fft_backend,
                    device_index=device_index,
                    use_scipy_fallback=bool(use_scipy_fallback),
                )
                cells.append(cell)
                radius_cells.append(cell)
                print(
                    f"normalize={int(bool(normalize_coords))} r={int(radius)} d={int(degree)} "
                    f"arc_rmse={cell['arc_grad_rmse']:.6e} step_rmse={cell['step_grad_rmse']:.6e} "
                    f"kappa={cell['kappa_design_matrix']:.6e} rank_def={int(cell['rank_deficient_count'])}"
                )
            zero_rank = [int(entry["degree"]) for entry in radius_cells if int(entry["rank_deficient_count"]) == 0]
            recommendation.append(
                {
                    "radius": int(radius),
                    "max_useful_degree": None if not zero_rank else int(max(zero_rank)),
                }
            )
        state_payloads.append(
            {
                "normalize_coords": bool(normalize_coords),
                "label": "normalized" if normalize_coords else "unnormalized",
                "cells": cells,
                "recommendation": recommendation,
            }
        )

    payload = {
        "title": "Section 7.5.3 degree-radius interaction grid",
        "subtitle": "Disk support, arcs/S-curves/steps, precomputed-kernel FFT application, clean local-patch evaluation",
        "config": {
            "image_size_px": int(IMAGE_SIZE),
            "patch_size_px": int(PATCH_SIZE),
            "apparatus_reduction": "Metrics are measured on centered local patches rather than full 1024^2 frames. The support radius never exceeds 50 px, the evaluation masks are confined to local tangent-normal neighborhoods, and each tiled FFT canvas fully contains every neighborhood that can influence the reported errors.",
            "radii": [int(value) for value in radii],
            "degrees": [int(value) for value in degrees],
            "curvature_radii_px": [int(value) for value in CURVATURE_RADII],
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "batch_cases": int(BATCH_CASES),
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "scipy_fallback_enabled": bool(use_scipy_fallback),
            "metrics": {
                "arc_grad_rmse": "Mean gradient-vector RMSE over circular arcs only.",
                "step_grad_rmse": "Mean gradient-vector RMSE over smoothed step edges.",
                "kappa_design_matrix": "Condition number sigma_max / sigma_min of the weighted design matrix.",
                "rank_deficient_count": "Number of singular values below the scaled-epsilon pseudoinverse cutoff.",
                "sigma_min": "Smallest singular value of the design matrix.",
                "white_noise_gain": "Sum of squared x-derivative weights.",
            },
            "recommendation_rule": "At each radius, max useful degree is the largest degree with rank_deficient_count == 0.",
        },
        "states": state_payloads,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return {"summary_json": summary_json}


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.5.3 interaction figure cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.5.3 degree-radius interaction grid.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_radius_interaction",
        help="Directory for interaction-grid outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_degree_radius_interaction" / "sec07_degree_radius_interaction_summary.json",
        help="Path for the combined interaction-grid summary JSON.",
    )
    parser.add_argument("--radii", type=str, default=None, help="Optional comma-separated radius subset.")
    parser.add_argument("--degrees", type=str, default=None, help="Optional comma-separated degree subset.")
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for the FFT backend.")
    parser.add_argument("--use-scipy-fallback", action="store_true", help="Allow SciPy correlation fallback if the native FFT path fails.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ heatmap overlay after writing JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radii = tuple(int(value) for value in (_parse_int_list(args.radii) or RADII))
    degrees = tuple(int(value) for value in (_parse_int_list(args.degrees) or DEGREES))
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        radii=radii,
        degrees=degrees,
        use_scipy_fallback=bool(args.use_scipy_fallback),
    )
    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_degree_radius_interaction_grid.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_degree_radius_interaction_grid.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["plot_pdf"] = figure_pdf
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
