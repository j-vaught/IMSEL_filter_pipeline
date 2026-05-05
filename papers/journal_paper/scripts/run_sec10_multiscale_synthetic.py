#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import KernelSpec, build_wvf
from core.taylor import build_taylor_matrix, compute_pseudoinverse, rotate_coordinates
from sec09_wvf_grid import wvf_conditioning_diagnostics
from section8_common import (
    CurvedCase,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    STEP_TANGENTIAL_SPAN_FACTOR,
    StepCase,
    add_awgn,
    apply_cases_batched,
    case_gradient_metrics,
    compile_plot,
    generate_curved_cases,
    generate_step_cases,
    mean,
    orientation_values,
    peak_metrics_from_profile,
    phase_values,
    sample_profile,
)
from wvf.radius import disk_offsets


TITLE = "Section 10 multi-scale synthetic validation"
SUBTITLE = "Bias-aware follow-up on the edge-width transfer bank with a five-scale active stack"
ACTIVE_STACK = (
    {"radius": 3, "degree": 5},
    {"radius": 5, "degree": 9},
    {"radius": 9, "degree": 11},
    {"radius": 15, "degree": 11},
    {"radius": 25, "degree": 11},
    {"radius": 50, "degree": 11},
)
SINGLE_SCALE_TRACE = ACTIVE_STACK
EDGE_WIDTHS_PX = (1.0, 3.0, 9.0, 27.0)
BEST_SINGLE_RADIUS_BY_WIDTH = {
    1.0: 15,
    3.0: 15,
    9.0: 25,
    27.0: 50,
}
EDGE_SIGMA_FACTOR = 0.25
SNR_DB = 10.0
NOISE_DRAWS = 100
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
NORMALIZE_COORDS = True
DEFAULT_FFT_BACKEND = "vkfft"
DEFAULT_BATCH_CASES = 256
EVAL_SUPPORT_SCALE = 50.0
PATCH_HALF_SIZE = 204
STEERABILITY_ANGLES_DEG = tuple(range(0, 180, 5))
EPS64 = float(np.finfo(np.float64).eps)
PASS_MULTIPLIER = 100.0
ABSOLUTE_THRESHOLD_FLOOR = 1.0e-12
NOISE_SEED_BASE = 10120
COMPOSITE_LABEL = "multi_scale_composite"
COMPOSITE_WIDTHS = EDGE_WIDTHS_PX
COMPOSITE_TILE_ORDER = EDGE_WIDTHS_PX
COMPOSITE_DIAGNOSTIC_MAX_WIDTH_PX = 400


@dataclass(frozen=True)
class GradientFieldCase:
    stimulus_name: str
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    true_gx: np.ndarray
    true_gy: np.ndarray
    eval_mask: np.ndarray


@dataclass(frozen=True)
class StimulusSpec:
    key: str
    label: str
    step_cases: tuple[StepCase | GradientFieldCase, ...]
    arc_cases: tuple[CurvedCase | GradientFieldCase, ...]
    edge_width_px: float | None
    curvature_radius_px: float | None
    component_widths_px: tuple[float, ...] | None = None
    component_curvature_radii_px: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ScaleRecord:
    radius: int
    degree: int
    kernel: KernelSpec
    diagnostics: dict[str, object]
    intrinsic_fwhm: float

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.radius), int(self.degree))


def _edge_sigma_px(edge_width_px: float) -> float:
    return float(edge_width_px) * float(EDGE_SIGMA_FACTOR)


def _strategy_label(key: str) -> str:
    if key == "l2_variance_inverse":
        return "L2 variance-inverse"
    if key == "l2_equal":
        return "L2 equal"
    if key == "l2_fwhm":
        return "L2 FWHM-inverse"
    if key == "l3_max":
        return "L3 max-across-scales"
    return key


def _stimulus_label(edge_width_px: float | None, composite: bool = False) -> str:
    if composite:
        return "Composite"
    if edge_width_px is None:
        return "Unknown"
    return f"w={int(round(float(edge_width_px)))} px"


def _parse_key_list(text: str | None) -> tuple[str, ...]:
    if text is None:
        return tuple()
    keys: list[str] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if item:
            keys.append(str(item))
    return tuple(keys)


def _resize_image_if_needed(image: Image.Image, max_width_px: int | None) -> Image.Image:
    if max_width_px is None or image.width <= int(max_width_px):
        return image
    new_width = int(max_width_px)
    new_height = max(1, int(round(image.height * (new_width / float(image.width)))))
    return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)


def _orientation_error_deg_map(
    true_gx: np.ndarray,
    true_gy: np.ndarray,
    est_gx: np.ndarray,
    est_gy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    true_angle = np.mod(np.arctan2(np.asarray(true_gy, dtype=np.float64), np.asarray(true_gx, dtype=np.float64)), np.pi)
    est_angle = np.mod(np.arctan2(np.asarray(est_gy, dtype=np.float64), np.asarray(est_gx, dtype=np.float64)), np.pi)
    diff = np.abs((est_angle - true_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    diff_deg = np.degrees(diff)
    masked = np.asarray(diff_deg, dtype=np.float64)
    masked[~np.asarray(mask, dtype=bool)] = np.nan
    return masked


def _error_map_to_rgb(
    error_deg: np.ndarray,
    clip_deg: float = 45.0,
) -> np.ndarray:
    errors = np.asarray(error_deg, dtype=np.float64)
    valid = np.isfinite(errors)
    t = np.zeros_like(errors, dtype=np.float64)
    t[valid] = np.clip(errors[valid] / float(clip_deg), 0.0, 1.0)
    rgb = np.full(errors.shape + (3,), 255, dtype=np.uint8)
    garnet = np.asarray((115, 0, 10), dtype=np.float64)
    white = np.asarray((255, 255, 255), dtype=np.float64)
    blended = (1.0 - t[..., None]) * white + t[..., None] * garnet
    rgb[valid] = np.clip(np.round(blended[valid]), 0.0, 255.0).astype(np.uint8)
    rgb[~valid] = np.asarray((236, 236, 236), dtype=np.uint8)
    return rgb


def _write_composite_orientation_diagnostic(
    path: Path,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_label: str,
    right_label: str,
    max_width_px: int,
) -> None:
    gap_px = 12
    title_band = 28
    left_image = Image.fromarray(np.asarray(left_rgb, dtype=np.uint8), mode="RGB")
    right_image = Image.fromarray(np.asarray(right_rgb, dtype=np.uint8), mode="RGB")
    canvas = Image.new(
        "RGB",
        (left_image.width + right_image.width + gap_px, max(left_image.height, right_image.height) + title_band),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), str(left_label), fill=(54, 54, 54))
    draw.text((left_image.width + gap_px + 8, 6), str(right_label), fill=(54, 54, 54))
    canvas.paste(left_image, (0, title_band))
    canvas.paste(right_image, (left_image.width + gap_px, title_band))
    path.parent.mkdir(parents=True, exist_ok=True)
    _resize_image_if_needed(canvas, int(max_width_px)).save(path)


def _clone_case(
    case: StepCase | CurvedCase | GradientFieldCase,
    image: np.ndarray,
) -> StepCase | CurvedCase | GradientFieldCase:
    return replace(case, image=np.asarray(image, dtype=np.float32))


def _method_slug(radius: int, degree: int) -> str:
    return f"r{int(radius)}_d{int(degree)}"


def _stable_key_seed(text: str) -> int:
    total = 0
    for index, char in enumerate(str(text)):
        total += (index + 1) * ord(char)
    return int(total)


def _variance_prefactor(degree: int) -> float:
    degree_f = float(degree)
    return ((degree_f + 1.0) ** 2 * (degree_f + 3.0) ** 2) / (16.0 * math.pi)


def _threshold_for(kernel_max: float) -> float:
    return max(ABSOLUTE_THRESHOLD_FLOOR, PASS_MULTIPLIER * EPS64 * float(kernel_max))


def _direct_rotated_weights(
    radius: int,
    degree: int,
    theta_deg: float,
    normalize_coords: bool,
) -> np.ndarray:
    offsets_xy = disk_offsets(int(radius), include_center=False)
    theta_rad = math.radians(float(theta_deg))
    rotated_xy = rotate_coordinates(offsets_xy, theta_rad)
    design = build_taylor_matrix(
        rotated_xy,
        order=int(degree),
        normalize_radius=float(radius) if normalize_coords else None,
    )
    pinv = compute_pseudoinverse(design)
    derivative_scale = 1.0 / float(radius) if normalize_coords else 1.0
    return np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)


def _pad_kernel(kernel: np.ndarray, target_radius: int) -> np.ndarray:
    target_size = 2 * int(target_radius) + 1
    source = np.asarray(kernel, dtype=np.float64)
    canvas = np.zeros((target_size, target_size), dtype=np.float64)
    row0 = (target_size - source.shape[0]) // 2
    col0 = (target_size - source.shape[1]) // 2
    canvas[row0 : row0 + source.shape[0], col0 : col0 + source.shape[1]] = source
    return canvas


def _build_linear_combined_kernel(
    strategy_key: str,
    weights: list[dict[str, float]],
    scale_records: list[ScaleRecord],
) -> KernelSpec:
    max_radius = max(record.radius for record in scale_records)
    target_size = 2 * int(max_radius) + 1
    kernel_x = np.zeros((target_size, target_size), dtype=np.float64)
    kernel_y = np.zeros((target_size, target_size), dtype=np.float64)
    for record, weight_info in zip(scale_records, weights, strict=True):
        alpha = float(weight_info["weight"])
        kernel_x += alpha * _pad_kernel(record.kernel.kernel_x, max_radius)
        kernel_y += alpha * _pad_kernel(record.kernel.kernel_y, max_radius)
    return KernelSpec(
        method=strategy_key,
        label=_strategy_label(strategy_key),
        config={"weights": strategy_key},
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        support_half_extent=int(max_radius),
        white_noise_gain=float(np.sum(kernel_x**2)),
    )


def _evaluate_intrinsic_fwhm(
    kernel: KernelSpec,
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
    fft_backend: str,
    device_index: int | None,
    batch_cases: int,
) -> float:
    cases = generate_step_cases(
        support_scale=float(EVAL_SUPPORT_SCALE),
        orientations_deg=orientations_deg,
        phases_px=phases_px,
        half_size=int(PATCH_HALF_SIZE),
        width_px=float(EDGE_WIDTH_PX),
    )
    responses = apply_cases_batched(
        list(cases),
        kernel,
        fft_backend,
        device_index,
        batch_cases=int(batch_cases),
    )
    fwhm_values: list[float] = []
    for case, (gx, gy) in zip(cases, responses, strict=True):
        theta = math.radians(float(case.orientation_deg))
        directional = np.asarray(gx, dtype=np.float64) * math.cos(theta) + np.asarray(gy, dtype=np.float64) * math.sin(theta)
        profile = sample_profile(directional, case.line_xs, case.line_ys)
        _, _, fwhm = peak_metrics_from_profile(
            profile=profile,
            x_coords=case.line_coords,
            phase_px=float(case.phase_px),
            support_scale=float(kernel.support_half_extent),
            width_px=float(EDGE_WIDTH_PX),
        )
        fwhm_values.append(float(fwhm))
    return float(mean(fwhm_values))


def _prepare_scale_records(
    specs: tuple[dict[str, int], ...],
    fft_backend: str,
    device_index: int | None,
    batch_cases: int,
) -> tuple[list[ScaleRecord], list[dict[str, object]]]:
    orientations = orientation_values(float(ORIENTATION_STEP_DEG), span_deg=180.0)
    phases = phase_values(int(PHASE_COUNT), float(PHASE_STEP_PX))
    included: list[ScaleRecord] = []
    excluded: list[dict[str, object]] = []
    for spec in specs:
        radius = int(spec["radius"])
        degree = int(spec["degree"])
        diagnostics = wvf_conditioning_diagnostics(radius=radius, degree=degree, normalize_coords=bool(NORMALIZE_COORDS))
        if int(diagnostics["rank_deficient_count"]) != 0 or str(diagnostics["status"]) != "ok":
            excluded.append(dict(diagnostics))
            continue
        kernel = build_wvf(radius=radius, degree=degree, normalize_coords=bool(NORMALIZE_COORDS))
        intrinsic_fwhm = _evaluate_intrinsic_fwhm(
            kernel=kernel,
            orientations_deg=orientations,
            phases_px=phases,
            fft_backend=fft_backend,
            device_index=device_index,
            batch_cases=batch_cases,
        )
        included.append(
            ScaleRecord(
                radius=radius,
                degree=degree,
                kernel=kernel,
                diagnostics=dict(diagnostics),
                intrinsic_fwhm=float(intrinsic_fwhm),
            )
        )
    return included, excluded


def _linear_weights_l2(scale_records: list[ScaleRecord]) -> list[dict[str, float]]:
    raw = []
    for record in scale_records:
        mass = float(record.radius) ** 4 / float(_variance_prefactor(int(record.degree)))
        raw.append(mass)
    total = float(sum(raw))
    return [
        {
            "radius": int(record.radius),
            "degree": int(record.degree),
            "weight": float(value / total),
        }
        for record, value in zip(scale_records, raw, strict=True)
    ]


def _linear_weights_equal(scale_records: list[ScaleRecord]) -> list[dict[str, float]]:
    weight = 1.0 / float(len(scale_records))
    return [
        {
            "radius": int(record.radius),
            "degree": int(record.degree),
            "weight": float(weight),
        }
        for record in scale_records
    ]


def _linear_weights_fwhm(scale_records: list[ScaleRecord]) -> list[dict[str, float]]:
    raw = [1.0 / max(float(record.intrinsic_fwhm), 1.0e-12) for record in scale_records]
    total = float(sum(raw))
    return [
        {
            "radius": int(record.radius),
            "degree": int(record.degree),
            "weight": float(value / total),
        }
        for record, value in zip(scale_records, raw, strict=True)
    ]


def _linear_steerability_summary(
    strategy_key: str,
    weights: list[dict[str, float]],
    scale_records: list[ScaleRecord],
) -> dict[str, object]:
    combined_kernel = _build_linear_combined_kernel(strategy_key, weights, scale_records)
    max_radius = int(combined_kernel.support_half_extent)
    records: list[dict[str, float]] = []
    for theta_deg in STEERABILITY_ANGLES_DEG:
        theta_rad = math.radians(float(theta_deg))
        synth = np.asarray(combined_kernel.kernel_x, dtype=np.float64) * math.cos(theta_rad) + np.asarray(combined_kernel.kernel_y, dtype=np.float64) * math.sin(theta_rad)
        direct = np.zeros_like(synth, dtype=np.float64)
        for record, weight_info in zip(scale_records, weights, strict=True):
            directional_weights = _direct_rotated_weights(
                radius=int(record.radius),
                degree=int(record.degree),
                theta_deg=float(theta_deg),
                normalize_coords=bool(NORMALIZE_COORDS),
            )
            support = disk_offsets(int(record.radius), include_center=False)
            directional_kernel = np.zeros((2 * int(max_radius) + 1, 2 * int(max_radius) + 1), dtype=np.float64)
            center = int(max_radius)
            for offset_xy, value in zip(support, directional_weights, strict=True):
                x = int(center + int(offset_xy[0]))
                y = int(center + int(offset_xy[1]))
                directional_kernel[y, x] = float(value)
            direct += float(weight_info["weight"]) * directional_kernel
        residual = float(np.max(np.abs(synth - direct)))
        kernel_max = float(np.max(np.abs(synth)))
        threshold = _threshold_for(kernel_max)
        records.append(
            {
                "theta_deg": float(theta_deg),
                "residual": float(residual),
                "kernel_max": float(kernel_max),
                "threshold": float(threshold),
                "passed": bool(residual < threshold),
            }
        )
    worst = max(records, key=lambda row: float(row["residual"]))
    return {
        "max_residual": float(worst["residual"]),
        "worst_theta_deg": float(worst["theta_deg"]),
        "pass_count": int(sum(1 for row in records if bool(row["passed"]))),
        "total_count": int(len(records)),
        "records": records,
    }


def _make_single_scale_stimuli(
    widths_px: tuple[float, ...],
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
) -> list[StimulusSpec]:
    stimuli: list[StimulusSpec] = []
    for edge_width_px in widths_px:
        sigma_px = _edge_sigma_px(float(edge_width_px))
        curvature_radius_px = int(4 * int(BEST_SINGLE_RADIUS_BY_WIDTH[float(edge_width_px)]))
        step_cases = tuple(
            generate_step_cases(
                support_scale=float(EVAL_SUPPORT_SCALE),
                orientations_deg=orientations_deg,
                phases_px=phases_px,
                half_size=int(PATCH_HALF_SIZE),
                width_px=float(sigma_px),
            )
        )
        arc_cases = tuple(
            case
            for case in generate_curved_cases(
                support_scale=float(EVAL_SUPPORT_SCALE),
                curvature_radii=(int(curvature_radius_px),),
                orientations_deg=orientations_deg,
                phases_px=phases_px,
                half_size=int(PATCH_HALF_SIZE),
                width_px=float(sigma_px),
            )
            if str(case.stimulus_class) == "arc"
        )
        stimuli.append(
            StimulusSpec(
                key=f"w{int(round(float(edge_width_px)))}",
                label=_stimulus_label(float(edge_width_px)),
                step_cases=step_cases,
                arc_cases=arc_cases,
                edge_width_px=float(edge_width_px),
                curvature_radius_px=float(curvature_radius_px),
            )
        )
    return stimuli


def _tile_component_cases(
    cases_by_width: dict[float, tuple[StepCase | CurvedCase, ...]],
    stimulus_name: str,
) -> tuple[GradientFieldCase, ...]:
    widths = tuple(float(width) for width in COMPOSITE_TILE_ORDER)
    reference_length = len(cases_by_width[widths[0]])
    tiled: list[GradientFieldCase] = []
    for index in range(reference_length):
        row_cases = [cases_by_width[width][index] for width in widths]
        tile_h, tile_w = row_cases[0].image.shape
        full_image = np.zeros((2 * tile_h, 2 * tile_w), dtype=np.float32)
        full_gx = np.zeros((2 * tile_h, 2 * tile_w), dtype=np.float64)
        full_gy = np.zeros((2 * tile_h, 2 * tile_w), dtype=np.float64)
        full_mask = np.zeros((2 * tile_h, 2 * tile_w), dtype=bool)
        placements = (
            (slice(0, tile_h), slice(0, tile_w)),
            (slice(0, tile_h), slice(tile_w, 2 * tile_w)),
            (slice(tile_h, 2 * tile_h), slice(0, tile_w)),
            (slice(tile_h, 2 * tile_h), slice(tile_w, 2 * tile_w)),
        )
        for case, (row_slice, col_slice) in zip(row_cases, placements, strict=True):
            full_image[row_slice, col_slice] = np.asarray(case.image, dtype=np.float32)
            full_gx[row_slice, col_slice] = np.asarray(case.true_gx, dtype=np.float64)
            full_gy[row_slice, col_slice] = np.asarray(case.true_gy, dtype=np.float64)
            full_mask[row_slice, col_slice] = np.asarray(case.eval_mask, dtype=bool)
        tiled.append(
            GradientFieldCase(
                stimulus_name=str(stimulus_name),
                orientation_deg=float(row_cases[0].orientation_deg),
                phase_px=float(row_cases[0].phase_px),
                image=np.asarray(full_image, dtype=np.float32),
                true_gx=np.asarray(full_gx, dtype=np.float64),
                true_gy=np.asarray(full_gy, dtype=np.float64),
                eval_mask=np.asarray(full_mask, dtype=bool),
            )
        )
    return tuple(tiled)


def _make_composite_stimulus(
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
) -> StimulusSpec:
    step_cases_by_width: dict[float, tuple[StepCase, ...]] = {}
    arc_cases_by_width: dict[float, tuple[CurvedCase, ...]] = {}
    component_curvatures: list[float] = []
    for edge_width_px in COMPOSITE_WIDTHS:
        sigma_px = _edge_sigma_px(float(edge_width_px))
        curvature_radius_px = int(4 * int(BEST_SINGLE_RADIUS_BY_WIDTH[float(edge_width_px)]))
        component_curvatures.append(float(curvature_radius_px))
        step_cases_by_width[float(edge_width_px)] = tuple(
            generate_step_cases(
                support_scale=float(EVAL_SUPPORT_SCALE),
                orientations_deg=orientations_deg,
                phases_px=phases_px,
                half_size=int(PATCH_HALF_SIZE),
                width_px=float(sigma_px),
            )
        )
        arc_cases_by_width[float(edge_width_px)] = tuple(
            case
            for case in generate_curved_cases(
                support_scale=float(EVAL_SUPPORT_SCALE),
                curvature_radii=(int(curvature_radius_px),),
                orientations_deg=orientations_deg,
                phases_px=phases_px,
                half_size=int(PATCH_HALF_SIZE),
                width_px=float(sigma_px),
            )
            if str(case.stimulus_class) == "arc"
        )
    return StimulusSpec(
        key=str(COMPOSITE_LABEL),
        label=_stimulus_label(None, composite=True),
        step_cases=_tile_component_cases(step_cases_by_width, f"{COMPOSITE_LABEL}_step"),
        arc_cases=_tile_component_cases(arc_cases_by_width, f"{COMPOSITE_LABEL}_arc"),
        edge_width_px=None,
        curvature_radius_px=None,
        component_widths_px=tuple(float(width) for width in COMPOSITE_WIDTHS),
        component_curvature_radii_px=tuple(component_curvatures),
    )


def _zero_response_buffers(
    cases: tuple[StepCase | CurvedCase | GradientFieldCase, ...],
) -> dict[str, list[np.ndarray]]:
    return {
        "gx": [np.zeros_like(np.asarray(case.true_gx, dtype=np.float64)) for case in cases],
        "gy": [np.zeros_like(np.asarray(case.true_gy, dtype=np.float64)) for case in cases],
    }


def _l3_response_buffers(
    cases: tuple[StepCase | CurvedCase | GradientFieldCase, ...],
) -> dict[str, list[np.ndarray]]:
    return {
        "mag": [np.full_like(np.asarray(case.true_gx, dtype=np.float64), -np.inf, dtype=np.float64) for case in cases],
        "gx": [np.zeros_like(np.asarray(case.true_gx, dtype=np.float64)) for case in cases],
        "gy": [np.zeros_like(np.asarray(case.true_gy, dtype=np.float64)) for case in cases],
    }


def _accumulate_linear_buffers(
    buffers: dict[str, list[np.ndarray]],
    outputs: list[tuple[np.ndarray, np.ndarray]],
    weight: float,
) -> None:
    for index, (gx, gy) in enumerate(outputs):
        buffers["gx"][index] += float(weight) * np.asarray(gx, dtype=np.float64)
        buffers["gy"][index] += float(weight) * np.asarray(gy, dtype=np.float64)


def _update_l3_buffers(
    buffers: dict[str, list[np.ndarray]],
    outputs: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    for index, (gx, gy) in enumerate(outputs):
        gx_arr = np.asarray(gx, dtype=np.float64)
        gy_arr = np.asarray(gy, dtype=np.float64)
        mag_arr = np.sqrt(gx_arr**2 + gy_arr**2)
        selector = mag_arr > buffers["mag"][index]
        buffers["mag"][index][selector] = mag_arr[selector]
        buffers["gx"][index][selector] = gx_arr[selector]
        buffers["gy"][index][selector] = gy_arr[selector]


def _buffers_to_outputs(
    buffers: dict[str, list[np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        for gx, gy in zip(buffers["gx"], buffers["gy"], strict=True)
    ]


def _step_rmse_sum(
    cases: tuple[StepCase | GradientFieldCase, ...],
    responses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int]:
    values = [
        float(case_gradient_metrics(case, gx, gy)["grad_rmse"])
        for case, (gx, gy) in zip(cases, responses, strict=True)
    ]
    return float(sum(values)), int(len(values))


def _arc_orientation_mae_sum(
    cases: tuple[CurvedCase | GradientFieldCase, ...],
    responses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int]:
    values = [
        float(case_gradient_metrics(case, gx, gy)["ang_mae_deg"])
        for case, (gx, gy) in zip(cases, responses, strict=True)
    ]
    return float(sum(values)), int(len(values))


def _split_outputs(
    outputs: list[tuple[np.ndarray, np.ndarray]],
    step_count: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    return outputs[: int(step_count)], outputs[int(step_count) :]


def _evaluate_stimulus(
    stimulus: StimulusSpec,
    baseline_scales: list[ScaleRecord],
    active_scales: list[ScaleRecord],
    linear_weight_map: dict[str, list[dict[str, float]]],
    fft_backend: str,
    device_index: int | None,
    batch_cases: int,
    noise_draws: int,
    snr_db: float,
) -> dict[str, object]:
    baseline_step_draws: dict[tuple[int, int], list[float]] = {record.key: [] for record in baseline_scales}
    baseline_arc_draws: dict[tuple[int, int], list[float]] = {record.key: [] for record in baseline_scales}
    strategy_step_draws: dict[str, list[float]] = {key: [] for key in ("l2_variance_inverse", "l2_equal", "l2_fwhm", "l3_max")}
    strategy_arc_draws: dict[str, list[float]] = {key: [] for key in ("l2_variance_inverse", "l2_equal", "l2_fwhm", "l3_max")}
    active_weight_lookup = {
        strategy_key: {
            (int(row["radius"]), int(row["degree"])): float(row["weight"])
            for row in weight_rows
        }
        for strategy_key, weight_rows in linear_weight_map.items()
    }

    stimulus_seed = _stable_key_seed(str(stimulus.key))
    for draw_index in range(int(noise_draws)):
        rng = np.random.default_rng(int(NOISE_SEED_BASE + 1000 * stimulus_seed + draw_index))
        baseline_step_sums = {record.key: 0.0 for record in baseline_scales}
        baseline_arc_sums = {record.key: 0.0 for record in baseline_scales}
        strategy_step_sums = {key: 0.0 for key in strategy_step_draws}
        strategy_arc_sums = {key: 0.0 for key in strategy_arc_draws}
        total_step_cases = 0
        total_arc_cases = 0

        for start in range(0, len(stimulus.step_cases), int(batch_cases)):
            batch_cases_clean = stimulus.step_cases[start : start + int(batch_cases)]
            noisy_cases = [
                _clone_case(case, add_awgn(np.asarray(case.image, dtype=np.float64), float(snr_db), rng))
                for case in batch_cases_clean
            ]
            linear_step_buffers = {key: _zero_response_buffers(tuple(batch_cases_clean)) for key in linear_weight_map}
            l3_step_buffers = _l3_response_buffers(tuple(batch_cases_clean))
            for record in baseline_scales:
                outputs = apply_cases_batched(
                    noisy_cases,
                    record.kernel,
                    fft_backend,
                    device_index,
                    batch_cases=int(batch_cases),
                )
                metric_sum, metric_count = _step_rmse_sum(tuple(batch_cases_clean), outputs)
                baseline_step_sums[record.key] += float(metric_sum)
                total_step_cases += 0 if record != baseline_scales[0] else int(metric_count)
                if record.key in active_weight_lookup["l2_variance_inverse"]:
                    for strategy_key, weights_by_key in active_weight_lookup.items():
                        _accumulate_linear_buffers(linear_step_buffers[strategy_key], outputs, float(weights_by_key[record.key]))
                    _update_l3_buffers(l3_step_buffers, outputs)
            for strategy_key in linear_weight_map:
                metric_sum, _ = _step_rmse_sum(tuple(batch_cases_clean), _buffers_to_outputs(linear_step_buffers[strategy_key]))
                strategy_step_sums[strategy_key] += float(metric_sum)
            metric_sum, _ = _step_rmse_sum(tuple(batch_cases_clean), _buffers_to_outputs(l3_step_buffers))
            strategy_step_sums["l3_max"] += float(metric_sum)

        for start in range(0, len(stimulus.arc_cases), int(batch_cases)):
            batch_cases_clean = stimulus.arc_cases[start : start + int(batch_cases)]
            noisy_cases = [
                _clone_case(case, add_awgn(np.asarray(case.image, dtype=np.float64), float(snr_db), rng))
                for case in batch_cases_clean
            ]
            linear_arc_buffers = {key: _zero_response_buffers(tuple(batch_cases_clean)) for key in linear_weight_map}
            l3_arc_buffers = _l3_response_buffers(tuple(batch_cases_clean))
            for record in baseline_scales:
                outputs = apply_cases_batched(
                    noisy_cases,
                    record.kernel,
                    fft_backend,
                    device_index,
                    batch_cases=int(batch_cases),
                )
                metric_sum, metric_count = _arc_orientation_mae_sum(tuple(batch_cases_clean), outputs)
                baseline_arc_sums[record.key] += float(metric_sum)
                total_arc_cases += 0 if record != baseline_scales[0] else int(metric_count)
                if record.key in active_weight_lookup["l2_variance_inverse"]:
                    for strategy_key, weights_by_key in active_weight_lookup.items():
                        _accumulate_linear_buffers(linear_arc_buffers[strategy_key], outputs, float(weights_by_key[record.key]))
                    _update_l3_buffers(l3_arc_buffers, outputs)
            for strategy_key in linear_weight_map:
                metric_sum, _ = _arc_orientation_mae_sum(tuple(batch_cases_clean), _buffers_to_outputs(linear_arc_buffers[strategy_key]))
                strategy_arc_sums[strategy_key] += float(metric_sum)
            metric_sum, _ = _arc_orientation_mae_sum(tuple(batch_cases_clean), _buffers_to_outputs(l3_arc_buffers))
            strategy_arc_sums["l3_max"] += float(metric_sum)

        for record in baseline_scales:
            baseline_step_draws[record.key].append(float(baseline_step_sums[record.key] / max(total_step_cases, 1)))
            baseline_arc_draws[record.key].append(float(baseline_arc_sums[record.key] / max(total_arc_cases, 1)))
        for strategy_key in strategy_step_draws:
            strategy_step_draws[strategy_key].append(float(strategy_step_sums[strategy_key] / max(total_step_cases, 1)))
            strategy_arc_draws[strategy_key].append(float(strategy_arc_sums[strategy_key] / max(total_arc_cases, 1)))

    baseline_rows = []
    for record in baseline_scales:
        baseline_rows.append(
            {
                "radius": int(record.radius),
                "degree": int(record.degree),
                "step_grad_rmse": float(mean(baseline_step_draws[record.key])),
                "arc_orientation_mae_deg": float(mean(baseline_arc_draws[record.key])),
            }
        )
    best_step = min(baseline_rows, key=lambda row: float(row["step_grad_rmse"]))
    best_arc = min(baseline_rows, key=lambda row: float(row["arc_orientation_mae_deg"]))

    strategy_rows: dict[str, dict[str, object]] = {}
    for strategy_key in strategy_step_draws:
        step_rmse = float(mean(strategy_step_draws[strategy_key]))
        arc_mae = float(mean(strategy_arc_draws[strategy_key]))
        delta = float(best_step["step_grad_rmse"]) - float(step_rmse)
        rel = delta / max(float(best_step["step_grad_rmse"]), 1.0e-15)
        arc_delta = float(best_arc["arc_orientation_mae_deg"]) - float(arc_mae)
        strategy_rows[strategy_key] = {
            "label": _strategy_label(strategy_key),
            "step_grad_rmse": float(step_rmse),
            "arc_orientation_mae_deg": float(arc_mae),
            "step_rmse_delta_to_best_single": float(delta),
            "step_rmse_relative_improvement": float(rel),
            "arc_orientation_delta_to_best_single_deg": float(arc_delta),
        }
    return {
        "stimulus_key": str(stimulus.key),
        "label": str(stimulus.label),
        "edge_width_px": None if stimulus.edge_width_px is None else float(stimulus.edge_width_px),
        "curvature_radius_px": None if stimulus.curvature_radius_px is None else float(stimulus.curvature_radius_px),
        "component_widths_px": None if stimulus.component_widths_px is None else [float(value) for value in stimulus.component_widths_px],
        "component_curvature_radii_px": None
        if stimulus.component_curvature_radii_px is None
        else [float(value) for value in stimulus.component_curvature_radii_px],
        "best_single_scale": {
            "step": best_step,
            "arc": best_arc,
            "all_trace_rows": baseline_rows,
        },
        "strategies": strategy_rows,
    }


def _composite_orientation_diagnostic(
    stimulus: StimulusSpec,
    active_scales: list[ScaleRecord],
    trace_scales: list[ScaleRecord],
    l2_weights: list[dict[str, float]],
    best_single_arc: dict[str, object],
    fft_backend: str,
    device_index: int | None,
    output_dir: Path,
) -> dict[str, object]:
    representative = None
    for case in stimulus.arc_cases:
        if abs(float(case.orientation_deg)) <= 1.0e-12 and abs(float(case.phase_px)) <= 1.0e-12:
            representative = case
            break
    if representative is None:
        representative = stimulus.arc_cases[0]

    seed = _stable_key_seed(str(stimulus.key))
    rng = np.random.default_rng(int(NOISE_SEED_BASE + 1000 * seed))
    noisy_case = _clone_case(
        representative,
        add_awgn(np.asarray(representative.image, dtype=np.float64), float(SNR_DB), rng),
    )

    best_radius = int(best_single_arc["radius"])
    best_degree = int(best_single_arc["degree"])
    best_record = next(record for record in trace_scales if record.key == (best_radius, best_degree))

    best_outputs = apply_cases_batched([noisy_case], best_record.kernel, fft_backend, device_index, batch_cases=1)
    best_gx, best_gy = best_outputs[0]

    l2_weight_lookup = {
        (int(row["radius"]), int(row["degree"])): float(row["weight"])
        for row in l2_weights
    }
    combined_gx = None
    combined_gy = None
    for record in active_scales:
        outputs = apply_cases_batched([noisy_case], record.kernel, fft_backend, device_index, batch_cases=1)
        gx, gy = outputs[0]
        weight = float(l2_weight_lookup[record.key])
        if combined_gx is None:
            combined_gx = weight * np.asarray(gx, dtype=np.float64)
            combined_gy = weight * np.asarray(gy, dtype=np.float64)
        else:
            combined_gx += weight * np.asarray(gx, dtype=np.float64)
            combined_gy += weight * np.asarray(gy, dtype=np.float64)
    assert combined_gx is not None and combined_gy is not None

    mask = np.asarray(representative.eval_mask, dtype=bool)
    l2_error = _orientation_error_deg_map(
        representative.true_gx,
        representative.true_gy,
        combined_gx,
        combined_gy,
        mask,
    )
    best_error = _orientation_error_deg_map(
        representative.true_gx,
        representative.true_gy,
        best_gx,
        best_gy,
        mask,
    )

    asset_dir = output_dir / f"assets_w{int(COMPOSITE_DIAGNOSTIC_MAX_WIDTH_PX)}"
    asset_path = asset_dir / "composite_orientation_error_l2_vs_best_single.png"
    _write_composite_orientation_diagnostic(
        asset_path,
        _error_map_to_rgb(l2_error),
        _error_map_to_rgb(best_error),
        left_label="L2 variance-inverse",
        right_label=f"Best single WVF (r={best_radius}, d={best_degree})",
        max_width_px=int(COMPOSITE_DIAGNOSTIC_MAX_WIDTH_PX),
    )
    return {
        "asset_path": str(asset_path),
        "asset_dir_name": str(asset_dir.name),
        "asset_max_width_px": int(COMPOSITE_DIAGNOSTIC_MAX_WIDTH_PX),
        "case": {
            "orientation_deg": float(representative.orientation_deg),
            "phase_px": float(representative.phase_px),
            "snr_db": float(SNR_DB),
            "draw_index": 0,
        },
        "l2_orientation_mae_deg_mean": float(np.nanmean(l2_error)),
        "best_single_orientation_mae_deg_mean": float(np.nanmean(best_error)),
        "best_single_config": {"radius": int(best_radius), "degree": int(best_degree)},
        "clip_deg": 45.0,
    }


def _decision(summary_rows: list[dict[str, object]]) -> dict[str, object]:
    linear_keys = ("l2_variance_inverse", "l2_equal", "l2_fwhm")
    nonlinear_key = "l3_max"
    best_linear: dict[str, object] | None = None
    best_l3: dict[str, object] | None = None

    for row in summary_rows:
        for strategy_key in linear_keys:
            candidate = {
                "stimulus_key": str(row["stimulus_key"]),
                "strategy_key": str(strategy_key),
                "step_rmse_relative_improvement": float(row["strategies"][strategy_key]["step_rmse_relative_improvement"]),
            }
            if best_linear is None or float(candidate["step_rmse_relative_improvement"]) > float(best_linear["step_rmse_relative_improvement"]):
                best_linear = candidate
        candidate_l3 = {
            "stimulus_key": str(row["stimulus_key"]),
            "strategy_key": str(nonlinear_key),
            "step_rmse_relative_improvement": float(row["strategies"][nonlinear_key]["step_rmse_relative_improvement"]),
        }
        if best_l3 is None or float(candidate_l3["step_rmse_relative_improvement"]) > float(best_l3["step_rmse_relative_improvement"]):
            best_l3 = candidate_l3

    best_linear = best_linear or {"stimulus_key": None, "strategy_key": None, "step_rmse_relative_improvement": 0.0}
    best_l3 = best_l3 or {"stimulus_key": None, "strategy_key": None, "step_rmse_relative_improvement": 0.0}
    proceed = False
    rationale = "stop"
    if float(best_linear["step_rmse_relative_improvement"]) >= 0.02:
        proceed = True
        rationale = "linear_strategy_clears_gate"
    elif float(best_l3["step_rmse_relative_improvement"]) >= 0.02:
        proceed = True
        rationale = "nonlinear_strategy_only"
    return {
        "best_linear": best_linear,
        "best_l3": best_l3,
        "proceed_to_phase2": bool(proceed),
        "decision_rule": "Proceed if the best linear strategy beats best single-scale by at least 2% on any stimulus. If only L3 clears 2%, proceed but frame the contribution as nonlinear.",
        "outcome": str(rationale),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
    batch_cases: int,
    noise_draws: int,
    stimulus_keys: tuple[str, ...] = tuple(),
) -> dict[str, Path]:
    start_time = time.perf_counter()
    output_dir = summary_json.parent
    active_records, excluded_scales = _prepare_scale_records(
        specs=tuple(dict(item) for item in ACTIVE_STACK),
        fft_backend=fft_backend,
        device_index=device_index,
        batch_cases=batch_cases,
    )
    if not active_records:
        raise RuntimeError("conditioning gate excluded every active multi-scale WVF cell")

    trace_records, excluded_trace = _prepare_scale_records(
        specs=tuple(dict(item) for item in SINGLE_SCALE_TRACE),
        fft_backend=fft_backend,
        device_index=device_index,
        batch_cases=batch_cases,
    )
    if not trace_records:
        raise RuntimeError("conditioning gate excluded every single-scale baseline cell")

    active_keys = {record.key for record in active_records}
    trace_records = sorted(trace_records, key=lambda record: (record.radius, record.degree))
    active_records = [record for record in trace_records if record.key in active_keys]

    l2_weights = _linear_weights_l2(active_records)
    l2_equal_weights = _linear_weights_equal(active_records)
    l2_fwhm_weights = _linear_weights_fwhm(active_records)
    linear_weight_map = {
        "l2_variance_inverse": l2_weights,
        "l2_equal": l2_equal_weights,
        "l2_fwhm": l2_fwhm_weights,
    }

    orientations = orientation_values(float(ORIENTATION_STEP_DEG), span_deg=180.0)
    phases = phase_values(int(PHASE_COUNT), float(PHASE_STEP_PX))
    stimuli = _make_single_scale_stimuli(EDGE_WIDTHS_PX, orientations, phases)
    stimuli.append(_make_composite_stimulus(orientations, phases))
    if stimulus_keys:
        requested = set(str(key) for key in stimulus_keys)
        stimuli = [stimulus for stimulus in stimuli if str(stimulus.key) in requested]
        if not stimuli:
            raise RuntimeError("stimulus filter removed every synthetic validation stimulus")

    summary_rows: list[dict[str, object]] = []
    for stimulus in stimuli:
        row = _evaluate_stimulus(
            stimulus=stimulus,
            baseline_scales=trace_records,
            active_scales=active_records,
            linear_weight_map=linear_weight_map,
            fft_backend=fft_backend,
            device_index=device_index,
            batch_cases=batch_cases,
            noise_draws=noise_draws,
            snr_db=float(SNR_DB),
        )
        summary_rows.append(row)
        best_strategy = max(
            row["strategies"].items(),
            key=lambda item: float(item[1]["step_rmse_relative_improvement"]),
        )
        print(
            f"sec10 {row['stimulus_key']} best={best_strategy[0]} "
            f"rel={float(best_strategy[1]['step_rmse_relative_improvement']):.4%} "
            f"delta={float(best_strategy[1]['step_rmse_delta_to_best_single']):.6e}"
        )

    steerability = {
        key: _linear_steerability_summary(key, weights, active_records)
        for key, weights in linear_weight_map.items()
    }
    decision = _decision(summary_rows)
    composite_diagnostic = None
    composite_row = next((row for row in summary_rows if str(row["stimulus_key"]) == str(COMPOSITE_LABEL)), None)
    composite_stimulus = next((stimulus for stimulus in stimuli if str(stimulus.key) == str(COMPOSITE_LABEL)), None)
    if composite_row is not None and composite_stimulus is not None:
        composite_diagnostic = _composite_orientation_diagnostic(
            stimulus=composite_stimulus,
            active_scales=active_records,
            trace_scales=trace_records,
            l2_weights=l2_weights,
            best_single_arc=dict(composite_row["best_single_scale"]["arc"]),
            fft_backend=fft_backend,
            device_index=device_index,
            output_dir=output_dir,
        )

    heatmap_stimuli = [row["stimulus_key"] for row in summary_rows]
    heatmap_labels = [row["label"] for row in summary_rows]
    heatmap_strategy_keys = ["l2_variance_inverse", "l2_equal", "l2_fwhm", "l3_max"]
    rmse_delta_matrix = [
        [float(next(row for row in summary_rows if str(row["stimulus_key"]) == stimulus_key)["strategies"][strategy_key]["step_rmse_delta_to_best_single"]) for stimulus_key in heatmap_stimuli]
        for strategy_key in heatmap_strategy_keys
    ]
    percent_matrix = [
        [
            100.0
            * float(
                next(row for row in summary_rows if str(row["stimulus_key"]) == stimulus_key)["strategies"][strategy_key]["step_rmse_relative_improvement"]
            )
            for stimulus_key in heatmap_stimuli
        ]
        for strategy_key in heatmap_strategy_keys
    ]

    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "config": {
            "active_stack_candidates": [dict(item) for item in ACTIVE_STACK],
            "single_scale_trace": [dict(item) for item in SINGLE_SCALE_TRACE],
            "edge_widths_px": [float(value) for value in EDGE_WIDTHS_PX],
            "snr_db": float(SNR_DB),
            "noise_draws": int(noise_draws),
            "orientation_step_deg": float(ORIENTATION_STEP_DEG),
            "phase_count": int(PHASE_COUNT),
            "phase_step_px": float(PHASE_STEP_PX),
            "fft_backend": str(fft_backend),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "patch_half_size": int(PATCH_HALF_SIZE),
            "stimulus_filter": [str(key) for key in stimulus_keys],
        },
        "conditioning_gate": "Cells are included only when rank_deficient_count == 0 under the scaled-epsilon SVD cutoff.",
        "excluded_scales": {
            "active_stack": excluded_scales,
            "single_scale_trace": excluded_trace,
        },
        "scale_calibration": [
            {
                "radius": int(record.radius),
                "degree": int(record.degree),
                "intrinsic_fwhm": float(record.intrinsic_fwhm),
                "white_noise_gain": float(record.kernel.white_noise_gain),
                "kappa_design_matrix": float(record.diagnostics["kappa_design_matrix"]),
                "sigma_min": float(record.diagnostics["sigma_min"]),
                "rank_deficient_count": int(record.diagnostics["rank_deficient_count"]),
            }
            for record in active_records
        ],
        "strategies": {
            "l2_variance_inverse": {
                "label": _strategy_label("l2_variance_inverse"),
                "class": "linear",
                "weights": l2_weights,
                "steerability": steerability["l2_variance_inverse"],
            },
            "l2_equal": {
                "label": _strategy_label("l2_equal"),
                "class": "linear",
                "weights": l2_equal_weights,
                "steerability": steerability["l2_equal"],
            },
            "l2_fwhm": {
                "label": _strategy_label("l2_fwhm"),
                "class": "linear",
                "weights": l2_fwhm_weights,
                "steerability": steerability["l2_fwhm"],
            },
            "l3_max": {
                "label": _strategy_label("l3_max"),
                "class": "nonlinear",
            },
        },
        "stimuli": summary_rows,
        "heatmap": {
            "strategy_keys": heatmap_strategy_keys,
            "strategy_labels": [_strategy_label(key) for key in heatmap_strategy_keys],
            "stimulus_keys": heatmap_stimuli,
            "stimulus_labels": heatmap_labels,
            "rmse_delta_matrix": rmse_delta_matrix,
            "percent_improvement_matrix": percent_matrix,
        },
        "decision": decision,
        "composite_orientation_diagnostic": composite_diagnostic,
        "wall_clock_seconds": float(time.perf_counter() - start_time),
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec10_multiscale_synthetic.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec10_multiscale_synthetic.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec10_multiscale_synthetic" / "sec10_multiscale_synthetic_summary.json",
        help="Path to the output summary JSON.",
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        default=DEFAULT_FFT_BACKEND,
        help="FFT backend for WVF application (vkfft or cpu).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional GPU device index for the FFT backend.",
    )
    parser.add_argument(
        "--batch-cases",
        type=int,
        default=DEFAULT_BATCH_CASES,
        help="Number of cases to tile into one FFT application batch.",
    )
    parser.add_argument(
        "--noise-draws",
        type=int,
        default=NOISE_DRAWS,
        help="Number of AWGN draws per stimulus cell.",
    )
    parser.add_argument(
        "--compile-plots",
        action="store_true",
        help="Compile the CeTZ figure after writing the summary JSON.",
    )
    parser.add_argument(
        "--stimulus-keys",
        type=str,
        default=None,
        help="Optional comma-separated subset of stimulus keys to evaluate.",
    )
    args = parser.parse_args(argv)

    outputs = run_experiment(
        summary_json=args.summary_json,
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
        batch_cases=int(args.batch_cases),
        noise_draws=int(args.noise_draws),
        stimulus_keys=_parse_key_list(args.stimulus_keys),
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
