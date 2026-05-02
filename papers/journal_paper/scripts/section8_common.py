#!/usr/bin/env python3
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import KernelSpec
from wvf_metal.metal import fft_gradients_with_kernel


CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
STEP_PATCH_HALF_SIZE = 192
CURVE_PATCH_HALF_SIZE = 192
STEP_NORMAL_BAND_HALF_PX = 6.0
STEP_TANGENTIAL_SPAN_FACTOR = 2.0
DEFAULT_BATCH_CASES = 64

BRANCH_SAMPLE_DISTANCES = (4.0, 6.0, 8.0)


@dataclass(frozen=True)
class StepCase:
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    true_gx: np.ndarray
    true_gy: np.ndarray
    eval_mask: np.ndarray
    line_coords: np.ndarray
    line_xs: np.ndarray
    line_ys: np.ndarray


@dataclass(frozen=True)
class CurvedCase:
    stimulus_class: str
    curvature_radius: int
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    true_gx: np.ndarray
    true_gy: np.ndarray
    eval_mask: np.ndarray


@dataclass(frozen=True)
class JunctionCase:
    junction_name: str
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    center_xy: tuple[float, float]
    branch_dirs: tuple[np.ndarray, ...]


def orientation_values(step_deg: float, span_deg: float = 180.0) -> tuple[float, ...]:
    count = int(round(float(span_deg) / float(step_deg)))
    return tuple(float(step_deg) * i for i in range(count))


def phase_values(count: int, step_px: float) -> tuple[float, ...]:
    return tuple(float(step_px) * i for i in range(int(count)))


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst_bin = shutil.which("typst") or str(Path.home() / "bin" / "typst")
    subprocess.run(
        [typst_bin, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        check=True,
        cwd=str(ROOT),
    )


def awgn_sigma(snr_db: float, contrast: float = CONTRAST) -> float:
    return float(contrast) / (10.0 ** (float(snr_db) / 20.0))


def add_awgn(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sigma = awgn_sigma(float(snr_db))
    return np.asarray(image, dtype=np.float64) + sigma * rng.normal(size=np.asarray(image).shape)


def _local_grid(half_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-int(half_size), int(half_size) + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return axis, xx, yy


def _rotate_to_local(xx: np.ndarray, yy: np.ndarray, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(theta_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    u = xx * cos_t + yy * sin_t
    v = -xx * sin_t + yy * cos_t
    return u, v


def _tanh_factor(phi: np.ndarray, width_px: float = EDGE_WIDTH_PX, contrast: float = CONTRAST) -> np.ndarray:
    normalized = np.tanh(np.asarray(phi, dtype=np.float64) / float(width_px))
    return 0.5 * float(contrast) / float(width_px) * (1.0 - normalized * normalized)


def render_smoothed_step(projection: np.ndarray, phase_px: float, contrast: float = CONTRAST, width_px: float = EDGE_WIDTH_PX) -> np.ndarray:
    return 0.5 * float(contrast) * (1.0 + np.tanh((np.asarray(projection, dtype=np.float64) - float(phase_px)) / float(width_px)))


def generate_step_cases(
    support_scale: float,
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
    half_size: int = STEP_PATCH_HALF_SIZE,
    contrast: float = CONTRAST,
    width_px: float = EDGE_WIDTH_PX,
) -> list[StepCase]:
    line_coords, xx, yy = _local_grid(int(half_size))
    center = float(half_size)
    cases: list[StepCase] = []
    for orientation_deg in orientations_deg:
        theta = math.radians(float(orientation_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        u, v = _rotate_to_local(xx, yy, float(orientation_deg))
        xs = center + line_coords * cos_t
        ys = center + line_coords * sin_t
        for phase_px in phases_px:
            phi = np.asarray(u, dtype=np.float64) - float(phase_px)
            factor = _tanh_factor(phi, width_px=width_px, contrast=contrast)
            image = render_smoothed_step(u, float(phase_px), contrast=contrast, width_px=width_px)
            true_gx = factor * cos_t
            true_gy = factor * sin_t
            eval_mask = (
                (np.abs(phi) <= float(STEP_NORMAL_BAND_HALF_PX))
                & (np.abs(v) <= float(STEP_TANGENTIAL_SPAN_FACTOR) * float(support_scale))
            )
            cases.append(
                StepCase(
                    orientation_deg=float(orientation_deg),
                    phase_px=float(phase_px),
                    image=np.asarray(image, dtype=np.float32),
                    true_gx=np.asarray(true_gx, dtype=np.float64),
                    true_gy=np.asarray(true_gy, dtype=np.float64),
                    eval_mask=np.asarray(eval_mask, dtype=bool),
                    line_coords=np.asarray(line_coords, dtype=np.float64),
                    line_xs=np.asarray(xs, dtype=np.float64),
                    line_ys=np.asarray(ys, dtype=np.float64),
                )
            )
    return cases


def _arc_level_set(u: np.ndarray, v: np.ndarray, rho: float, phase_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(u, dtype=np.float64) - float(phase_px)
    vv = np.asarray(v, dtype=np.float64)
    denom = np.sqrt((up - float(rho)) ** 2 + vv**2)
    phi = float(rho) - denom
    dphi_du = (float(rho) - up) / np.maximum(denom, 1.0e-12)
    dphi_dv = -vv / np.maximum(denom, 1.0e-12)
    return phi, dphi_du, dphi_dv


def _s_curve_level_set(u: np.ndarray, v: np.ndarray, rho: float, phase_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = np.asarray(u, dtype=np.float64) - float(phase_px)
    vv = np.asarray(v, dtype=np.float64)
    phi = up - (vv**3) / (3.0 * float(rho) ** 2)
    dphi_du = np.ones_like(phi, dtype=np.float64)
    dphi_dv = -(vv**2) / (float(rho) ** 2)
    return phi, dphi_du, dphi_dv


def generate_curved_cases(
    support_scale: float,
    curvature_radii: tuple[int, ...],
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
    half_size: int = CURVE_PATCH_HALF_SIZE,
    contrast: float = CONTRAST,
    width_px: float = EDGE_WIDTH_PX,
) -> list[CurvedCase]:
    _, xx, yy = _local_grid(int(half_size))
    cases: list[CurvedCase] = []
    for curvature_radius in curvature_radii:
        for orientation_deg in orientations_deg:
            u, v = _rotate_to_local(xx, yy, float(orientation_deg))
            theta = math.radians(float(orientation_deg))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            for phase_px in phases_px:
                for stimulus_class in ("arc", "s_curve"):
                    if stimulus_class == "arc":
                        phi, dphi_du, dphi_dv = _arc_level_set(u, v, float(curvature_radius), float(phase_px))
                    else:
                        phi, dphi_du, dphi_dv = _s_curve_level_set(u, v, float(curvature_radius), float(phase_px))
                    factor = _tanh_factor(phi, width_px=width_px, contrast=contrast)
                    image = 0.5 * float(contrast) * (1.0 + np.tanh(phi / float(width_px)))
                    true_gx = factor * (dphi_du * cos_t - dphi_dv * sin_t)
                    true_gy = factor * (dphi_du * sin_t + dphi_dv * cos_t)
                    eval_mask = (
                        (np.abs(phi) <= float(STEP_NORMAL_BAND_HALF_PX))
                        & (np.abs(v) <= float(STEP_TANGENTIAL_SPAN_FACTOR) * float(support_scale))
                    )
                    cases.append(
                        CurvedCase(
                            stimulus_class=stimulus_class,
                            curvature_radius=int(curvature_radius),
                            orientation_deg=float(orientation_deg),
                            phase_px=float(phase_px),
                            image=np.asarray(image, dtype=np.float32),
                            true_gx=np.asarray(true_gx, dtype=np.float64),
                            true_gy=np.asarray(true_gy, dtype=np.float64),
                            eval_mask=np.asarray(eval_mask, dtype=bool),
                        )
                    )
    return cases


def _sigmoid(value: np.ndarray, width_px: float = EDGE_WIDTH_PX) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(value, dtype=np.float64) / float(width_px)))


def _center_shift_world(theta_rad: float, phase_px: float) -> tuple[float, float]:
    return (
        float(phase_px * (math.cos(theta_rad) - math.sin(theta_rad))),
        float(phase_px * (math.sin(theta_rad) + math.cos(theta_rad))),
    )


def _render_l_corner(u: np.ndarray, v: np.ndarray, width_px: float = EDGE_WIDTH_PX) -> np.ndarray:
    return _sigmoid(u, width_px) * _sigmoid(v, width_px)


def _render_x_junction(u: np.ndarray, v: np.ndarray, width_px: float = EDGE_WIDTH_PX) -> np.ndarray:
    return 0.5 * (_sigmoid(u, width_px) + _sigmoid(v, width_px))


def generate_junction_cases(
    junction_name: str,
    orientations_deg: tuple[float, ...],
    phases_px: tuple[float, ...],
    image_size: int = 1024,
    contrast: float = CONTRAST,
    width_px: float = EDGE_WIDTH_PX,
) -> list[JunctionCase]:
    coords = np.arange(int(image_size), dtype=np.float64) - (int(image_size) - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    cases: list[JunctionCase] = []
    for orientation_deg in orientations_deg:
        theta_rad = math.radians(float(orientation_deg))
        for phase_px in phases_px:
            center_xy = _center_shift_world(theta_rad, float(phase_px))
            shifted_x = xx - float(center_xy[0])
            shifted_y = yy - float(center_xy[1])
            u = shifted_x * math.cos(theta_rad) + shifted_y * math.sin(theta_rad)
            v = -shifted_x * math.sin(theta_rad) + shifted_y * math.cos(theta_rad)
            if junction_name == "l_corner":
                image = contrast * _render_l_corner(u, v, width_px)
                branch_angles = (0.0, 90.0)
            elif junction_name == "x_junction":
                image = contrast * _render_x_junction(u, v, width_px)
                branch_angles = (0.0, 90.0, 180.0, 270.0)
            else:
                raise ValueError(f"unsupported junction_name {junction_name!r}")
            branch_dirs = []
            for local_deg in branch_angles:
                world_angle = theta_rad + math.radians(float(local_deg))
                branch_dirs.append(np.asarray((math.cos(world_angle), math.sin(world_angle)), dtype=np.float64))
            cases.append(
                JunctionCase(
                    junction_name=str(junction_name),
                    orientation_deg=float(orientation_deg),
                    phase_px=float(phase_px),
                    image=np.asarray(image, dtype=np.float32),
                    center_xy=(float(center_xy[0]), float(center_xy[1])),
                    branch_dirs=tuple(branch_dirs),
                )
            )
    return cases


def tile_images(images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    if not images:
        raise ValueError("cannot tile an empty image list")
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


def apply_images_batched(
    images: list[np.ndarray],
    kernel: KernelSpec,
    fft_backend: str,
    device_index: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    canvas, placements = tile_images(images)
    gx_canvas, gy_canvas = fft_gradients_with_kernel(
        canvas,
        radius=int(kernel.support_half_extent),
        kernel_x=np.asarray(kernel.kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel.kernel_y, dtype=np.float64),
        fft_backend=str(fft_backend),
        device_index=device_index,
    )
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for row_slice, col_slice in placements:
        outputs.append(
            (
                np.asarray(gx_canvas[row_slice, col_slice], dtype=np.float64).copy(),
                np.asarray(gy_canvas[row_slice, col_slice], dtype=np.float64).copy(),
            )
        )
    return outputs


def apply_cases_batched(
    cases: list[StepCase | CurvedCase | JunctionCase],
    kernel: KernelSpec,
    fft_backend: str,
    device_index: int | None,
    batch_cases: int = DEFAULT_BATCH_CASES,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(cases), int(batch_cases)):
        batch = cases[start : start + int(batch_cases)]
        batch_images = [np.asarray(case.image, dtype=np.float32) for case in batch]
        outputs.extend(apply_images_batched(batch_images, kernel, fft_backend, device_index))
    return outputs


def orientation_mae_deg(true_gx: np.ndarray, true_gy: np.ndarray, est_gx: np.ndarray, est_gy: np.ndarray) -> float:
    true_angle = np.mod(np.arctan2(np.asarray(true_gy, dtype=np.float64), np.asarray(true_gx, dtype=np.float64)), np.pi)
    est_angle = np.mod(np.arctan2(np.asarray(est_gy, dtype=np.float64), np.asarray(est_gx, dtype=np.float64)), np.pi)
    diff = np.abs((est_angle - true_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(np.degrees(np.mean(diff)))


def case_gradient_metrics(case: StepCase | CurvedCase, gx: np.ndarray, gy: np.ndarray) -> dict[str, float]:
    mask = np.asarray(case.eval_mask, dtype=bool)
    true_gx = np.asarray(case.true_gx, dtype=np.float64)[mask]
    true_gy = np.asarray(case.true_gy, dtype=np.float64)[mask]
    est_gx = np.asarray(gx, dtype=np.float64)[mask]
    est_gy = np.asarray(gy, dtype=np.float64)[mask]
    true_mag = np.sqrt(true_gx**2 + true_gy**2)
    est_mag = np.sqrt(est_gx**2 + est_gy**2)
    return {
        "grad_rmse": float(np.sqrt(np.mean((est_gx - true_gx) ** 2 + (est_gy - true_gy) ** 2))),
        "ang_mae_deg": float(orientation_mae_deg(true_gx, true_gy, est_gx, est_gy)),
        "mag_bias": float(np.mean(est_mag - true_mag)),
    }


def quadratic_peak_refinement(x_coords: np.ndarray, profile: np.ndarray, peak_index: int) -> tuple[float, float]:
    index = int(peak_index)
    if index <= 0 or index >= profile.shape[0] - 1:
        return float(x_coords[index]), float(profile[index])
    y_prev = float(profile[index - 1])
    y_mid = float(profile[index])
    y_next = float(profile[index + 1])
    denom = y_prev - 2.0 * y_mid + y_next
    if abs(denom) <= 1.0e-15:
        return float(x_coords[index]), y_mid
    delta = 0.5 * (y_prev - y_next) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    peak_x = float(x_coords[index]) + delta
    peak_y = y_mid - 0.25 * (y_prev - y_next) * delta
    return peak_x, float(peak_y)


def _crossing_x(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if abs(y1 - y0) <= 1.0e-15:
        return 0.5 * (float(x0) + float(x1))
    alpha = (float(target) - float(y0)) / (float(y1) - float(y0))
    return float(x0) + float(np.clip(alpha, 0.0, 1.0)) * (float(x1) - float(x0))


def fwhm_from_profile(
    profile: np.ndarray,
    x_coords: np.ndarray,
    peak_index: int,
    peak_height: float,
    baseline: float,
    search_start: int,
    search_stop: int,
) -> float:
    if peak_height <= baseline:
        return 0.0
    target = float(baseline) + 0.5 * (float(peak_height) - float(baseline))
    left = int(peak_index)
    while left > int(search_start) and float(profile[left]) >= target:
        left -= 1
    if float(profile[left]) >= target:
        left_x = float(x_coords[left])
    else:
        left_x = _crossing_x(float(x_coords[left]), float(profile[left]), float(x_coords[left + 1]), float(profile[left + 1]), target)

    right = int(peak_index)
    while right < int(search_stop) - 1 and float(profile[right]) >= target:
        right += 1
    if float(profile[right]) >= target:
        right_x = float(x_coords[right])
    else:
        right_x = _crossing_x(float(x_coords[right - 1]), float(profile[right - 1]), float(x_coords[right]), float(profile[right]), target)
    return max(0.0, float(right_x - left_x))


def peak_metrics_from_profile(
    profile: np.ndarray,
    x_coords: np.ndarray,
    phase_px: float,
    support_scale: float,
    width_px: float = EDGE_WIDTH_PX,
) -> tuple[float, float, float]:
    search_half = int(max(16, int(math.ceil(float(support_scale))), int(math.ceil(12.0 * float(width_px)))))
    center_pos = float(phase_px)
    center_index = int(round(center_pos + float(x_coords.shape[0] // 2)))
    search_start = max(1, center_index - search_half)
    search_stop = min(x_coords.shape[0] - 1, center_index + search_half + 1)
    local = np.asarray(profile[search_start:search_stop], dtype=np.float64)
    peak_index = int(search_start + int(np.argmax(local)))
    peak_x, peak_height = quadratic_peak_refinement(x_coords, profile, peak_index)
    far_mask = np.abs(np.asarray(x_coords, dtype=np.float64) - center_pos) > float(search_half)
    baseline = float(np.median(np.asarray(profile, dtype=np.float64)[far_mask])) if np.any(far_mask) else 0.0
    fwhm = fwhm_from_profile(profile, x_coords, peak_index, peak_height, baseline, search_start, search_stop)
    return float(peak_height), float(peak_x - center_pos), float(fwhm)


def sample_profile(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.asarray(
        ndimage.map_coordinates(
            np.asarray(image, dtype=np.float64),
            np.vstack((np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64))),
            order=1,
            mode="reflect",
        ),
        dtype=np.float64,
    )


def bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
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


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))
