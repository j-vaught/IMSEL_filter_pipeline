#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_dog, build_farid_simoncelli, build_sobel, build_wvf
from run_sec09_real_image_drive import (
    _add_awgn,
    _boundary_normal_field,
    _boundary_soft_mask,
    _centerline_mask,
    _centerline_tangent_angles,
    _fov_mask_from_green,
    _load_drive_input,
    _load_vessel_mask,
    _noise_slug,
    _normalize_magnitude,
    _orientation_entropy_from_angles,
    _orientation_mae_tangent,
    _orientation_rgb,
    _vector_rmse,
)
from run_sec09_real_image_hrf import _class_alias, _ensure_hrf_root, _role_alias
from sec09_wvf_grid import WVF_GRID_DEGREES, WVF_GRID_RADII, feasible_wvf_grid
from section8_common import apply_images_batched, compile_plot


TITLE = "Section 9 zoom-stack headline"
SUBTITLE = "One HRF retinal image across four effective vessel-width regimes"
SELECTION_IMAGE_STEM = "10_dr"
OUTPUT_SIZE_PX = 512
DEFAULT_ASSET_MAX_WIDTH_PX = 400
SNR_LEVELS = (math.inf, 10.0)
NOISE_DRAWS = 100
PRIMARY_METRIC_KEY = "orientation_mae_deg_mean"
WVF_METHOD_ORDER = ("sobel", "dog", "farid_simoncelli", "wvf")
BASELINE_METHOD_ORDER = ("sobel", "dog", "farid_simoncelli")

ZOOM_SPECS = (
    {
        "slug": "zoom1_out_x8",
        "label": "Zoom 1 out x8",
        "downsample_factor": 8,
        "effective_vessel_diameter_px": "~1-3 px",
    },
    {
        "slug": "zoom2_x4",
        "label": "Zoom 2 x4",
        "downsample_factor": 4,
        "effective_vessel_diameter_px": "~2-6 px",
    },
    {
        "slug": "zoom3_x2",
        "label": "Zoom 3 x2",
        "downsample_factor": 2,
        "effective_vessel_diameter_px": "~4-12 px",
    },
    {
        "slug": "zoom4_full",
        "label": "Zoom 4 full",
        "downsample_factor": 1,
        "effective_vessel_diameter_px": "~8-30 px",
    },
)


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    return np.asarray(image.resize((int(width), int(height)), resample=Image.Resampling.BICUBIC), dtype=np.uint8)


def _resize_gray_unit(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    image_u8 = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(image_u8, mode="L")
    resized = np.asarray(image.resize((int(width), int(height)), resample=Image.Resampling.BICUBIC), dtype=np.float64)
    return resized / 255.0


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image_u8 = np.asarray(mask, dtype=np.uint8) * 255
    image = Image.fromarray(image_u8, mode="L")
    resized = np.asarray(image.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR), dtype=np.float64)
    return np.asarray(resized >= 127.5, dtype=bool)


def _resize_image_if_needed(image: Image.Image, max_width_px: int | None) -> Image.Image:
    if max_width_px is None:
        return image
    width, height = image.size
    target_width = int(max_width_px)
    if target_width <= 0 or width <= target_width:
        return image
    target_height = max(1, int(round(height * (target_width / float(width)))))
    return image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)


def _write_rgb_asset(path: Path, rgb: np.ndarray, max_width_px: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    _resize_image_if_needed(image, max_width_px).save(path)


def _write_gray_asset(path: Path, gray: np.ndarray, max_width_px: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(image_u8, mode="L")
    _resize_image_if_needed(image, max_width_px).save(path)


def _reflect_pad_center_crop(array: np.ndarray, size_px: int) -> np.ndarray:
    target = int(size_px)
    height, width = array.shape[:2]
    pad_h = max(0, target - height)
    pad_w = max(0, target - width)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h > 0 or pad_w > 0:
        if array.ndim == 2:
            pad_spec = ((top, bottom), (left, right))
        else:
            pad_spec = ((top, bottom), (left, right), (0, 0))
        array = np.pad(array, pad_spec, mode="reflect")
    start_y = (array.shape[0] - target) // 2
    start_x = (array.shape[1] - target) // 2
    return np.asarray(array[start_y : start_y + target, start_x : start_x + target]).copy()


def _parse_zoom_filter(raw: str) -> set[str]:
    return {token.strip() for token in str(raw).split(",") if token.strip()}


def _select_zoom_specs(zoom_filter: set[str]) -> list[dict[str, object]]:
    selected = []
    for spec in ZOOM_SPECS:
        if zoom_filter and str(spec["slug"]) not in zoom_filter:
            continue
        selected.append(dict(spec))
    if not selected:
        raise RuntimeError("zoom filter removed every zoom level")
    return selected


def _select_hrf_image(data_root: Path, image_stem: str) -> dict[str, object]:
    target = str(image_stem).lower()
    image_path: Path | None = None
    label_path: Path | None = None
    fov_path: Path | None = None
    condition_class: str | None = None
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            continue
        if path.stem.lower() != target:
            continue
        class_name = _class_alias(str(path))
        role = _role_alias(path)
        if class_name is None or role is None:
            continue
        condition_class = class_name
        if role == "image" and image_path is None:
            image_path = path
        elif role == "label" and label_path is None:
            label_path = path
        elif role == "fov" and fov_path is None:
            fov_path = path
    if image_path is not None and label_path is not None and condition_class is not None:
        _, green = _load_drive_input(image_path)
        vessel_mask = _load_vessel_mask(label_path)
        centerline = _centerline_mask(vessel_mask)
        tangent_angles, tangent_valid = _centerline_tangent_angles(centerline)
        vessel_pixels = int(np.sum(vessel_mask))
        orientation_entropy = _orientation_entropy_from_angles(tangent_angles, tangent_valid)
        return {
            "image_id": f"{condition_class}:{image_path.stem}",
            "condition_class": str(condition_class),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "fov_path": None if fov_path is None else str(fov_path),
            "selection_score": 0.0,
            "vessel_pixels": vessel_pixels,
            "orientation_entropy": float(orientation_entropy),
        }
    raise FileNotFoundError(f"HRF image stem {image_stem!r} was not found under {data_root}")


def _prepare_zoom_view(
    rgb: np.ndarray,
    green: np.ndarray,
    vessel_mask: np.ndarray,
    zoom_spec: dict[str, object],
) -> dict[str, object]:
    factor = int(zoom_spec["downsample_factor"])
    height, width = green.shape
    resized_width = max(1, int(round(width / factor)))
    resized_height = max(1, int(round(height / factor)))
    if factor == 1:
        rgb_resized = np.asarray(rgb, dtype=np.uint8)
        green_resized = np.asarray(green, dtype=np.float64)
        mask_resized = np.asarray(vessel_mask, dtype=bool)
    else:
        rgb_resized = _resize_rgb(rgb, resized_width, resized_height)
        green_resized = _resize_gray_unit(green, resized_width, resized_height)
        mask_resized = _resize_mask(vessel_mask, resized_width, resized_height)

    rgb_crop = _reflect_pad_center_crop(rgb_resized, OUTPUT_SIZE_PX)
    green_crop = _reflect_pad_center_crop(green_resized, OUTPUT_SIZE_PX)
    mask_crop = _reflect_pad_center_crop(mask_resized, OUTPUT_SIZE_PX)
    fov_mask = _fov_mask_from_green(green_crop)
    soft_boundary = _boundary_soft_mask(mask_crop, fov_mask)
    boundary_normals, boundary_valid = _boundary_normal_field(soft_boundary)
    centerline = _centerline_mask(mask_crop)
    tangent_angles, tangent_valid = _centerline_tangent_angles(centerline)

    return {
        "slug": str(zoom_spec["slug"]),
        "label": str(zoom_spec["label"]),
        "downsample_factor": factor,
        "effective_vessel_diameter_px": str(zoom_spec["effective_vessel_diameter_px"]),
        "resized_shape_px": [int(rgb_resized.shape[1]), int(rgb_resized.shape[0])],
        "crop_shape_px": [int(rgb_crop.shape[1]), int(rgb_crop.shape[0])],
        "rgb": np.asarray(rgb_crop, dtype=np.uint8),
        "green": np.asarray(green_crop, dtype=np.float32),
        "vessel_mask": np.asarray(mask_crop, dtype=bool),
        "fov_mask": np.asarray(fov_mask, dtype=bool),
        "soft_boundary": np.asarray(soft_boundary, dtype=np.float64),
        "boundary_normals": np.asarray(boundary_normals, dtype=np.float64),
        "boundary_valid": np.asarray(boundary_valid & fov_mask, dtype=bool),
        "tangent_angles": np.asarray(tangent_angles, dtype=np.float64),
        "tangent_valid": np.asarray(tangent_valid & fov_mask, dtype=bool),
    }


def _save_method_assets_for_image(
    *,
    method_name: str,
    image_key: str,
    green_image: np.ndarray,
    kernel,
    assets_dir: Path,
    fft_backend: str,
    device_index: int | None,
    asset_max_width_px: int | None,
) -> dict[str, str]:
    gx, gy = apply_images_batched([np.asarray(green_image, dtype=np.float32)], kernel, fft_backend, device_index)[0]
    magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    mag_norm, _ = _normalize_magnitude(magnitude)
    mag_path = assets_dir / f"{method_name}_{image_key}_magnitude.png"
    ori_path = assets_dir / f"{method_name}_{image_key}_orientation.png"
    _write_gray_asset(mag_path, mag_norm, max_width_px=asset_max_width_px)
    _write_rgb_asset(
        ori_path,
        _orientation_rgb(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)),
        max_width_px=asset_max_width_px,
    )
    return {
        "magnitude_path": str(mag_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
        "orientation_path": str(ori_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
    }


def _evaluate_metrics(
    *,
    kernel,
    green_image: np.ndarray,
    soft_boundary: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_valid: np.ndarray,
    tangent_angles: np.ndarray,
    tangent_valid: np.ndarray,
    fov_mask: np.ndarray,
    snr_db: float,
    noise_draws: int,
    fft_backend: str,
    device_index: int | None,
    seed_offset: int,
) -> dict[str, object]:
    draw_count = 1 if math.isinf(float(snr_db)) else int(noise_draws)
    rmse_values = []
    angle_values = []
    clean = np.asarray(green_image, dtype=np.float64)
    for draw_index in range(draw_count):
        if math.isinf(float(snr_db)):
            noisy = clean.astype(np.float32)
        else:
            rng = np.random.default_rng(880000 + int(seed_offset) + 10000 * draw_index + 1000 * int(round(float(snr_db) * 10.0)))
            noisy = _add_awgn(clean, float(snr_db), rng)
        gx, gy = apply_images_batched([np.asarray(noisy, dtype=np.float32)], kernel, fft_backend, device_index)[0]
        rmse_values.append(
            _vector_rmse(
                gx=np.asarray(gx, dtype=np.float64),
                gy=np.asarray(gy, dtype=np.float64),
                soft_gt=np.asarray(soft_boundary, dtype=np.float64),
                normals=np.asarray(boundary_normals, dtype=np.float64),
                valid_mask=np.asarray(boundary_valid, dtype=bool),
            )
        )
        angle_values.append(
            _orientation_mae_tangent(
                gx=np.asarray(gx, dtype=np.float64),
                gy=np.asarray(gy, dtype=np.float64),
                gt_tangent_angles=np.asarray(tangent_angles, dtype=np.float64),
                gt_tangent_valid=np.asarray(tangent_valid, dtype=bool),
                fov_mask=np.asarray(fov_mask, dtype=bool),
            )
        )
    return {
        "gradient_vector_rmse_mean": float(np.mean(np.asarray(rmse_values, dtype=np.float64))),
        "orientation_mae_deg_mean": float(np.mean(np.asarray(angle_values, dtype=np.float64))),
        "noise_draws": int(draw_count),
    }


def _best_wvf_by_primary(cells: list[dict[str, object]], snr_slug: str) -> dict[str, object]:
    best = min(cells, key=lambda cell: float(cell["snr_metrics"][snr_slug][PRIMARY_METRIC_KEY]))
    metrics = dict(best["snr_metrics"][snr_slug])
    return {
        "radius": int(best["radius"]),
        "degree": int(best["degree"]),
        "metrics": metrics,
        "label": f"r={int(best['radius'])}, d={int(best['degree'])}",
        "primary_metric_key": PRIMARY_METRIC_KEY,
        "primary_metric_value": float(metrics[PRIMARY_METRIC_KEY]),
    }


def _best_baseline_by_primary(methods_payload: dict[str, object], snr_slug: str) -> dict[str, object]:
    candidates = []
    for method_name in BASELINE_METHOD_ORDER:
        method_payload = methods_payload[method_name]
        metrics = dict(method_payload["snr_metrics"][snr_slug])
        candidates.append(
            {
                "method": method_name,
                "label": str(method_payload["label"]),
                "config": dict(method_payload["config"]),
                "metrics": metrics,
                "primary_metric_value": float(metrics[PRIMARY_METRIC_KEY]),
            }
        )
    return min(candidates, key=lambda row: float(row["primary_metric_value"]))


def _merge_partial_summaries(
    *,
    partial_paths: list[Path],
    summary_json: Path,
    compile_plots: bool,
) -> dict[str, Path]:
    partials = [json.loads(path.read_text()) for path in partial_paths]
    if not partials:
        raise RuntimeError("no shard summaries were provided for merge")
    first = partials[0]
    zoom_order = [str(spec["slug"]) for spec in ZOOM_SPECS]
    merged_zooms: dict[str, object] = {}
    for payload in partials:
        if payload["selected_image"] != first["selected_image"]:
            raise RuntimeError("zoom-stack shard merge failed because selected images differ")
        if payload["config"] != first["config"]:
            raise RuntimeError("zoom-stack shard merge failed because configs differ")
        for zoom_key, zoom_payload in payload.get("zooms", {}).items():
            merged_zooms[str(zoom_key)] = zoom_payload
    final_payload = dict(first)
    final_payload["zoom_order"] = [slug for slug in zoom_order if slug in merged_zooms]
    final_payload["zooms"] = {slug: merged_zooms[slug] for slug in final_payload["zoom_order"]}
    final_payload["partial_request"] = None
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(final_payload, handle, indent=2)
        handle.write("\n")
    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_zoom_stack_headline.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_zoom_stack_headline.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def run_experiment(
    *,
    dataset_root: Path,
    output_dir: Path,
    summary_json: Path,
    image_stem: str,
    fft_backend: str,
    device_index: int | None,
    noise_draws: int,
    asset_max_width_px: int | None,
    zoom_filter: set[str],
    compile_plots: bool,
    auto_download: bool,
) -> dict[str, Path]:
    data_root = _ensure_hrf_root(dataset_root, auto_download=bool(auto_download))
    selection = _select_hrf_image(data_root, image_stem=image_stem)
    print(
        f"zoom-stack start image={selection['image_id']} fft_backend={fft_backend} "
        f"noise_draws={noise_draws} asset_max_width_px={asset_max_width_px}",
        flush=True,
    )
    rgb, green = _load_drive_input(Path(selection["image_path"]))
    vessel_mask = _load_vessel_mask(Path(selection["label_path"]))

    selected_zooms = _select_zoom_specs(zoom_filter)
    assets_dir_name = "assets" if asset_max_width_px is None else f"assets_w{int(asset_max_width_px)}"
    assets_dir = output_dir / assets_dir_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    feasible_cells = feasible_wvf_grid(
        radii=WVF_GRID_RADII,
        degrees=WVF_GRID_DEGREES,
        normalize_coords=True,
    )
    baseline_specs = {
        "sobel": build_sobel(),
        "dog": build_dog(3.0),
        "farid_simoncelli": build_farid_simoncelli(),
    }

    zoom_payloads: dict[str, object] = {}
    for zoom_index, zoom_spec in enumerate(selected_zooms):
        zoom_view = _prepare_zoom_view(rgb, green, vessel_mask, zoom_spec)
        zoom_key = str(zoom_view["slug"])
        print(
            f"zoom={zoom_key} start downsample_factor={zoom_view['downsample_factor']} "
            f"effective_vessel_diameter_px={zoom_view['effective_vessel_diameter_px']}",
            flush=True,
        )
        input_path = assets_dir / f"{zoom_key}_input.png"
        mask_path = assets_dir / f"{zoom_key}_vessel_mask.png"
        _write_rgb_asset(input_path, np.asarray(zoom_view["rgb"], dtype=np.uint8), max_width_px=asset_max_width_px)
        _write_gray_asset(mask_path, np.asarray(zoom_view["vessel_mask"], dtype=np.float64), max_width_px=asset_max_width_px)

        methods_payload: dict[str, object] = {}
        for method_index, (method_name, kernel) in enumerate(baseline_specs.items()):
            metrics_by_snr = {}
            for snr_db in SNR_LEVELS:
                snr_slug = _noise_slug(float(snr_db))
                metrics_by_snr[snr_slug] = _evaluate_metrics(
                    kernel=kernel,
                    green_image=np.asarray(zoom_view["green"], dtype=np.float32),
                    soft_boundary=np.asarray(zoom_view["soft_boundary"], dtype=np.float64),
                    boundary_normals=np.asarray(zoom_view["boundary_normals"], dtype=np.float64),
                    boundary_valid=np.asarray(zoom_view["boundary_valid"], dtype=bool),
                    tangent_angles=np.asarray(zoom_view["tangent_angles"], dtype=np.float64),
                    tangent_valid=np.asarray(zoom_view["tangent_valid"], dtype=bool),
                    fov_mask=np.asarray(zoom_view["fov_mask"], dtype=bool),
                    snr_db=float(snr_db),
                    noise_draws=int(noise_draws),
                    fft_backend=fft_backend,
                    device_index=device_index,
                    seed_offset=100000 * zoom_index + 1000 * method_index,
                )
            clean_assets = _save_method_assets_for_image(
                method_name=method_name,
                image_key=zoom_key,
                green_image=np.asarray(zoom_view["green"], dtype=np.float32),
                kernel=kernel,
                assets_dir=assets_dir,
                fft_backend=fft_backend,
                device_index=device_index,
                asset_max_width_px=asset_max_width_px,
            )
            methods_payload[method_name] = {
                "label": str(kernel.label),
                "config": dict(kernel.config),
                "clean_assets": clean_assets,
                "snr_metrics": metrics_by_snr,
            }
        print(f"zoom={zoom_key} baselines_done methods={','.join(baseline_specs.keys())}", flush=True)

        wvf_cells: list[dict[str, object]] = []
        total_cells = len(feasible_cells)
        for cell_index, cell in enumerate(feasible_cells):
            radius = int(cell["radius"])
            degree = int(cell["degree"])
            kernel = build_wvf(radius=radius, degree=degree, normalize_coords=True)
            snr_metrics = {}
            for snr_db in SNR_LEVELS:
                snr_slug = _noise_slug(float(snr_db))
                snr_metrics[snr_slug] = _evaluate_metrics(
                    kernel=kernel,
                    green_image=np.asarray(zoom_view["green"], dtype=np.float32),
                    soft_boundary=np.asarray(zoom_view["soft_boundary"], dtype=np.float64),
                    boundary_normals=np.asarray(zoom_view["boundary_normals"], dtype=np.float64),
                    boundary_valid=np.asarray(zoom_view["boundary_valid"], dtype=bool),
                    tangent_angles=np.asarray(zoom_view["tangent_angles"], dtype=np.float64),
                    tangent_valid=np.asarray(zoom_view["tangent_valid"], dtype=bool),
                    fov_mask=np.asarray(zoom_view["fov_mask"], dtype=bool),
                    snr_db=float(snr_db),
                    noise_draws=int(noise_draws),
                    fft_backend=fft_backend,
                    device_index=device_index,
                    seed_offset=1000000 * zoom_index + 1000 * cell_index,
                )
            wvf_cells.append(
                {
                    "radius": radius,
                    "degree": degree,
                    "config": {"r": radius, "d": degree, "normalize_coords": True},
                    "support_cardinality": int(cell["support_cardinality"]),
                    "coefficient_count": int(cell["coefficient_count"]),
                    "kappa_design_matrix": float(cell["kappa_design_matrix"]),
                    "sigma_min": float(cell["sigma_min"]),
                    "rank_deficient_count": int(cell["rank_deficient_count"]),
                    "snr_metrics": snr_metrics,
                }
            )
            if (cell_index + 1) % 10 == 0 or (cell_index + 1) == total_cells:
                print(
                    f"zoom={zoom_key} wvf_progress {cell_index + 1}/{total_cells} "
                    f"last_r={radius} last_d={degree}",
                    flush=True,
                )

        best_wvf_by_snr = {}
        for snr_db in SNR_LEVELS:
            snr_slug = _noise_slug(float(snr_db))
            best_wvf_by_snr[snr_slug] = _best_wvf_by_primary(wvf_cells, snr_slug)

        clean_best = best_wvf_by_snr["inf"]
        clean_best_kernel = build_wvf(
            radius=int(clean_best["radius"]),
            degree=int(clean_best["degree"]),
            normalize_coords=True,
        )
        methods_payload["wvf"] = {
            "label": "WVF",
            "config": {
                "clean_preview_r": int(clean_best["radius"]),
                "clean_preview_d": int(clean_best["degree"]),
                "normalize_coords": True,
            },
            "clean_assets": _save_method_assets_for_image(
                method_name="wvf",
                image_key=zoom_key,
                green_image=np.asarray(zoom_view["green"], dtype=np.float32),
                kernel=clean_best_kernel,
                assets_dir=assets_dir,
                fft_backend=fft_backend,
                device_index=device_index,
                asset_max_width_px=asset_max_width_px,
            ),
            "best_by_snr": best_wvf_by_snr,
            "cells": wvf_cells,
        }

        best_baseline_by_snr = {}
        deltas = {}
        for snr_db in SNR_LEVELS:
            snr_slug = _noise_slug(float(snr_db))
            best_baseline = _best_baseline_by_primary(methods_payload, snr_slug)
            best_baseline_by_snr[snr_slug] = best_baseline
            best_metrics = best_wvf_by_snr[snr_slug]["metrics"]
            baseline_metrics = best_baseline["metrics"]
            deltas[snr_slug] = {
                "baseline_method": str(best_baseline["method"]),
                "baseline_label": str(best_baseline["label"]),
                "baseline_config": dict(best_baseline["config"]),
                "orientation_mae_deg_mean": float(baseline_metrics["orientation_mae_deg_mean"] - best_metrics["orientation_mae_deg_mean"]),
                "gradient_vector_rmse_mean": float(baseline_metrics["gradient_vector_rmse_mean"] - best_metrics["gradient_vector_rmse_mean"]),
                "best_wvf_label": str(best_wvf_by_snr[snr_slug]["label"]),
            }

        zoom_payloads[zoom_key] = {
            "label": str(zoom_view["label"]),
            "downsample_factor": int(zoom_view["downsample_factor"]),
            "effective_vessel_diameter_px": str(zoom_view["effective_vessel_diameter_px"]),
            "resized_shape_px": list(zoom_view["resized_shape_px"]),
            "crop_shape_px": list(zoom_view["crop_shape_px"]),
            "input_asset_path": str(input_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            "vessel_mask_asset_path": str(mask_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            "methods": methods_payload,
            "best_baseline_by_snr": best_baseline_by_snr,
            "delta_small_stencil_minus_best_wvf": deltas,
            "primary_metric_key": PRIMARY_METRIC_KEY,
            "conditioning_gate": "Cells are included only when rank_deficient_count == 0 under the scaled-epsilon SVD cutoff.",
        }
        print(
            f"zoom={zoom_key} complete clean_best={best_wvf_by_snr['inf']['label']} "
            f"snr10_best={best_wvf_by_snr['10']['label']}",
            flush=True,
        )

    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "scenario": "zoom_stack_headline",
        "dataset": {
            "name": "HRF",
            "data_root": str(data_root),
            "image_shape_px": [3504, 2336],
        },
        "selected_image": selection,
        "config": {
            "snr_levels": [_noise_slug(float(value)) for value in SNR_LEVELS],
            "noise_draws": int(noise_draws),
            "asset_max_width_px": None if asset_max_width_px is None else int(asset_max_width_px),
            "output_size_px": int(OUTPUT_SIZE_PX),
            "baseline_method_order": list(BASELINE_METHOD_ORDER),
            "wvf_candidate_radii": [int(value) for value in WVF_GRID_RADII],
            "wvf_candidate_degrees": [int(value) for value in WVF_GRID_DEGREES],
            "wvf_primary_selection_metric": PRIMARY_METRIC_KEY,
            "fft_backend": str(fft_backend),
        },
        "asset_rendering": {
            "asset_dir_name": str(assets_dir_name),
            "asset_max_width_px": None if asset_max_width_px is None else int(asset_max_width_px),
            "rendered_preview_width_px": None if asset_max_width_px is None else int(asset_max_width_px),
        },
        "method_order": list(WVF_METHOD_ORDER),
        "zoom_order": [str(spec["slug"]) for spec in selected_zooms],
        "zooms": zoom_payloads,
        "partial_request": {
            "zoom_filter": sorted(zoom_filter),
        } if zoom_filter else None,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"zoom-stack summary_written path={summary_json}", flush=True)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_zoom_stack_headline.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_zoom_stack_headline.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "HRF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_zoom_stack",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_zoom_stack" / "sec09_real_image_zoom_stack_summary.json",
    )
    parser.add_argument("--image-stem", type=str, default=SELECTION_IMAGE_STEM)
    parser.add_argument("--fft-backend", type=str, default="vkfft", choices=("vkfft", "cpu"))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--noise-draws", type=int, default=NOISE_DRAWS)
    parser.add_argument("--asset-max-width-px", type=int, default=DEFAULT_ASSET_MAX_WIDTH_PX)
    parser.add_argument("--zoom-filter", type=str, default="")
    parser.add_argument("--merge-shard-jsons", type=Path, nargs="+", default=None)
    parser.add_argument("--compile-plots", action="store_true")
    parser.add_argument("--auto-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_shard_jsons:
        _merge_partial_summaries(
            partial_paths=[path.resolve() for path in args.merge_shard_jsons],
            summary_json=args.summary_json.resolve(),
            compile_plots=bool(args.compile_plots),
        )
        return 0
    run_experiment(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        image_stem=str(args.image_stem),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        noise_draws=int(args.noise_draws),
        asset_max_width_px=args.asset_max_width_px,
        zoom_filter=_parse_zoom_filter(str(args.zoom_filter)),
        compile_plots=bool(args.compile_plots),
        auto_download=bool(args.auto_download),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
