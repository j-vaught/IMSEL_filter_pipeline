#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_farid_simoncelli, build_wvf
from run_sec09_real_image_drive import (
    _boundary_normal_field,
    _boundary_soft_mask,
    _centerline_mask,
    _centerline_tangent_angles,
    _fov_mask_from_green,
    _load_drive_input,
    _load_vessel_mask,
    _normalize_magnitude,
    _orientation_rgb,
    _orientation_mae_tangent,
    _save_gray,
    _save_rgb,
    _vector_rmse,
)
from run_sec09_real_image_hrf import _class_alias, _ensure_hrf_root, _role_alias
from sec09_wvf_grid import wvf_conditioning_diagnostics
from section8_common import apply_images_batched


TITLE = "Section 9 HD WVF tuning viewer"
SUBTITLE = "Full-image manual tuning on one HRF diabetic-retinopathy frame"
DEFAULT_IMAGE_STEM = "10_dr"
DEFAULT_ASSET_MAX_WIDTH_PX = 1500
RADIUS_VALUES = tuple(range(2, 31))
DEGREE_VALUES = (1, 3, 5, 7, 9, 11)
CROP_SIZE_PX = 420
PREVIEW_CROP_KEYS = (
    "full_image",
    "top_left_quadrant",
    "top_right_quadrant",
    "vessel_junction_crop",
    "optic_disc_crop",
    "peripheral_crop",
)
EPS = 1.0e-12


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
    if image_path is None or label_path is None or condition_class is None:
        raise FileNotFoundError(f"HRF image stem {image_stem!r} was not found under {data_root}")
    return {
        "image_id": f"{condition_class}:{image_path.stem}",
        "condition_class": str(condition_class),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "fov_path": None if fov_path is None else str(fov_path),
    }


def _resize_rgb_array(rgb: np.ndarray, max_width_px: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    width, height = image.size
    if width <= int(max_width_px):
        return np.asarray(image, dtype=np.uint8)
    target_height = max(1, int(round(height * (int(max_width_px) / float(width)))))
    resized = image.resize((int(max_width_px), target_height), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def _resize_gray_array(gray: np.ndarray, max_width_px: int) -> np.ndarray:
    gray_u8 = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(gray_u8, mode="L")
    width, height = image.size
    if width <= int(max_width_px):
        return np.asarray(image, dtype=np.float64) / 255.0
    target_height = max(1, int(round(height * (int(max_width_px) / float(width)))))
    resized = image.resize((int(max_width_px), target_height), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float64) / 255.0


def _resize_mask_array(mask: np.ndarray, max_width_px: int) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    image = Image.fromarray(mask_u8, mode="L")
    width, height = image.size
    if width <= int(max_width_px):
        return np.asarray(mask, dtype=bool)
    target_height = max(1, int(round(height * (int(max_width_px) / float(width)))))
    resized = image.resize((int(max_width_px), target_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(np.asarray(resized, dtype=np.float64) >= 127.5, dtype=bool)


def _relative_figure_path(path: Path) -> str:
    return str(path.relative_to(ROOT / "papers" / "journal_paper" / "figures"))


def _apply_kernel(
    green_image: np.ndarray,
    kernel,
    fft_backend: str,
    device_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = apply_images_batched([np.asarray(green_image, dtype=np.float32)], kernel, fft_backend, device_index)[0]
    return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)


def _measure_full_image_metrics(
    *,
    gx: np.ndarray,
    gy: np.ndarray,
    soft_boundary: np.ndarray,
    boundary_normals: np.ndarray,
    boundary_valid: np.ndarray,
    tangent_angles: np.ndarray,
    tangent_valid: np.ndarray,
    fov_mask: np.ndarray,
) -> dict[str, float]:
    return {
        "gradient_vector_rmse_mean": float(
            _vector_rmse(
                gx=np.asarray(gx, dtype=np.float64),
                gy=np.asarray(gy, dtype=np.float64),
                soft_gt=np.asarray(soft_boundary, dtype=np.float64),
                normals=np.asarray(boundary_normals, dtype=np.float64),
                valid_mask=np.asarray(boundary_valid, dtype=bool),
            )
        ),
        "orientation_mae_deg_mean": float(
            _orientation_mae_tangent(
                gx=np.asarray(gx, dtype=np.float64),
                gy=np.asarray(gy, dtype=np.float64),
                gt_tangent_angles=np.asarray(tangent_angles, dtype=np.float64),
                gt_tangent_valid=np.asarray(tangent_valid, dtype=bool),
                fov_mask=np.asarray(fov_mask, dtype=bool),
            )
        ),
    }


def _write_response_assets(
    *,
    method_slug: str,
    gx: np.ndarray,
    gy: np.ndarray,
    assets_dir: Path,
    asset_max_width_px: int,
) -> dict[str, str]:
    magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    mag_norm, _ = _normalize_magnitude(magnitude)
    mag_path = assets_dir / f"{method_slug}_magnitude.png"
    ori_path = assets_dir / f"{method_slug}_orientation.png"
    _save_gray(mag_path, mag_norm, max_width_px=asset_max_width_px)
    _save_rgb(
        ori_path,
        _orientation_rgb(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)),
        max_width_px=asset_max_width_px,
    )
    return {
        "magnitude_path": _relative_figure_path(mag_path),
        "orientation_path": _relative_figure_path(ori_path),
    }


def _clamp_crop(center_x: int, center_y: int, crop_w: int, crop_h: int, image_w: int, image_h: int) -> list[int]:
    width = min(int(crop_w), int(image_w))
    height = min(int(crop_h), int(image_h))
    x0 = int(round(center_x - width / 2.0))
    y0 = int(round(center_y - height / 2.0))
    x0 = min(max(0, x0), max(0, int(image_w) - width))
    y0 = min(max(0, y0), max(0, int(image_h) - height))
    return [int(x0), int(y0), int(width), int(height)]


def _detect_optic_disc_center(green_preview: np.ndarray, fov_mask_preview: np.ndarray) -> tuple[int, int]:
    smoothed = ndimage.gaussian_filter(np.asarray(green_preview, dtype=np.float64), sigma=12.0, mode="reflect")
    interior = ndimage.binary_erosion(np.asarray(fov_mask_preview, dtype=bool), iterations=10)
    if not np.any(interior):
        interior = np.asarray(fov_mask_preview, dtype=bool)
    masked = np.where(interior, smoothed, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(x), int(y)


def _detect_junction_center(vessel_mask_preview: np.ndarray) -> tuple[int, int]:
    centerline = _centerline_mask(np.asarray(vessel_mask_preview, dtype=bool))
    neighbor_count = (
        ndimage.convolve(centerline.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant", cval=0)
        - centerline.astype(np.int32)
    )
    junction_mask = centerline & (neighbor_count >= 3)
    if np.any(junction_mask):
        ys, xs = np.nonzero(junction_mask)
        center_y = (junction_mask.shape[0] - 1) / 2.0
        center_x = (junction_mask.shape[1] - 1) / 2.0
        distances = (xs.astype(np.float64) - center_x) ** 2 + (ys.astype(np.float64) - center_y) ** 2
        index = int(np.argmin(distances))
        return int(xs[index]), int(ys[index])
    density = ndimage.gaussian_filter(np.asarray(vessel_mask_preview, dtype=np.float64), sigma=10.0, mode="reflect")
    y, x = np.unravel_index(int(np.argmax(density)), density.shape)
    return int(x), int(y)


def _detect_peripheral_center(vessel_mask_preview: np.ndarray, fov_mask_preview: np.ndarray) -> tuple[int, int]:
    height, width = vessel_mask_preview.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    norm = np.sqrt(((xx - center_x) / max(width / 2.0, 1.0)) ** 2 + ((yy - center_y) / max(height / 2.0, 1.0)) ** 2)
    density = ndimage.gaussian_filter(np.asarray(vessel_mask_preview, dtype=np.float64), sigma=8.0, mode="reflect")
    ring = (norm >= 0.55) & (norm <= 0.92) & np.asarray(fov_mask_preview, dtype=bool)
    if np.any(ring):
        masked = np.where(ring, density, -np.inf)
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        return int(x), int(y)
    return int(round(0.78 * width)), int(round(0.55 * height))


def _build_crop_presets(
    *,
    green_preview: np.ndarray,
    vessel_mask_preview: np.ndarray,
    fov_mask_preview: np.ndarray,
) -> dict[str, dict[str, object]]:
    height, width = green_preview.shape
    optic_x, optic_y = _detect_optic_disc_center(green_preview, fov_mask_preview)
    junction_x, junction_y = _detect_junction_center(vessel_mask_preview)
    peripheral_x, peripheral_y = _detect_peripheral_center(vessel_mask_preview, fov_mask_preview)
    return {
        "full_image": {
            "label": "Full image",
            "rect_xywh": [0, 0, int(width), int(height)],
        },
        "top_left_quadrant": {
            "label": "Top-left quadrant",
            "rect_xywh": [0, 0, int(width // 2), int(height // 2)],
        },
        "top_right_quadrant": {
            "label": "Top-right quadrant",
            "rect_xywh": [int(width // 2), 0, int(width - width // 2), int(height // 2)],
        },
        "vessel_junction_crop": {
            "label": "Vessel-junction crop",
            "rect_xywh": _clamp_crop(junction_x, junction_y, CROP_SIZE_PX, CROP_SIZE_PX, width, height),
        },
        "optic_disc_crop": {
            "label": "Optic-disc crop",
            "rect_xywh": _clamp_crop(optic_x, optic_y, CROP_SIZE_PX, CROP_SIZE_PX, width, height),
        },
        "peripheral_crop": {
            "label": "Peripheral crop",
            "rect_xywh": _clamp_crop(peripheral_x, peripheral_y, CROP_SIZE_PX, CROP_SIZE_PX, width, height),
        },
    }


def run_experiment(
    *,
    dataset_root: Path,
    output_dir: Path,
    summary_json: Path,
    image_stem: str,
    fft_backend: str,
    device_index: int | None,
    asset_max_width_px: int,
    auto_download: bool,
) -> dict[str, Path]:
    data_root = _ensure_hrf_root(dataset_root, auto_download=bool(auto_download))
    selection = _select_hrf_image(data_root, image_stem=image_stem)
    print(
        f"hd-viewer start image={selection['image_id']} fft_backend={fft_backend} "
        f"asset_max_width_px={asset_max_width_px}",
        flush=True,
    )
    rgb, green = _load_drive_input(Path(selection["image_path"]))
    vessel_mask = _load_vessel_mask(Path(selection["label_path"]))
    fov_mask = _fov_mask_from_green(green)
    soft_boundary = _boundary_soft_mask(vessel_mask, fov_mask)
    boundary_normals, boundary_valid = _boundary_normal_field(soft_boundary)
    centerline = _centerline_mask(vessel_mask)
    tangent_angles, tangent_valid = _centerline_tangent_angles(centerline)

    assets_dir_name = f"assets_w{int(asset_max_width_px)}"
    assets_dir = output_dir / assets_dir_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    input_path = assets_dir / f"{image_stem}_input.png"
    vessel_mask_path = assets_dir / f"{image_stem}_vessel_mask.png"
    _save_rgb(input_path, np.asarray(rgb, dtype=np.uint8), max_width_px=asset_max_width_px)
    _save_gray(vessel_mask_path, np.asarray(vessel_mask, dtype=np.float64), max_width_px=asset_max_width_px)

    preview_rgb = _resize_rgb_array(np.asarray(rgb, dtype=np.uint8), int(asset_max_width_px))
    preview_green = _resize_gray_array(np.asarray(green, dtype=np.float64), int(asset_max_width_px))
    preview_mask = _resize_mask_array(np.asarray(vessel_mask, dtype=bool), int(asset_max_width_px))
    preview_fov = _resize_mask_array(np.asarray(fov_mask, dtype=bool), int(asset_max_width_px))
    crop_presets = _build_crop_presets(
        green_preview=preview_green,
        vessel_mask_preview=preview_mask,
        fov_mask_preview=preview_fov,
    )

    baseline_kernel = build_farid_simoncelli()
    baseline_gx, baseline_gy = _apply_kernel(
        green_image=np.asarray(green, dtype=np.float32),
        kernel=baseline_kernel,
        fft_backend=fft_backend,
        device_index=device_index,
    )
    baseline_assets = _write_response_assets(
        method_slug=f"farid_simoncelli_{image_stem}",
        gx=baseline_gx,
        gy=baseline_gy,
        assets_dir=assets_dir,
        asset_max_width_px=int(asset_max_width_px),
    )
    baseline_metrics = _measure_full_image_metrics(
        gx=baseline_gx,
        gy=baseline_gy,
        soft_boundary=np.asarray(soft_boundary, dtype=np.float64),
        boundary_normals=np.asarray(boundary_normals, dtype=np.float64),
        boundary_valid=np.asarray(boundary_valid & fov_mask, dtype=bool),
        tangent_angles=np.asarray(tangent_angles, dtype=np.float64),
        tangent_valid=np.asarray(tangent_valid & fov_mask, dtype=bool),
        fov_mask=np.asarray(fov_mask, dtype=bool),
    )

    feasible_cells: list[dict[str, object]] = []
    skipped_cells: list[dict[str, object]] = []
    total_cells = len(RADIUS_VALUES) * len(DEGREE_VALUES)
    processed = 0
    for radius in RADIUS_VALUES:
        for degree in DEGREE_VALUES:
            diagnostics = wvf_conditioning_diagnostics(radius=radius, degree=degree, normalize_coords=True)
            processed += 1
            if str(diagnostics["status"]) != "ok":
                skipped_cells.append(dict(diagnostics))
                continue
            kernel = build_wvf(radius=radius, degree=degree, normalize_coords=True)
            gx, gy = _apply_kernel(
                green_image=np.asarray(green, dtype=np.float32),
                kernel=kernel,
                fft_backend=fft_backend,
                device_index=device_index,
            )
            metrics = _measure_full_image_metrics(
                gx=gx,
                gy=gy,
                soft_boundary=np.asarray(soft_boundary, dtype=np.float64),
                boundary_normals=np.asarray(boundary_normals, dtype=np.float64),
                boundary_valid=np.asarray(boundary_valid & fov_mask, dtype=bool),
                tangent_angles=np.asarray(tangent_angles, dtype=np.float64),
                tangent_valid=np.asarray(tangent_valid & fov_mask, dtype=bool),
                fov_mask=np.asarray(fov_mask, dtype=bool),
            )
            assets = _write_response_assets(
                method_slug=f"wvf_{image_stem}_r{int(radius)}_d{int(degree)}",
                gx=gx,
                gy=gy,
                assets_dir=assets_dir,
                asset_max_width_px=int(asset_max_width_px),
            )
            feasible_cells.append(
                {
                    "radius": int(radius),
                    "degree": int(degree),
                    "config": {"r": int(radius), "d": int(degree), "normalize_coords": True},
                    "support_cardinality": int(diagnostics["support_cardinality"]),
                    "coefficient_count": int(diagnostics["coefficient_count"]),
                    "kappa_design_matrix": float(diagnostics["kappa_design_matrix"]),
                    "sigma_min": float(diagnostics["sigma_min"]),
                    "rank_deficient_count": int(diagnostics["rank_deficient_count"]),
                    "metrics": metrics,
                    "assets": assets,
                }
            )
            print(
                f"hd-viewer progress {processed}/{total_cells} r={radius} d={degree} "
                f"rmse={metrics['gradient_vector_rmse_mean']:.6e} "
                f"ang={metrics['orientation_mae_deg_mean']:.4f}",
                flush=True,
            )

    if not feasible_cells:
        raise RuntimeError("conditioning gate removed every WVF cell from the HD viewer sweep")

    default_cell = min(
        feasible_cells,
        key=lambda cell: (
            float(cell["metrics"]["orientation_mae_deg_mean"]),
            float(cell["metrics"]["gradient_vector_rmse_mean"]),
            int(cell["radius"]),
            int(cell["degree"]),
        ),
    )

    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "scenario": "hd_tuning_viewer",
        "dataset": {
            "name": "HRF",
            "data_root": str(data_root),
            "image_shape_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        },
        "selected_image": selection,
        "config": {
            "radii": [int(value) for value in RADIUS_VALUES],
            "degrees": [int(value) for value in DEGREE_VALUES],
            "normalize_coords": True,
            "fft_backend": str(fft_backend),
            "asset_max_width_px": int(asset_max_width_px),
        },
        "asset_rendering": {
            "asset_dir_name": str(assets_dir_name),
            "asset_max_width_px": int(asset_max_width_px),
            "rendered_preview_width_px": int(asset_max_width_px),
            "preview_shape_px": [int(preview_rgb.shape[1]), int(preview_rgb.shape[0])],
        },
        "static_assets": {
            "input_path": _relative_figure_path(input_path),
            "ground_truth_path": _relative_figure_path(vessel_mask_path),
        },
        "crop_preset_order": list(PREVIEW_CROP_KEYS),
        "crop_presets": crop_presets,
        "baseline_reference": {
            "method": "farid_simoncelli",
            "label": "Farid-Simoncelli",
            "config": dict(baseline_kernel.config),
            "metrics": baseline_metrics,
            "assets": baseline_assets,
        },
        "wvf_grid": {
            "conditioning_gate": "Cells are included only when rank_deficient_count == 0 under the scaled-epsilon SVD cutoff.",
            "default_cell": {
                "radius": int(default_cell["radius"]),
                "degree": int(default_cell["degree"]),
            },
            "feasible_cells": feasible_cells,
            "skipped_cells": skipped_cells,
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"hd-viewer summary_written path={summary_json}", flush=True)
    return {"summary_json": summary_json}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "HRF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_hd_viewer",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_hd_viewer" / "sec09_real_image_hd_viewer_summary.json",
    )
    parser.add_argument("--image-stem", type=str, default=DEFAULT_IMAGE_STEM)
    parser.add_argument("--fft-backend", type=str, default="vkfft", choices=("vkfft", "cpu"))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--asset-max-width-px", type=int, default=DEFAULT_ASSET_MAX_WIDTH_PX)
    parser.add_argument("--auto-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        image_stem=str(args.image_stem),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        asset_max_width_px=int(args.asset_max_width_px),
        auto_download=bool(args.auto_download),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
