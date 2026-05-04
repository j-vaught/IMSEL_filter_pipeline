#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sec09_real_image_drive as drive_mod
from baseline_filters import build_method
from run_sec09_real_image_drive import (
    DISPLAY_PERCENTILE,
    EPS,
    ODS_TOLERANCE_PX,
    SNR_LEVELS as DRIVE_SNR_LEVELS,
    _boundary_normal_field,
    _boundary_soft_mask,
    _centerline_mask,
    _centerline_tangent_angles,
    _clean_assets_for_method,
    _evaluate_snr_bank,
    _fov_mask_from_green,
    _load_drive_input,
    _load_vessel_mask,
    _noise_slug,
    _orientation_entropy_from_angles,
    _save_gray,
    _save_rgb,
)
from sec09_wvf_grid import WVF_GRID_DEGREES, WVF_GRID_RADII, feasible_wvf_grid
from section8_common import compile_plot


TITLE = "Section 9 Scenario B-HD high-resolution retinal vessels"
SUBTITLE = "HRF real-image comparison on high-resolution retinal vasculature"
DATASET_PAGE_URL = "https://www5.cs.fau.de/research/data/fundus-images/"
HRF_NOISE_DRAWS = 100
HRF_ODS_THRESHOLD_COUNT = 201
DEFAULT_SNR_LEVELS = tuple(float(value) for value in DRIVE_SNR_LEVELS)
HRF_SELECTION_PLAN = (
    ("healthy", 2),
    ("diabetic_retinopathy", 2),
    ("glaucoma", 1),
)
TRACE_METRICS = (
    ("ods_f_score", True),
    ("gradient_vector_rmse_mean", False),
    ("orientation_mae_deg_mean", False),
)
GRID_PRIMARY_METRIC_KEY = "orientation_mae_deg_mean"
GRID_PRIMARY_SNR_SLUG = "10"
SMALL_STENCIL_METHODS = ("roberts", "prewitt", "sobel", "scharr")
WVF_TRACE_SPECS = (
    {"r": 3, "d": 5, "normalize_coords": True},
    {"r": 5, "d": 9, "normalize_coords": True},
    {"r": 9, "d": 11, "normalize_coords": True},
    {"r": 15, "d": 11, "normalize_coords": True},
    {"r": 25, "d": 11, "normalize_coords": True},
    {"r": 50, "d": 11, "normalize_coords": True},
)

ODS_THRESHOLDS = drive_mod.ODS_THRESHOLDS


def _set_ods_threshold_count(count: int) -> np.ndarray:
    global ODS_THRESHOLDS
    ODS_THRESHOLDS = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    drive_mod.ODS_THRESHOLDS = ODS_THRESHOLDS
    return ODS_THRESHOLDS


_set_ods_threshold_count(HRF_ODS_THRESHOLD_COUNT)


@dataclass(frozen=True)
class HrfSelection:
    image_key: str
    image_id: str
    condition_class: str
    image_path: str
    label_path: str
    fov_path: str | None
    selection_score: float
    vessel_pixels: int
    orientation_entropy: float


def _build_roster(validation_summary: dict[str, object]) -> list[dict[str, object]]:
    roster = []
    for row in validation_summary.get("method_roster", []):
        config = dict(row["config"])
        roster.append(
            {
                "method": str(row["method"]),
                "label": str(row["label"]),
                "config": config,
                "kernel": build_method(str(row["method"]), **config),
            }
        )
    return roster


def _parse_method_filter(raw: str) -> set[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    return set(values)


def _parse_trace_radii(raw: str) -> set[int]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    return {int(token) for token in values}


def _class_alias(raw: str) -> str | None:
    value = raw.lower()
    if "healthy" in value:
        return "healthy"
    if re.search(r"(^|[_/.-])h($|[_/.-])", value):
        return "healthy"
    if "glau" in value:
        return "glaucoma"
    if re.search(r"(^|[_/.-])g($|[_/.-])", value):
        return "glaucoma"
    if "diabetic" in value or "retinopathy" in value or re.search(r"(^|[^a-z])dr([^a-z]|$)", value):
        return "diabetic_retinopathy"
    return None


def _role_alias(path: Path) -> str | None:
    text = "/".join(part.lower() for part in path.parts)
    stem = path.stem.lower()
    token = text + "/" + stem
    if "fov" in token or "fieldofview" in token or re.search(r"(^|[^a-z])mask([^a-z]|$)", token):
        return "fov"
    if "manual" in token or "gold" in token or "label" in token or "vessel" in token or "ground" in token:
        return "label"
    return "image"


def _discover_zip_links() -> list[str]:
    html = subprocess.check_output(["curl", "-fLs", DATASET_PAGE_URL], text=True)
    links = re.findall(r'href="([^"]+\.zip)"', html, flags=re.IGNORECASE)
    resolved = []
    for href in links:
        if href.startswith("http://") or href.startswith("https://"):
            resolved.append(href)
        else:
            resolved.append(urljoin(DATASET_PAGE_URL, href))
    # preserve order while removing duplicates
    unique: list[str] = []
    seen: set[str] = set()
    for url in resolved:
        if url not in seen:
            unique.append(url)
            seen.add(url)
    return unique


def _extract_zip(archive_path: Path, target_root: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_root)


def _resolve_hrf_root(dataset_root: Path) -> Path | None:
    candidates = [dataset_root]
    if dataset_root.exists():
        candidates.extend(path for path in dataset_root.iterdir() if path.is_dir())
    best: Path | None = None
    best_count = -1
    for candidate in candidates:
        image_count = sum(
            1
            for path in candidate.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
            and _class_alias(str(path)) is not None
        )
        if image_count > best_count:
            best = candidate
            best_count = image_count
    if best is None or best_count <= 0:
        return None
    return best


def _ensure_hrf_root(dataset_root: Path, auto_download: bool) -> Path:
    resolved = _resolve_hrf_root(dataset_root)
    if resolved is not None:
        return resolved
    if not auto_download:
        raise FileNotFoundError(
            f"HRF dataset not found under {dataset_root}. Use --auto-download or point --dataset-root at an extracted HRF tree."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    zip_links = _discover_zip_links()
    preferred = [url for url in zip_links if Path(url).name.lower() == "all.zip"]
    fallback = [url for url in zip_links if "fundus-images" in url.lower() or "fileadmin" in url.lower()]
    download_urls = preferred or fallback
    if not download_urls:
        raise RuntimeError("could not discover any HRF zip archives from the official dataset page")
    for index, url in enumerate(download_urls, start=1):
        archive_path = dataset_root / f"download_{index:02d}_{Path(url).name}"
        if not archive_path.exists():
            subprocess.run(["curl", "-fL", url, "-o", str(archive_path)], check=True, cwd=str(dataset_root))
        _extract_zip(archive_path, dataset_root)
        resolved = _resolve_hrf_root(dataset_root)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(f"Unable to resolve extracted HRF files under {dataset_root}")


def _candidate_pairs(data_root: Path) -> list[HrfSelection]:
    grouped: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            continue
        class_name = _class_alias(str(path))
        role = _role_alias(path)
        if class_name is None or role is None:
            continue
        grouped.setdefault(class_name, {"image": [], "label": [], "fov": []})[role].append(path)

    candidates: list[HrfSelection] = []
    for class_name, buckets in grouped.items():
        images = sorted(buckets["image"])
        labels = sorted(buckets["label"])
        fovs = sorted(buckets["fov"])
        pair_count = min(len(images), len(labels))
        for idx in range(pair_count):
            image_path = images[idx]
            label_path = labels[idx]
            fov_path = fovs[idx] if idx < len(fovs) else None
            image_id = f"{class_name}:{image_path.stem}"
            _, green = _load_drive_input(image_path)
            vessel_mask = _load_vessel_mask(label_path)
            centerline = _centerline_mask(vessel_mask)
            tangent_angles, tangent_valid = _centerline_tangent_angles(centerline)
            vessel_pixels = int(np.sum(vessel_mask))
            entropy = _orientation_entropy_from_angles(tangent_angles, tangent_valid)
            mean_contrast = (
                float(np.std(np.asarray(green, dtype=np.float64)[np.asarray(vessel_mask, dtype=bool)]))
                if vessel_pixels > 0
                else 0.0
            )
            score = float(vessel_pixels) * (0.5 + entropy) * (1.0 + 0.25 * mean_contrast)
            candidates.append(
                HrfSelection(
                    image_key="",
                    image_id=image_id,
                    condition_class=class_name,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    fov_path=None if fov_path is None else str(fov_path),
                    selection_score=score,
                    vessel_pixels=vessel_pixels,
                    orientation_entropy=entropy,
                )
            )
    return candidates


def _select_images(data_root: Path) -> list[HrfSelection]:
    candidates = _candidate_pairs(data_root)
    if not candidates:
        raise RuntimeError("no HRF image/mask pairs were discovered after extraction")
    selected: list[HrfSelection] = []
    for class_name, count in HRF_SELECTION_PLAN:
        class_rows = [row for row in candidates if row.condition_class == class_name]
        if len(class_rows) < int(count):
            raise RuntimeError(f"only found {len(class_rows)} HRF pairs for class {class_name}, expected {count}")
        for row in sorted(class_rows, key=lambda item: item.selection_score, reverse=True)[: int(count)]:
            selected.append(row)
    result = []
    for index, row in enumerate(selected, start=1):
        result.append(
            HrfSelection(
                image_key=f"img{index:02d}",
                image_id=row.image_id,
                condition_class=row.condition_class,
                image_path=row.image_path,
                label_path=row.label_path,
                fov_path=row.fov_path,
                selection_score=row.selection_score,
                vessel_pixels=row.vessel_pixels,
                orientation_entropy=row.orientation_entropy,
            )
        )
    return result


def _selection_from_summary(summary: dict[str, object]) -> list[HrfSelection] | None:
    rows = summary.get("images")
    if not isinstance(rows, list):
        return None
    selections: list[HrfSelection] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        try:
            selections.append(
                HrfSelection(
                    image_key=str(row["image_key"]),
                    image_id=str(row["image_id"]),
                    condition_class=str(row["condition_class"]),
                    image_path=str(row["image_path"]),
                    label_path=str(row["label_path"]),
                    fov_path=None if row.get("fov_path") in (None, "") else str(row.get("fov_path")),
                    selection_score=float(row["selection_score"]),
                    vessel_pixels=int(row["vessel_pixels"]),
                    orientation_entropy=float(row["orientation_entropy"]),
                )
            )
        except KeyError:
            return None
    return selections


def _best_baseline_by_metric(
    methods_payload: dict[str, object],
    snr_slug: str,
    metric_key: str,
    higher_is_better: bool,
) -> dict[str, object]:
    best_method = ""
    best_label = ""
    best_value: float | None = None
    for method_key, method_data in methods_payload.items():
        if str(method_key) == "wvf":
            continue
        value = float(method_data["snr_metrics"][snr_slug][metric_key])
        if best_value is None or (value > best_value if higher_is_better else value < best_value):
            best_value = value
            best_method = str(method_key)
            best_label = str(method_data["label"])
    if best_value is None:
        raise RuntimeError(f"no baseline reference found for {metric_key} at SNR {snr_slug}")
    return {"method": best_method, "label": best_label, "value": float(best_value)}


def _parse_snr_levels(raw: str) -> tuple[float, ...]:
    levels: list[float] = []
    for token in [item.strip() for item in str(raw).split(",") if item.strip()]:
        if token.lower() in {"inf", "infinity"}:
            levels.append(math.inf)
        else:
            levels.append(float(token))
    if not levels:
        raise ValueError("at least one SNR level is required")
    return tuple(levels)


def _snr_slugs(snr_levels: tuple[float, ...]) -> tuple[str, ...]:
    return tuple(_noise_slug(float(value)) for value in snr_levels)


def _classify_optimum_driver(best_by_snr: dict[str, dict[str, object]]) -> dict[str, object]:
    ordered_slugs = ("inf", "20", "10", "5")
    radii = [int(best_by_snr[slug]["radius"]) for slug in ordered_slugs if slug in best_by_snr]
    if not radii:
        return {
            "classification": "unknown",
            "rationale": "no feasible WVF cells were evaluated for the requested SNR levels.",
        }
    if len(set(radii)) == 1:
        radius = int(radii[0])
        return {
            "classification": "bias_upper_bound",
            "rationale": f"the HRF orientation-MAE optimum stayed fixed at r={radius} across clean and noisy runs, so vessel scale dominates the choice.",
        }
    nondecreasing = all(radii[idx] <= radii[idx + 1] for idx in range(len(radii) - 1))
    spread = int(max(radii) - min(radii))
    if spread >= 4 and min(radii) <= 5:
        return {
            "classification": "both",
            "rationale": (
                f"the HRF optimum ranges from r={int(min(radii))} to r={int(max(radii))} across the SNR sweep, "
                "so vessel scale sets the baseline while the variance floor favors wider averaging under noise."
            ),
        }
    return {
        "classification": "variance_lower_bound",
        "rationale": (
            f"the preferred HRF radius varies across SNR levels with a total spread of {spread} px, indicating that the noise floor materially "
            "affects the best vessel-orientation operating point."
        ),
    }


def _select_roster_entries(roster: list[dict[str, object]], method_filter: set[str]) -> list[dict[str, object]]:
    if not method_filter:
        return list(roster)
    return [row for row in roster if str(row["method"]) in method_filter]


def _select_wvf_trace_specs(radii_filter: set[int]) -> list[dict[str, object]]:
    if not radii_filter:
        return [dict(spec) for spec in WVF_TRACE_SPECS]
    return [dict(spec) for spec in WVF_TRACE_SPECS if int(spec["r"]) in radii_filter]


def _decorate_wvf_trace(
    points: list[dict[str, object]],
    methods_payload: dict[str, object],
    snr_levels: tuple[float, ...],
) -> dict[str, object]:
    sorted_points = sorted(points, key=lambda row: (int(row["radius"]), int(row["degree"])))
    if not methods_payload:
        return {"points": sorted_points, "baseline_best": {}}

    baseline_best: dict[str, dict[str, object]] = {}
    best_by_snr: dict[str, dict[str, object]] = {}
    for snr_db in snr_levels:
        snr_slug = _noise_slug(float(snr_db))
        baseline_best[snr_slug] = {}
        for metric_key, higher_is_better in TRACE_METRICS:
            baseline_best[snr_slug][metric_key] = _best_baseline_by_metric(
                methods_payload=methods_payload,
                snr_slug=snr_slug,
                metric_key=metric_key,
                higher_is_better=bool(higher_is_better),
            )

    decorated_points = []
    for point in sorted_points:
        snr_metrics = {}
        for snr_db in snr_levels:
            snr_slug = _noise_slug(float(snr_db))
            metrics = dict(point["snr_metrics"][snr_slug])
            comparisons = {}
            for metric_key, higher_is_better in TRACE_METRICS:
                best = baseline_best[snr_slug][metric_key]
                value = float(metrics[metric_key])
                overtakes = value > float(best["value"]) if higher_is_better else value < float(best["value"])
                comparisons[metric_key] = {
                    "best_baseline_method": str(best["method"]),
                    "best_baseline_label": str(best["label"]),
                    "best_baseline_value": float(best["value"]),
                    "overtakes_best_baseline": bool(int(point["radius"]) < 50 and overtakes),
                }
            snr_metrics[snr_slug] = metrics | {"comparison": comparisons}
            candidate = {
                "radius": int(point["radius"]),
                "degree": int(point["degree"]),
                "value": float(metrics[GRID_PRIMARY_METRIC_KEY]),
                "overtakes_best_baseline": bool(
                    comparisons[GRID_PRIMARY_METRIC_KEY]["overtakes_best_baseline"]
                ),
            }
            current = best_by_snr.get(snr_slug)
            if current is None or float(candidate["value"]) < float(current["value"]):
                best_by_snr[snr_slug] = candidate
        decorated_points.append(dict(point) | {"snr_metrics": snr_metrics})

    optimum = None
    if GRID_PRIMARY_SNR_SLUG in best_by_snr:
        optimum = dict(best_by_snr[GRID_PRIMARY_SNR_SLUG])
        optimum["snr_slug"] = GRID_PRIMARY_SNR_SLUG
        optimum["metric_key"] = GRID_PRIMARY_METRIC_KEY
        optimum["label"] = f"r={int(optimum['radius'])}, d={int(optimum['degree'])}"

    payload = {
        "points": decorated_points,
        "baseline_best": baseline_best,
        "best_by_snr": best_by_snr,
    }
    if optimum is not None:
        payload["annotated_optimum"] = optimum
        payload["driver_assessment"] = _classify_optimum_driver(best_by_snr)
    return payload


def _fixed_wvf_from_trace(points: list[dict[str, object]]) -> dict[str, object] | None:
    for point in points:
        if int(point["radius"]) == 50 and int(point["degree"]) == 11:
            return {
                "label": "WVF",
                "config": {"r": 50, "d": 11, "normalize_coords": True},
                "clean_assets": dict(point.get("clean_assets", {})),
                "snr_metrics": dict(point["snr_metrics"]),
            }
    return None


def _evaluate_wvf_grid(
    green_images: dict[str, np.ndarray],
    soft_boundary_map: dict[str, np.ndarray],
    boundary_normals_map: dict[str, np.ndarray],
    boundary_valid_map: dict[str, np.ndarray],
    tangent_angle_map: dict[str, np.ndarray],
    tangent_valid_map: dict[str, np.ndarray],
    fov_mask_map: dict[str, np.ndarray],
    methods_payload: dict[str, object],
    fft_backend: str,
    device_index: int | None,
    noise_draws: int,
) -> dict[str, object]:
    feasible_cells = feasible_wvf_grid(normalize_coords=True)
    baseline_best: dict[str, dict[str, object]] = {}
    for snr_db in SNR_LEVELS:
        snr_slug = _noise_slug(float(snr_db))
        baseline_best[snr_slug] = {}
        for metric_key, higher_is_better in TRACE_METRICS:
            baseline_best[snr_slug][metric_key] = _best_baseline_by_metric(
                methods_payload=methods_payload,
                snr_slug=snr_slug,
                metric_key=metric_key,
                higher_is_better=bool(higher_is_better),
            )

    cells = []
    best_by_snr: dict[str, dict[str, object]] = {}
    for cell_info in feasible_cells:
        spec = {"r": int(cell_info["radius"]), "d": int(cell_info["degree"]), "normalize_coords": True}
        method_item = {
            "method": "wvf",
            "label": "WVF",
            "config": dict(spec),
            "kernel": build_method("wvf", **spec),
        }
        snr_metrics = {}
        for snr_db in SNR_LEVELS:
            snr_slug = _noise_slug(float(snr_db))
            metrics = _evaluate_snr_bank(
                method_item=method_item,
                green_images=green_images,
                soft_boundary_map=soft_boundary_map,
                boundary_normals_map=boundary_normals_map,
                boundary_valid_map=boundary_valid_map,
                tangent_angle_map=tangent_angle_map,
                tangent_valid_map=tangent_valid_map,
                fov_mask_map=fov_mask_map,
                snr_db=float(snr_db),
                noise_draws=int(noise_draws),
                fft_backend=fft_backend,
                device_index=device_index,
            )
            comparisons = {}
            for metric_key, higher_is_better in TRACE_METRICS:
                best = baseline_best[snr_slug][metric_key]
                value = float(metrics[metric_key])
                overtakes = value > float(best["value"]) if higher_is_better else value < float(best["value"])
                comparisons[metric_key] = {
                    "best_baseline_method": str(best["method"]),
                    "best_baseline_label": str(best["label"]),
                    "best_baseline_value": float(best["value"]),
                    "overtakes_best_baseline": bool(overtakes),
                }
            snr_metrics[snr_slug] = metrics | {"comparison": comparisons}
            candidate = {
                "radius": int(spec["r"]),
                "degree": int(spec["d"]),
                "value": float(metrics[GRID_PRIMARY_METRIC_KEY]),
                "overtakes_best_baseline": bool(
                    comparisons[GRID_PRIMARY_METRIC_KEY]["overtakes_best_baseline"]
                ),
            }
            current = best_by_snr.get(snr_slug)
            if current is None or float(candidate["value"]) < float(current["value"]):
                best_by_snr[snr_slug] = candidate
            print(
                f"sec09HRF-grid r={spec['r']} d={spec['d']} snr={snr_slug} "
                f"rmse={metrics['gradient_vector_rmse_mean']:.6e} ods={metrics['ods_f_score']:.6f} ang={metrics['orientation_mae_deg_mean']:.4f}"
            )
        cells.append(
            {
                "radius": int(spec["r"]),
                "degree": int(spec["d"]),
                "config": dict(spec),
                "support_cardinality": int(cell_info["support_cardinality"]),
                "coefficient_count": int(cell_info["coefficient_count"]),
                "kappa_design_matrix": float(cell_info["kappa_design_matrix"]),
                "sigma_min": float(cell_info["sigma_min"]),
                "rank_deficient_count": int(cell_info["rank_deficient_count"]),
                "white_noise_gain": float(method_item["kernel"].white_noise_gain),
                "snr_metrics": snr_metrics,
            }
        )
    optimum = dict(best_by_snr[GRID_PRIMARY_SNR_SLUG])
    optimum["snr_slug"] = GRID_PRIMARY_SNR_SLUG
    optimum["metric_key"] = GRID_PRIMARY_METRIC_KEY
    optimum["label"] = f"r={int(optimum['radius'])}, d={int(optimum['degree'])}"
    return {
        "primary_metric_key": GRID_PRIMARY_METRIC_KEY,
        "primary_metric_label": "Centerline orientation MAE",
        "primary_snr_slug": GRID_PRIMARY_SNR_SLUG,
        "grid_radii": [int(value) for value in WVF_GRID_RADII],
        "grid_degrees": [int(value) for value in WVF_GRID_DEGREES],
        "cells": cells,
        "best_by_snr": best_by_snr,
        "annotated_optimum": optimum,
        "driver_assessment": _classify_optimum_driver(best_by_snr),
        "conditioning_gate": "Cells are included only when rank_deficient_count == 0 under the scaled-epsilon SVD cutoff.",
    }


def _evaluate_wvf_trace(
    green_images: dict[str, np.ndarray],
    soft_boundary_map: dict[str, np.ndarray],
    boundary_normals_map: dict[str, np.ndarray],
    boundary_valid_map: dict[str, np.ndarray],
    tangent_angle_map: dict[str, np.ndarray],
    tangent_valid_map: dict[str, np.ndarray],
    fov_mask_map: dict[str, np.ndarray],
    trace_specs: list[dict[str, object]],
    assets_dir: Path,
    fft_backend: str,
    device_index: int | None,
    noise_draws: int,
    snr_levels: tuple[float, ...],
    asset_max_width_px: int | None,
) -> list[dict[str, object]]:
    points = []
    for spec in trace_specs:
        method_item = {
            "method": "wvf",
            "label": "WVF",
            "config": dict(spec),
            "kernel": build_method("wvf", **spec),
        }
        clean_assets = _clean_assets_for_method(
            method_item=method_item,
            green_images=green_images,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
            asset_max_width_px=asset_max_width_px,
        )
        snr_metrics = {}
        for snr_db in snr_levels:
            snr_slug = _noise_slug(float(snr_db))
            metrics = _evaluate_snr_bank(
                method_item=method_item,
                green_images=green_images,
                soft_boundary_map=soft_boundary_map,
                boundary_normals_map=boundary_normals_map,
                boundary_valid_map=boundary_valid_map,
                tangent_angle_map=tangent_angle_map,
                tangent_valid_map=tangent_valid_map,
                fov_mask_map=fov_mask_map,
                snr_db=float(snr_db),
                noise_draws=int(noise_draws),
                fft_backend=fft_backend,
                device_index=device_index,
            )
            snr_metrics[snr_slug] = metrics
            print(
                f"sec09HRF-trace r={spec['r']} d={spec['d']} snr={snr_slug} "
                f"rmse={metrics['gradient_vector_rmse_mean']:.6e} ods={metrics['ods_f_score']:.6f} ang={metrics['orientation_mae_deg_mean']:.4f}"
            )
        points.append(
            {
                "radius": int(spec["r"]),
                "degree": int(spec["d"]),
                "config": dict(spec),
                "white_noise_gain": float(method_item["kernel"].white_noise_gain),
                "support_half_extent": int(method_item["kernel"].support_half_extent),
                "clean_assets": clean_assets,
                "snr_metrics": snr_metrics,
            }
        )
    return points


def _best_small_stencil(methods_payload: dict[str, object], snr_slug: str, metric_key: str) -> dict[str, object]:
    best_method = ""
    best_value: float | None = None
    for method_name in SMALL_STENCIL_METHODS:
        if method_name not in methods_payload:
            continue
        value = float(methods_payload[method_name]["snr_metrics"][snr_slug][metric_key])
        if best_value is None or value < best_value:
            best_method = str(method_name)
            best_value = value
    if best_value is None:
        raise RuntimeError(f"no small-stencil baseline found for {metric_key} at SNR {snr_slug}")
    return {"method": best_method, "value": float(best_value)}


def _best_wvf_metric(wvf_payload: dict[str, object], snr_slug: str, metric_key: str) -> dict[str, object]:
    best_point: dict[str, object] | None = None
    points = wvf_payload.get("cells", wvf_payload.get("points", []))
    for point in points:
        value = float(point["snr_metrics"][snr_slug][metric_key])
        if best_point is None or value < float(best_point["value"]):
            best_point = {
                "radius": int(point["radius"]),
                "degree": int(point["degree"]),
                "value": float(value),
            }
    if best_point is None:
        raise RuntimeError(f"no WVF cells found for metric {metric_key} at SNR {snr_slug}")
    return best_point


def _drive_low_res_delta(drive_summary: dict[str, object], snr_slugs: tuple[str, ...]) -> dict[str, object]:
    metrics = {
        "gradient_vector_rmse_mean": "Vector RMSE delta",
        "orientation_mae_deg_mean": "Orientation MAE delta",
    }
    deltas: dict[str, dict[str, object]] = {}
    methods_payload = drive_summary["methods"]
    wvf_grid = drive_summary["wvf_grid"]
    for metric_key, label in metrics.items():
        per_snr = {}
        for snr_slug in snr_slugs:
            best_wvf = _best_wvf_metric(wvf_grid, snr_slug, metric_key)
            best_small = _best_small_stencil(methods_payload, snr_slug, metric_key)
            per_snr[snr_slug] = {
                "best_wvf": best_wvf,
                "best_small_stencil": best_small,
                "delta_small_minus_wvf": float(best_small["value"] - best_wvf["value"]),
            }
        deltas[metric_key] = {"label": label, "per_snr": per_snr}
    return deltas


def _hrf_delta_summary(
    methods_payload: dict[str, object],
    wvf_trace: dict[str, object],
    snr_slugs: tuple[str, ...],
) -> dict[str, object]:
    metrics = {
        "gradient_vector_rmse_mean": "Vector RMSE delta",
        "orientation_mae_deg_mean": "Orientation MAE delta",
    }
    deltas: dict[str, dict[str, object]] = {}
    for metric_key, label in metrics.items():
        per_snr = {}
        for snr_slug in snr_slugs:
            best_wvf = _best_wvf_metric(wvf_trace, snr_slug, metric_key)
            best_small = _best_small_stencil(methods_payload, snr_slug, metric_key)
            per_snr[snr_slug] = {
                "best_wvf": best_wvf,
                "best_small_stencil": best_small,
                "delta_small_minus_wvf": float(best_small["value"] - best_wvf["value"]),
            }
        deltas[metric_key] = {"label": label, "per_snr": per_snr}
    return deltas


def _assemble_summary_payload(
    *,
    validation_summary: dict[str, object],
    data_root: Path,
    drive_summary_json: Path,
    fft_backend: str,
    noise_draws: int,
    snr_levels: tuple[float, ...],
    image_payload: list[dict[str, object]],
    methods_payload: dict[str, object],
    wvf_points: list[dict[str, object]],
    asset_rendering: dict[str, object] | None,
    partial_request: dict[str, object] | None,
) -> dict[str, object]:
    methods_full = dict(methods_payload)
    baseline_methods = {key: value for key, value in methods_full.items() if str(key) != "wvf"}
    if "wvf" not in methods_full:
        fixed_wvf = _fixed_wvf_from_trace(wvf_points)
        if fixed_wvf is not None:
            methods_full["wvf"] = fixed_wvf

    wvf_trace = _decorate_wvf_trace(wvf_points, methods_full if baseline_methods else {}, snr_levels)
    snr_slugs = _snr_slugs(snr_levels)
    method_order = [
        str(row["method"])
        for row in validation_summary.get("method_roster", [])
        if str(row["method"]) in methods_full
    ]
    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "scenario": "B-HD",
        "dataset": {
            "name": "HRF",
            "data_root": str(data_root),
            "page_url": DATASET_PAGE_URL,
            "resolution_px": [3504, 2336],
            "input_channel": "green",
        },
        "config": {
            "snr_levels": list(snr_slugs),
            "noise_draws": int(noise_draws),
            "image_count": int(len(image_payload)),
            "ods_threshold_count": int(ODS_THRESHOLDS.shape[0]),
            "ods_tolerance_px": int(ODS_TOLERANCE_PX),
            "vector_rmse_reference": "manual vessel boundary map derived from segmentation",
            "orientation_reference": "manual vessel centerline tangent direction",
            "fft_backend": str(fft_backend),
            "display_percentile": float(DISPLAY_PERCENTILE),
        },
        "image_order": [str(row["image_key"]) for row in image_payload],
        "images": image_payload,
        "method_order": method_order,
        "methods": methods_full,
        "wvf_trace": wvf_trace,
    }
    if asset_rendering is not None:
        payload["asset_rendering"] = asset_rendering
    if partial_request is not None:
        payload["partial_request"] = partial_request
    if baseline_methods and wvf_trace.get("points"):
        drive_summary = json.loads(drive_summary_json.read_text())
        payload["comparison_to_drive"] = {
            "drive_image_shape_px": [565, 584],
            "drive_summary_json": str(drive_summary_json),
            "drive_deltas": _drive_low_res_delta(drive_summary, snr_slugs),
            "hrf_deltas": _hrf_delta_summary(methods_full, wvf_trace, snr_slugs),
        }
    return payload


def _merge_partial_summaries(
    *,
    validation_summary: dict[str, object],
    drive_summary_json: Path,
    partial_paths: list[Path],
    summary_json: Path,
    fft_backend: str,
    compile_plots: bool,
) -> dict[str, Path]:
    partials = [json.loads(path.read_text()) for path in partial_paths]
    if not partials:
        raise RuntimeError("no shard summaries were provided for merge")
    first = partials[0]
    dataset_root = Path(str(first["dataset"]["data_root"]))
    image_payload = list(first["images"])
    snr_slugs = tuple(str(value) for value in first["config"]["snr_levels"])
    noise_draws = int(first["config"]["noise_draws"])
    ods_threshold_count = int(first["config"]["ods_threshold_count"])
    asset_rendering = first.get("asset_rendering")
    _set_ods_threshold_count(ods_threshold_count)

    merged_methods: dict[str, object] = {}
    merged_points: dict[tuple[int, int], dict[str, object]] = {}
    for payload in partials:
        if list(payload["image_order"]) != list(first["image_order"]):
            raise RuntimeError("HRF shard merge failed because image selections differ")
        if tuple(str(value) for value in payload["config"]["snr_levels"]) != snr_slugs:
            raise RuntimeError("HRF shard merge failed because SNR schedules differ")
        if int(payload["config"]["noise_draws"]) != noise_draws:
            raise RuntimeError("HRF shard merge failed because noise draw counts differ")
        if payload.get("asset_rendering") != asset_rendering:
            raise RuntimeError("HRF shard merge failed because asset rendering settings differ")
        for method_name, method_payload in payload.get("methods", {}).items():
            merged_methods[str(method_name)] = method_payload
        for point in payload.get("wvf_trace", {}).get("points", []):
            merged_points[(int(point["radius"]), int(point["degree"]))] = point

    snr_levels = _parse_snr_levels(",".join(snr_slugs))
    merged_payload = _assemble_summary_payload(
        validation_summary=validation_summary,
        data_root=dataset_root,
        drive_summary_json=drive_summary_json,
        fft_backend=fft_backend,
        noise_draws=noise_draws,
        snr_levels=snr_levels,
        image_payload=image_payload,
        methods_payload=merged_methods,
        wvf_points=list(merged_points.values()),
        asset_rendering=asset_rendering,
        partial_request=None,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(merged_payload, handle, indent=2)
        handle.write("\n")

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_real_image_hrf.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_real_image_hrf.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def _refresh_assets_only_summary(
    *,
    existing_summary: dict[str, object],
    green_images: dict[str, np.ndarray],
    image_payload: list[dict[str, object]],
    assets_dir: Path,
    fft_backend: str,
    device_index: int | None,
    asset_max_width_px: int | None,
    asset_rendering: dict[str, object],
    summary_json: Path,
    compile_plots: bool,
) -> dict[str, Path]:
    for method_name, method_payload in existing_summary.get("methods", {}).items():
        config = dict(method_payload["config"])
        method_item = {
            "method": str(method_name),
            "label": str(method_payload["label"]),
            "config": config,
            "kernel": build_method(str(method_name), **config),
        }
        method_payload["clean_assets"] = _clean_assets_for_method(
            method_item=method_item,
            green_images=green_images,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
            asset_max_width_px=asset_max_width_px,
        )

    for point in existing_summary.get("wvf_trace", {}).get("points", []):
        config = dict(point["config"])
        method_item = {
            "method": "wvf",
            "label": "WVF",
            "config": config,
            "kernel": build_method("wvf", **config),
        }
        point["clean_assets"] = _clean_assets_for_method(
            method_item=method_item,
            green_images=green_images,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
            asset_max_width_px=asset_max_width_px,
        )

    existing_summary["image_order"] = [str(row["image_key"]) for row in image_payload]
    existing_summary["images"] = image_payload
    existing_summary["asset_rendering"] = asset_rendering

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(existing_summary, handle, indent=2)
        handle.write("\n")

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_real_image_hrf.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_real_image_hrf.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def run_experiment(
    validation_json: Path,
    dataset_root: Path,
    drive_summary_json: Path,
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
    noise_draws: int,
    ods_threshold_count: int,
    snr_levels: tuple[float, ...],
    asset_max_width_px: int | None,
    selection_summary_json: Path | None,
    skip_methods: bool,
    method_filter: set[str],
    skip_wvf_trace: bool,
    wvf_trace_radii: set[int],
    auto_download: bool,
) -> dict[str, Path]:
    _set_ods_threshold_count(int(ods_threshold_count))
    validation_summary = json.loads(validation_json.read_text())
    data_root = _ensure_hrf_root(dataset_root, auto_download=bool(auto_download))
    selection_source = selection_summary_json.resolve() if selection_summary_json is not None else summary_json.resolve()
    existing_summary = json.loads(selection_source.read_text()) if selection_source.exists() else None
    existing_selection = None if existing_summary is None else _selection_from_summary(existing_summary)
    selections = existing_selection if existing_selection is not None else _select_images(data_root)
    asset_max_width_px = None if asset_max_width_px is None else int(asset_max_width_px)
    assets_dir_name = "assets" if asset_max_width_px is None else f"assets_w{asset_max_width_px}"
    asset_rendering = {
        "asset_dir_name": assets_dir_name,
        "asset_max_width_px": asset_max_width_px,
        "full_resolution_assets_available": asset_max_width_px is None,
        "note": (
            "When asset_max_width_px is set, saved preview PNGs are downsampled for figure embedding while metrics remain unchanged."
        ),
    }
    refresh_assets_only = (
        asset_max_width_px is not None
        and existing_summary is not None
        and not bool(skip_methods)
        and not bool(method_filter)
        and not bool(skip_wvf_trace)
        and not bool(wvf_trace_radii)
    )
    roster: list[dict[str, object]] = []
    if not refresh_assets_only:
        roster = _build_roster(validation_summary)
        roster = _select_roster_entries(roster, method_filter)

    green_images: dict[str, np.ndarray] = {}
    soft_boundary_map: dict[str, np.ndarray] = {}
    boundary_normals_map: dict[str, np.ndarray] = {}
    boundary_valid_map: dict[str, np.ndarray] = {}
    tangent_angle_map: dict[str, np.ndarray] = {}
    tangent_valid_map: dict[str, np.ndarray] = {}
    fov_mask_map: dict[str, np.ndarray] = {}
    assets_dir = output_dir / assets_dir_name
    image_payload = []

    for selection in selections:
        rgb, green = _load_drive_input(Path(selection.image_path))
        vessel_mask = _load_vessel_mask(Path(selection.label_path))
        if selection.fov_path is not None and Path(selection.fov_path).exists():
            fov_mask = _load_vessel_mask(Path(selection.fov_path))
        else:
            fov_mask = _fov_mask_from_green(green)
        soft_boundary = _boundary_soft_mask(vessel_mask, fov_mask)
        boundary_normals, boundary_valid = _boundary_normal_field(soft_boundary)
        centerline = _centerline_mask(vessel_mask)
        tangent_angles, tangent_valid = _centerline_tangent_angles(centerline)

        green_images[selection.image_key] = np.asarray(green, dtype=np.float32)
        soft_boundary_map[selection.image_key] = np.asarray(soft_boundary, dtype=np.float64)
        boundary_normals_map[selection.image_key] = np.asarray(boundary_normals, dtype=np.float64)
        boundary_valid_map[selection.image_key] = np.asarray(boundary_valid & fov_mask, dtype=bool)
        tangent_angle_map[selection.image_key] = np.asarray(tangent_angles, dtype=np.float64)
        tangent_valid_map[selection.image_key] = np.asarray(tangent_valid & fov_mask, dtype=bool)
        fov_mask_map[selection.image_key] = np.asarray(fov_mask, dtype=bool)

        input_path = assets_dir / f"{selection.image_key}_input.png"
        mask_path = assets_dir / f"{selection.image_key}_vessel_mask.png"
        _save_rgb(input_path, rgb, max_width_px=asset_max_width_px)
        _save_gray(mask_path, np.asarray(vessel_mask, dtype=np.float64), max_width_px=asset_max_width_px)
        image_payload.append(
            {
                "image_key": str(selection.image_key),
                "image_id": str(selection.image_id),
                "condition_class": str(selection.condition_class),
                "image_path": str(selection.image_path),
                "label_path": str(selection.label_path),
                "fov_path": None if selection.fov_path is None else str(selection.fov_path),
                "selection_score": float(selection.selection_score),
                "vessel_pixels": int(selection.vessel_pixels),
                "orientation_entropy": float(selection.orientation_entropy),
                "asset_max_width_px": asset_max_width_px,
                "input_asset_path": str(input_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
                "vessel_mask_asset_path": str(mask_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            }
        )

    if refresh_assets_only:
        return _refresh_assets_only_summary(
            existing_summary=existing_summary,
            green_images=green_images,
            image_payload=image_payload,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
            asset_max_width_px=asset_max_width_px,
            asset_rendering=asset_rendering,
            summary_json=summary_json,
            compile_plots=compile_plots,
        )

    methods_payload = {}
    if not skip_methods:
        for method_item in roster:
            clean_assets = _clean_assets_for_method(
                method_item=method_item,
                green_images=green_images,
                assets_dir=assets_dir,
                fft_backend=fft_backend,
                device_index=device_index,
                asset_max_width_px=asset_max_width_px,
            )
            snr_metrics = {}
            for snr_db in snr_levels:
                slug = _noise_slug(float(snr_db))
                metrics = _evaluate_snr_bank(
                    method_item=method_item,
                    green_images=green_images,
                    soft_boundary_map=soft_boundary_map,
                    boundary_normals_map=boundary_normals_map,
                    boundary_valid_map=boundary_valid_map,
                    tangent_angle_map=tangent_angle_map,
                    tangent_valid_map=tangent_valid_map,
                    fov_mask_map=fov_mask_map,
                    snr_db=float(snr_db),
                    noise_draws=int(noise_draws),
                    fft_backend=fft_backend,
                    device_index=device_index,
                )
                snr_metrics[slug] = metrics
                print(
                    f"sec09HRF {method_item['method']} snr={slug} rmse={metrics['gradient_vector_rmse_mean']:.6e} "
                    f"ods={metrics['ods_f_score']:.6f} ang={metrics['orientation_mae_deg_mean']:.4f}"
                )
            methods_payload[str(method_item["method"])] = {
                "label": str(method_item["label"]),
                "config": dict(method_item["config"]),
                "clean_assets": clean_assets,
                "snr_metrics": snr_metrics,
            }

    trace_specs = [] if skip_wvf_trace else _select_wvf_trace_specs(wvf_trace_radii)
    wvf_points = _evaluate_wvf_trace(
        green_images=green_images,
        soft_boundary_map=soft_boundary_map,
        boundary_normals_map=boundary_normals_map,
        boundary_valid_map=boundary_valid_map,
        tangent_angle_map=tangent_angle_map,
        tangent_valid_map=tangent_valid_map,
        fov_mask_map=fov_mask_map,
        trace_specs=trace_specs,
        assets_dir=assets_dir,
        fft_backend=fft_backend,
        device_index=device_index,
        noise_draws=int(noise_draws),
        snr_levels=snr_levels,
        asset_max_width_px=asset_max_width_px,
    )

    partial_request = {
        "skip_methods": bool(skip_methods),
        "method_filter": sorted(method_filter),
        "skip_wvf_trace": bool(skip_wvf_trace),
        "wvf_trace_radii": sorted(int(value) for value in wvf_trace_radii),
    }
    payload = _assemble_summary_payload(
        validation_summary=validation_summary,
        data_root=data_root,
        drive_summary_json=drive_summary_json,
        fft_backend=fft_backend,
        noise_draws=int(noise_draws),
        snr_levels=snr_levels,
        image_payload=image_payload,
        methods_payload=methods_payload,
        wvf_points=wvf_points,
        asset_rendering=asset_rendering,
        partial_request=partial_request,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_real_image_hrf.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_real_image_hrf.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_baseline_validation" / "sec08_baseline_validation_summary.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "HRF",
    )
    parser.add_argument(
        "--drive-summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_drive" / "sec09_real_image_drive_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_hrf",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_hrf" / "sec09_real_image_hrf_summary.json",
    )
    parser.add_argument("--selection-summary-json", type=Path, default=None)
    parser.add_argument("--fft-backend", type=str, default="vkfft", choices=("vkfft", "cpu"))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--compile-plots", action="store_true")
    parser.add_argument("--noise-draws", type=int, default=HRF_NOISE_DRAWS)
    parser.add_argument("--ods-threshold-count", type=int, default=HRF_ODS_THRESHOLD_COUNT)
    parser.add_argument("--snr-levels", type=str, default="inf,20,10,5")
    parser.add_argument("--asset-max-width-px", type=int, default=None)
    parser.add_argument("--skip-methods", action="store_true")
    parser.add_argument("--method-filter", type=str, default="")
    parser.add_argument("--skip-wvf-trace", action="store_true")
    parser.add_argument("--wvf-trace-radii", type=str, default="")
    parser.add_argument("--merge-shard-jsons", type=Path, nargs="+", default=None)
    parser.add_argument("--auto-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_shard_jsons:
        validation_summary = json.loads(args.validation_json.resolve().read_text())
        _merge_partial_summaries(
            validation_summary=validation_summary,
            drive_summary_json=args.drive_summary_json.resolve(),
            partial_paths=[path.resolve() for path in args.merge_shard_jsons],
            summary_json=args.summary_json.resolve(),
            fft_backend=str(args.fft_backend),
            compile_plots=bool(args.compile_plots),
        )
        return 0
    run_experiment(
        validation_json=args.validation_json.resolve(),
        dataset_root=args.dataset_root.resolve(),
        drive_summary_json=args.drive_summary_json.resolve(),
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        selection_summary_json=None if args.selection_summary_json is None else args.selection_summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
        noise_draws=int(args.noise_draws),
        ods_threshold_count=int(args.ods_threshold_count),
        snr_levels=_parse_snr_levels(str(args.snr_levels)),
        asset_max_width_px=args.asset_max_width_px,
        skip_methods=bool(args.skip_methods),
        method_filter=_parse_method_filter(str(args.method_filter)),
        skip_wvf_trace=bool(args.skip_wvf_trace),
        wvf_trace_radii=_parse_trace_radii(str(args.wvf_trace_radii)),
        auto_download=bool(args.auto_download),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
