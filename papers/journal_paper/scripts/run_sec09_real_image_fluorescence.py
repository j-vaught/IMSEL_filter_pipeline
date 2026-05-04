#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_method
from sec09_wvf_grid import WVF_GRID_DEGREES, WVF_GRID_RADII, feasible_wvf_grid
from section8_common import apply_images_batched, compile_plot


TITLE = "Section 9 Scenario C fluorescence microscopy"
SUBTITLE = "BBBC039 real-image comparison on native noisy fluorescence microscopy"
DATASET_URL = "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"
MIN_ARCHIVE_BYTES = 10_000_000
IMAGE_COUNT = 5
BACKGROUND_PERCENTILE = 35.0
DISPLAY_PERCENTILE = 99.5
EPS = 1.0e-12
WVF_TRACE_SPECS = (
    {"r": 3, "d": 5, "normalize_coords": True},
    {"r": 5, "d": 9, "normalize_coords": True},
    {"r": 9, "d": 11, "normalize_coords": True},
    {"r": 15, "d": 11, "normalize_coords": True},
    {"r": 25, "d": 11, "normalize_coords": True},
    {"r": 50, "d": 11, "normalize_coords": True},
)
TRACE_METRICS = (
    "white_noise_gain",
    "background_gradient_mad_mean",
    "background_gradient_median_mean",
)
GRID_PRIMARY_METRIC_KEY = "background_gradient_mad_mean"


@dataclass(frozen=True)
class FluorSelection:
    image_key: str
    image_name: str
    tif_path: str
    selection_score: float
    intensity_mean: float
    intensity_std: float
    entropy: float


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


def _fluor_selection_from_summary(summary: dict[str, object], data_root: Path) -> list[FluorSelection] | None:
    rows = summary.get("images")
    if not isinstance(rows, list):
        return None
    path_map = {path.stem: path for path in data_root.glob("*.tif")}
    selections: list[FluorSelection] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        image_name = str(row["image_name"])
        tif_path = path_map.get(image_name)
        if tif_path is None:
            return None
        try:
            selections.append(
                FluorSelection(
                    image_key=str(row["image_key"]),
                    image_name=image_name,
                    tif_path=str(tif_path),
                    selection_score=float(row["selection_score"]),
                    intensity_mean=float(row["intensity_mean"]),
                    intensity_std=float(row["intensity_std"]),
                    entropy=float(row["entropy"]),
                )
            )
        except KeyError:
            return None
    return selections


def _resolve_bbbc_root(dataset_root: Path) -> Path | None:
    candidates = [dataset_root]
    if dataset_root.exists():
        candidates.extend(path for path in dataset_root.iterdir() if path.is_dir())
    best: Path | None = None
    best_count = -1
    for candidate in candidates:
        tif_count = sum(1 for _ in candidate.glob("*.tif"))
        if tif_count > best_count:
            best = candidate
            best_count = tif_count
    if best is None or best_count <= 0:
        return None
    return best


def _ensure_bbbc_root(dataset_root: Path, auto_download: bool) -> Path:
    resolved = _resolve_bbbc_root(dataset_root)
    if resolved is not None and sum(1 for _ in resolved.glob("*.tif")) >= int(IMAGE_COUNT):
        return resolved
    if not auto_download:
        raise FileNotFoundError(
            f"BBBC039 TIFFs not found under {dataset_root}. "
            "Use --auto-download or point --dataset-root at an extracted BBBC039 image directory."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / "images.zip"
    if (not archive_path.exists()) or archive_path.stat().st_size < int(MIN_ARCHIVE_BYTES):
        subprocess.run(["curl", "-fL", DATASET_URL, "-o", str(archive_path)], check=True, cwd=str(dataset_root))
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_root)
    resolved = _resolve_bbbc_root(dataset_root)
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve extracted BBBC039 TIFFs under {dataset_root}")
    return resolved


def _load_tif(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., 0]
    maximum = float(np.max(array))
    if maximum <= EPS:
        return np.zeros_like(array, dtype=np.float32)
    return np.asarray(array / maximum, dtype=np.float32)


def _save_gray(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    percentile = float(np.percentile(finite, DISPLAY_PERCENTILE))
    maximum = float(np.max(finite))
    return percentile if percentile > EPS else max(maximum, 1.0)


def _normalize_magnitude(magnitude: np.ndarray) -> tuple[np.ndarray, float]:
    scale = _robust_scale(magnitude)
    return np.clip(np.asarray(magnitude, dtype=np.float64) / float(scale), 0.0, 1.0), float(scale)


def _orientation_rgb(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    value, _ = _normalize_magnitude(magnitude)
    hue = np.mod(np.arctan2(np.asarray(gy, dtype=np.float64), np.asarray(gx, dtype=np.float64)), math.pi) / math.pi
    saturation = np.ones_like(value, dtype=np.float64)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    frac = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    rgb = np.zeros(value.shape + (3,), dtype=np.float64)
    for idx, comps in enumerate(
        (
            (value, t, p),
            (q, value, p),
            (p, value, t),
            (p, q, value),
            (t, p, value),
            (value, p, q),
        )
    ):
        mask = sector == idx
        rgb[mask, 0] = comps[0][mask]
        rgb[mask, 1] = comps[1][mask]
        rgb[mask, 2] = comps[2][mask]
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _intensity_entropy(image: np.ndarray) -> float:
    hist, _ = np.histogram(np.asarray(image, dtype=np.float64), bins=32, range=(0.0, 1.0))
    probs = hist.astype(np.float64) / max(float(np.sum(hist)), EPS)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(32.0))


def _select_images(data_root: Path, image_count: int) -> list[FluorSelection]:
    candidates: list[FluorSelection] = []
    for tif_path in sorted(data_root.glob("*.tif")):
        image = _load_tif(tif_path)
        mean_intensity = float(np.mean(np.asarray(image, dtype=np.float64)))
        std_intensity = float(np.std(np.asarray(image, dtype=np.float64)))
        entropy = _intensity_entropy(image)
        score = float((std_intensity / max(mean_intensity + 0.03, EPS)) * (0.5 + entropy))
        candidates.append(
            FluorSelection(
                image_key=f"img{len(candidates) + 1:02d}",
                image_name=tif_path.stem,
                tif_path=str(tif_path),
                selection_score=float(score),
                intensity_mean=float(mean_intensity),
                intensity_std=float(std_intensity),
                entropy=float(entropy),
            )
        )
    if len(candidates) < int(image_count):
        raise RuntimeError(f"only found {len(candidates)} BBBC039 images, expected at least {image_count}")
    selected = sorted(candidates, key=lambda item: item.selection_score, reverse=True)[: int(image_count)]
    result = []
    for index, item in enumerate(selected, start=1):
        result.append(
            FluorSelection(
                image_key=f"img{index:02d}",
                image_name=item.image_name,
                tif_path=item.tif_path,
                selection_score=item.selection_score,
                intensity_mean=item.intensity_mean,
                intensity_std=item.intensity_std,
                entropy=item.entropy,
            )
        )
    return result


def _background_stats(magnitude: np.ndarray, image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(magnitude, dtype=np.float64)
    background_mask = np.asarray(image, dtype=np.float64) <= float(np.percentile(np.asarray(image, dtype=np.float64), BACKGROUND_PERCENTILE))
    background = values[np.asarray(background_mask, dtype=bool)]
    if background.size == 0:
        return 0.0, 0.0
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    return median, mad


def _clean_assets_for_method(
    method_item: dict[str, object],
    images: dict[str, np.ndarray],
    assets_dir: Path,
    fft_backend: str,
    device_index: int | None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, float]]]:
    kernel = method_item["kernel"]
    image_keys = list(images.keys())
    bank = [np.asarray(images[image_key], dtype=np.float32) for image_key in image_keys]
    responses = apply_images_batched(bank, kernel, fft_backend, device_index)
    assets: dict[str, dict[str, str]] = {}
    stats: dict[str, dict[str, float]] = {}
    for image_key, (gx, gy) in zip(image_keys, responses, strict=True):
        magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        mag_norm, _ = _normalize_magnitude(magnitude)
        mag_path = assets_dir / f"{method_item['method']}_{image_key}_magnitude.png"
        ori_path = assets_dir / f"{method_item['method']}_{image_key}_orientation.png"
        _save_gray(mag_path, mag_norm)
        _save_rgb(ori_path, _orientation_rgb(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)))
        bg_median, bg_mad = _background_stats(magnitude, np.asarray(images[image_key], dtype=np.float64))
        assets[image_key] = {
            "magnitude_path": str(mag_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            "orientation_path": str(ori_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
        }
        stats[image_key] = {
            "background_gradient_median": float(bg_median),
            "background_gradient_mad": float(bg_mad),
        }
    return assets, stats


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _best_baseline_by_metric(methods_payload: dict[str, object], metric_key: str) -> dict[str, object]:
    best_method = ""
    best_label = ""
    best_value: float | None = None
    for method_key, method_data in methods_payload.items():
        if str(method_key) == "wvf":
            continue
        value = float(method_data[metric_key])
        if best_value is None or value < best_value:
            best_value = value
            best_method = str(method_key)
            best_label = str(method_data["label"])
    if best_value is None:
        raise RuntimeError(f"no baseline reference found for {metric_key}")
    return {
        "method": best_method,
        "label": best_label,
        "value": float(best_value),
    }


def _classify_optimum_driver(optimum_radius: int) -> dict[str, object]:
    radius = int(optimum_radius)
    if radius <= 5:
        return {
            "classification": "bias_upper_bound",
            "rationale": f"the native-noise fluorescence optimum lands at a very narrow support r={radius}, indicating that fine feature scale dominates over additional averaging.",
        }
    if radius >= 25:
        return {
            "classification": "variance_lower_bound",
            "rationale": f"the native-noise fluorescence optimum stays in the widest tested regime at r={radius}, indicating that noise averaging dominates this modality.",
        }
    return {
        "classification": "both",
        "rationale": (
            f"the native-noise fluorescence optimum is intermediate at r={radius}, which is consistent with a compromise between fine transition scale and the modality's empirical noise floor."
        ),
    }


def _evaluate_wvf_grid(
    images: dict[str, np.ndarray],
    methods_payload: dict[str, object],
    fft_backend: str,
    device_index: int | None,
) -> dict[str, object]:
    feasible_cells = feasible_wvf_grid(normalize_coords=True)
    baseline_best = {metric_key: _best_baseline_by_metric(methods_payload, metric_key) for metric_key in TRACE_METRICS}
    assets_dir = ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_fluorescence" / "assets"
    cells = []
    optimum: dict[str, object] | None = None
    for cell_info in feasible_cells:
        spec = {
            "r": int(cell_info["radius"]),
            "d": int(cell_info["degree"]),
            "normalize_coords": True,
        }
        method_item = {
            "method": "wvf",
            "label": "WVF",
            "config": dict(spec),
            "kernel": build_method("wvf", **spec),
        }
        _, clean_stats = _clean_assets_for_method(
            method_item=method_item,
            images=images,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        bg_medians = [float(clean_stats[key]["background_gradient_median"]) for key in clean_stats]
        bg_mads = [float(clean_stats[key]["background_gradient_mad"]) for key in clean_stats]
        cell = {
            "radius": int(spec["r"]),
            "degree": int(spec["d"]),
            "config": dict(spec),
            "support_cardinality": int(cell_info["support_cardinality"]),
            "coefficient_count": int(cell_info["coefficient_count"]),
            "kappa_design_matrix": float(cell_info["kappa_design_matrix"]),
            "sigma_min": float(cell_info["sigma_min"]),
            "rank_deficient_count": int(cell_info["rank_deficient_count"]),
            "white_noise_gain": float(method_item["kernel"].white_noise_gain),
            "background_gradient_median_mean": float(np.mean(np.asarray(bg_medians, dtype=np.float64))),
            "background_gradient_mad_mean": float(np.mean(np.asarray(bg_mads, dtype=np.float64))),
        }
        comparison = {}
        for metric_key in TRACE_METRICS:
            best = baseline_best[metric_key]
            value = float(cell[metric_key])
            comparison[metric_key] = {
                "best_baseline_method": str(best["method"]),
                "best_baseline_label": str(best["label"]),
                "best_baseline_value": float(best["value"]),
                "overtakes_best_baseline": bool(value < float(best["value"])),
            }
        cell["comparison"] = comparison
        if optimum is None or float(cell[GRID_PRIMARY_METRIC_KEY]) < float(optimum["value"]):
            optimum = {
                "radius": int(spec["r"]),
                "degree": int(spec["d"]),
                "value": float(cell[GRID_PRIMARY_METRIC_KEY]),
                "overtakes_best_baseline": bool(comparison[GRID_PRIMARY_METRIC_KEY]["overtakes_best_baseline"]),
            }
        print(
            f"sec09C-grid r={spec['r']} d={spec['d']} "
            f"wng={cell['white_noise_gain']:.6e} "
            f"bgmad={cell['background_gradient_mad_mean']:.6e} "
            f"bgmed={cell['background_gradient_median_mean']:.6e}"
        )
        cells.append(cell)
    if optimum is None:
        raise RuntimeError("no feasible WVF grid cells were evaluated for fluorescence scenario")
    optimum["metric_key"] = GRID_PRIMARY_METRIC_KEY
    optimum["label"] = f"r={int(optimum['radius'])}, d={int(optimum['degree'])}"
    return {
        "primary_metric_key": GRID_PRIMARY_METRIC_KEY,
        "primary_metric_label": "Background gradient MAD",
        "grid_radii": [int(value) for value in WVF_GRID_RADII],
        "grid_degrees": [int(value) for value in WVF_GRID_DEGREES],
        "cells": cells,
        "annotated_optimum": optimum,
        "driver_assessment": _classify_optimum_driver(int(optimum["radius"])),
        "conditioning_gate": "Cells are included only when rank_deficient_count == 0 under the scaled-epsilon SVD cutoff.",
    }


def _evaluate_wvf_trace(
    images: dict[str, np.ndarray],
    methods_payload: dict[str, object],
    fft_backend: str,
    device_index: int | None,
) -> dict[str, object]:
    baseline_best = {metric_key: _best_baseline_by_metric(methods_payload, metric_key) for metric_key in TRACE_METRICS}
    assets_dir = ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_fluorescence" / "assets"
    points = []
    for spec in WVF_TRACE_SPECS:
        method_item = {
            "method": "wvf",
            "label": "WVF",
            "config": dict(spec),
            "kernel": build_method("wvf", **spec),
        }
        _, clean_stats = _clean_assets_for_method(
            method_item=method_item,
            images=images,
            assets_dir=assets_dir,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        bg_medians = [float(clean_stats[key]["background_gradient_median"]) for key in clean_stats]
        bg_mads = [float(clean_stats[key]["background_gradient_mad"]) for key in clean_stats]
        point = {
            "radius": int(spec["r"]),
            "degree": int(spec["d"]),
            "config": dict(spec),
            "white_noise_gain": float(method_item["kernel"].white_noise_gain),
            "background_gradient_median_mean": float(np.mean(np.asarray(bg_medians, dtype=np.float64))),
            "background_gradient_mad_mean": float(np.mean(np.asarray(bg_mads, dtype=np.float64))),
        }
        comparison = {}
        for metric_key in TRACE_METRICS:
            best = baseline_best[metric_key]
            value = float(point[metric_key])
            comparison[metric_key] = {
                "best_baseline_method": str(best["method"]),
                "best_baseline_label": str(best["label"]),
                "best_baseline_value": float(best["value"]),
                "overtakes_best_baseline": bool(spec["r"] < 50 and value < float(best["value"])),
            }
        point["comparison"] = comparison
        print(
            f"sec09C-trace r={spec['r']} d={spec['d']} "
            f"wng={point['white_noise_gain']:.6e} "
            f"bgmad={point['background_gradient_mad_mean']:.6e} "
            f"bgmed={point['background_gradient_median_mean']:.6e}"
        )
        points.append(point)
    return {
        "points": points,
        "baseline_best": baseline_best,
    }


def run_experiment(
    validation_json: Path,
    dataset_root: Path,
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
    auto_download: bool,
    image_count: int,
) -> dict[str, Path]:
    validation_summary = json.loads(validation_json.read_text())
    roster = _build_roster(validation_summary)
    data_root = _ensure_bbbc_root(dataset_root, auto_download=bool(auto_download))
    existing_summary = json.loads(summary_json.read_text()) if summary_json.exists() else None
    existing_selection = None if existing_summary is None else _fluor_selection_from_summary(existing_summary, data_root)
    selections = existing_selection if existing_selection is not None else _select_images(data_root, int(image_count))

    images: dict[str, np.ndarray] = {}
    assets_dir = output_dir / "assets"
    image_payload = []
    for selection in selections:
        image = _load_tif(Path(selection.tif_path))
        images[selection.image_key] = np.asarray(image, dtype=np.float32)
        input_path = assets_dir / f"{selection.image_key}_input.png"
        _save_gray(input_path, np.asarray(image, dtype=np.float64))
        image_payload.append(
            {
                "image_key": str(selection.image_key),
                "image_name": str(selection.image_name),
                "selection_score": float(selection.selection_score),
                "intensity_mean": float(selection.intensity_mean),
                "intensity_std": float(selection.intensity_std),
                "entropy": float(selection.entropy),
                "input_asset_path": str(input_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            }
        )

    if existing_summary is not None and isinstance(existing_summary.get("methods"), dict):
        methods_payload = existing_summary["methods"]
    else:
        methods_payload = {}
        for method_item in roster:
            clean_assets, clean_stats = _clean_assets_for_method(
                method_item=method_item,
                images=images,
                assets_dir=assets_dir,
                fft_backend=fft_backend,
                device_index=device_index,
            )
            bg_medians = [float(clean_stats[key]["background_gradient_median"]) for key in clean_stats]
            bg_mads = [float(clean_stats[key]["background_gradient_mad"]) for key in clean_stats]
            print(
                f"sec09C {method_item['method']} "
                f"wng={float(method_item['kernel'].white_noise_gain):.6e} "
                f"bgmad={float(np.mean(np.asarray(bg_mads, dtype=np.float64))):.6e}"
            )
            methods_payload[str(method_item["method"])] = {
                "label": str(method_item["label"]),
                "config": dict(method_item["config"]),
                "white_noise_gain": float(method_item["kernel"].white_noise_gain),
                "background_gradient_median_mean": float(np.mean(np.asarray(bg_medians, dtype=np.float64))),
                "background_gradient_mad_mean": float(np.mean(np.asarray(bg_mads, dtype=np.float64))),
                "clean_assets": clean_assets,
                "background_stability": {
                    "per_image": clean_stats,
                },
            }

    wvf_trace = _evaluate_wvf_trace(
        images=images,
        methods_payload=methods_payload,
        fft_backend=fft_backend,
        device_index=device_index,
    )
    wvf_grid = _evaluate_wvf_grid(
        images=images,
        methods_payload=methods_payload,
        fft_backend=fft_backend,
        device_index=device_index,
    )

    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "scenario": "C",
        "dataset": {
            "name": "BBBC039",
            "data_root": str(data_root),
            "modality": "Hoechst fluorescence microscopy",
        },
        "config": {
            "image_count": int(image_count),
            "background_percentile": float(BACKGROUND_PERCENTILE),
            "visual_stability_proxy": "background gradient median absolute deviation on darkest percentile pixels",
            "fft_backend": str(fft_backend),
        },
        "image_order": [str(row["image_key"]) for row in image_payload],
        "images": image_payload,
        "method_order": [str(method_item["method"]) for method_item in roster],
        "methods": methods_payload,
        "wvf_trace": wvf_trace,
        "wvf_grid": wvf_grid,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {
        "summary_json": summary_json,
    }
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_real_image_fluorescence.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_real_image_fluorescence.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_baseline_validation" / "sec08_baseline_validation_summary.json",
        help="Path to the Section 8.1 validation summary JSON.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "BBBC039",
        help="Root directory containing BBBC039 images or a location where they can be downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_fluorescence",
        help="Directory for JSON summaries and image assets.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_fluorescence" / "sec09_real_image_fluorescence_summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", choices=("vkfft", "cpu"), help="FFT backend to use for fluorescence filtering.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figure.")
    parser.add_argument("--auto-download", action="store_true", help="Download BBBC039 if the dataset root is missing or incomplete.")
    parser.add_argument("--image-count", type=int, default=IMAGE_COUNT, help="Number of fluorescence images to select.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(
        validation_json=args.validation_json.resolve(),
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
        auto_download=bool(args.auto_download),
        image_count=int(args.image_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
