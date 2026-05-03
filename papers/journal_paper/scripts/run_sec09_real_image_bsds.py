#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import io as sio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_method
from section8_common import apply_images_batched, compile_plot


TITLE = "Section 9 Scenario A natural images"
SUBTITLE = "BSDS500 real-image comparison with validation-tuned classical baselines"
DATASET_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
MIN_ARCHIVE_BYTES = 1_000_000
SNR_LEVELS = (math.inf, 20.0, 10.0, 5.0)
NOISE_DRAWS = 10
NOISE_SEED_BASE = 9100
ODS_TOLERANCE_PX = 3
ODS_THRESHOLDS = np.linspace(0.0, 1.0, 201, dtype=np.float64)
CROP_SIZE_PX = 160
CROP_COUNT = 5
CROP_STRIDE_PX = 48
BOUNDARY_WEIGHT_MIN = 0.05
NORMAL_WINDOW_RADIUS = 3
SELECTION_BINS = 12
SELECTION_EDGE_MASS_MIN = 80.0
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
    ("ods_f_score", True),
    ("gradient_vector_rmse_mean", False),
)


@dataclass(frozen=True)
class CropSelection:
    crop_key: str
    label: str
    image_id: str
    x0: int
    y0: int
    width: int
    height: int
    selection_score: float
    edge_mass: float
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


def _crop_selection_from_summary(summary: dict[str, object]) -> list[CropSelection] | None:
    rows = summary.get("crops")
    if not isinstance(rows, list):
        return None
    selections: list[CropSelection] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        try:
            selections.append(
                CropSelection(
                    crop_key=str(row["crop_key"]),
                    label=str(row["label"]),
                    image_id=str(row["image_id"]),
                    x0=int(row["x0"]),
                    y0=int(row["y0"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    selection_score=float(row["selection_score"]),
                    edge_mass=float(row["edge_mass"]),
                    orientation_entropy=float(row["orientation_entropy"]),
                )
            )
        except KeyError:
            return None
    return selections


def _noise_slug(snr_db: float) -> str:
    if math.isinf(float(snr_db)):
        return "inf"
    return f"{float(snr_db):g}".replace(".", "p")


def _resolve_bsds_data_root(dataset_root: Path) -> Path | None:
    candidates = (
        dataset_root,
        dataset_root / "data",
        dataset_root / "BSDS500" / "data",
        dataset_root / "BSDS500" / "BSDS500" / "data",
        dataset_root / "BSR" / "BSDS500" / "data",
    )
    best_path: Path | None = None
    best_count = -1
    for candidate in candidates:
        image_dir = candidate / "images" / "test"
        gt_dir = candidate / "groundTruth" / "test"
        if not image_dir.is_dir() or not gt_dir.is_dir():
            continue
        image_count = sum(1 for _ in image_dir.glob("*.jpg"))
        gt_count = sum(1 for _ in gt_dir.glob("*.mat"))
        count = min(image_count, gt_count)
        if count > best_count:
            best_count = count
            best_path = candidate
    return best_path


def _ensure_bsds_data_root(dataset_root: Path, auto_download: bool) -> Path:
    resolved = _resolve_bsds_data_root(dataset_root)
    if resolved is not None:
        count = sum(1 for _ in (resolved / "images" / "test").glob("*.jpg"))
        if count >= 5:
            return resolved
    if not auto_download:
        raise FileNotFoundError(
            f"BSDS500 test split not found under {dataset_root}. "
            "Use --auto-download or point --dataset-root at a full BSDS500 install."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / "BSR_bsds500.tgz"
    if (not archive_path.exists()) or archive_path.stat().st_size < int(MIN_ARCHIVE_BYTES):
        subprocess.run(
            ["curl", "-fL", DATASET_URL, "-o", str(archive_path)],
            check=True,
            cwd=str(dataset_root),
        )
    subprocess.run(
        ["tar", "-xzf", str(archive_path), "-C", str(dataset_root)],
        check=True,
        cwd=str(dataset_root),
    )
    resolved = _resolve_bsds_data_root(dataset_root)
    if resolved is None:
        raise FileNotFoundError(f"Unable to resolve BSDS500 after downloading into {dataset_root}")
    return resolved


def _load_soft_boundaries(gt_path: Path) -> np.ndarray:
    mat = sio.loadmat(gt_path)
    gt_all = mat["groundTruth"]
    boundaries = [gt_all[0, idx]["Boundaries"][0, 0].astype(np.float64) for idx in range(gt_all.shape[1])]
    return np.mean(boundaries, axis=0)


def _load_image_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    return rgb, gray


def _orientation_entropy(soft_gt: np.ndarray) -> tuple[float, float]:
    smooth = ndimage.gaussian_filter(np.asarray(soft_gt, dtype=np.float64), sigma=1.0, mode="reflect")
    gx = ndimage.sobel(smooth, axis=1, mode="reflect")
    gy = ndimage.sobel(smooth, axis=0, mode="reflect")
    weights = np.hypot(gx, gy)
    weight_sum = float(np.sum(weights))
    if weight_sum <= EPS:
        return 0.0, 0.0
    orientation = np.mod(np.arctan2(gy, gx), math.pi)
    hist, _ = np.histogram(
        orientation,
        bins=int(SELECTION_BINS),
        range=(0.0, math.pi),
        weights=weights,
    )
    hist_sum = float(np.sum(hist))
    if hist_sum <= EPS:
        return 0.0, weight_sum
    probs = hist / hist_sum
    probs = probs[probs > 0.0]
    entropy = float(-np.sum(probs * np.log(probs)) / math.log(float(SELECTION_BINS)))
    return entropy, weight_sum


def _select_crops(data_root: Path, crop_size: int, crop_count: int, stride: int) -> list[CropSelection]:
    image_dir = data_root / "images" / "test"
    gt_dir = data_root / "groundTruth" / "test"
    candidates: list[CropSelection] = []
    image_ids = sorted(path.stem for path in image_dir.glob("*.jpg"))
    for image_id in image_ids:
        soft_gt = _load_soft_boundaries(gt_dir / f"{image_id}.mat")
        height, width = soft_gt.shape
        for y0 in range(0, max(1, height - crop_size + 1), int(stride)):
            for x0 in range(0, max(1, width - crop_size + 1), int(stride)):
                y1 = y0 + int(crop_size)
                x1 = x0 + int(crop_size)
                if y1 > height or x1 > width:
                    continue
                crop_gt = soft_gt[y0:y1, x0:x1]
                edge_mass = float(np.sum(crop_gt))
                if edge_mass < float(SELECTION_EDGE_MASS_MIN):
                    continue
                entropy, grad_weight = _orientation_entropy(crop_gt)
                score = edge_mass * (0.5 + entropy) * (1.0 + 0.05 * math.log1p(max(grad_weight, 0.0)))
                crop_index = len(candidates) + 1
                candidates.append(
                    CropSelection(
                        crop_key=f"crop{crop_index:02d}",
                        label=f"{image_id}",
                        image_id=image_id,
                        x0=int(x0),
                        y0=int(y0),
                        width=int(crop_size),
                        height=int(crop_size),
                        selection_score=float(score),
                        edge_mass=float(edge_mass),
                        orientation_entropy=float(entropy),
                    )
                )
    if not candidates:
        raise RuntimeError("crop selector did not find any BSDS500 test crops with usable boundary mass")
    selected: list[CropSelection] = []
    used_images: set[str] = set()
    for record in sorted(candidates, key=lambda item: item.selection_score, reverse=True):
        if record.image_id in used_images:
            continue
        selected.append(
            CropSelection(
                crop_key=f"crop{len(selected) + 1:02d}",
                label=f"{record.image_id}-{len(selected) + 1}",
                image_id=record.image_id,
                x0=record.x0,
                y0=record.y0,
                width=record.width,
                height=record.height,
                selection_score=record.selection_score,
                edge_mass=record.edge_mass,
                orientation_entropy=record.orientation_entropy,
            )
        )
        used_images.add(record.image_id)
        if len(selected) == int(crop_count):
            break
    if len(selected) < int(crop_count):
        for record in sorted(candidates, key=lambda item: item.selection_score, reverse=True):
            if any(
                record.image_id == existing.image_id
                and abs(record.x0 - existing.x0) < int(crop_size)
                and abs(record.y0 - existing.y0) < int(crop_size)
                for existing in selected
            ):
                continue
            selected.append(
                CropSelection(
                    crop_key=f"crop{len(selected) + 1:02d}",
                    label=f"{record.image_id}-{len(selected) + 1}",
                    image_id=record.image_id,
                    x0=record.x0,
                    y0=record.y0,
                    width=record.width,
                    height=record.height,
                    selection_score=record.selection_score,
                    edge_mass=record.edge_mass,
                    orientation_entropy=record.orientation_entropy,
                )
            )
            if len(selected) == int(crop_count):
                break
    if len(selected) < int(crop_count):
        raise RuntimeError(f"only selected {len(selected)} crops, expected {crop_count}")
    return selected


def _crop_image(image: np.ndarray, selection: CropSelection) -> np.ndarray:
    y1 = selection.y0 + selection.height
    x1 = selection.x0 + selection.width
    return np.asarray(image[selection.y0:y1, selection.x0:x1], copy=True)


def _signal_sigma(image: np.ndarray, snr_db: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    signal_std = float(np.std(np.asarray(image, dtype=np.float64)))
    return signal_std / (10.0 ** (float(snr_db) / 20.0))


def _add_awgn(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sigma = _signal_sigma(image, float(snr_db))
    noisy = np.asarray(image, dtype=np.float64) + sigma * rng.normal(size=np.asarray(image).shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _boundary_normal_field(soft_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(soft_gt, dtype=np.float64)
    height, width = weight.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    size = 2 * int(NORMAL_WINDOW_RADIUS) + 1
    kernel = np.ones((size, size), dtype=np.float64)
    sum_w = ndimage.convolve(weight, kernel, mode="reflect")
    sum_x = ndimage.convolve(weight * xx, kernel, mode="reflect")
    sum_y = ndimage.convolve(weight * yy, kernel, mode="reflect")
    sum_xx = ndimage.convolve(weight * xx * xx, kernel, mode="reflect")
    sum_xy = ndimage.convolve(weight * xx * yy, kernel, mode="reflect")
    sum_yy = ndimage.convolve(weight * yy * yy, kernel, mode="reflect")
    safe = np.maximum(sum_w, EPS)
    mu_x = sum_x / safe
    mu_y = sum_y / safe
    cov_xx = sum_xx / safe - mu_x * mu_x
    cov_xy = sum_xy / safe - mu_x * mu_y
    cov_yy = sum_yy / safe - mu_y * mu_y
    trace = cov_xx + cov_yy
    disc = np.sqrt(np.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy, 0.0))
    lambda1 = 0.5 * (trace + disc)
    tangent_x = cov_xy
    tangent_y = lambda1 - cov_xx
    alt_x = np.where(cov_xx >= cov_yy, 1.0, 0.0)
    alt_y = np.where(cov_xx >= cov_yy, 0.0, 1.0)
    use_alt = (np.abs(tangent_x) + np.abs(tangent_y)) <= EPS
    tangent_x = np.where(use_alt, alt_x, tangent_x)
    tangent_y = np.where(use_alt, alt_y, tangent_y)
    tangent_norm = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
    tangent_x = tangent_x / np.maximum(tangent_norm, EPS)
    tangent_y = tangent_y / np.maximum(tangent_norm, EPS)
    normals = np.stack((-tangent_y, tangent_x), axis=-1)
    valid = (weight >= float(BOUNDARY_WEIGHT_MIN)) & (sum_w > 0.25) & (lambda1 > 1.0e-6)
    return normals, valid


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


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def _vector_rmse(
    gx: np.ndarray,
    gy: np.ndarray,
    soft_gt: np.ndarray,
    normals: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    pred_mag = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    scale = _robust_scale(pred_mag)
    pred_vx = np.asarray(gx, dtype=np.float64) / float(scale)
    pred_vy = np.asarray(gy, dtype=np.float64) / float(scale)
    gt_weight = np.asarray(soft_gt, dtype=np.float64)
    gt_vx = gt_weight * np.asarray(normals[..., 0], dtype=np.float64)
    gt_vy = gt_weight * np.asarray(normals[..., 1], dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    err_pos = (pred_vx - gt_vx) ** 2 + (pred_vy - gt_vy) ** 2
    err_neg = (pred_vx + gt_vx) ** 2 + (pred_vy + gt_vy) ** 2
    err = np.minimum(err_pos, err_neg)
    weights = gt_weight[mask]
    if weights.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(weights * err[mask]) / max(np.sum(weights), EPS)))


def _soft_ods(pred_maps: list[np.ndarray], gt_maps: list[np.ndarray], thresholds: np.ndarray, tolerance_px: int) -> tuple[float, float]:
    gt_dilated = [
        ndimage.maximum_filter(np.asarray(gt, dtype=np.float64), size=2 * int(tolerance_px) + 1, mode="reflect")
        for gt in gt_maps
    ]
    n_gt = float(sum(np.sum(np.asarray(gt, dtype=np.float64)) for gt in gt_maps))
    best_f = -1.0
    best_t = 0.0
    for threshold in np.asarray(thresholds, dtype=np.float64):
        tp_p = 0.0
        tp_r = 0.0
        n_pred = 0.0
        for pred_map, gt_map, gt_dil in zip(pred_maps, gt_maps, gt_dilated, strict=True):
            pred_binary = np.asarray(pred_map, dtype=np.float64) >= float(threshold)
            pred_float = pred_binary.astype(np.float64)
            pred_dilated = ndimage.maximum_filter(pred_float, size=2 * int(tolerance_px) + 1, mode="reflect")
            tp_p += float(np.sum(pred_float * gt_dil))
            tp_r += float(np.sum(np.asarray(gt_map, dtype=np.float64) * pred_dilated))
            n_pred += float(np.sum(pred_float))
        precision = tp_p / max(n_pred, EPS)
        recall = tp_r / max(n_gt, EPS)
        f_score = 2.0 * precision * recall / max(precision + recall, EPS)
        if f_score > best_f:
            best_f = float(f_score)
            best_t = float(threshold)
    return float(best_f), float(best_t)


def _evaluate_snr_bank(
    method_item: dict[str, object],
    crop_gray_map: dict[str, np.ndarray],
    crop_gt_map: dict[str, np.ndarray],
    crop_normal_map: dict[str, np.ndarray],
    crop_valid_map: dict[str, np.ndarray],
    snr_db: float,
    noise_draws: int,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, object]:
    kernel = method_item["kernel"]
    crop_keys = list(crop_gray_map.keys())
    if math.isinf(float(snr_db)):
        draw_count = 1
    else:
        draw_count = int(noise_draws)
    crop_vector_rmses: dict[str, list[float]] = {crop_key: [] for crop_key in crop_keys}
    crop_mean_magnitude = {
        crop_key: np.zeros_like(np.asarray(crop_gray_map[crop_key], dtype=np.float64), dtype=np.float64)
        for crop_key in crop_keys
    }
    for draw_index in range(draw_count):
        images = []
        for crop_index, crop_key in enumerate(crop_keys):
            clean = np.asarray(crop_gray_map[crop_key], dtype=np.float64)
            if math.isinf(float(snr_db)):
                noisy = clean.astype(np.float32)
            else:
                rng = np.random.default_rng(
                    NOISE_SEED_BASE
                    + 10000 * draw_index
                    + 1000 * int(round(float(snr_db) * 10.0))
                    + 31 * crop_index
                )
                noisy = _add_awgn(clean, float(snr_db), rng)
            images.append(np.asarray(noisy, dtype=np.float32))
        responses = apply_images_batched(images, kernel, fft_backend, device_index)
        for crop_key, (gx, gy) in zip(crop_keys, responses, strict=True):
            crop_vector_rmses[crop_key].append(
                _vector_rmse(
                    gx=np.asarray(gx, dtype=np.float64),
                    gy=np.asarray(gy, dtype=np.float64),
                    soft_gt=np.asarray(crop_gt_map[crop_key], dtype=np.float64),
                    normals=np.asarray(crop_normal_map[crop_key], dtype=np.float64),
                    valid_mask=np.asarray(crop_valid_map[crop_key], dtype=bool),
                )
            )
            mag_norm, _ = _normalize_magnitude(np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)))
            crop_mean_magnitude[crop_key] += np.asarray(mag_norm, dtype=np.float64)
    averaged_pred_maps = [crop_mean_magnitude[crop_key] / float(draw_count) for crop_key in crop_keys]
    gt_maps = [np.asarray(crop_gt_map[crop_key], dtype=np.float64) for crop_key in crop_keys]
    ods_f_score, ods_threshold = _soft_ods(averaged_pred_maps, gt_maps, ODS_THRESHOLDS, ODS_TOLERANCE_PX)
    per_crop = {}
    crop_rmse_means = []
    for crop_key in crop_keys:
        mean_rmse = float(np.mean(np.asarray(crop_vector_rmses[crop_key], dtype=np.float64)))
        crop_rmse_means.append(mean_rmse)
        per_crop[crop_key] = {
            "gradient_vector_rmse_mean": float(mean_rmse),
            "noise_draws": int(draw_count),
        }
    return {
        "gradient_vector_rmse_mean": float(np.mean(np.asarray(crop_rmse_means, dtype=np.float64))),
        "ods_f_score": float(ods_f_score),
        "ods_threshold": float(ods_threshold),
        "noise_draws": int(draw_count),
        "per_crop": per_crop,
    }


def _clean_assets_for_method(
    method_item: dict[str, object],
    crop_gray_map: dict[str, np.ndarray],
    assets_dir: Path,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, dict[str, str]]:
    kernel = method_item["kernel"]
    crop_keys = list(crop_gray_map.keys())
    images = [np.asarray(crop_gray_map[crop_key], dtype=np.float32) for crop_key in crop_keys]
    responses = apply_images_batched(images, kernel, fft_backend, device_index)
    outputs: dict[str, dict[str, str]] = {}
    for crop_key, (gx, gy) in zip(crop_keys, responses, strict=True):
        magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        mag_norm, _ = _normalize_magnitude(magnitude)
        mag_path = assets_dir / f"{method_item['method']}_{crop_key}_magnitude.png"
        ori_path = assets_dir / f"{method_item['method']}_{crop_key}_orientation.png"
        _save_gray(mag_path, mag_norm)
        _save_rgb(ori_path, _orientation_rgb(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)))
        outputs[crop_key] = {
            "magnitude_path": str(mag_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            "orientation_path": str(ori_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
        }
    return outputs


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


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
    return {
        "method": best_method,
        "label": best_label,
        "value": float(best_value),
    }


def _evaluate_wvf_trace(
    crop_gray_map: dict[str, np.ndarray],
    crop_gt_map: dict[str, np.ndarray],
    crop_normal_map: dict[str, np.ndarray],
    crop_valid_map: dict[str, np.ndarray],
    methods_payload: dict[str, object],
    fft_backend: str,
    device_index: int | None,
    noise_draws: int,
) -> dict[str, object]:
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

    points = []
    for spec in WVF_TRACE_SPECS:
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
                crop_gray_map=crop_gray_map,
                crop_gt_map=crop_gt_map,
                crop_normal_map=crop_normal_map,
                crop_valid_map=crop_valid_map,
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
                    "overtakes_best_baseline": bool(spec["r"] < 50 and overtakes),
                }
            snr_metrics[snr_slug] = metrics | {"comparison": comparisons}
            print(
                f"sec09A-trace r={spec['r']} d={spec['d']} snr={snr_slug} "
                f"rmse={metrics['gradient_vector_rmse_mean']:.6e} ods={metrics['ods_f_score']:.6f}"
            )
        points.append(
            {
                "radius": int(spec["r"]),
                "degree": int(spec["d"]),
                "config": dict(spec),
                "white_noise_gain": float(method_item["kernel"].white_noise_gain),
                "support_half_extent": int(method_item["kernel"].support_half_extent),
                "snr_metrics": snr_metrics,
            }
        )
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
    crop_size_px: int,
    crop_count: int,
    crop_stride_px: int,
    noise_draws: int,
    auto_download: bool,
) -> dict[str, Path]:
    validation_summary = json.loads(validation_json.read_text())
    roster = _build_roster(validation_summary)
    data_root = _ensure_bsds_data_root(dataset_root, auto_download=bool(auto_download))
    existing_summary = json.loads(summary_json.read_text()) if summary_json.exists() else None
    existing_selection = None if existing_summary is None else _crop_selection_from_summary(existing_summary)
    selections = (
        existing_selection
        if existing_selection is not None
        else _select_crops(data_root, int(crop_size_px), int(crop_count), int(crop_stride_px))
    )

    image_dir = data_root / "images" / "test"
    gt_dir = data_root / "groundTruth" / "test"
    crop_rgb_map: dict[str, np.ndarray] = {}
    crop_gray_map: dict[str, np.ndarray] = {}
    crop_gt_map: dict[str, np.ndarray] = {}
    crop_normal_map: dict[str, np.ndarray] = {}
    crop_valid_map: dict[str, np.ndarray] = {}

    assets_dir = output_dir / "assets"
    crop_payload = []
    for selection in selections:
        rgb, gray = _load_image_pair(image_dir / f"{selection.image_id}.jpg")
        soft_gt = _load_soft_boundaries(gt_dir / f"{selection.image_id}.mat")
        crop_rgb = _crop_image(rgb, selection)
        crop_gray = _crop_image(gray, selection)
        crop_gt = _crop_image(soft_gt, selection)
        normals, valid = _boundary_normal_field(crop_gt)
        crop_rgb_map[selection.crop_key] = np.asarray(crop_rgb, dtype=np.uint8)
        crop_gray_map[selection.crop_key] = np.asarray(crop_gray, dtype=np.float32)
        crop_gt_map[selection.crop_key] = np.asarray(crop_gt, dtype=np.float64)
        crop_normal_map[selection.crop_key] = np.asarray(normals, dtype=np.float64)
        crop_valid_map[selection.crop_key] = np.asarray(valid, dtype=bool)
        input_path = assets_dir / f"{selection.crop_key}_input.png"
        gt_path = assets_dir / f"{selection.crop_key}_soft_gt.png"
        _save_rgb(input_path, crop_rgb)
        gt_norm = crop_gt / max(float(np.max(crop_gt)), 1.0)
        _save_gray(gt_path, gt_norm)
        crop_payload.append(
            {
                "crop_key": str(selection.crop_key),
                "label": str(selection.label),
                "image_id": str(selection.image_id),
                "x0": int(selection.x0),
                "y0": int(selection.y0),
                "width": int(selection.width),
                "height": int(selection.height),
                "selection_score": float(selection.selection_score),
                "edge_mass": float(selection.edge_mass),
                "orientation_entropy": float(selection.orientation_entropy),
                "input_asset_path": str(input_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
                "soft_gt_asset_path": str(gt_path.relative_to(ROOT / "papers" / "journal_paper" / "figures")),
            }
        )

    if existing_summary is not None and isinstance(existing_summary.get("methods"), dict):
        methods_payload = existing_summary["methods"]
    else:
        methods_payload = {}
        for method_item in roster:
            clean_assets = _clean_assets_for_method(
                method_item=method_item,
                crop_gray_map=crop_gray_map,
                assets_dir=assets_dir,
                fft_backend=fft_backend,
                device_index=device_index,
            )
            snr_metrics = {}
            for snr_db in SNR_LEVELS:
                slug = _noise_slug(float(snr_db))
                metrics = _evaluate_snr_bank(
                    method_item=method_item,
                    crop_gray_map=crop_gray_map,
                    crop_gt_map=crop_gt_map,
                    crop_normal_map=crop_normal_map,
                    crop_valid_map=crop_valid_map,
                    snr_db=float(snr_db),
                    noise_draws=int(noise_draws),
                    fft_backend=fft_backend,
                    device_index=device_index,
                )
                snr_metrics[slug] = metrics
                print(
                    f"sec09A {method_item['method']} snr={slug} "
                    f"rmse={metrics['gradient_vector_rmse_mean']:.6e} ods={metrics['ods_f_score']:.6f}"
                )
            methods_payload[str(method_item["method"])] = {
                "label": str(method_item["label"]),
                "config": dict(method_item["config"]),
                "clean_assets": clean_assets,
                "snr_metrics": snr_metrics,
            }

    wvf_trace = _evaluate_wvf_trace(
        crop_gray_map=crop_gray_map,
        crop_gt_map=crop_gt_map,
        crop_normal_map=crop_normal_map,
        crop_valid_map=crop_valid_map,
        methods_payload=methods_payload,
        fft_backend=fft_backend,
        device_index=device_index,
        noise_draws=int(noise_draws),
    )

    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "scenario": "A",
        "dataset": {
            "name": "BSDS500",
            "data_root": str(data_root),
        },
        "config": {
            "crop_size_px": int(crop_size_px),
            "crop_count": int(crop_count),
            "crop_stride_px": int(crop_stride_px),
            "snr_levels": ["inf", "20", "10", "5"],
            "noise_draws": int(noise_draws),
            "ods_threshold_count": int(ODS_THRESHOLDS.shape[0]),
            "ods_tolerance_px": int(ODS_TOLERANCE_PX),
            "vector_rmse_reference": "soft-boundary local-normal field with sign-invariant orientation",
            "ods_noise_aggregation": "draw-averaged normalized magnitude map per crop",
            "fft_backend": str(fft_backend),
            "display_percentile": float(DISPLAY_PERCENTILE),
        },
        "crop_order": [str(row["crop_key"]) for row in crop_payload],
        "crops": crop_payload,
        "method_order": [str(method_item["method"]) for method_item in roster],
        "methods": methods_payload,
        "wvf_trace": wvf_trace,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {
        "summary_json": summary_json,
    }
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec09_real_image_bsds.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec09_real_image_bsds.pdf"
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
        default=ROOT / "datasets" / "BSDS500",
        help="Root directory containing BSDS500 or a location where it can be downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_bsds",
        help="Directory for JSON summaries and image assets.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec09_real_image_bsds" / "sec09_real_image_bsds_summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", choices=("vkfft", "cpu"), help="FFT backend to use for real-image filtering.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figure.")
    parser.add_argument("--crop-size-px", type=int, default=CROP_SIZE_PX, help="Square crop size in pixels.")
    parser.add_argument("--crop-count", type=int, default=CROP_COUNT, help="Number of BSDS500 crops to select.")
    parser.add_argument("--crop-stride-px", type=int, default=CROP_STRIDE_PX, help="Stride used during automatic crop selection.")
    parser.add_argument("--noise-draws", type=int, default=NOISE_DRAWS, help="Noise draws per crop for noisy SNR levels.")
    parser.add_argument("--auto-download", action="store_true", help="Download and extract BSDS500 if the dataset root is missing or incomplete.")
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
        crop_size_px=int(args.crop_size_px),
        crop_count=int(args.crop_count),
        crop_stride_px=int(args.crop_stride_px),
        noise_draws=int(args.noise_draws),
        auto_download=bool(args.auto_download),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
