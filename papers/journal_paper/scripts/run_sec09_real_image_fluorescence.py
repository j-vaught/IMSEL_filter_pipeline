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
from section8_common import apply_images_batched, compile_plot


TITLE = "Section 9 Scenario C fluorescence microscopy"
SUBTITLE = "BBBC039 real-image comparison on native noisy fluorescence microscopy"
DATASET_URL = "https://data.broadinstitute.org/bbbc/BBBC039/images.zip"
MIN_ARCHIVE_BYTES = 10_000_000
IMAGE_COUNT = 5
BACKGROUND_PERCENTILE = 35.0
DISPLAY_PERCENTILE = 99.5
EPS = 1.0e-12


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
    selections = _select_images(data_root, int(image_count))

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

    methods_payload: dict[str, object] = {}
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
