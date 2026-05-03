#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_method
from section8_common import CONTRAST, EDGE_WIDTH_PX, apply_cases_batched, compile_plot, orientation_values, phase_values


PATCH_HALF_SIZE = 128
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 16
PHASE_STEP_PX = 0.0625
SNR_LEVELS = (math.inf, 20.0, 15.0, 10.0, 5.0, 0.0)
NOISE_DRAWS = 100
PROFILE_STEP_PX = 0.25
PROFILE_MARGIN_PX = 8.0
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = 64
SEED_BASE = 8450


@dataclass(frozen=True)
class StepCase:
    orientation_deg: float
    phase_px: float
    image: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    t_coords: np.ndarray


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


def _noise_slug(snr_db: float) -> str:
    if math.isinf(float(snr_db)):
        return "inf"
    return f"{float(snr_db):g}".replace(".", "p")


def _local_grid() -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(-PATCH_HALF_SIZE, PATCH_HALF_SIZE + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _render_smoothed_step(xx: np.ndarray, yy: np.ndarray, orientation_deg: float, phase_px: float) -> np.ndarray:
    theta = math.radians(float(orientation_deg))
    normal = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta) - float(phase_px)
    return np.asarray(0.5 * float(CONTRAST) * (1.0 + np.tanh(normal / float(EDGE_WIDTH_PX))), dtype=np.float32)


def _profile_geometry(orientation_deg: float, phase_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = math.radians(float(orientation_deg))
    search_half = float(MAX_SUPPORT_SCALE) + 12.0 * float(EDGE_WIDTH_PX) + float(PROFILE_MARGIN_PX)
    t_coords = np.arange(-search_half, search_half + 0.5 * float(PROFILE_STEP_PX), float(PROFILE_STEP_PX), dtype=np.float64)
    center = float(PATCH_HALF_SIZE)
    center_x = center + float(phase_px) * math.cos(theta)
    center_y = center + float(phase_px) * math.sin(theta)
    sample_x = center_x + t_coords * math.cos(theta)
    sample_y = center_y + t_coords * math.sin(theta)
    return t_coords, sample_x, sample_y


def _build_cases() -> list[StepCase]:
    xx, yy = _local_grid()
    cases: list[StepCase] = []
    for orientation_deg in orientation_values(ORIENTATION_STEP_DEG, span_deg=180.0):
        for phase_px in phase_values(PHASE_COUNT, PHASE_STEP_PX):
            t_coords, sample_x, sample_y = _profile_geometry(float(orientation_deg), float(phase_px))
            cases.append(
                StepCase(
                    orientation_deg=float(orientation_deg),
                    phase_px=float(phase_px),
                    image=_render_smoothed_step(xx, yy, float(orientation_deg), float(phase_px)),
                    sample_x=np.asarray(sample_x, dtype=np.float64),
                    sample_y=np.asarray(sample_y, dtype=np.float64),
                    t_coords=np.asarray(t_coords, dtype=np.float64),
                )
            )
    return cases


def _sample_profile(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    return np.asarray(
        ndimage.map_coordinates(
            np.asarray(image, dtype=np.float64),
            np.vstack((np.asarray(sample_y, dtype=np.float64), np.asarray(sample_x, dtype=np.float64))),
            order=1,
            mode="nearest",
        ),
        dtype=np.float64,
    )


def _quadratic_peak_refinement(x_coords: np.ndarray, profile: np.ndarray, peak_index: int) -> tuple[float, float]:
    idx = int(peak_index)
    if idx <= 0 or idx >= profile.shape[0] - 1:
        return float(x_coords[idx]), float(profile[idx])
    y_prev = float(profile[idx - 1])
    y_mid = float(profile[idx])
    y_next = float(profile[idx + 1])
    denom = y_prev - 2.0 * y_mid + y_next
    if abs(denom) <= 1.0e-15:
        return float(x_coords[idx]), y_mid
    delta = 0.5 * (y_prev - y_next) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    peak_x = float(x_coords[idx]) + delta * float(x_coords[idx + 1] - x_coords[idx])
    peak_y = y_mid - 0.25 * (y_prev - y_next) * delta
    return float(peak_x), float(peak_y)


def _localization_offset(profile: np.ndarray, t_coords: np.ndarray) -> float:
    values = np.asarray(profile, dtype=np.float64)
    t = np.asarray(t_coords, dtype=np.float64)
    peak_index = int(np.argmax(values))
    peak_t, _ = _quadratic_peak_refinement(t, values, peak_index)
    return float(peak_t)


def _signal_std_reference(cases: list[StepCase]) -> float:
    clean = np.concatenate([np.asarray(case.image, dtype=np.float64).ravel() for case in cases])
    return float(np.std(clean))


def _sigma_from_snr_db(snr_db: float | str, signal_std: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    return float(signal_std) / (10.0 ** (float(snr_db) / 20.0))


def _evaluate_bank(method_item, cases: list[StepCase], images: list[np.ndarray], fft_backend: str, device_index: int | None) -> list[float]:
    eval_cases = [
        StepCase(
            orientation_deg=case.orientation_deg,
            phase_px=case.phase_px,
            image=np.asarray(image, dtype=np.float32),
            sample_x=case.sample_x,
            sample_y=case.sample_y,
            t_coords=case.t_coords,
        )
        for case, image in zip(cases, images, strict=True)
    ]
    responses = apply_cases_batched(eval_cases, method_item["kernel"], fft_backend, device_index, batch_cases=BATCH_CASES)
    offsets: list[float] = []
    for case, (gx, gy) in zip(cases, responses, strict=True):
        magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        offsets.append(float(_localization_offset(_sample_profile(magnitude, case.sample_x, case.sample_y), case.t_coords)))
    return offsets


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    validation_json: Path,
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
) -> dict[str, Path]:
    validation_summary = json.loads(validation_json.read_text())
    roster = _build_roster(validation_summary)
    cases = _build_cases()
    signal_std = _signal_std_reference(cases)

    rms_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    methods_payload: dict[str, object] = {}

    clean_images = [np.asarray(case.image, dtype=np.float32) for case in cases]
    for method_index, item in enumerate(roster):
        clean_offsets = _evaluate_bank(item, cases, clean_images, fft_backend, device_index)
        rms_by_snr = {
            "inf": float(np.sqrt(np.mean(np.asarray(clean_offsets, dtype=np.float64) ** 2)))
        }
        print(f"sec834 {item['method']} snr=inf rms={rms_by_snr['inf']:.6e}")
        per_phase: dict[float, list[float]] = {}
        for case, offset in zip(cases, clean_offsets, strict=True):
            per_phase.setdefault(float(case.phase_px), []).append(float(offset))
        phase_profile = [
            {"phase_px": float(phase_px), "mean_offset": float(np.mean(np.asarray(values, dtype=np.float64)))}
            for phase_px, values in sorted(per_phase.items(), key=lambda item: item[0])
        ]
        for row in phase_profile:
            phase_rows.append(
                {
                    "method": str(item["method"]),
                    "label": str(item["label"]),
                    "phase_px": f"{row['phase_px']:.4f}",
                    "mean_offset": f"{row['mean_offset']:.17e}",
                }
            )

        for snr_db in SNR_LEVELS:
            slug = _noise_slug(float(snr_db))
            if slug != "inf":
                sigma = _sigma_from_snr_db(snr_db, signal_std)
                offsets = []
                for draw_index in range(int(NOISE_DRAWS)):
                    rng = np.random.default_rng(SEED_BASE + 1000 * method_index + 100 * (SNR_LEVELS.index(snr_db)) + draw_index)
                    noisy_images = [
                        np.asarray(case.image, dtype=np.float64) + sigma * rng.normal(size=case.image.shape)
                        for case in cases
                    ]
                    offsets.extend(_evaluate_bank(item, cases, noisy_images, fft_backend, device_index))
                    if (draw_index + 1) % 20 == 0:
                        print(f"sec834 {item['method']} snr={slug} draw={draw_index + 1}/{int(NOISE_DRAWS)}")
                rms_by_snr[slug] = float(np.sqrt(np.mean(np.asarray(offsets, dtype=np.float64) ** 2)))
                print(f"sec834 {item['method']} snr={slug} rms={rms_by_snr[slug]:.6e}")

            rms_rows.append(
                {
                    "method": str(item["method"]),
                    "label": str(item["label"]),
                    "snr_db": slug,
                    "localisation_rms": f"{rms_by_snr[slug]:.17e}",
                }
            )

        methods_payload[str(item["method"])] = {
            "label": str(item["label"]),
            "config": dict(item["config"]),
            "rms_by_snr": rms_by_snr,
            "clean_phase_profile": phase_profile,
        }

    _write_csv(
        output_dir / "sec08_subpixel_localization_rms.csv",
        ("method", "label", "snr_db", "localisation_rms"),
        rms_rows,
    )
    _write_csv(
        output_dir / "sec08_subpixel_localization_phase_bias.csv",
        ("method", "label", "phase_px", "mean_offset"),
        phase_rows,
    )

    payload = {
        "title": "Section 8.3.4 sub-pixel localisation",
        "subtitle": "Validation-tuned head-to-head comparison on smoothed step edges",
        "config": {
            "orientation_step_deg": ORIENTATION_STEP_DEG,
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "snr_levels": ["inf", "20", "15", "10", "5", "0"],
            "noise_draws": int(NOISE_DRAWS),
            "fft_backend": str(fft_backend),
        },
        "method_order": [str(item["method"]) for item in roster],
        "methods": methods_payload,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {
        "summary_json": summary_json,
        "rms_csv": output_dir / "sec08_subpixel_localization_rms.csv",
        "phase_csv": output_dir / "sec08_subpixel_localization_phase_bias.csv",
    }
    if compile_plots:
        figures = (
            ("fig_sec08_subpixel_localization_rms.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_subpixel_localization_rms.pdf"),
            ("fig_sec08_subpixel_localization_phase_bias.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_subpixel_localization_phase_bias.pdf"),
        )
        for src_name, pdf_path in figures:
            src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / src_name
            compile_plot(src, pdf_path)
            outputs[pdf_path.stem] = pdf_path
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_baseline_validation" / "sec08_baseline_validation_summary.json",
        help="Path to the Section 8.1 validation summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_subpixel_localization",
        help="Directory for Section 8.3.4 CSV outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_subpixel_localization" / "sec08_subpixel_localization_summary.json",
        help="Path for the Section 8.3.4 summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", help="FFT backend for applying precomputed kernels.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for FFT execution.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the Typst plots after writing outputs.")
    args = parser.parse_args(argv)
    run_experiment(
        validation_json=args.validation_json.resolve(),
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
