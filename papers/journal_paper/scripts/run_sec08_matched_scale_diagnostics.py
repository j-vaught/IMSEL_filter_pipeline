#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_square_sg, build_wvf, recommended_wvf_degree
from section8_common import (
    CONTRAST,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    add_awgn,
    apply_cases_batched,
    compile_plot,
    generate_step_cases,
    mean,
    orientation_values,
    peak_metrics_from_profile,
    phase_values,
    sample_profile,
)


RADIUS_SCHEDULE = (3, 5, 9, 15, 25, 50)
MATCH_RULES = ("bounding_radius", "effective_response_width", "support_cardinality", "white_noise_gain", "effective_second_moment")
CALIBRATION_ORIENTATION_STEP_DEG = 5.0
EVAL_ORIENTATION_STEP_DEG = 0.5
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
SNR_DB = 10.0
NOISE_SEED = 8560
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = DEFAULT_BATCH_CASES
SQUARE_WINDOWS = tuple(range(3, 122, 2))


def _noise_slug(snr_db: float) -> str:
    return f"{float(snr_db):g}".replace(".", "p")


def _effective_second_moment(kernel) -> float:
    weights = np.asarray(kernel.kernel_x, dtype=np.float64)
    half_y = weights.shape[0] // 2
    half_x = weights.shape[1] // 2
    yy, xx = np.meshgrid(
        np.arange(-half_y, half_y + 1, dtype=np.float64),
        np.arange(-half_x, half_x + 1, dtype=np.float64),
        indexing="ij",
    )
    radial_sq = xx**2 + yy**2
    return float(np.sum(radial_sq * (weights**2)) / np.sum(weights**2))


def _step_peak_profile(case, gx: np.ndarray, gy: np.ndarray, support_scale: float) -> tuple[float, float]:
    theta = math.radians(float(case.orientation_deg))
    directional = np.asarray(gx, dtype=np.float64) * math.cos(theta) + np.asarray(gy, dtype=np.float64) * math.sin(theta)
    profile = sample_profile(directional, case.line_xs, case.line_ys)
    peak_height, _, fwhm = peak_metrics_from_profile(
        profile=profile,
        x_coords=case.line_coords,
        phase_px=float(case.phase_px),
        support_scale=float(support_scale),
        width_px=EDGE_WIDTH_PX,
    )
    return float(peak_height), float(fwhm)


def _evaluate_step_bank(kernel, cases, image_bank, fft_backend: str, device_index: int | None) -> dict[str, float]:
    eval_cases = [
        type(case)(
            orientation_deg=case.orientation_deg,
            phase_px=case.phase_px,
            image=np.asarray(image, dtype=np.float32),
            true_gx=case.true_gx,
            true_gy=case.true_gy,
            eval_mask=case.eval_mask,
            line_coords=case.line_coords,
            line_xs=case.line_xs,
            line_ys=case.line_ys,
        )
        for case, image in zip(cases, image_bank, strict=True)
    ]
    responses = apply_cases_batched(eval_cases, kernel, fft_backend, device_index, batch_cases=BATCH_CASES)
    per_orientation: dict[float, list[float]] = {}
    fwhm_values: list[float] = []
    for case, (gx, gy) in zip(cases, responses, strict=True):
        peak_height, fwhm = _step_peak_profile(case, gx, gy, kernel.support_half_extent)
        per_orientation.setdefault(float(case.orientation_deg), []).append(float(peak_height))
        fwhm_values.append(float(fwhm))
    curve = [
        {"theta_deg": float(theta_deg), "response": float(mean(values))}
        for theta_deg, values in sorted(per_orientation.items(), key=lambda item: item[0])
    ]
    response_values = np.asarray([row["response"] for row in curve], dtype=np.float64)
    return {
        "anisotropy_ratio": float(np.max(response_values) / max(np.min(response_values), 1.0e-15)),
        "fwhm": float(mean(fwhm_values)),
    }


def _square_candidate_table(degree: int, calibration_cases, calibration_images, fft_backend: str, device_index: int | None) -> list[dict[str, object]]:
    candidates = []
    for window_size in SQUARE_WINDOWS:
        kernel = build_square_sg(window_size=int(window_size), degree=int(degree), normalize_coords=True)
        metrics = _evaluate_step_bank(kernel, calibration_cases, calibration_images, fft_backend, device_index)
        candidates.append(
            {
                "window_size": int(window_size),
                "half_side": int(window_size // 2),
                "degree": int(degree),
                "kernel": kernel,
                "support_cardinality": int(kernel.support_cardinality) if kernel.support_cardinality is not None else None,
                "white_noise_gain": float(kernel.white_noise_gain),
                "effective_second_moment": float(_effective_second_moment(kernel)),
                "clean_fwhm": float(metrics["fwhm"]),
            }
        )
        print(f"sec84 square_cal degree={degree} N={window_size} fwhm={metrics['fwhm']:.6e}")
    return candidates


def _match_candidate(rule: str, wvf_info: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    if rule == "bounding_radius":
        target_half_side = int(wvf_info["radius"])
        return min(candidates, key=lambda item: (abs(int(item["half_side"]) - target_half_side), int(item["window_size"])))
    metric_key = {
        "effective_response_width": "clean_fwhm",
        "support_cardinality": "support_cardinality",
        "white_noise_gain": "white_noise_gain",
        "effective_second_moment": "effective_second_moment",
    }[rule]
    target = float(wvf_info[metric_key])
    return min(candidates, key=lambda item: (abs(float(item[metric_key]) - target), int(item["window_size"])))


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
) -> dict[str, Path]:
    calibration_cases = generate_step_cases(
        support_scale=MAX_SUPPORT_SCALE,
        orientations_deg=orientation_values(CALIBRATION_ORIENTATION_STEP_DEG, span_deg=180.0),
        phases_px=phase_values(PHASE_COUNT, PHASE_STEP_PX),
    )
    calibration_images = [np.asarray(case.image, dtype=np.float32) for case in calibration_cases]

    eval_cases = generate_step_cases(
        support_scale=MAX_SUPPORT_SCALE,
        orientations_deg=orientation_values(EVAL_ORIENTATION_STEP_DEG, span_deg=180.0),
        phases_px=phase_values(PHASE_COUNT, PHASE_STEP_PX),
    )
    rng = np.random.default_rng(NOISE_SEED)
    noisy_eval_images = [
        np.asarray(add_awgn(case.image, SNR_DB, rng), dtype=np.float32)
        for case in eval_cases
    ]

    candidate_tables: dict[int, list[dict[str, object]]] = {}
    eval_cache: dict[tuple[str, int, int], dict[str, float]] = {}
    wvf_infos = []
    for radius in RADIUS_SCHEDULE:
        degree = recommended_wvf_degree(int(radius))
        if degree not in candidate_tables:
            candidate_tables[int(degree)] = _square_candidate_table(
                degree=int(degree),
                calibration_cases=calibration_cases,
                calibration_images=calibration_images,
                fft_backend=fft_backend,
                device_index=device_index,
            )
        kernel = build_wvf(radius=int(radius), degree=int(degree), normalize_coords=True)
        metrics = _evaluate_step_bank(kernel, calibration_cases, calibration_images, fft_backend, device_index)
        wvf_infos.append(
            {
                "radius": int(radius),
                "degree": int(degree),
                "kernel": kernel,
                "support_cardinality": int(kernel.support_cardinality) if kernel.support_cardinality is not None else None,
                "white_noise_gain": float(kernel.white_noise_gain),
                "effective_second_moment": float(_effective_second_moment(kernel)),
                "clean_fwhm": float(metrics["fwhm"]),
            }
        )
        print(f"sec84 wvf_cal r={radius} d={degree} fwhm={metrics['fwhm']:.6e}")

    rules_payload = {}
    for rule in MATCH_RULES:
        rows = []
        for wvf_info in wvf_infos:
            square_match = _match_candidate(str(rule), wvf_info, candidate_tables[int(wvf_info["degree"])])
            wvf_key = ("wvf", int(wvf_info["radius"]), int(wvf_info["degree"]))
            if wvf_key not in eval_cache:
                eval_cache[wvf_key] = _evaluate_step_bank(wvf_info["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)
            square_key = ("square_sg", int(square_match["window_size"]), int(square_match["degree"]))
            if square_key not in eval_cache:
                eval_cache[square_key] = _evaluate_step_bank(square_match["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)
            wvf_eval = eval_cache[wvf_key]
            square_eval = eval_cache[square_key]
            row = {
                "radius": int(wvf_info["radius"]),
                "degree": int(wvf_info["degree"]),
                "wvf": {
                    "anisotropy_ratio": float(wvf_eval["anisotropy_ratio"]),
                    "clean_fwhm": float(wvf_info["clean_fwhm"]),
                    "support_cardinality": int(wvf_info["support_cardinality"]),
                    "white_noise_gain": float(wvf_info["white_noise_gain"]),
                    "effective_second_moment": float(wvf_info["effective_second_moment"]),
                },
                "square_sg": {
                    "window_size": int(square_match["window_size"]),
                    "half_side": int(square_match["half_side"]),
                    "degree": int(square_match["degree"]),
                    "anisotropy_ratio": float(square_eval["anisotropy_ratio"]),
                    "clean_fwhm": float(square_match["clean_fwhm"]),
                    "support_cardinality": int(square_match["support_cardinality"]),
                    "white_noise_gain": float(square_match["white_noise_gain"]),
                    "effective_second_moment": float(square_match["effective_second_moment"]),
                },
            }
            print(
                f"sec84 rule={rule} r={row['radius']} "
                f"wvfA={row['wvf']['anisotropy_ratio']:.6f} squareA={row['square_sg']['anisotropy_ratio']:.6f}"
            )
            rows.append(row)
        rules_payload[str(rule)] = {"rows": rows}

    payload = {
        "title": "Section 8.4 matched-scale diagnostics",
        "subtitle": "WVF versus square SG on smoothed step edges at AWGN 10 dB",
        "config": {
            "radius_schedule": list(RADIUS_SCHEDULE),
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "calibration_orientation_step_deg": CALIBRATION_ORIENTATION_STEP_DEG,
            "evaluation_orientation_step_deg": EVAL_ORIENTATION_STEP_DEG,
            "snr_db": SNR_DB,
            "fft_backend": str(fft_backend),
        },
        "rule_order": list(MATCH_RULES),
        "rules": rules_payload,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec08_matched_scale_diagnostics.typ"
        pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_matched_scale_diagnostics.pdf"
        compile_plot(src, pdf)
        outputs[pdf.stem] = pdf
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_matched_scale_diagnostics" / "sec08_matched_scale_diagnostics_summary.json",
        help="Path for the matched-scale diagnostics summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", help="FFT backend for applying precomputed kernels.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for FFT execution.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the Typst plot after writing the summary JSON.")
    args = parser.parse_args(argv)
    run_experiment(
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
