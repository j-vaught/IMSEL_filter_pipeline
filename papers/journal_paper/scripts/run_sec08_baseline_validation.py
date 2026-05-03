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

from baseline_filters import (
    build_dog,
    build_farid_simoncelli,
    build_prewitt,
    build_roberts,
    build_scharr,
    build_sobel,
    build_square_sg,
    build_wvf,
    fixed_method_order,
    recommended_wvf_degree,
)
from section8_common import (
    CONTRAST,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    STEP_PATCH_HALF_SIZE,
    add_awgn,
    apply_cases_batched,
    case_gradient_metrics,
    generate_step_cases,
    mean,
    orientation_values,
    peak_metrics_from_profile,
    phase_values,
    sample_profile,
)


VALIDATION_SNR_DB = 10.0
ORIENTATION_STEP_DEG = 0.5
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
NOISE_SEED = 8101
BATCH_CASES = DEFAULT_BATCH_CASES
MAX_SUPPORT_SCALE = 50.0
DOG_SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SQUARE_WINDOWS = (3, 5, 7, 9, 11, 13, 15)
SQUARE_DEGREES = (1, 3, 5)
WVF_RADII = (3, 5, 9, 15, 25, 50)
CONSTRAINTS = {
    "localization_rms": 0.5,
    "mag_bias_abs": 0.05,
    "white_noise_gain": 0.001,
}
RELAX_FACTORS = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0)
THRESHOLD_SCENARIOS = (
    ("loc_half", {"localization_rms": 0.5}),
    ("loc_double", {"localization_rms": 2.0}),
    ("bias_half", {"mag_bias_abs": 0.5}),
    ("bias_double", {"mag_bias_abs": 2.0}),
    ("wng_half", {"white_noise_gain": 0.5}),
    ("wng_double", {"white_noise_gain": 2.0}),
    ("all_half", {"localization_rms": 0.5, "mag_bias_abs": 0.5, "white_noise_gain": 0.5}),
    ("all_double", {"localization_rms": 2.0, "mag_bias_abs": 2.0, "white_noise_gain": 2.0}),
)


def _directional_profile(case, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    theta = math.radians(float(case.orientation_deg))
    directional = np.asarray(gx, dtype=np.float64) * math.cos(theta) + np.asarray(gy, dtype=np.float64) * math.sin(theta)
    return sample_profile(directional, case.line_xs, case.line_ys)


def _evaluate_kernel(kernel, cases, noisy_images, fft_backend: str, device_index: int | None) -> dict[str, float]:
    noisy_cases = [
        type(case)(
            orientation_deg=case.orientation_deg,
            phase_px=case.phase_px,
            image=np.asarray(noisy_image, dtype=np.float32),
            true_gx=case.true_gx,
            true_gy=case.true_gy,
            eval_mask=case.eval_mask,
            line_coords=case.line_coords,
            line_xs=case.line_xs,
            line_ys=case.line_ys,
        )
        for case, noisy_image in zip(cases, noisy_images, strict=True)
    ]
    responses = apply_cases_batched(noisy_cases, kernel, fft_backend, device_index, batch_cases=BATCH_CASES)
    grad_rmses: list[float] = []
    mag_biases: list[float] = []
    localization_offsets: list[float] = []
    fwhm_values: list[float] = []
    peak_values: list[float] = []
    for case, (gx, gy) in zip(cases, responses, strict=True):
        metrics = case_gradient_metrics(case, gx, gy)
        grad_rmses.append(float(metrics["grad_rmse"]))
        mag_biases.append(float(metrics["mag_bias"]))
        profile = _directional_profile(case, gx, gy)
        peak_height, localization_offset, fwhm = peak_metrics_from_profile(
            profile=profile,
            x_coords=case.line_coords,
            phase_px=float(case.phase_px),
            support_scale=float(kernel.support_half_extent),
            width_px=EDGE_WIDTH_PX,
        )
        peak_values.append(float(peak_height))
        localization_offsets.append(float(localization_offset))
        fwhm_values.append(float(fwhm))
    localization_rms = float(np.sqrt(np.mean(np.asarray(localization_offsets, dtype=np.float64) ** 2)))
    return {
        "grad_rmse": mean(grad_rmses),
        "magnitude_bias": mean(mag_biases),
        "mag_bias_abs": float(abs(mean(mag_biases))),
        "localization_rms": localization_rms,
        "fwhm": mean(fwhm_values),
        "peak_amplitude": mean(peak_values),
        "white_noise_gain": float(kernel.white_noise_gain),
    }


def _candidate_payload(kernel) -> dict[str, object]:
    return {
        "method": str(kernel.method),
        "label": str(kernel.label),
        "config": dict(kernel.config),
        "support_half_extent": int(kernel.support_half_extent),
        "white_noise_gain": float(kernel.white_noise_gain),
        "support_cardinality": None if kernel.support_cardinality is None else int(kernel.support_cardinality),
        "kappa_design_matrix": None if kernel.kappa_design_matrix is None else float(kernel.kappa_design_matrix),
    }


def _pick_best(records: list[dict[str, object]], constraints: dict[str, float] | None = None) -> dict[str, object]:
    limits = dict(CONSTRAINTS if constraints is None else constraints)
    for relax_factor in RELAX_FACTORS:
        loc_limit = float(limits["localization_rms"]) * float(relax_factor)
        mag_limit = float(limits["mag_bias_abs"]) * float(relax_factor)
        wng_limit = float(limits["white_noise_gain"]) * float(relax_factor)
        feasible = [
            record
            for record in records
            if float(record["metrics"]["localization_rms"]) < loc_limit
            and float(record["metrics"]["mag_bias_abs"]) < mag_limit
            and float(record["metrics"]["white_noise_gain"]) < wng_limit
        ]
        if feasible:
            chosen = min(feasible, key=lambda item: float(item["metrics"]["grad_rmse"]))
            return {
                "selection_mode": "constrained_rmse",
                "relax_factor": float(relax_factor),
                "constraints_used": {
                    "localization_rms": float(loc_limit),
                    "mag_bias_abs": float(mag_limit),
                    "white_noise_gain": float(wng_limit),
                },
                "chosen": chosen,
                "feasible_count": int(len(feasible)),
            }
    chosen = min(records, key=lambda item: float(item["metrics"]["grad_rmse"]))
    return {
        "selection_mode": "rmse_only",
        "relax_factor": None,
        "constraints_used": None,
        "chosen": chosen,
        "feasible_count": 0,
    }


def _threshold_sensitivity(baseline_selection: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for scenario_name, multipliers in THRESHOLD_SCENARIOS:
        scenario_constraints = {
            key: float(CONSTRAINTS[key]) * float(multipliers.get(key, 1.0))
            for key in CONSTRAINTS
        }
        scenario_row = {
            "scenario": str(scenario_name),
            "constraints": scenario_constraints,
            "methods": {},
        }
        for method_name, baseline in baseline_selection.items():
            selected = _pick_best(list(baseline["candidates"]), constraints=scenario_constraints)
            chosen = dict(selected["chosen"])
            baseline_config = dict(baseline["chosen"]["config"])
            scenario_row["methods"][str(method_name)] = {
                "changed": bool(dict(chosen["config"]) != baseline_config),
                "baseline_config": baseline_config,
                "selected_config": dict(chosen["config"]),
                "selection_mode": str(selected["selection_mode"]),
                "relax_factor": selected["relax_factor"],
                "grad_rmse_delta": float(chosen["metrics"]["grad_rmse"] - baseline["chosen"]["metrics"]["grad_rmse"]),
            }
        table.append(scenario_row)
    return table


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(summary_json: Path, fft_backend: str, device_index: int | None) -> Path:
    orientations_deg = orientation_values(ORIENTATION_STEP_DEG, span_deg=180.0)
    phases_px = phase_values(PHASE_COUNT, PHASE_STEP_PX)
    clean_cases = generate_step_cases(
        support_scale=MAX_SUPPORT_SCALE,
        orientations_deg=orientations_deg,
        phases_px=phases_px,
        half_size=STEP_PATCH_HALF_SIZE,
        contrast=CONTRAST,
        width_px=EDGE_WIDTH_PX,
    )
    rng = np.random.default_rng(NOISE_SEED)
    noisy_images = [
        np.asarray(add_awgn(case.image, VALIDATION_SNR_DB, rng), dtype=np.float32)
        for case in clean_cases
    ]

    fixed_records: dict[str, dict[str, object]] = {}
    skipped_methods: list[dict[str, str]] = []
    fixed_builders = {
        "roberts": build_roberts,
        "prewitt": build_prewitt,
        "sobel": build_sobel,
        "scharr": build_scharr,
        "farid_simoncelli": build_farid_simoncelli,
    }
    for method in fixed_method_order():
        if method not in fixed_builders:
            continue
        try:
            kernel = fixed_builders[method]()
            metrics = _evaluate_kernel(kernel, clean_cases, noisy_images, fft_backend, device_index)
            fixed_records[method] = {
                **_candidate_payload(kernel),
                "metrics": metrics,
            }
            print(
                f"sec81 fixed {method} rmse={metrics['grad_rmse']:.6e} "
                f"loc={metrics['localization_rms']:.6e} bias={metrics['mag_bias_abs']:.6e}"
            )
        except Exception as exc:
            skipped_methods.append({"method": method, "reason": f"{type(exc).__name__}: {exc}"})

    tuning_results: dict[str, dict[str, object]] = {}

    try:
        dog_candidates = []
        for sigma in DOG_SIGMAS:
            kernel = build_dog(float(sigma))
            metrics = _evaluate_kernel(kernel, clean_cases, noisy_images, fft_backend, device_index)
            dog_candidates.append({**_candidate_payload(kernel), "metrics": metrics})
            print(f"sec81 dog sigma={sigma:g} rmse={metrics['grad_rmse']:.6e}")
        tuning_results["dog"] = {
            "candidates": dog_candidates,
            **_pick_best(dog_candidates),
        }
    except Exception as exc:
        skipped_methods.append({"method": "dog", "reason": f"{type(exc).__name__}: {exc}"})

    try:
        square_candidates = []
        for window_size in SQUARE_WINDOWS:
            for degree in SQUARE_DEGREES:
                if degree >= window_size:
                    continue
                kernel = build_square_sg(window_size=window_size, degree=degree, normalize_coords=True)
                metrics = _evaluate_kernel(kernel, clean_cases, noisy_images, fft_backend, device_index)
                square_candidates.append({**_candidate_payload(kernel), "metrics": metrics})
                print(f"sec81 square_sg N={window_size} d={degree} rmse={metrics['grad_rmse']:.6e}")
        tuning_results["square_sg"] = {
            "candidates": square_candidates,
            **_pick_best(square_candidates),
        }
    except Exception as exc:
        skipped_methods.append({"method": "square_sg", "reason": f"{type(exc).__name__}: {exc}"})

    try:
        wvf_candidates = []
        for radius in WVF_RADII:
            degree = recommended_wvf_degree(int(radius))
            kernel = build_wvf(radius=int(radius), degree=int(degree), normalize_coords=True)
            metrics = _evaluate_kernel(kernel, clean_cases, noisy_images, fft_backend, device_index)
            wvf_candidates.append({**_candidate_payload(kernel), "metrics": metrics})
            print(f"sec81 wvf r={radius} d={degree} rmse={metrics['grad_rmse']:.6e}")
        tuning_results["wvf"] = {
            "candidates": wvf_candidates,
            **_pick_best(wvf_candidates),
        }
    except Exception as exc:
        skipped_methods.append({"method": "wvf", "reason": f"{type(exc).__name__}: {exc}"})

    method_roster = []
    for method in fixed_method_order():
        if method in fixed_records:
            item = fixed_records[method]
            method_roster.append(
                {
                    "method": str(item["method"]),
                    "label": str(item["label"]),
                    "config": dict(item["config"]),
                    "metrics": dict(item["metrics"]),
                }
            )
        elif method in tuning_results:
            chosen = dict(tuning_results[method]["chosen"])
            method_roster.append(
                {
                    "method": str(chosen["method"]),
                    "label": str(chosen["label"]),
                    "config": dict(chosen["config"]),
                    "metrics": dict(chosen["metrics"]),
                }
            )

    baseline_selection = {
        name: {
            "candidates": list(result["candidates"]),
            "chosen": dict(result["chosen"]),
        }
        for name, result in tuning_results.items()
    }

    payload = {
        "title": "Section 8.1 baseline infrastructure and validation tuning",
        "subtitle": "Validation-tuned head-to-head method roster under AWGN 10 dB on smoothed step edges",
        "config": {
            "contrast": CONTRAST,
            "edge_width_px": EDGE_WIDTH_PX,
            "validation_snr_db": VALIDATION_SNR_DB,
            "orientation_step_deg": ORIENTATION_STEP_DEG,
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "patch_half_size": STEP_PATCH_HALF_SIZE,
            "fft_backend": str(fft_backend),
            "noise_seed": NOISE_SEED,
        },
        "constraints": dict(CONSTRAINTS),
        "method_order": list(fixed_method_order()),
        "method_roster": method_roster,
        "fixed_methods": fixed_records,
        "tuning_results": tuning_results,
        "threshold_sensitivity": _threshold_sensitivity(baseline_selection),
        "skipped_methods": skipped_methods,
    }
    _write_json(summary_json, payload)
    print(f"wrote {summary_json}")
    return summary_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_baseline_validation" / "sec08_baseline_validation_summary.json",
        help="Path for the validation summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", help="FFT backend for applying precomputed kernels.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional CUDA device index for VkFFT execution.")
    args = parser.parse_args(argv)
    run_experiment(summary_json=args.summary_json.resolve(), fft_backend=str(args.fft_backend), device_index=args.device_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
