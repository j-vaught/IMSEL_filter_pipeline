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

from baseline_filters import build_dog, build_square_sg, build_wvf, recommended_wvf_degree
from section8_common import (
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
COMPARATOR_ORDER = ("square_sg", "dog")
CALIBRATION_ORIENTATION_STEP_DEG = 5.0
EVAL_ORIENTATION_STEP_DEG = 0.5
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
SNR_DB = 10.0
NOISE_SEED = 8560
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = DEFAULT_BATCH_CASES
SQUARE_WINDOWS = tuple(range(3, 122, 2))
DOG_SIGMAS = tuple(0.25 * value for value in range(2, 51))
OVERLAY_FWHM_TARGETS = (4.0, 8.0, 14.0)
OVERLAY_SQUARE_WINDOWS = (3, 5, 7, 9, 11, 13, 15)
OVERLAY_SQUARE_DEGREES = (1, 3, 5)


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


def _kernel_support_cardinality(kernel) -> int:
    return int(np.asarray(kernel.kernel_x).size)


def _candidate_key(family: str, config: dict[str, object]) -> tuple[object, ...]:
    if family == "square_sg":
        return (family, int(config["N"]), int(config["d"]))
    if family == "dog":
        return (family, float(config["sigma"]))
    if family == "wvf":
        return (family, int(config["r"]), int(config["d"]))
    raise ValueError(f"unsupported candidate family {family!r}")


def _candidate_record(
    family: str,
    kernel,
    config: dict[str, object],
    bounding_radius: int,
    calibration_cases,
    calibration_images,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, object]:
    metrics = _evaluate_step_bank(kernel, calibration_cases, calibration_images, fft_backend, device_index)
    record = {
        "family": str(family),
        "config": config,
        "kernel": kernel,
        "key": _candidate_key(str(family), config),
        "bounding_radius": int(bounding_radius),
        "support_cardinality": _kernel_support_cardinality(kernel),
        "white_noise_gain": float(kernel.white_noise_gain),
        "effective_second_moment": float(_effective_second_moment(kernel)),
        "clean_fwhm": float(metrics["fwhm"]),
    }
    return record


def _square_candidate_table(
    degree: int,
    windows: tuple[int, ...],
    calibration_cases,
    calibration_images,
    fft_backend: str,
    device_index: int | None,
    log_prefix: str,
) -> list[dict[str, object]]:
    candidates = []
    for window_size in windows:
        kernel = build_square_sg(window_size=int(window_size), degree=int(degree), normalize_coords=True)
        record = _candidate_record(
            family="square_sg",
            kernel=kernel,
            config={"N": int(window_size), "d": int(degree), "normalize_coords": True},
            bounding_radius=int(window_size // 2),
            calibration_cases=calibration_cases,
            calibration_images=calibration_images,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        candidates.append(record)
        print(f"{log_prefix} degree={degree} N={window_size} fwhm={record['clean_fwhm']:.6e}")
    return candidates


def _dog_candidate_table(
    sigmas: tuple[float, ...],
    calibration_cases,
    calibration_images,
    fft_backend: str,
    device_index: int | None,
) -> list[dict[str, object]]:
    candidates = []
    for sigma in sigmas:
        kernel = build_dog(float(sigma))
        record = _candidate_record(
            family="dog",
            kernel=kernel,
            config={"sigma": float(sigma)},
            bounding_radius=int(kernel.support_half_extent),
            calibration_cases=calibration_cases,
            calibration_images=calibration_images,
            fft_backend=fft_backend,
            device_index=device_index,
        )
        candidates.append(record)
        print(f"sec84 dog_cal sigma={float(sigma):g} h={kernel.support_half_extent} fwhm={record['clean_fwhm']:.6e}")
    return candidates


def _match_candidate(rule: str, wvf_info: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    if rule == "bounding_radius":
        target = int(wvf_info["radius"])
        return min(candidates, key=lambda item: (abs(int(item["bounding_radius"]) - target), abs(float(item["clean_fwhm"]) - float(wvf_info["clean_fwhm"]))))
    metric_key = {
        "effective_response_width": "clean_fwhm",
        "support_cardinality": "support_cardinality",
        "white_noise_gain": "white_noise_gain",
        "effective_second_moment": "effective_second_moment",
    }[rule]
    target = float(wvf_info[metric_key])
    return min(candidates, key=lambda item: (abs(float(item[metric_key]) - target), abs(float(item["clean_fwhm"]) - float(wvf_info["clean_fwhm"]))))


def _serialize_candidate(candidate: dict[str, object], anisotropy_ratio: float) -> dict[str, object]:
    return {
        "config": dict(candidate["config"]),
        "bounding_radius": int(candidate["bounding_radius"]),
        "anisotropy_ratio": float(anisotropy_ratio),
        "clean_fwhm": float(candidate["clean_fwhm"]),
        "support_cardinality": int(candidate["support_cardinality"]),
        "white_noise_gain": float(candidate["white_noise_gain"]),
        "effective_second_moment": float(candidate["effective_second_moment"]),
    }


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
    noisy_eval_images = [np.asarray(add_awgn(case.image, SNR_DB, rng), dtype=np.float32) for case in eval_cases]

    wvf_infos = []
    degree_matched_square_tables: dict[int, list[dict[str, object]]] = {}
    overlay_square_candidates: list[dict[str, object]] = []
    seen_overlay_square_keys: set[tuple[object, ...]] = set()
    dog_candidates = _dog_candidate_table(
        sigmas=DOG_SIGMAS,
        calibration_cases=calibration_cases,
        calibration_images=calibration_images,
        fft_backend=fft_backend,
        device_index=device_index,
    )
    eval_cache: dict[tuple[object, ...], dict[str, float]] = {}

    for radius in RADIUS_SCHEDULE:
        degree = recommended_wvf_degree(int(radius))
        if degree not in degree_matched_square_tables:
            degree_matched_square_tables[int(degree)] = _square_candidate_table(
                degree=int(degree),
                windows=SQUARE_WINDOWS,
                calibration_cases=calibration_cases,
                calibration_images=calibration_images,
                fft_backend=fft_backend,
                device_index=device_index,
                log_prefix="sec84 square_cal",
            )
        kernel = build_wvf(radius=int(radius), degree=int(degree), normalize_coords=True)
        metrics = _evaluate_step_bank(kernel, calibration_cases, calibration_images, fft_backend, device_index)
        wvf_infos.append(
            {
                "radius": int(radius),
                "degree": int(degree),
                "kernel": kernel,
                "key": _candidate_key("wvf", {"r": int(radius), "d": int(degree)}),
                "support_cardinality": int(kernel.support_cardinality) if kernel.support_cardinality is not None else None,
                "white_noise_gain": float(kernel.white_noise_gain),
                "effective_second_moment": float(_effective_second_moment(kernel)),
                "clean_fwhm": float(metrics["fwhm"]),
            }
        )
        print(f"sec84 wvf_cal r={radius} d={degree} fwhm={metrics['fwhm']:.6e}")

    for degree in OVERLAY_SQUARE_DEGREES:
        table = _square_candidate_table(
            degree=int(degree),
            windows=OVERLAY_SQUARE_WINDOWS,
            calibration_cases=calibration_cases,
            calibration_images=calibration_images,
            fft_backend=fft_backend,
            device_index=device_index,
            log_prefix="sec84 square_overlay",
        )
        for candidate in table:
            key = tuple(candidate["key"])
            if key in seen_overlay_square_keys:
                continue
            seen_overlay_square_keys.add(key)
            overlay_square_candidates.append(candidate)

    comparator_payload: dict[str, object] = {
        "square_sg": {
            "label": "Square SG",
            "rules": {},
        },
        "dog": {
            "label": "Derivative of Gaussian",
            "rules": {},
        },
    }

    for rule in MATCH_RULES:
        square_rows = []
        dog_rows = []
        for wvf_info in wvf_infos:
            wvf_key = tuple(wvf_info["key"])
            if wvf_key not in eval_cache:
                eval_cache[wvf_key] = _evaluate_step_bank(wvf_info["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)
            wvf_eval = eval_cache[wvf_key]

            square_match = _match_candidate(str(rule), wvf_info, degree_matched_square_tables[int(wvf_info["degree"])])
            square_key = tuple(square_match["key"])
            if square_key not in eval_cache:
                eval_cache[square_key] = _evaluate_step_bank(square_match["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)
            square_eval = eval_cache[square_key]
            square_row = {
                "radius": int(wvf_info["radius"]),
                "degree": int(wvf_info["degree"]),
                "wvf": _serialize_candidate(
                    {
                        "config": {"r": int(wvf_info["radius"]), "d": int(wvf_info["degree"]), "normalize_coords": True},
                        "bounding_radius": int(wvf_info["radius"]),
                        "clean_fwhm": float(wvf_info["clean_fwhm"]),
                        "support_cardinality": int(wvf_info["support_cardinality"]),
                        "white_noise_gain": float(wvf_info["white_noise_gain"]),
                        "effective_second_moment": float(wvf_info["effective_second_moment"]),
                    },
                    anisotropy_ratio=float(wvf_eval["anisotropy_ratio"]),
                ),
                "comparator": _serialize_candidate(square_match, anisotropy_ratio=float(square_eval["anisotropy_ratio"])),
            }
            square_rows.append(square_row)

            dog_match = _match_candidate(str(rule), wvf_info, dog_candidates)
            dog_key = tuple(dog_match["key"])
            if dog_key not in eval_cache:
                eval_cache[dog_key] = _evaluate_step_bank(dog_match["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)
            dog_eval = eval_cache[dog_key]
            dog_row = {
                "radius": int(wvf_info["radius"]),
                "degree": int(wvf_info["degree"]),
                "wvf": square_row["wvf"],
                "comparator": _serialize_candidate(dog_match, anisotropy_ratio=float(dog_eval["anisotropy_ratio"])),
            }
            dog_rows.append(dog_row)

            print(
                f"sec84 rule={rule} r={int(wvf_info['radius'])} "
                f"wvfA={float(wvf_eval['anisotropy_ratio']):.6f} "
                f"squareA={float(square_eval['anisotropy_ratio']):.6f} "
                f"dogA={float(dog_eval['anisotropy_ratio']):.6f}"
            )
        comparator_payload["square_sg"]["rules"][str(rule)] = {"rows": square_rows}
        comparator_payload["dog"]["rules"][str(rule)] = {"rows": dog_rows}

    overlay_rows = []
    for target_fwhm in OVERLAY_FWHM_TARGETS:
        wvf_match = min(wvf_infos, key=lambda item: abs(float(item["clean_fwhm"]) - float(target_fwhm)))
        dog_match = min(dog_candidates, key=lambda item: abs(float(item["clean_fwhm"]) - float(target_fwhm)))
        square_match = min(overlay_square_candidates, key=lambda item: abs(float(item["clean_fwhm"]) - float(target_fwhm)))

        for item in (wvf_match, dog_match, square_match):
            key = tuple(item["key"])
            if key not in eval_cache:
                eval_cache[key] = _evaluate_step_bank(item["kernel"], eval_cases, noisy_eval_images, fft_backend, device_index)

        overlay_row = {
            "target_fwhm": float(target_fwhm),
            "wvf": _serialize_candidate(
                {
                    "config": {"r": int(wvf_match["radius"]), "d": int(wvf_match["degree"]), "normalize_coords": True},
                    "bounding_radius": int(wvf_match["radius"]),
                    "clean_fwhm": float(wvf_match["clean_fwhm"]),
                    "support_cardinality": int(wvf_match["support_cardinality"]),
                    "white_noise_gain": float(wvf_match["white_noise_gain"]),
                    "effective_second_moment": float(wvf_match["effective_second_moment"]),
                },
                anisotropy_ratio=float(eval_cache[tuple(wvf_match["key"])]["anisotropy_ratio"]),
            ),
            "dog": _serialize_candidate(dog_match, anisotropy_ratio=float(eval_cache[tuple(dog_match["key"])]["anisotropy_ratio"])),
            "square_sg": _serialize_candidate(square_match, anisotropy_ratio=float(eval_cache[tuple(square_match["key"])]["anisotropy_ratio"])),
        }
        overlay_rows.append(overlay_row)
        print(
            f"sec84 overlay fwhm={float(target_fwhm):.1f} "
            f"wvfA={overlay_row['wvf']['anisotropy_ratio']:.6f} "
            f"dogA={overlay_row['dog']['anisotropy_ratio']:.6f} "
            f"squareA={overlay_row['square_sg']['anisotropy_ratio']:.6f}"
        )

    payload = {
        "title": "Section 8.4 matched-scale diagnostics",
        "subtitle": "WVF versus square SG and derivative of Gaussian on smoothed step edges at AWGN 10 dB",
        "config": {
            "radius_schedule": list(RADIUS_SCHEDULE),
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "calibration_orientation_step_deg": CALIBRATION_ORIENTATION_STEP_DEG,
            "evaluation_orientation_step_deg": EVAL_ORIENTATION_STEP_DEG,
            "snr_db": SNR_DB,
            "fft_backend": str(fft_backend),
            "dog_sigmas": [float(value) for value in DOG_SIGMAS],
            "overlay_square_windows": list(OVERLAY_SQUARE_WINDOWS),
            "overlay_square_degrees": list(OVERLAY_SQUARE_DEGREES),
        },
        "rule_order": list(MATCH_RULES),
        "comparator_order": list(COMPARATOR_ORDER),
        "comparators": comparator_payload,
        "matched_fwhm_overlay": {
            "targets_px": list(OVERLAY_FWHM_TARGETS),
            "rows": overlay_rows,
        },
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec08_matched_scale_diagnostics.typ"
        pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_matched_scale_diagnostics.pdf"
        compile_plot(src, pdf)
        outputs[pdf.stem] = pdf

        overlay_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec08_matched_fwhm_overlay.typ"
        overlay_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_matched_fwhm_overlay.pdf"
        compile_plot(overlay_src, overlay_pdf)
        outputs[overlay_pdf.stem] = overlay_pdf
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
    parser.add_argument("--compile-plots", action="store_true", help="Compile the Typst plots after writing the summary JSON.")
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
