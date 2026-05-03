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

from baseline_filters import build_method, build_square_sg
from section8_common import (
    CONTRAST,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    add_awgn,
    apply_cases_batched,
    bilinear_sample,
    compile_plot,
    generate_junction_cases,
    generate_step_cases,
    mean,
    orientation_values,
    peak_metrics_from_profile,
    phase_values,
    sample_profile,
)


STEP_ORIENTATION_STEP_DEG = 0.5
STEP_PHASE_COUNT = 4
STEP_PHASE_STEP_PX = 0.25
STEP_SNR_LEVELS = (math.inf, 20.0, 10.0)
JUNCTION_ORIENTATION_STEP_DEG = 10.0
JUNCTION_PHASE_COUNT = 4
JUNCTION_PHASE_STEP_PX = 0.25
NOISE_SEED_BASE = 8210
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = DEFAULT_BATCH_CASES
DEGREE_MATCHED_SQUARES = (
    ("square_sg_degmatch_n21_d11", "Square SG N=21 d=11", 21, 11),
    ("square_sg_degmatch_n25_d11", "Square SG N=25 d=11", 25, 11),
)


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
                "comparison_status": "validated",
            }
        )
    for method_name, label, window_size, degree in DEGREE_MATCHED_SQUARES:
        roster.append(
            {
                "method": str(method_name),
                "label": str(label),
                "config": {"N": int(window_size), "d": int(degree), "normalize_coords": True},
                "kernel": build_square_sg(window_size=int(window_size), degree=int(degree), normalize_coords=True),
                "comparison_status": "degree_matched_not_constraint_feasible",
            }
        )
    return roster


def _noise_slug(snr_db: float) -> str:
    if math.isinf(float(snr_db)):
        return "inf"
    return f"{float(snr_db):g}".replace(".", "p")


def _directional_peak(case, gx: np.ndarray, gy: np.ndarray, support_scale: float) -> float:
    magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    profile = sample_profile(magnitude, case.line_xs, case.line_ys)
    peak_height, _, _ = peak_metrics_from_profile(
        profile=profile,
        x_coords=case.line_coords,
        phase_px=float(case.phase_px),
        support_scale=float(support_scale),
        width_px=EDGE_WIDTH_PX,
    )
    return float(peak_height)


def _evaluate_step_bank(kernel, cases, image_bank, fft_backend: str, device_index: int | None) -> tuple[list[dict[str, float]], float]:
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
    for case, (gx, gy) in zip(cases, responses, strict=True):
        peak = _directional_peak(case, gx, gy, kernel.support_half_extent)
        per_orientation.setdefault(float(case.orientation_deg), []).append(float(peak))
    curve = [
        {"theta_deg": float(theta_deg), "response": float(mean(values))}
        for theta_deg, values in sorted(per_orientation.items(), key=lambda item: item[0])
    ]
    response_values = np.asarray([row["response"] for row in curve], dtype=np.float64)
    anisotropy = float(np.max(response_values) / max(np.min(response_values), 1.0e-15))
    return curve, anisotropy


def _evaluate_junction_bank(kernel, cases, fft_backend: str, device_index: int | None) -> tuple[list[dict[str, float]], float]:
    responses = apply_cases_batched(cases, kernel, fft_backend, device_index, batch_cases=BATCH_CASES)
    per_orientation: dict[float, list[float]] = {}
    for case, (gx, gy) in zip(cases, responses, strict=True):
        magnitude = np.hypot(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
        branch_values = []
        for direction in case.branch_dirs:
            samples = [
                bilinear_sample(
                    magnitude,
                    case.center_xy[0] + float(distance) * float(direction[0]),
                    case.center_xy[1] + float(distance) * float(direction[1]),
                )
                for distance in (4.0, 6.0, 8.0)
            ]
            branch_values.append(float(np.mean(np.asarray(samples, dtype=np.float64))))
        ratio = float(max(branch_values) / max(min(branch_values), 1.0e-15))
        per_orientation.setdefault(float(case.orientation_deg), []).append(ratio)
    series = [
        {"theta_deg": float(theta_deg), "branch_isotropy_ratio": float(mean(values))}
        for theta_deg, values in sorted(per_orientation.items(), key=lambda item: item[0])
    ]
    return series, float(mean([row["branch_isotropy_ratio"] for row in series]))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_experiment(
    validation_json: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
) -> dict[str, Path]:
    validation_summary = json.loads(validation_json.read_text())
    roster = _build_roster(validation_summary)
    validation_wng = {
        str(row["method"]): float(row["metrics"]["white_noise_gain"])
        for row in validation_summary.get("method_roster", [])
    }

    step_orientations = orientation_values(STEP_ORIENTATION_STEP_DEG, span_deg=180.0)
    step_phases = phase_values(STEP_PHASE_COUNT, STEP_PHASE_STEP_PX)
    clean_step_cases = generate_step_cases(
        support_scale=MAX_SUPPORT_SCALE,
        orientations_deg=step_orientations,
        phases_px=step_phases,
    )
    clean_images = [np.asarray(case.image, dtype=np.float32) for case in clean_step_cases]
    noisy_step_banks: dict[str, list[np.ndarray]] = {"inf": clean_images}
    for idx, snr_db in enumerate(STEP_SNR_LEVELS):
        if math.isinf(float(snr_db)):
            continue
        rng = np.random.default_rng(NOISE_SEED_BASE + idx)
        noisy_step_banks[_noise_slug(float(snr_db))] = [
            np.asarray(add_awgn(case.image, float(snr_db), rng), dtype=np.float32)
            for case in clean_step_cases
        ]

    junction_orientations = orientation_values(JUNCTION_ORIENTATION_STEP_DEG, span_deg=360.0)
    junction_phases = phase_values(JUNCTION_PHASE_COUNT, JUNCTION_PHASE_STEP_PX)
    l_cases = generate_junction_cases("l_corner", junction_orientations, junction_phases)
    x_cases = generate_junction_cases("x_junction", junction_orientations, junction_phases)

    methods_payload = {}
    for item in roster:
        kernel = item["kernel"]
        step_payload = {"anisotropy_by_snr": {}, "clean_curve": []}
        for snr_db in STEP_SNR_LEVELS:
            slug = _noise_slug(float(snr_db))
            curve, anisotropy = _evaluate_step_bank(
                kernel=kernel,
                cases=clean_step_cases,
                image_bank=noisy_step_banks[slug],
                fft_backend=fft_backend,
                device_index=device_index,
            )
            step_payload["anisotropy_by_snr"][slug] = float(anisotropy)
            if math.isinf(float(snr_db)):
                step_payload["clean_curve"] = curve
            print(f"sec831 {item['method']} step snr={slug} anisotropy={anisotropy:.6f}")

        l_series, l_iso = _evaluate_junction_bank(kernel, l_cases, fft_backend, device_index)
        x_series, x_iso = _evaluate_junction_bank(kernel, x_cases, fft_backend, device_index)
        print(f"sec831 {item['method']} junction L={l_iso:.6f} X={x_iso:.6f}")
        methods_payload[str(item["method"])] = {
            "label": str(item["label"]),
            "config": dict(item["config"]),
            "comparison_status": str(item["comparison_status"]),
            "step": step_payload,
            "junctions": {
                "l_corner": {"branch_isotropy_mean": float(l_iso), "series": l_series},
                "x_junction": {"branch_isotropy_mean": float(x_iso), "series": x_series},
            },
        }

    reporting_table = []
    for item in roster:
        method_key = str(item["method"])
        kernel = item["kernel"]
        reporting_table.append(
            {
                "method": method_key,
                "label": str(item["label"]),
                "comparison_status": str(item["comparison_status"]),
                "geometric_anisotropy_clean": float(methods_payload[method_key]["step"]["anisotropy_by_snr"]["inf"]),
                "noise_robustness_wng": float(validation_wng.get(method_key, kernel.white_noise_gain)),
                "junction_branch_isotropy_l_corner": float(methods_payload[method_key]["junctions"]["l_corner"]["branch_isotropy_mean"]),
                "junction_branch_isotropy_x_junction": float(methods_payload[method_key]["junctions"]["x_junction"]["branch_isotropy_mean"]),
            }
        )

    payload = {
        "title": "Section 8.3.1 rotational equivariance and anisotropy",
        "subtitle": "Validation-tuned head-to-head comparison on step edges, L-corners, and X-junctions",
        "config": {
            "step_orientation_step_deg": STEP_ORIENTATION_STEP_DEG,
            "step_phase_count": STEP_PHASE_COUNT,
            "step_phase_step_px": STEP_PHASE_STEP_PX,
            "step_snr_levels": ["inf", "20", "10"],
            "junction_orientation_step_deg": JUNCTION_ORIENTATION_STEP_DEG,
            "junction_phase_count": JUNCTION_PHASE_COUNT,
            "junction_phase_step_px": JUNCTION_PHASE_STEP_PX,
            "fft_backend": str(fft_backend),
        },
        "method_order": [str(item["method"]) for item in roster],
        "reporting_table": reporting_table,
        "methods": methods_payload,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figures = (
            ("fig_sec08_rotational_response_curves.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_rotational_response_curves.pdf"),
            ("fig_sec08_rotational_anisotropy_bar.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_rotational_anisotropy_bar.pdf"),
            ("fig_sec08_junction_branch_isotropy.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_junction_branch_isotropy.pdf"),
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
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_rotational_equivariance" / "sec08_rotational_equivariance_summary.json",
        help="Path for the rotational-equivariance summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", help="FFT backend for applying precomputed kernels.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for FFT execution.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the Typst plots after writing the summary JSON.")
    args = parser.parse_args(argv)
    run_experiment(
        validation_json=args.validation_json.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
