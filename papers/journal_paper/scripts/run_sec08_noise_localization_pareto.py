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

from baseline_filters import build_method, build_wvf, recommended_wvf_degree
from section8_common import (
    CONTRAST,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    add_awgn,
    apply_cases_batched,
    case_gradient_metrics,
    compile_plot,
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
SNR_LEVELS = (math.inf, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 7.5, 5.0, 2.5, 1.0, 0.5, 0.0)
NOISE_SEED_BASE = 8320
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = DEFAULT_BATCH_CASES
WVF_TRACE_RADII = (3, 5, 9, 15, 25, 50)


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


def _evaluate_step_metrics(kernel, cases, image_bank, fft_backend: str, device_index: int | None) -> dict[str, float]:
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
    grad_rmses: list[float] = []
    ang_maes: list[float] = []
    fwhm_values: list[float] = []
    for case, (gx, gy) in zip(cases, responses, strict=True):
        metrics = case_gradient_metrics(case, gx, gy)
        grad_rmses.append(float(metrics["grad_rmse"]))
        ang_maes.append(float(metrics["ang_mae_deg"]))
        theta = math.radians(float(case.orientation_deg))
        directional = np.asarray(gx, dtype=np.float64) * math.cos(theta) + np.asarray(gy, dtype=np.float64) * math.sin(theta)
        profile = sample_profile(directional, case.line_xs, case.line_ys)
        _, _, fwhm = peak_metrics_from_profile(
            profile=profile,
            x_coords=case.line_coords,
            phase_px=float(case.phase_px),
            support_scale=float(kernel.support_half_extent),
            width_px=EDGE_WIDTH_PX,
        )
        fwhm_values.append(float(fwhm))
    return {
        "white_noise_gain": float(kernel.white_noise_gain),
        "fwhm": float(mean(fwhm_values)),
        "grad_rmse": float(mean(grad_rmses)),
        "ang_mae_deg": float(mean(ang_maes)),
    }


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

    orientations = orientation_values(STEP_ORIENTATION_STEP_DEG, span_deg=180.0)
    phases = phase_values(STEP_PHASE_COUNT, STEP_PHASE_STEP_PX)
    clean_cases = generate_step_cases(
        support_scale=MAX_SUPPORT_SCALE,
        orientations_deg=orientations,
        phases_px=phases,
    )
    image_banks: dict[str, list[np.ndarray]] = {"inf": [np.asarray(case.image, dtype=np.float32) for case in clean_cases]}
    for idx, snr_db in enumerate(SNR_LEVELS):
        if math.isinf(float(snr_db)):
            continue
        rng = np.random.default_rng(NOISE_SEED_BASE + idx)
        image_banks[_noise_slug(float(snr_db))] = [
            np.asarray(add_awgn(case.image, float(snr_db), rng), dtype=np.float32)
            for case in clean_cases
        ]

    methods_payload = {}
    for item in roster:
        metrics_by_snr = {}
        for snr_db in SNR_LEVELS:
            slug = _noise_slug(float(snr_db))
            metrics = _evaluate_step_metrics(item["kernel"], clean_cases, image_banks[slug], fft_backend, device_index)
            metrics_by_snr[slug] = metrics
            print(f"sec832 {item['method']} snr={slug} rmse={metrics['grad_rmse']:.6e} fwhm={metrics['fwhm']:.6e}")
        methods_payload[str(item["method"])] = {
            "label": str(item["label"]),
            "config": dict(item["config"]),
            "snr_metrics": metrics_by_snr,
        }

    wvf_trace = []
    for radius in WVF_TRACE_RADII:
        degree = recommended_wvf_degree(int(radius))
        kernel = build_wvf(radius=int(radius), degree=int(degree), normalize_coords=True)
        metrics = _evaluate_step_metrics(kernel, clean_cases, image_banks["inf"], fft_backend, device_index)
        wvf_trace.append(
            {
                "radius": int(radius),
                "degree": int(degree),
                "white_noise_gain": float(metrics["white_noise_gain"]),
                "fwhm": float(metrics["fwhm"]),
            }
        )
        print(f"sec832 wvf_trace r={radius} d={degree} fwhm={metrics['fwhm']:.6e} wng={metrics['white_noise_gain']:.6e}")

    payload = {
        "title": "Section 8.3.2 noise-localisation Pareto",
        "subtitle": "Validation-tuned head-to-head comparison on smoothed step edges under AWGN",
        "config": {
            "orientation_step_deg": STEP_ORIENTATION_STEP_DEG,
            "phase_count": STEP_PHASE_COUNT,
            "phase_step_px": STEP_PHASE_STEP_PX,
            "snr_levels": ["inf", "30", "25", "20", "15", "12", "10", "7p5", "5", "2p5", "1", "0p5", "0"],
            "fft_backend": str(fft_backend),
        },
        "method_order": [str(item["method"]) for item in roster],
        "methods": methods_payload,
        "wvf_trace": wvf_trace,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figures = (
            ("fig_sec08_noise_localization_pareto.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_noise_localization_pareto.pdf"),
            ("fig_sec08_noise_rmse_vs_snr.typ", ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_noise_rmse_vs_snr.pdf"),
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
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_noise_localization_pareto" / "sec08_noise_localization_pareto_summary.json",
        help="Path for the noise-localisation summary JSON.",
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
