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

from baseline_filters import build_method
from section8_common import (
    CONTRAST,
    CURVE_PATCH_HALF_SIZE,
    DEFAULT_BATCH_CASES,
    EDGE_WIDTH_PX,
    add_awgn,
    apply_cases_batched,
    case_gradient_metrics,
    compile_plot,
    generate_curved_cases,
    mean,
    orientation_values,
    phase_values,
)


CURVATURE_RADII = (20, 50, 100, 200)
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
SNR_DB = 10.0
NOISE_SEED = 8430
MAX_SUPPORT_SCALE = 50.0
BATCH_CASES = DEFAULT_BATCH_CASES


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


def _evaluate_curved_metrics(kernel, cases, image_bank, fft_backend: str, device_index: int | None) -> dict[int, dict[str, float]]:
    eval_cases = [
        type(case)(
            stimulus_class=case.stimulus_class,
            curvature_radius=case.curvature_radius,
            orientation_deg=case.orientation_deg,
            phase_px=case.phase_px,
            image=np.asarray(image, dtype=np.float32),
            true_gx=case.true_gx,
            true_gy=case.true_gy,
            eval_mask=case.eval_mask,
        )
        for case, image in zip(cases, image_bank, strict=True)
    ]
    responses = apply_cases_batched(eval_cases, kernel, fft_backend, device_index, batch_cases=BATCH_CASES)
    grouped: dict[int, dict[str, list[float]]] = {}
    for case, (gx, gy) in zip(cases, responses, strict=True):
        metrics = case_gradient_metrics(case, gx, gy)
        slot = grouped.setdefault(
            int(case.curvature_radius),
            {"grad_rmse": [], "ang_mae_deg": [], "mag_bias": []},
        )
        slot["grad_rmse"].append(float(metrics["grad_rmse"]))
        slot["ang_mae_deg"].append(float(metrics["ang_mae_deg"]))
        slot["mag_bias"].append(float(metrics["mag_bias"]))
    return {
        int(curvature_radius): {
            "grad_rmse": float(mean(values["grad_rmse"])),
            "ang_mae_deg": float(mean(values["ang_mae_deg"])),
            "mag_bias": float(mean(values["mag_bias"])),
        }
        for curvature_radius, values in grouped.items()
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

    orientations = orientation_values(ORIENTATION_STEP_DEG, span_deg=180.0)
    phases = phase_values(PHASE_COUNT, PHASE_STEP_PX)
    clean_cases = generate_curved_cases(
        support_scale=MAX_SUPPORT_SCALE,
        curvature_radii=CURVATURE_RADII,
        orientations_deg=orientations,
        phases_px=phases,
        half_size=CURVE_PATCH_HALF_SIZE,
        contrast=CONTRAST,
        width_px=EDGE_WIDTH_PX,
    )
    rng = np.random.default_rng(NOISE_SEED)
    noisy_images = [
        np.asarray(add_awgn(case.image, SNR_DB, rng), dtype=np.float32)
        for case in clean_cases
    ]

    methods_payload = {}
    for item in roster:
        metrics_by_curvature = _evaluate_curved_metrics(item["kernel"], clean_cases, noisy_images, fft_backend, device_index)
        methods_payload[str(item["method"])] = {
            "label": str(item["label"]),
            "config": dict(item["config"]),
            "curvature_metrics": metrics_by_curvature,
        }
        for curvature_radius in CURVATURE_RADII:
            print(
                f"sec833 {item['method']} rho={curvature_radius} "
                f"rmse={metrics_by_curvature[int(curvature_radius)]['grad_rmse']:.6e}"
            )

    payload = {
        "title": "Section 8.3.3 curvature handling",
        "subtitle": "Validation-tuned head-to-head comparison on smoothed arcs and S-curves at AWGN 10 dB",
        "config": {
            "curvature_radii": list(CURVATURE_RADII),
            "orientation_step_deg": ORIENTATION_STEP_DEG,
            "phase_count": PHASE_COUNT,
            "phase_step_px": PHASE_STEP_PX,
            "snr_db": SNR_DB,
            "fft_backend": str(fft_backend),
        },
        "method_order": [str(item["method"]) for item in roster],
        "methods": methods_payload,
    }
    _write_json(summary_json, payload)

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec08_curvature_handling.typ"
        pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec08_curvature_handling.pdf"
        compile_plot(src, pdf)
        outputs[pdf.stem] = pdf
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
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec08_curvature_handling" / "sec08_curvature_handling_summary.json",
        help="Path for the curvature-handling summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default="vkfft", help="FFT backend for applying precomputed kernels.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index for FFT execution.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the Typst plot after writing the summary JSON.")
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
