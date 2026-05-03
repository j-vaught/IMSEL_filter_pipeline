#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baseline_filters import build_method
from section8_common import (
    DEFAULT_BATCH_CASES,
    CurvedCase,
    StepCase,
    add_awgn,
    apply_cases_batched,
    case_gradient_metrics,
    compile_plot,
    generate_curved_cases,
    generate_step_cases,
    mean,
    orientation_values,
    phase_values,
)


TITLE = "Section 7 edge-width transfer sweep"
SUBTITLE = "WVF radius-width interaction under fixed-noise smoothed-edge conditions"
EDGE_WIDTHS_PX = (1.0, 3.0, 9.0, 27.0)
WVF_TRACE = (
    {"radius": 3, "degree": 5},
    {"radius": 5, "degree": 9},
    {"radius": 9, "degree": 11},
    {"radius": 15, "degree": 11},
    {"radius": 25, "degree": 11},
    {"radius": 50, "degree": 11},
)
SNR_DB = 10.0
NOISE_DRAWS = 100
ORIENTATION_STEP_DEG = 5.0
PHASE_COUNT = 4
PHASE_STEP_PX = 0.25
CURVATURE_RADIUS_FACTOR = 4.0
EDGE_SIGMA_FACTOR = 0.25
NOISE_SEED_BASE = 9700
NORMALIZE_COORDS = True
DEFAULT_FFT_BACKEND = "vkfft"
DEFAULT_BATCH_CASES = 256


def _parse_float_list(text: str | None) -> tuple[float, ...]:
    if text is None:
        return tuple()
    values = []
    for raw in text.split(","):
        item = raw.strip()
        if item:
            values.append(float(item))
    return tuple(values)


def _parse_radius_degree_pairs(text: str | None) -> tuple[dict[str, int], ...]:
    if text is None:
        return tuple()
    pairs: list[dict[str, int]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        radius_text, degree_text = item.split(":", 1)
        pairs.append({"radius": int(radius_text), "degree": int(degree_text)})
    return tuple(pairs)


def _edge_sigma_px(edge_width_px: float) -> float:
    return float(edge_width_px) * float(EDGE_SIGMA_FACTOR)


def _patch_half_size(radius: int, edge_width_px: float) -> int:
    return int(max(64, math.ceil(3.0 * float(radius) + 2.0 * float(edge_width_px))))


def _clone_step_case(case: StepCase, image: np.ndarray) -> StepCase:
    return replace(case, image=np.asarray(image, dtype=np.float32))


def _clone_curved_case(case: CurvedCase, image: np.ndarray) -> CurvedCase:
    return replace(case, image=np.asarray(image, dtype=np.float32))


def _evaluate_step_rmse(
    kernel,
    clean_cases: list[StepCase],
    snr_db: float,
    noise_draws: int,
    fft_backend: str,
    device_index: int | None,
    seed_offset: int,
    batch_cases: int,
) -> float:
    draw_means: list[float] = []
    for draw_index in range(int(noise_draws)):
        rng = np.random.default_rng(int(NOISE_SEED_BASE + seed_offset + draw_index))
        noisy_cases = [
            _clone_step_case(case, add_awgn(np.asarray(case.image, dtype=np.float64), float(snr_db), rng))
            for case in clean_cases
        ]
        responses = apply_cases_batched(
            noisy_cases,
            kernel,
            fft_backend,
            device_index,
            batch_cases=int(max(batch_cases, len(noisy_cases))),
        )
        case_rmses = [
            float(case_gradient_metrics(clean_case, gx, gy)["grad_rmse"])
            for clean_case, (gx, gy) in zip(clean_cases, responses, strict=True)
        ]
        draw_means.append(float(mean(case_rmses)))
    return float(mean(draw_means))


def _evaluate_arc_orientation_mae(
    kernel,
    clean_cases: list[CurvedCase],
    snr_db: float,
    noise_draws: int,
    fft_backend: str,
    device_index: int | None,
    seed_offset: int,
    batch_cases: int,
) -> float:
    draw_means: list[float] = []
    for draw_index in range(int(noise_draws)):
        rng = np.random.default_rng(int(NOISE_SEED_BASE + 500000 + seed_offset + draw_index))
        noisy_cases = [
            _clone_curved_case(case, add_awgn(np.asarray(case.image, dtype=np.float64), float(snr_db), rng))
            for case in clean_cases
        ]
        responses = apply_cases_batched(
            noisy_cases,
            kernel,
            fft_backend,
            device_index,
            batch_cases=int(max(batch_cases, len(noisy_cases))),
        )
        case_maes = [
            float(case_gradient_metrics(clean_case, gx, gy)["ang_mae_deg"])
            for clean_case, (gx, gy) in zip(clean_cases, responses, strict=True)
        ]
        draw_means.append(float(mean(case_maes)))
    return float(mean(draw_means))


def _best_by_width(cells: list[dict[str, object]], widths_px: tuple[float, ...]) -> list[dict[str, object]]:
    rows = []
    for edge_width_px in widths_px:
        width_cells = [cell for cell in cells if float(cell["edge_width_px"]) == float(edge_width_px)]
        step_best = min(width_cells, key=lambda cell: float(cell["step_grad_rmse"]))
        arc_best = min(width_cells, key=lambda cell: float(cell["arc_orientation_mae_deg"]))
        lower = 3.0 * float(edge_width_px)
        upper = 5.0 * float(edge_width_px)
        rows.append(
            {
                "edge_width_px": float(edge_width_px),
                "hypothesis_radius_band_px": [float(lower), float(upper)],
                "best_step_radius": int(step_best["radius"]),
                "best_step_degree": int(step_best["degree"]),
                "best_step_grad_rmse": float(step_best["step_grad_rmse"]),
                "best_step_in_band": bool(lower <= float(step_best["radius"]) <= upper),
                "best_arc_radius": int(arc_best["radius"]),
                "best_arc_degree": int(arc_best["degree"]),
                "best_arc_orientation_mae_deg": float(arc_best["arc_orientation_mae_deg"]),
                "best_arc_in_band": bool(lower <= float(arc_best["radius"]) <= upper),
            }
        )
    return rows


def run_experiment(
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
    compile_plots: bool,
    edge_widths_px: tuple[float, ...],
    trace_points: tuple[dict[str, int], ...],
    snr_db: float,
    noise_draws: int,
    orientation_step_deg: float,
    phase_count: int,
    phase_step_px: float,
    curvature_radius_factor: float,
    batch_cases: int,
) -> dict[str, Path]:
    widths = tuple(float(value) for value in edge_widths_px)
    trace = tuple({"radius": int(item["radius"]), "degree": int(item["degree"])} for item in trace_points)
    orientations = orientation_values(float(orientation_step_deg), span_deg=180.0)
    phases = phase_values(int(phase_count), float(phase_step_px))

    cells: list[dict[str, object]] = []
    for width_index, edge_width_px in enumerate(widths):
        sigma_px = _edge_sigma_px(float(edge_width_px))
        for trace_index, trace_point in enumerate(trace):
            radius = int(trace_point["radius"])
            degree = int(trace_point["degree"])
            curvature_radius_px = int(round(float(curvature_radius_factor) * float(radius)))
            kernel = build_method("wvf", r=radius, d=degree, normalize_coords=bool(NORMALIZE_COORDS))
            half_size = _patch_half_size(int(radius), float(edge_width_px))

            step_cases = generate_step_cases(
                support_scale=float(radius),
                orientations_deg=orientations,
                phases_px=phases,
                half_size=int(half_size),
                width_px=float(sigma_px),
            )
            arc_cases = [
                case
                for case in generate_curved_cases(
                    support_scale=float(radius),
                    curvature_radii=(int(curvature_radius_px),),
                    orientations_deg=orientations,
                    phases_px=phases,
                    half_size=int(half_size),
                    width_px=float(sigma_px),
                )
                if str(case.stimulus_class) == "arc"
            ]

            seed_offset = 10000 * width_index + 1000 * trace_index
            step_grad_rmse = _evaluate_step_rmse(
                kernel=kernel,
                clean_cases=step_cases,
                snr_db=float(snr_db),
                noise_draws=int(noise_draws),
                fft_backend=fft_backend,
                device_index=device_index,
                seed_offset=int(seed_offset),
                batch_cases=int(batch_cases),
            )
            arc_orientation_mae_deg = _evaluate_arc_orientation_mae(
                kernel=kernel,
                clean_cases=arc_cases,
                snr_db=float(snr_db),
                noise_draws=int(noise_draws),
                fft_backend=fft_backend,
                device_index=device_index,
                seed_offset=int(seed_offset),
                batch_cases=int(batch_cases),
            )
            row = {
                "edge_width_px": float(edge_width_px),
                "edge_sigma_px": float(sigma_px),
                "radius": int(radius),
                "degree": int(degree),
                "curvature_radius_px": int(curvature_radius_px),
                "support_half_extent": int(kernel.support_half_extent),
                "white_noise_gain": float(kernel.white_noise_gain),
                "step_grad_rmse": float(step_grad_rmse),
                "arc_orientation_mae_deg": float(arc_orientation_mae_deg),
            }
            cells.append(row)
            print(
                f"sec07width w={edge_width_px:g} r={radius} d={degree} "
                f"step_rmse={step_grad_rmse:.6e} arc_ang={arc_orientation_mae_deg:.6e}"
            )

    best_rows = _best_by_width(cells, widths)
    payload = {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "config": {
            "edge_widths_px": list(widths),
            "wvf_trace": [dict(item) for item in trace],
            "snr_db": float(snr_db),
            "noise_draws": int(noise_draws),
            "orientation_step_deg": float(orientation_step_deg),
            "phase_count": int(phase_count),
            "phase_step_px": float(phase_step_px),
            "curvature_radius_rule": f"rho = {float(curvature_radius_factor):g} * r",
            "edge_profile_rule": "tanh profile with sigma_e = w / 4",
            "normalize_coords": bool(NORMALIZE_COORDS),
            "fft_backend": str(fft_backend),
        },
        "cells": cells,
        "best_by_width": best_rows,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    outputs: dict[str, Path] = {"summary_json": summary_json}
    if compile_plots:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_edge_width_transfer.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_edge_width_transfer.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["figure_pdf"] = figure_pdf
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_edge_width_transfer" / "sec07_edge_width_transfer_summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        default=DEFAULT_FFT_BACKEND,
        choices=("vkfft", "cpu"),
        help="FFT backend for the WVF application path.",
    )
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plots", action="store_true", help="Compile the checked-in Typst/CeTZ figure.")
    parser.add_argument("--edge-widths", type=str, default=None, help="Comma-separated conceptual edge widths in pixels.")
    parser.add_argument("--wvf-trace", type=str, default=None, help="Comma-separated radius:degree pairs.")
    parser.add_argument("--snr-db", type=float, default=float(SNR_DB), help="AWGN SNR in dB.")
    parser.add_argument("--noise-draws", type=int, default=int(NOISE_DRAWS), help="Number of AWGN draws per cell.")
    parser.add_argument("--orientation-step-deg", type=float, default=float(ORIENTATION_STEP_DEG), help="Orientation sampling step in degrees.")
    parser.add_argument("--phase-count", type=int, default=int(PHASE_COUNT), help="Number of sub-pixel phases.")
    parser.add_argument("--phase-step-px", type=float, default=float(PHASE_STEP_PX), help="Phase spacing in pixels.")
    parser.add_argument(
        "--curvature-radius-factor",
        type=float,
        default=float(CURVATURE_RADIUS_FACTOR),
        help="Curvature radius factor relative to support radius.",
    )
    parser.add_argument("--batch-cases", type=int, default=int(DEFAULT_BATCH_CASES), help="Minimum batch size request for FFT tiling.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    edge_widths = _parse_float_list(args.edge_widths) or tuple(float(value) for value in EDGE_WIDTHS_PX)
    trace_points = _parse_radius_degree_pairs(args.wvf_trace) or tuple(dict(item) for item in WVF_TRACE)
    run_experiment(
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
        compile_plots=bool(args.compile_plots),
        edge_widths_px=edge_widths,
        trace_points=trace_points,
        snr_db=float(args.snr_db),
        noise_draws=int(args.noise_draws),
        orientation_step_deg=float(args.orientation_step_deg),
        phase_count=int(args.phase_count),
        phase_step_px=float(args.phase_step_px),
        curvature_radius_factor=float(args.curvature_radius_factor),
        batch_cases=int(args.batch_cases),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
