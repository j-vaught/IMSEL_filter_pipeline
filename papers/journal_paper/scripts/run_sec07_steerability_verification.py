#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.taylor import build_taylor_matrix, compute_pseudoinverse, rotate_coordinates
from wvf.radius import build_wvf_radius_kernels


ANGLES_DEG = tuple(range(0, 180, 5))
EPS64 = float(np.finfo(np.float64).eps)
DEFAULT_CONFIGS = (
    (5, 3, True),
    (15, 5, True),
    (50, 7, True),
    (5, 3, False),
)


@dataclass(frozen=True)
class SteerabilityConfig:
    radius: int
    degree: int
    normalize_coords: bool

    @property
    def slug(self) -> str:
        mode = "normalized" if self.normalize_coords else "unnormalized"
        return f"disk_r{self.radius}_d{self.degree}_{mode}"

    @property
    def label(self) -> str:
        mode = "normalize_coords=True" if self.normalize_coords else "normalize_coords=False"
        return f"(r={self.radius}, d={self.degree}, {mode})"


def _direct_rotated_weights(
    offsets_xy: np.ndarray,
    radius: int,
    degree: int,
    theta_deg: float,
    normalize_coords: bool,
) -> np.ndarray:
    theta_rad = math.radians(float(theta_deg))
    rotated_xy = rotate_coordinates(offsets_xy, theta_rad)
    design = build_taylor_matrix(
        rotated_xy,
        order=int(degree),
        normalize_radius=float(radius) if normalize_coords else None,
    )
    pinv = compute_pseudoinverse(design)
    derivative_scale = 1.0 / float(radius) if normalize_coords else 1.0
    return np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)


def _plot_range(values: list[float]) -> tuple[float, float, float, list[float]]:
    positives = [value for value in values if value > 0.0]
    floor = min(positives) if positives else EPS64
    plot_floor = max(floor / 10.0, EPS64 * 1.0e-2)
    floored = [max(value, plot_floor) for value in values]

    log_min = math.floor(math.log10(min(floored)))
    log_max = math.ceil(math.log10(max(floored)))
    if log_min == log_max:
        log_min -= 1
        log_max += 1
    ticks = [10.0**exponent for exponent in range(log_min, log_max + 1)]
    return plot_floor, float(log_min), float(log_max), ticks


def _write_csv(csv_path: Path, records: list[dict[str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("theta_deg", "residual", "kernel_max"))
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "theta_deg": int(row["theta_deg"]),
                    "residual": f"{row['residual']:.17e}",
                    "kernel_max": f"{row['kernel_max']:.17e}",
                }
            )


def _write_json(json_path: Path, payload: dict[str, object]) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_config(config: SteerabilityConfig, output_dir: Path) -> tuple[Path, Path, int, int]:
    kernels = build_wvf_radius_kernels(
        radius=config.radius,
        order=config.degree,
        normalize_coords=config.normalize_coords,
    )
    weights_x = np.asarray(kernels.weights_x, dtype=np.float64)
    weights_y = np.asarray(kernels.weights_y, dtype=np.float64)
    offsets_xy = np.asarray(kernels.offsets_xy, dtype=np.float64)

    records: list[dict[str, float]] = []
    pass_count = 0
    total_count = len(ANGLES_DEG)

    for theta_deg in ANGLES_DEG:
        theta_rad = math.radians(float(theta_deg))
        synth = weights_x * math.cos(theta_rad) + weights_y * math.sin(theta_rad)
        direct = _direct_rotated_weights(
            offsets_xy,
            config.radius,
            config.degree,
            float(theta_deg),
            config.normalize_coords,
        )
        residual = float(np.max(np.abs(synth - direct)))
        kernel_max = float(np.max(np.abs(synth)))
        threshold = 10.0 * EPS64 * kernel_max
        passed = residual < threshold
        if passed:
            pass_count += 1
        records.append(
            {
                "theta_deg": float(theta_deg),
                "residual": residual,
                "kernel_max": kernel_max,
                "threshold": threshold,
                "passed": 1.0 if passed else 0.0,
            }
        )

    plot_floor, log10_min, log10_max, y_ticks = _plot_range(
        [float(record["residual"]) for record in records]
    )
    for record in records:
        plot_value = max(float(record["residual"]), plot_floor)
        record["plot_residual"] = plot_value
        record["log10_plot_residual"] = float(math.log10(plot_value))

    csv_path = output_dir / f"sec07_steerability_{config.slug}.csv"
    json_path = output_dir / f"sec07_steerability_{config.slug}.json"

    _write_csv(csv_path, records)
    _write_json(
        json_path,
        {
            "radius": config.radius,
            "degree": config.degree,
            "normalize_coords": config.normalize_coords,
            "pass_count": pass_count,
            "total_count": total_count,
            "plot": {
                "x_ticks": [0, 25, 50, 75, 100, 125, 150, 175],
                "y_ticks": [
                    {
                        "value": float(value),
                        "label": f"1e{int(round(math.log10(value)))}",
                        "log10_value": float(math.log10(value)),
                    }
                    for value in y_ticks
                ],
                "log10_min": log10_min,
                "log10_max": log10_max,
                "plot_floor": float(plot_floor),
            },
            "records": records,
        },
    )

    print(f"{config.label}: {pass_count}/{total_count} passed")
    return csv_path, json_path, pass_count, total_count


def compile_plots(figures_dir: Path, configs: tuple[SteerabilityConfig, ...]) -> list[Path]:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so plots cannot be compiled")

    outputs: list[Path] = []
    cetz_src = figures_dir / "cetz_src"
    for config in configs:
        src_path = cetz_src / f"fig_sec07_steerability_{config.slug}.typ"
        out_path = figures_dir / f"fig_sec07_steerability_{config.slug}.pdf"
        subprocess.run(
            [typst, "compile", "--root", str(ROOT), str(src_path), str(out_path)],
            check=True,
            cwd=ROOT,
        )
        outputs.append(out_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 7.2 steerability verification for disk WVF kernels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_steerability",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--compile-plots",
        action="store_true",
        help="Compile the checked-in Typst/CeTZ plot wrappers to PDF after data generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = tuple(SteerabilityConfig(*values) for values in DEFAULT_CONFIGS)
    csv_paths: list[Path] = []
    json_paths: list[Path] = []
    failures = 0

    for config in configs:
        csv_path, json_path, pass_count, total_count = run_config(config, output_dir)
        csv_paths.append(csv_path)
        json_paths.append(json_path)
        if pass_count != total_count:
            failures += 1

    if args.compile_plots:
        figure_paths = compile_plots(ROOT / "papers" / "journal_paper" / "figures", configs)
        for figure_path in figure_paths:
            print(f"wrote {figure_path}")

    for csv_path in csv_paths:
        print(f"wrote {csv_path}")
    for json_path in json_paths:
        print(f"wrote {json_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
