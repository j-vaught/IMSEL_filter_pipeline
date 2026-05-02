#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wvf.radius import build_wvf_radius_kernels
from wvf_metal.metal import fft_gradients_with_kernel

IMAGE_SIZE = 1024
RADIUS = 15
DEGREE = 11
NORMALIZE_COORDS = True
CONTRAST = 1.0
EDGE_WIDTH_PX = 1.5
ROTATION_STEP_DEG = 15.0
BATCH_CASES = 4
VALID_MARGIN_PX = 64.0
DEFAULT_FFT_BACKEND = "vkfft"


@dataclass(frozen=True)
class SceneCase:
    angle_deg: float
    image: np.ndarray


def _rotation_values(step_deg: float) -> tuple[float, ...]:
    count = int(round(360.0 / float(step_deg)))
    return tuple(float(step_deg) * idx for idx in range(count))


def _local_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float64) - 0.5 * float(size - 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def _sigmoid(value: np.ndarray, width_px: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(value, dtype=np.float64) / float(width_px)))


def _component_junction(u: np.ndarray, v: np.ndarray, components: tuple[tuple[str, float], ...]) -> np.ndarray:
    accum = np.zeros_like(u, dtype=np.float64)
    for kind, angle_deg in components:
        angle_rad = math.radians(float(angle_deg))
        tangent = np.asarray(u, dtype=np.float64) * math.cos(angle_rad) + np.asarray(v, dtype=np.float64) * math.sin(angle_rad)
        normal = -np.asarray(u, dtype=np.float64) * math.sin(angle_rad) + np.asarray(v, dtype=np.float64) * math.cos(angle_rad)
        component = _sigmoid(normal, EDGE_WIDTH_PX)
        if kind == "ray":
            component = component * _sigmoid(tangent, EDGE_WIDTH_PX)
        accum += component
    return accum / float(len(components))


def _rotated_local(xx: np.ndarray, yy: np.ndarray, center_xy: tuple[float, float], theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(theta_deg))
    shifted_x = np.asarray(xx, dtype=np.float64) - float(center_xy[0])
    shifted_y = np.asarray(yy, dtype=np.float64) - float(center_xy[1])
    u = shifted_x * math.cos(theta) + shifted_y * math.sin(theta)
    v = -shifted_x * math.sin(theta) + shifted_y * math.cos(theta)
    return u, v


def _canonical_scene(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    image = np.zeros_like(xx, dtype=np.float64)

    junction_specs = (
        ((-260.0, -170.0), 20.0, (("full", 90.0), ("ray", 180.0))),   # T
        ((220.0, -190.0), -15.0, (("full", 0.0), ("full", 90.0))),    # X
        ((-10.0, 160.0), 35.0, (("full", 90.0), ("ray", 180.0), ("ray", 270.0))),  # mixed Y-like but lattice-safe T blend
        ((260.0, 220.0), 0.0, (("ray", 0.0), ("full", 90.0))),        # L-ish
    )
    for center_xy, angle_deg, components in junction_specs:
        u, v = _rotated_local(xx, yy, center_xy, angle_deg)
        image = np.maximum(image, CONTRAST * _component_junction(u, v, components))

    disk_specs = (
        ((-300.0, 260.0), 16.0),
        ((-180.0, 300.0), 32.0),
        ((120.0, -20.0), 24.0),
        ((320.0, 60.0), 48.0),
        ((60.0, 300.0), 72.0),
        ((-320.0, 20.0), 96.0),
    )
    for center_xy, diameter in disk_specs:
        radius = 0.5 * float(diameter)
        mask = (np.asarray(xx, dtype=np.float64) - float(center_xy[0])) ** 2 + (np.asarray(yy, dtype=np.float64) - float(center_xy[1])) ** 2 <= radius * radius
        image[mask] = float(CONTRAST)

    return np.asarray(np.clip(image, 0.0, float(CONTRAST)), dtype=np.float32)


def _render_rotated_scene(xx: np.ndarray, yy: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    inv_x = np.asarray(xx, dtype=np.float64) * math.cos(theta) + np.asarray(yy, dtype=np.float64) * math.sin(theta)
    inv_y = -np.asarray(xx, dtype=np.float64) * math.sin(theta) + np.asarray(yy, dtype=np.float64) * math.cos(theta)
    return _canonical_scene(inv_x, inv_y)


def _build_cases() -> list[SceneCase]:
    xx, yy = _local_grid(IMAGE_SIZE)
    return [SceneCase(angle_deg=float(angle), image=_render_rotated_scene(xx, yy, float(angle))) for angle in _rotation_values(ROTATION_STEP_DEG)]


def _build_kernel() -> tuple[np.ndarray, np.ndarray]:
    kernels = build_wvf_radius_kernels(int(RADIUS), order=int(DEGREE), normalize_coords=bool(NORMALIZE_COORDS))
    return np.asarray(kernels.kernel_x, dtype=np.float64), np.asarray(kernels.kernel_y, dtype=np.float64)


def _tile_cases(images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    tile_h, tile_w = images[0].shape
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=np.float32)
    placements: list[tuple[slice, slice]] = []
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        row_slice = slice(row * tile_h, (row + 1) * tile_h)
        col_slice = slice(col * tile_w, (col + 1) * tile_w)
        canvas[row_slice, col_slice] = np.asarray(image, dtype=np.float32)
        placements.append((row_slice, col_slice))
    return canvas, placements


def _apply_batched_gradients(
    images: list[np.ndarray],
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    fft_backend: str,
    device_index: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    canvas, placements = _tile_cases(images)
    gx_canvas, gy_canvas = fft_gradients_with_kernel(
        canvas,
        radius=int(RADIUS),
        kernel_x=np.asarray(kernel_x, dtype=np.float64),
        kernel_y=np.asarray(kernel_y, dtype=np.float64),
        fft_backend=fft_backend,
        device_index=device_index,
    )
    outputs = []
    for row_slice, col_slice in placements:
        outputs.append(
            (
                np.asarray(gx_canvas[row_slice, col_slice], dtype=np.float64).copy(),
                np.asarray(gy_canvas[row_slice, col_slice], dtype=np.float64).copy(),
            )
        )
    return outputs


def _rotate_field_back(gx: np.ndarray, gy: np.ndarray, angle_deg: float, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(angle_deg))
    sample_x = np.asarray(xx, dtype=np.float64) * math.cos(theta) - np.asarray(yy, dtype=np.float64) * math.sin(theta)
    sample_y = np.asarray(xx, dtype=np.float64) * math.sin(theta) + np.asarray(yy, dtype=np.float64) * math.cos(theta)
    col = sample_x + 0.5 * float(gx.shape[1] - 1)
    row = sample_y + 0.5 * float(gx.shape[0] - 1)
    coords = np.vstack((row.ravel(), col.ravel()))
    gx_sampled = ndimage.map_coordinates(np.asarray(gx, dtype=np.float64), coords, order=1, mode="nearest").reshape(gx.shape)
    gy_sampled = ndimage.map_coordinates(np.asarray(gy, dtype=np.float64), coords, order=1, mode="nearest").reshape(gy.shape)
    gx_back = math.cos(theta) * gx_sampled + math.sin(theta) * gy_sampled
    gy_back = -math.sin(theta) * gx_sampled + math.cos(theta) * gy_sampled
    return np.asarray(gx_back, dtype=np.float64), np.asarray(gy_back, dtype=np.float64)


def _valid_mask(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    radius = 0.5 * float(IMAGE_SIZE) - float(VALID_MARGIN_PX)
    return (np.asarray(xx, dtype=np.float64) ** 2 + np.asarray(yy, dtype=np.float64) ** 2) <= radius * radius


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def compile_plot(figure_src: Path, figure_pdf: Path) -> None:
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("typst is not installed, so the Section 7.11 figure cannot be compiled")
    subprocess.run(
        [typst, "compile", "--root", str(ROOT), str(figure_src), str(figure_pdf)],
        cwd=ROOT,
        check=True,
    )


def run_experiment(
    output_dir: Path,
    summary_json: Path,
    fft_backend: str,
    device_index: int | None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    cases = _build_cases()
    kernel_x, kernel_y = _build_kernel()
    xx, yy = _local_grid(IMAGE_SIZE)
    valid_mask = _valid_mask(xx, yy)

    gradients: list[tuple[np.ndarray, np.ndarray]] = []
    images = [case.image for case in cases]
    for batch_start in range(0, len(images), int(BATCH_CASES)):
        gradients.extend(
            _apply_batched_gradients(
                images[batch_start : batch_start + int(BATCH_CASES)],
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                fft_backend=fft_backend,
                device_index=device_index,
            )
        )

    rotated_back = []
    for case, (gx, gy) in zip(cases, gradients, strict=True):
        gx_back, gy_back = _rotate_field_back(gx, gy, float(case.angle_deg), xx, yy)
        rotated_back.append((gx_back, gy_back))

    gx_ref, gy_ref = rotated_back[0]
    per_rotation_error = []
    gx_stack = np.stack([pair[0] for pair in rotated_back], axis=0)
    gy_stack = np.stack([pair[1] for pair in rotated_back], axis=0)
    gx_mean = np.mean(gx_stack, axis=0)
    gy_mean = np.mean(gy_stack, axis=0)
    variance_map = np.mean((gx_stack - gx_mean[None, :, :]) ** 2 + (gy_stack - gy_mean[None, :, :]) ** 2, axis=0)
    masked_variance = np.asarray(variance_map[valid_mask], dtype=np.float64)

    for case, (gx_back, gy_back) in zip(cases, rotated_back, strict=True):
        error = np.sqrt((np.asarray(gx_back, dtype=np.float64) - np.asarray(gx_ref, dtype=np.float64)) ** 2 + (np.asarray(gy_back, dtype=np.float64) - np.asarray(gy_ref, dtype=np.float64)) ** 2)
        per_rotation_error.append(
            {
                "angle_deg": float(case.angle_deg),
                "mean_error_magnitude": float(np.mean(np.asarray(error[valid_mask], dtype=np.float64))),
            }
        )

    payload = {
        "title": "Section 7.11 rotated-content equivariance",
        "subtitle": "Disk support, (r, d) = (15, 11), normalize_coords = True, content-rich synthetic scene",
        "config": {
            "radius": int(RADIUS),
            "degree": int(DEGREE),
            "normalize_coords": bool(NORMALIZE_COORDS),
            "image_size_px": int(IMAGE_SIZE),
            "rotation_angles_deg": [float(case.angle_deg) for case in cases],
            "contrast": float(CONTRAST),
            "edge_width_px": float(EDGE_WIDTH_PX),
            "fft_backend": str(fft_backend),
            "device_index": None if device_index is None else int(device_index),
            "valid_mask": "Metrics are evaluated on a centered circular mask to avoid rotation-induced boundary interpolation artifacts.",
        },
        "mean_cross_rotation_variance": float(np.mean(masked_variance)),
        "p95_cross_rotation_variance": float(np.percentile(masked_variance, 95.0)),
        "per_rotation_error": per_rotation_error,
        "square_overlay_included": False,
    }
    _write_json(summary_json, payload)
    return {"summary_json": summary_json}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Section 7.11 rotated-content equivariance experiment.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_rotated_content_equivariance",
        help="Directory for the Section 7.11 outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "papers" / "journal_paper" / "figures" / "data" / "sec07_rotated_content_equivariance" / "sec07_rotated_content_equivariance_summary_r15_d11_normalized.json",
        help="Path for the Section 7.11 summary JSON.",
    )
    parser.add_argument("--fft-backend", type=str, default=DEFAULT_FFT_BACKEND, choices=("auto", "cpu", "vkfft"), help="FFT backend to use.")
    parser.add_argument("--device-index", type=int, default=None, help="Optional GPU device index.")
    parser.add_argument("--compile-plot", action="store_true", help="Compile the checked-in Typst/CeTZ polar plot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_experiment(
        output_dir=args.output_dir.resolve(),
        summary_json=args.summary_json.resolve(),
        fft_backend=str(args.fft_backend),
        device_index=args.device_index,
    )
    if args.compile_plot:
        figure_src = ROOT / "papers" / "journal_paper" / "figures" / "cetz_src" / "fig_sec07_rotated_content_equivariance_polar.typ"
        figure_pdf = ROOT / "papers" / "journal_paper" / "figures" / "fig_sec07_rotated_content_equivariance_polar.pdf"
        compile_plot(figure_src, figure_pdf)
        outputs["plot_pdf"] = figure_pdf
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
