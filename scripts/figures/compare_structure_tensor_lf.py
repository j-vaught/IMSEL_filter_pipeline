"""Compare LF+spline orientation against WVF-component structure tensors."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from edgecritic.nms_gmm import (
    enhanced_nonmax_suppression,
    hysteresis_threshold,
    line_filter_response_stack,
    spline_orientation_map,
)
from edgecritic.structure_tensor import structure_tensor_orientation
from edgecritic.synthetic import create_multi_angle_line_image, create_step_edge_image


GARNET = "#73000A"
ROSE = "#CC2E40"
ATLANTIC = "#466A9F"
CONGAREE = "#1F414D"
BLACK90 = "#363636"
BLACK10 = "#ECECEC"


@dataclass(frozen=True)
class ImageCase:
    name: str
    image: np.ndarray
    source: str


def _read_gray(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L")
        gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float64) / 255.0


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(arr)
    return np.clip(arr / max_value, 0.0, 1.0)


def _angle_rgb(angle: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    hue = (np.asarray(angle) % np.pi) / np.pi
    value = np.maximum(_normalize(magnitude), 0.20)
    hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
    return hsv_to_rgb(hsv)


def _axial_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = (np.asarray(a) - np.asarray(b) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return np.abs(np.rad2deg(diff))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.size < 2 or np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _auto_edges(magnitude: np.ndarray, angle: np.ndarray, high_quantile: float) -> np.ndarray:
    nms = enhanced_nonmax_suppression(magnitude, angle, n_directions=8)
    positive = nms[nms > 0.0]
    if positive.size == 0:
        return np.zeros_like(magnitude, dtype=bool)
    high = float(np.quantile(positive, high_quantile))
    low = 0.4 * high
    return hysteresis_threshold(nms, low, high)


def _edge_f1(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_mask = np.asarray(pred, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    tp = int(np.sum(pred_mask & target_mask))
    fp = int(np.sum(pred_mask & ~target_mask))
    fn = int(np.sum(~pred_mask & target_mask))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _load_cases(max_side: int) -> list[ImageCase]:
    step, _, _ = create_step_edge_image(
        size=max_side,
        edge_angle_deg=35,
        snr=1.5,
        high_val=1.0,
        low_val=0.0,
    )
    multiline, _, _ = create_multi_angle_line_image(
        size=max_side,
        line_width=2,
        snr=2.0,
        background=0.45,
    )
    multiline = np.clip(multiline / np.max(multiline), 0.0, 1.0)

    candidates = [
        ImageCase("synthetic_step_snr1p5", step, "synthetic"),
        ImageCase("synthetic_multiline_snr2", multiline, "synthetic"),
    ]

    real_paths = [
        ("biped_rgb008", Path("papers/filter-critique/presentation/figures/datasets/biped_RGB008.png")),
        ("uded_02", Path("papers/filter-critique/presentation/figures/datasets/uded_02.png")),
        ("aquatic_dark_wake", Path("example_images/Screenshot 2026-04-24 at 11.36.02\u202fPM.png")),
    ]
    for name, path in real_paths:
        if path.exists():
            candidates.append(ImageCase(name, _read_gray(path, max_side=max_side), str(path)))

    return candidates


def _lf_spline(image: np.ndarray, half_width: int, np_count: int, order: int, n_orientations: int):
    start = time.perf_counter()
    responses = line_filter_response_stack(
        image,
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
    )
    result = spline_orientation_map(
        responses.responses,
        responses.angles,
        range_threshold_rel=0.0,
        refine=True,
    )
    return result, time.perf_counter() - start


def _edge_disagreement_rgb(lf_edges: np.ndarray, tensor_edges: np.ndarray) -> np.ndarray:
    lf_mask = np.asarray(lf_edges, dtype=bool)
    tensor_mask = np.asarray(tensor_edges, dtype=bool)
    rgb = np.ones((*lf_mask.shape, 3), dtype=np.float64)
    rgb[lf_mask & tensor_mask] = np.array([0.0, 0.0, 0.0])
    rgb[lf_mask & ~tensor_mask] = np.array([204.0, 46.0, 64.0]) / 255.0
    rgb[~lf_mask & tensor_mask] = np.array([70.0, 106.0, 159.0]) / 255.0
    return rgb


def _plot_panel(
    case: ImageCase,
    lf,
    tensor,
    diff: np.ndarray,
    lf_edges: np.ndarray,
    tensor_edges: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 8.4), dpi=180)
    fig.patch.set_facecolor("white")

    panels = [
        ("Input", case.image, "gray"),
        ("LF magnitude", _normalize(lf.magnitude), "gray"),
        ("Tensor magnitude", _normalize(tensor.magnitude_b), "gray"),
        ("LF orientation", _angle_rgb(lf.angle, lf.magnitude), None),
        ("Tensor orientation", _angle_rgb(tensor.orientation, tensor.magnitude_b), None),
        ("|angle diff| deg", diff, "magma"),
        ("LF final edges", lf_edges, "gray"),
        ("Tensor final edges", tensor_edges, "gray"),
        ("Edge disagreement", _edge_disagreement_rgb(lf_edges, tensor_edges), None),
    ]
    for ax, (label, data, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(label, fontsize=9, color=BLACK90)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BLACK10)

    fig.suptitle(title, color=GARNET, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _compare_one(
    case: ImageCase,
    output_dir: Path,
    half_width: int,
    np_count: int,
    order: int,
    n_orientations: int,
    radii: list[int],
    high_quantile: float,
) -> dict:
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    lf, lf_seconds = _lf_spline(case.image, half_width, np_count, order, n_orientations)
    gate = lf.magnitude > 0.1 * float(np.max(lf.magnitude))
    lf_edges = _auto_edges(lf.magnitude, lf.angle, high_quantile=high_quantile)

    results = {
        "name": case.name,
        "source": case.source,
        "shape": list(case.image.shape),
        "lf_seconds": lf_seconds,
        "sweeps": [],
    }

    for radius in radii:
        for weight_type in ("disk", "gaussian"):
            start = time.perf_counter()
            tensor = structure_tensor_orientation(
                case.image,
                radius=radius,
                weight_type=weight_type,
                np_count=np_count,
                order=order,
            )
            tensor_seconds = time.perf_counter() - start

            diff = _axial_diff_deg(tensor.orientation, lf.angle)
            tangent_diff = _axial_diff_deg((tensor.orientation + np.pi / 2.0) % np.pi, lf.angle)
            gated_diff = diff[gate]
            gated_tangent_diff = tangent_diff[gate]
            corr_a = _pearson(lf.magnitude[gate], tensor.magnitude_a[gate])
            corr_b = _pearson(lf.magnitude[gate], tensor.magnitude_b[gate])
            tensor_edges = _auto_edges(tensor.magnitude_b, tensor.orientation, high_quantile=high_quantile)
            edge_agreement = _edge_f1(tensor_edges, lf_edges)

            label = f"{weight_type}_r{radius}"
            _plot_panel(
                case,
                lf,
                tensor,
                diff,
                lf_edges,
                tensor_edges,
                case_dir / f"{label}_panel.png",
                f"{case.name}: {weight_type}, radius={radius}",
            )

            convention = "gradient"
            if float(np.mean(gated_tangent_diff)) < float(np.mean(gated_diff)):
                convention = "tangent"

            results["sweeps"].append(
                {
                    "radius": radius,
                    "weight_type": weight_type,
                    "gate_pixels": int(np.sum(gate)),
                    "tensor_seconds": tensor_seconds,
                    "speedup_vs_lf": lf_seconds / max(tensor_seconds, 1e-12),
                    "mean_angle_error_deg": float(np.mean(gated_diff)),
                    "median_angle_error_deg": float(np.median(gated_diff)),
                    "p90_angle_error_deg": float(np.percentile(gated_diff, 90)),
                    "frac_over_5deg": float(np.mean(gated_diff > 5.0)),
                    "mean_tangent_convention_error_deg": float(np.mean(gated_tangent_diff)),
                    "median_tangent_convention_error_deg": float(np.median(gated_tangent_diff)),
                    "best_orientation_convention": convention,
                    "pearson_magnitude_a": corr_a,
                    "pearson_magnitude_b": corr_b,
                    "edge_f1_vs_lf": edge_agreement["f1"],
                    "edge_precision_vs_lf": edge_agreement["precision"],
                    "edge_recall_vs_lf": edge_agreement["recall"],
                    "panel": str(case_dir / f"{label}_panel.png"),
                }
            )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/structure_tensor_comparison"))
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--half-width", type=int, default=7)
    parser.add_argument("--np-count", type=int, default=15)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--orientations", type=int, default=36)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_radius = 2 * args.half_width + 1
    radii = sorted({max(1, int(round(base_radius / 2))), base_radius, 2 * base_radius})

    summary = {
        "config": {
            "max_side": args.max_side,
            "half_width": args.half_width,
            "base_radius": base_radius,
            "radii": radii,
            "np_count": args.np_count,
            "order": args.order,
            "orientations": args.orientations,
            "high_quantile": args.high_quantile,
        },
        "cases": [],
    }

    for case in _load_cases(max_side=args.max_side):
        print(f"Comparing {case.name}", flush=True)
        summary["cases"].append(
            _compare_one(
                case,
                output_dir=args.output_dir,
                half_width=args.half_width,
                np_count=args.np_count,
                order=args.order,
                n_orientations=args.orientations,
                radii=radii,
                high_quantile=args.high_quantile,
            )
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
