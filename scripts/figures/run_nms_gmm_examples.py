"""Run the NMS/GMM edge detector on the example image folder."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from edgecritic.nms_gmm import NMSGMMConfig, detect_edges


GARNET = "#73000A"
ROSE = "#CC2E40"
BLACK_90 = "#363636"
BLACK_10 = "#ECECEC"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "image"


def _read_preview(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as im:
        original_size = im.size
        rgb = im.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(rgb), original_size


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / max_value * 255.0, 0.0, 255.0).astype(np.uint8)


def _edge_overlay(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float64) / 255.0
    edge_color = np.array([204.0, 46.0, 64.0]) / 255.0
    overlay[edges] = 0.35 * overlay[edges] + 0.65 * edge_color
    return np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_panel(
    image: np.ndarray,
    nms: np.ndarray,
    edges: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=180)
    fig.patch.set_facecolor("white")

    panels = [
        ("Input", image, None),
        ("NMS magnitude", nms, "gray"),
        ("Binary edges", edges, "gray"),
        ("Overlay", overlay, None),
    ]
    for ax, (label, data, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=0 if cmap == "gray" else None)
        ax.set_title(label, color=BLACK_90, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BLACK_10)
            spine.set_linewidth(1.0)

    fig.suptitle(title, color=GARNET, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _run_one(
    path: Path,
    output_dir: Path,
    index: int,
    config: NMSGMMConfig,
    max_side: int,
) -> dict:
    image, original_size = _read_preview(path, max_side=max_side)
    stem = f"{index:02d}_{_slugify(path.stem)}"
    image_dir = output_dir / stem
    image_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result = detect_edges(image, config=config)
    elapsed = time.perf_counter() - start

    nms_u8 = _normalize_to_uint8(result.nms)
    edges_u8 = (result.edges.astype(np.uint8) * 255)
    overlay = _edge_overlay(image, result.edges)

    _save_image(image_dir / "input_preview.png", image)
    _save_image(image_dir / "nms_magnitude.png", nms_u8)
    _save_image(image_dir / "edges.png", edges_u8)
    _save_image(image_dir / "overlay.png", overlay)
    _save_panel(
        image=image,
        nms=nms_u8,
        edges=edges_u8,
        overlay=overlay,
        output_path=image_dir / "panel.png",
        title=path.name,
    )

    return {
        "input": str(path),
        "output_dir": str(image_dir),
        "original_size": list(original_size),
        "preview_shape": list(image.shape),
        "seconds": elapsed,
        "edge_pixels": int(result.edges.sum()),
        "edge_fraction": float(result.edges.mean()),
        "low_threshold": result.low_threshold,
        "high_threshold": result.high_threshold,
        "labels": list(result.labels),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("example_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nms_gmm_examples"))
    parser.add_argument("--max-side", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", choices=("default", "aquatic"), default="aquatic")
    parser.add_argument("--half-widths", default="3,7,11")
    parser.add_argument("--domains", default="auto")
    parser.add_argument("--np-count", type=int, default=15)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--orientations", type=int, default=36)
    parser.add_argument("--high-quantile", type=float, default=None)
    parser.add_argument("--low-ratio", type=float, default=None)
    parser.add_argument("--no-link-gaps", action="store_true")
    parser.add_argument("--max-link-gap", type=int, default=None)
    parser.add_argument("--link-candidate-ratio", type=float, default=None)
    parser.add_argument("--min-component-size", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = [
        path
        for path in sorted(args.input_dir.glob("*"))
        if path.is_file() and not path.name.startswith("._")
    ]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No input images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "half_widths": _parse_int_tuple(args.half_widths),
        "domains": args.domains,
        "np_count": args.np_count,
        "order": args.order,
        "n_orientations": args.orientations,
    }
    if args.high_quantile is not None:
        overrides["high_quantile"] = args.high_quantile
    if args.low_ratio is not None:
        overrides["low_ratio"] = args.low_ratio
    if args.no_link_gaps:
        overrides["link_gaps"] = False
    if args.max_link_gap is not None:
        overrides["max_link_gap"] = args.max_link_gap
    if args.link_candidate_ratio is not None:
        overrides["link_candidate_ratio"] = args.link_candidate_ratio
    if args.min_component_size is not None:
        overrides["min_component_size"] = args.min_component_size

    if args.preset == "aquatic":
        config = NMSGMMConfig.aquatic(**overrides)
    else:
        config = NMSGMMConfig(**overrides)

    summary = {
        "config": {
            "half_widths": list(config.half_widths),
            "domains": config.domains,
            "np_count": config.np_count,
            "order": config.order,
            "n_orientations": config.n_orientations,
            "gmm_bins": config.gmm_bins,
            "nms_directions": config.nms_directions,
            "high_quantile": config.high_quantile,
            "low_ratio": config.low_ratio,
            "link_gaps": config.link_gaps,
            "max_link_gap": config.max_link_gap,
            "link_candidate_ratio": config.link_candidate_ratio,
            "min_component_size": config.min_component_size,
            "max_side": args.max_side,
        },
        "results": [],
    }

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        summary["results"].append(
            _run_one(
                path=path,
                output_dir=args.output_dir,
                index=index,
                config=config,
                max_side=args.max_side,
            )
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
