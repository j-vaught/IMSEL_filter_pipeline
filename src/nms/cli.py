"""Command line interface for the NMS edge detector."""

from __future__ import annotations

import argparse

import numpy as np

from nms.pipeline import NMSConfig, detect_edges


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.uint8)
    max_value = float(np.max(arr))
    if max_value <= 0.0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip(arr / max_value * 255.0, 0.0, 255.0).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the edgecritic NMS edge detector.")
    parser.add_argument("input", help="Input image path.")
    parser.add_argument("output", help="Output edge-map path.")
    parser.add_argument("--half-widths", default="3,7,11", help="Comma-separated LF half-widths.")
    parser.add_argument("--domains", default="auto", help="Domain selector: auto, rgb, or comma list.")
    parser.add_argument("--np-count", type=int, default=15)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--orientations", type=int, default=36)
    parser.add_argument("--high-threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--low-ratio", type=float, default=0.40)
    parser.add_argument("--save-nms", default=None, help="Optional path for the thinned NMS magnitude map.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise SystemExit("The CLI requires imageio. Install it with `pip install imageio`.") from exc

    image = iio.imread(args.input)
    config = NMSConfig(
        half_widths=_parse_int_tuple(args.half_widths),
        domains=args.domains,
        np_count=args.np_count,
        order=args.order,
        n_orientations=args.orientations,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        high_quantile=args.high_quantile,
        low_ratio=args.low_ratio,
    )
    result = detect_edges(image, config=config)
    iio.imwrite(args.output, (result.edges.astype(np.uint8) * 255))
    if args.save_nms:
        iio.imwrite(args.save_nms, _to_uint8(result.nms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
