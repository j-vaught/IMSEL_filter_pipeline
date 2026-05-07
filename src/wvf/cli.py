"""Pure Python WVF reference command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wvf.metal import wvf_magnitude_angle_metal


def _load_array(path: Path, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        archive = np.load(path)
        if key is not None:
            return archive[key]
        names = list(archive.files)
        if not names:
            raise ValueError(f"{path} does not contain arrays")
        return archive[names[0]]

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise SystemExit(
            "Image inputs require imageio. Use .npy/.npz or install imageio."
        ) from exc
    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3].astype(np.float32)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError(f"unsupported image shape {image.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone WVF gradients with the Metal backend."
    )
    parser.add_argument("input", type=Path, help="Input .npy, .npz, or image path.")
    parser.add_argument("output", type=Path, help="Output .npz path.")
    parser.add_argument("-r", "--radius", type=int, required=True)
    parser.add_argument("-d", "--degree", type=int, required=True)
    parser.add_argument("--key", default=None, help="Array key when input is .npz.")
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--variant",
        choices=("split", "antipodal", "direct"),
        default="split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = _load_array(args.input, args.key)
    dtype = np.dtype(args.output_dtype)
    gx, gy, mag, angle = wvf_magnitude_angle_metal(
        image,
        radius=args.radius,
        degree=args.degree,
        output_dtype=dtype,
        variant=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gx=gx,
        gy=gy,
        magnitude=mag,
        angle=angle,
        radius=np.int32(args.radius),
        degree=np.int32(args.degree),
        variant=args.variant,
        input_path=str(args.input),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
