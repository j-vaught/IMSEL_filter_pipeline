"""Command line interface for the standalone WVF native package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import backend_info, components, gradients, magnitude, magnitude_orientation


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
        description="Run standalone WVF outputs with native GPU or Rust CPU FFT backends."
    )
    parser.add_argument("input", type=Path, help="Input .npy, .npz, or image path.")
    parser.add_argument("output", type=Path, help="Output .npz path.")
    parser.add_argument("-r", "--radius", type=int, required=True)
    parser.add_argument("-d", "--degree", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=("split", "antipodal", "direct", "fft"),
        default="split",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("auto", "cpu", "vkfft"),
        default="auto",
        help="FFT backend for variant=fft. auto benchmarks CPU and VkFFT once per workload.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="GPU device index for native GPU execution.",
    )
    parser.add_argument(
        "--mode",
        choices=("gradients", "magnitude", "magnitude-angle", "all"),
        default="all",
        help="Requested output mode.",
    )
    parser.add_argument("--key", default=None, help="Array key when input is .npz.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = _load_array(args.input, args.key)

    payload: dict[str, object] = {
        "radius": np.int32(args.radius),
        "degree": np.int32(args.degree),
        "variant": args.variant,
        "mode": args.mode,
        "input_path": str(args.input),
    }
    common_kwargs = {
        "radius": args.radius,
        "degree": args.degree,
        "variant": args.variant,
        "fft_backend": args.fft_backend,
        "device_index": args.device_index,
    }
    if args.mode == "gradients":
        result = gradients(image, **common_kwargs)
        payload["gx"] = result.gx
        payload["gy"] = result.gy
    elif args.mode == "magnitude":
        payload["magnitude"] = magnitude(image, **common_kwargs)
    elif args.mode == "magnitude-angle":
        result = magnitude_orientation(image, **common_kwargs)
        payload["magnitude"] = result.magnitude
        payload["angle"] = result.angle
    else:
        result = components(image, **common_kwargs)
        payload["gx"] = result.gx
        payload["gy"] = result.gy
        payload["magnitude"] = result.magnitude
        payload["angle"] = result.angle

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    return 0


def doctor_main(argv: list[str] | None = None) -> int:
    del argv
    print(json.dumps(backend_info(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
