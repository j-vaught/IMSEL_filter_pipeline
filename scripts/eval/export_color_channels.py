"""Export L / R / G / B channels of a color image as standalone PNGs.

Loads a color image, decomposes it into a BT.709 luminance channel and
the three RGB channels, and writes each as a single-channel grayscale
PNG plus a manifest. Intended to feed CeTZ figures showing how a color
image splits into separate inputs for downstream edge detectors.

Usage::

    python scripts/eval/export_color_channels.py \\
        --image example_images/synthetic_nested_shapes/clean/1024/...png \\
        --out-dir cetz_figures/data/color_channels/1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    arr = np.asarray(pil)
    h, w = arr.shape[:2]
    print(f"Loaded {args.image.name}: {h} x {w}")

    # Save original (re-encoded so the figure's path layout is uniform).
    pil.save(args.out_dir / "original.png")

    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]
    # BT.709 luminance.
    L = (0.2126 * R + 0.7152 * G + 0.0722 * B).clip(0, 255).astype(np.uint8)

    for name, ch in (("L", L), ("R", R), ("G", G), ("B", B)):
        Image.fromarray(ch.astype(np.uint8), mode="L").save(
            args.out_dir / f"channel_{name}.png")
        print(f"  channel_{name}: min={int(ch.min())} max={int(ch.max())}")

    manifest = {
        "source_image": str(args.image),
        "image_shape": [int(h), int(w)],
        "luminance_formula": "BT.709 Y = 0.2126 R + 0.7152 G + 0.0722 B",
        "files": {
            "original": "original.png",
            "L": "channel_L.png",
            "R": "channel_R.png",
            "G": "channel_G.png",
            "B": "channel_B.png",
        },
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exported channels to {args.out_dir}")


if __name__ == "__main__":
    main()
