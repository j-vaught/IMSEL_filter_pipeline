"""Score every LF param-sweep edge map against synthetic GT.

GT is derived from the original RGB image: a pixel is on an edge iff
any of its 4-connected neighbours has a different color value. This is
exact for the noiseless flat-shaded nested-shapes images.

Each gray_r PNG in the sweep dir is loaded and inverted (255 - u8) to
recover a magnitude proxy; ODS / OIS are scale-invariant so this is a
faithful score even though the PNG is 8-bit clipped at vmax.

Usage::

    python scripts/eval/analyze_lf_param_sweep.py \\
        --sweep-dir "cetz_figures/data/lf_param_sweep/L_1024" \\
        --rgb       "cetz_figures/data/color_channels/1024/original.png" \\
        --out       "cetz_figures/data/lf_param_sweep/L_1024/scores.json"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from edgecritic.evaluation.metrics import compute_ods_ois


_FNAME_RX = re.compile(r"edge_r(\d+)_d(\d+)_m(\d+)\.png")


def derive_gt(rgb_path: Path,
              crop_origin: tuple[int, int] | None = None,
              crop_size: int = 0) -> np.ndarray:
    """Boolean GT edge map: any pixel adjacent to a different color.
    Optionally crops the RGB to the same region as the sweep maps so
    GT and prediction share an identical pixel grid."""
    arr = np.asarray(Image.open(rgb_path).convert("RGB"))
    if crop_size > 0 and crop_origin is not None:
        x0, y0 = crop_origin
        arr = arr[y0:y0 + crop_size, x0:x0 + crop_size]
    diff_x = np.any(arr[:, 1:] != arr[:, :-1], axis=-1)
    diff_y = np.any(arr[1:, :] != arr[:-1, :], axis=-1)
    edges = np.zeros(arr.shape[:2], dtype=bool)
    edges[:, :-1] |= diff_x
    edges[:, 1:]  |= diff_x
    edges[1:, :]  |= diff_y
    edges[:-1, :] |= diff_y
    return edges


def load_magnitude(png_path: Path) -> np.ndarray:
    """Convert a gray_r PNG (high response = dark) back to a magnitude
    proxy (high response = high value)."""
    u8 = np.asarray(Image.open(png_path).convert("L"))
    return (255 - u8).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--rgb",       required=True, type=Path)
    parser.add_argument("--out",       required=True, type=Path)
    parser.add_argument("--match-radius", type=int, default=2,
                        help="Pixel tolerance for ODS edge matching.")
    args = parser.parse_args()

    # If the sweep was cropped, mirror the crop on the GT.
    manifest_path = args.sweep_dir / "manifest.json"
    crop_origin = None
    crop_size = 0
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        crop_size = int(m.get("crop_size", 0))
        crop_origin = m.get("crop_origin_xy")
        if crop_origin is not None:
            crop_origin = tuple(crop_origin)
    gt = derive_gt(args.rgb, crop_origin=crop_origin, crop_size=crop_size)
    print(f"GT image: {args.rgb.name}, "
          f"{int(gt.sum()):,} edge pixels in {gt.shape}"
          + (f" (cropped to {crop_size}x{crop_size} "
             f"at origin {crop_origin})" if crop_size > 0 else ""))

    rows = []
    for png in sorted(args.sweep_dir.glob("edge_*.png")):
        match = _FNAME_RX.match(png.name)
        if not match:
            continue
        r, d, m = (int(g) for g in match.groups())
        mag = load_magnitude(png)
        ods, ois, _, _ = compute_ods_ois(
            mag, gt.astype(np.uint8),
            n_thresholds=80, match_radius=args.match_radius)
        rows.append({"r": r, "d": d, "m": m,
                     "ods": float(ods), "ois": float(ois)})

    rows.sort(key=lambda x: -x["ods"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "rgb": str(args.rgb),
            "match_radius": args.match_radius,
            "scores": rows,
        }, f, indent=2)

    print(f"\nTop 10 (r, d, m) by ODS:")
    print(f"  {'r':>2}  {'d':>1}  {'m':>1}    ODS     OIS")
    for row in rows[:10]:
        print(f"  {row['r']:>2}  {row['d']:>1}  {row['m']:>1}  "
              f"{row['ods']:.4f}  {row['ois']:.4f}")
    print(f"\nBottom 5 (r, d, m) by ODS:")
    for row in rows[-5:]:
        print(f"  {row['r']:>2}  {row['d']:>1}  {row['m']:>1}  "
              f"{row['ods']:.4f}  {row['ois']:.4f}")

    # Marginal effects: best ODS aggregated by varying one axis at a time.
    def by(key):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["ods"])
        return {k: float(np.mean(v)) for k, v in sorted(groups.items())}

    print(f"\nMean ODS by d  (over all r, m): {by('d')}")
    print(f"Mean ODS by r  (over all d, m): {by('r')}")
    print(f"Mean ODS by m  (over all r, d): {by('m')}")
    print(f"\nWrote scores -> {args.out}")


if __name__ == "__main__":
    main()
