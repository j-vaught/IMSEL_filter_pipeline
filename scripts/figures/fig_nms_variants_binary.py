"""Render raw binary NMS edge maps (no dilation, no orientation colour).

Each panel is a (H, W) uint8 mask: 255 where NMS kept the pixel, 0 elsewhere.
Saved as PNGs into cetz_figures/data/cgmm_nms_binary_panels/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from cgmm_nms import enhanced_nms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",     required=True, type=Path)
    p.add_argument("--out-dir",  required=True, type=Path)
    p.add_argument("--invert", action="store_true",
                   help="white edges on black bg by default; pass to "
                        "flip to black edges on white bg")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dump, allow_pickle=False)
    M_p = d["M_fused"]; th_p = d["theta_fused"]
    M_s = d["M_fused_sec"]; th_s = d["theta_fused_sec"]
    v   = d["v_fused"]

    bg = 0
    fg = 255
    if args.invert:
        bg, fg = 255, 0

    for nbh in (1, 2, 3):
        for fid in ("A8", "A16", "Acont"):
            out = enhanced_nms(th_p, M_p, th_s, M_s, v,
                               neighborhood=nbh, angular_fidelity=fid,
                               corner_method="or")
            kept = out > 0
            n = int(kept.sum())
            img = np.full(kept.shape, bg, dtype=np.uint8)
            img[kept] = fg
            label = f"N{nbh}_{fid}"
            png = args.out_dir / f"binary_{label}.png"
            Image.fromarray(img, mode="L").save(png, optimize=True)
            print(f"  {label}: kept={n:>7}  -> {png.name}")


if __name__ == "__main__":
    main()
