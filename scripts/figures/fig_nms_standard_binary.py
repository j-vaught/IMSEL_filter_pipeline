"""Render raw binary STANDARD NMS edge maps (no corner-OR rule).

For visual comparison against enhanced (corner-OR) NMS:
each panel uses the same c-GMM fused (theta_primary, M_primary, v_fused)
inputs, but only does the single primary-direction local-max check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from cgmm_nms import standard_nms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",     required=True, type=Path)
    p.add_argument("--out-dir",  required=True, type=Path)
    p.add_argument("--angular-fidelity", default="Acont",
                   choices=["A8", "A16", "Acont"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dump, allow_pickle=False)
    M_p = d["M_fused"]; th_p = d["theta_fused"]
    v   = d["v_fused"]

    fid = args.angular_fidelity
    for nbh in (1, 2, 3, 4, 5):
        out = standard_nms(th_p, M_p, v,
                           neighborhood=nbh,
                           angular_fidelity=fid)
        kept = out > 0
        n = int(kept.sum())
        img = np.zeros(kept.shape, dtype=np.uint8)
        img[kept] = 255
        label = f"standard_N{nbh}_{fid}"
        png = args.out_dir / f"binary_{label}.png"
        Image.fromarray(img, mode="L").save(png, optimize=True)
        print(f"  {label}: kept={n:>7}  -> {png.name}")


if __name__ == "__main__":
    main()
