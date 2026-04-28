"""Render skeletonized N4/Acont enhanced NMS edge maps (1-pixel ridges).

Takes the N4/Acont enhanced-NMS binary mask, applies morphological
skeletonize (skimage), and saves the 1-pixel-wide result.  Saved into
cetz_figures/data/cgmm_nms_binary_panels/ alongside the existing panels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from cgmm_nms import enhanced_nms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",     required=True, type=Path)
    p.add_argument("--out-dir",  required=True, type=Path)
    p.add_argument("--label",    required=True, type=str,
                   help="e.g. 'clean' or 'noisy'")
    p.add_argument("--neighborhood", type=int, default=4)
    p.add_argument("--angular-fidelity", default="Acont",
                   choices=["A8", "A16", "Acont"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dump, allow_pickle=False)
    M_p = d["M_fused"]; th_p = d["theta_fused"]
    M_s = d["M_fused_sec"]; th_s = d["theta_fused_sec"]
    v   = d["v_fused"]

    out = enhanced_nms(th_p, M_p, th_s, M_s, v,
                       neighborhood=args.neighborhood,
                       angular_fidelity=args.angular_fidelity,
                       corner_method="or")
    kept = out > 0
    skel = skeletonize(kept, method="lee")

    img_kept = np.zeros(kept.shape, dtype=np.uint8); img_kept[kept] = 255
    img_skel = np.zeros(skel.shape, dtype=np.uint8); img_skel[skel] = 255

    tag = f"N{args.neighborhood}_{args.angular_fidelity}"
    png_kept = args.out_dir / f"binary_{args.label}_{tag}.png"
    png_skel = args.out_dir / f"binary_skel_{args.label}_{tag}.png"
    Image.fromarray(img_kept, mode="L").save(png_kept, optimize=True)
    Image.fromarray(img_skel, mode="L").save(png_skel, optimize=True)

    n_kept = int(kept.sum())
    n_skel = int(skel.sum())
    thick = n_kept / max(n_skel, 1)
    print(f"  {args.label} {tag}: kept={n_kept:>7}  skel={n_skel:>7}  "
          f"thick={thick:.3f}")
    print(f"    -> {png_kept.name}")
    print(f"    -> {png_skel.name}")


if __name__ == "__main__":
    main()
