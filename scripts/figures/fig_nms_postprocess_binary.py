"""Compare ringing-suppression post-processes on N4/Acont enhanced NMS.

Renders four panels per condition (clean, noisy):
    1. raw N4/Acont          (baseline kept mask)
    2. raw + skeleton        (thick -> 1px, no ringing fix)
    3. magnitude threshold   (drop kept pixels below percentile p) + skeleton
    4. component prune       (skeleton, drop components shorter than L)

Saved into cetz_figures/data/cgmm_nms_binary_panels/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from cgmm_nms import enhanced_nms


def save_mask(mask, path):
    img = np.zeros(mask.shape, dtype=np.uint8)
    img[mask] = 255
    Image.fromarray(img, mode="L").save(path, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",     required=True, type=Path)
    p.add_argument("--out-dir",  required=True, type=Path)
    p.add_argument("--label",    required=True, type=str)
    p.add_argument("--mag-pct",  type=float, default=30.0,
                   help="drop kept pixels below this percentile of M")
    p.add_argument("--min-len",  type=int, default=30,
                   help="drop skeleton components shorter than this")
    p.add_argument("--neighborhood", type=int, default=4)
    p.add_argument("--angular-fidelity", default="Acont",
                   choices=["A8", "A16", "Acont"])
    p.add_argument("--crop", type=int, default=0,
                   help="center crop side (e.g. 512). 0 = full image. "
                        "Threshold percentile is computed within the crop.")
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

    if args.crop > 0:
        H, W = out.shape
        s = args.crop
        r0, c0 = (H - s) // 2, (W - s) // 2
        out = out[r0:r0 + s, c0:c0 + s].copy()

    kept = out > 0

    # 1. raw kept
    n_raw = int(kept.sum())

    # 2. raw -> skeleton
    skel_raw = skeletonize(kept, method="lee")
    n_skel_raw = int(skel_raw.sum())

    # 3. magnitude threshold then skeleton
    mag_vals = out[kept]
    thr = float(np.percentile(mag_vals, args.mag_pct))
    kept_thr = kept & (out >= thr)
    skel_thr = skeletonize(kept_thr, method="lee")
    n_kept_thr = int(kept_thr.sum())
    n_skel_thr = int(skel_thr.sum())

    # 4. component prune on raw skeleton (8-conn)
    lbl, n_comp = ndimage.label(skel_raw, structure=np.ones((3, 3)))
    sizes = ndimage.sum(skel_raw, lbl, index=np.arange(1, n_comp + 1))
    keep_ids = np.where(sizes >= args.min_len)[0] + 1
    keep_set = np.zeros(n_comp + 1, dtype=bool)
    keep_set[keep_ids] = True
    skel_pruned = keep_set[lbl]
    n_skel_pruned = int(skel_pruned.sum())

    tag = f"N{args.neighborhood}_{args.angular_fidelity}"
    crop_tag = f"_c{args.crop}" if args.crop > 0 else ""
    pct_tag = f"magpct{args.mag_pct:g}".replace(".", "p")
    paths = {
        "raw":         args.out_dir / f"binary_{args.label}_{tag}{crop_tag}.png",
        "skel":        args.out_dir / f"binary_skel_{args.label}_{tag}{crop_tag}.png",
        "mag_skel":    args.out_dir / f"binary_skel_{pct_tag}_{args.label}_{tag}{crop_tag}.png",
        "prune_skel":  args.out_dir / f"binary_skel_prune{args.min_len}_{args.label}_{tag}{crop_tag}.png",
    }
    save_mask(kept,         paths["raw"])
    save_mask(skel_raw,     paths["skel"])
    save_mask(skel_thr,     paths["mag_skel"])
    save_mask(skel_pruned,  paths["prune_skel"])

    print(f"[{args.label} {tag}]")
    print(f"  raw kept              : {n_raw:>7}")
    print(f"  skeleton              : {n_skel_raw:>7}  (thickness {n_raw/max(n_skel_raw,1):.2f})")
    print(f"  mag>={thr:.4f} (p{args.mag_pct:.0f}) kept: {n_kept_thr:>7}  -> skel {n_skel_thr:>7}")
    print(f"  components total      : {n_comp:>7}, kept >= {args.min_len}px: {len(keep_ids):>5}")
    print(f"  pruned skeleton       : {n_skel_pruned:>7}")
    for k, v in paths.items():
        print(f"    -> {v.name}")


if __name__ == "__main__":
    main()
