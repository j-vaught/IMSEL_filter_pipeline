"""Sweep magnitude-percentile thresholds on the (e) enhanced-NMS panel.

For each percentile p in --pcts, takes the c-GMM dump, runs enhanced NMS
(N4/Acont, corner-OR), then keeps only pixels with M_fused >= p-th
percentile of valid pixels' M_fused.  Saves the binary panels (full and
central crop) so we can pick the best operating point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from cgmm_nms import enhanced_nms, standard_nms


def crop_center(arr, side):
    H, W = arr.shape[:2]
    r0 = (H - side) // 2
    c0 = (W - side) // 2
    return arr[r0:r0 + side, c0:c0 + side]


def save_binary(mask, path):
    img = np.zeros(mask.shape, dtype=np.uint8)
    img[mask] = 255
    Image.fromarray(img, mode="L").save(path, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--pcts", default="90,95,97.5,99,99.5")
    p.add_argument("--crop", type=int, default=512)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.dump} ...")
    d = np.load(args.dump, allow_pickle=False)
    th_p = d["theta_fused"]
    th_s = d["theta_fused_sec"]
    M_p  = d["M_fused"]
    M_s  = d["M_fused_sec"]
    v    = d["v_fused"]
    H, W = th_p.shape

    print("running enhanced NMS N4/Acont (corner-OR) ...")
    out_enh = enhanced_nms(th_p, M_p, th_s, M_s, v,
                           neighborhood=4, angular_fidelity="Acont",
                           corner_method="or")
    kept_enh = out_enh > 0
    print(f"  kept by NMS alone: {int(kept_enh.sum()):,} "
          f"({kept_enh.mean()*100:.2f}%)")

    print("running standard NMS N1/A8 (Canny baseline) ...")
    out_std = standard_nms(th_p, M_p, v,
                           neighborhood=1, angular_fidelity="A8")
    kept_std = out_std > 0
    print(f"  kept by NMS alone: {int(kept_std.sum()):,} "
          f"({kept_std.mean()*100:.2f}%)")

    pcts = [float(x) for x in args.pcts.split(",")]
    Mv = M_p[v.astype(bool)]
    print(f"\nM_fused distribution: median={np.median(Mv):.2f}  "
          f"p99={np.percentile(Mv, 99):.2f}  max={Mv.max():.2f}")

    for pct in pcts:
        thr = float(np.percentile(Mv, pct))
        # apply threshold AFTER NMS (Canny-style)
        kept_enh_t = kept_enh & (M_p >= thr)
        kept_std_t = kept_std & (M_p >= thr)
        print(f"  p{pct:>5}: thr={thr:7.3f}  "
              f"enhanced={int(kept_enh_t.sum()):>8} "
              f"({kept_enh_t.mean()*100:5.2f}%)  "
              f"standard={int(kept_std_t.sum()):>8} "
              f"({kept_std_t.mean()*100:5.2f}%)")

        ptag = f"p{pct:g}".replace(".", "p")
        save_binary(kept_enh_t,
                    args.out_dir / f"binary_N4_Acont_{ptag}.png")
        save_binary(crop_center(kept_enh_t, args.crop),
                    args.out_dir / f"binary_N4_Acont_{ptag}_c{args.crop}.png")
        save_binary(kept_std_t,
                    args.out_dir / f"binary_standard_N1_A8_{ptag}.png")
        save_binary(crop_center(kept_std_t, args.crop),
                    args.out_dir / f"binary_standard_N1_A8_{ptag}_c{args.crop}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
