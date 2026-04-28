"""Generate inputs.npz and expected.npz for the orientation-recovery
Metal acceptance tests.

Builds a realistic slab of LF response curves by running the Metal LF
front-end on a noisy synthetic 4096x4096 image, then sampling N pixels
across the image to keep the file size manageable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "agent_workspaces" / "orientation_recovery_metal"))

from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.lf._metal import lf_orientation_stack_metal
from reference_impl import find_two_peaks


def main():
    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/4096"
                / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png")
    print(f"image: {img_path.name}")
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]

    rng = np.random.default_rng(0)
    sigma = 13.0
    rgb_n = np.clip(rgb.astype(np.float32)
                    + rng.normal(0.0, sigma, rgb.shape).astype(np.float32),
                    0.0, 255.0)

    # luminance only -- one channel is enough for reference data
    L = (0.2126 * rgb_n[..., 0]
         + 0.7152 * rgb_n[..., 1]
         + 0.0722 * rgb_n[..., 2]).astype(np.float32)

    r, d, m = 9, 3, 60
    n_orient = 64
    print(f"building WVF (r={r}, d={d}) and LF stack "
          f"(m={m}, n_orient={n_orient})")
    kernels = build_wvf_radius_kernels(radius=r, order=d)
    gx, gy = wvf_radius_gradients_metal(L, kernels, output_dtype=np.float32)

    t0 = time.perf_counter()
    stack = lf_orientation_stack_metal(gx, gy, m=m,
                                       n_orientations=n_orient,
                                       output_dtype=np.float32,
                                       method="box")
    print(f"  LF stack: {time.perf_counter()-t0:.1f}s, shape {stack.shape}")

    # Sample 200,000 random pixels.  Mix of high and low magnitude for
    # coverage of single-peak and multi-peak rows.
    rng2 = np.random.default_rng(1)
    n_samples = 200_000
    ys = rng2.integers(0, H, size=n_samples).astype(np.int32)
    xs = rng2.integers(0, W, size=n_samples).astype(np.int32)

    angles = np.linspace(0.0, np.pi, n_orient, endpoint=False)
    # response curve per pixel: shape (n_samples, n_orient) float32
    resp = stack[:, ys, xs].T.copy().astype(np.float32)
    print(f"  sampled response slab: {resp.shape} {resp.dtype}")

    # Reference output via spline.
    t0 = time.perf_counter()
    th_p, M_p, th_s, M_s, v = find_two_peaks(angles,
                                              resp.astype(np.float64),
                                              tau_sec_floor=0.40,
                                              tau_validity=0.10,
                                              dense_n=500,
                                              min_sep_frac=0.125)
    print(f"  reference find_two_peaks: {time.perf_counter()-t0:.1f}s")
    print(f"  validity v=1 rows: {int(v.sum()):,} / {v.size:,} "
          f"({v.mean()*100:.1f}%)")

    out = ROOT / "agent_workspaces/orientation_recovery_metal/reference"
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "inputs.npz",
                        angles=angles,
                        response=resp,
                        sample_xs=xs,
                        sample_ys=ys,
                        config_dense_n=np.int32(500),
                        config_tau_sec_floor=np.float64(0.40),
                        config_tau_validity=np.float64(0.10),
                        config_min_sep_frac=np.float64(0.125),
                        config_n_orientations=np.int32(n_orient),
                        config_image=str(img_path.name),
                        config_sigma=np.float64(sigma),
                        config_r=np.int32(r),
                        config_d=np.int32(d),
                        config_m=np.int32(m))
    np.savez_compressed(out / "expected.npz",
                        theta_primary=th_p,
                        M_primary=M_p,
                        theta_secondary=th_s,
                        M_secondary=M_s,
                        v=v)

    print()
    print(f"wrote {out / 'inputs.npz'}  "
          f"(uncompressed slab ~{resp.nbytes/1e6:.0f} MB)")
    print(f"wrote {out / 'expected.npz'}")
    print(f"  primary M range  : [{M_p.min():.3f}, {M_p.max():.3f}]")
    print(f"  secondary kept   : "
          f"{int((~np.isnan(th_s)).sum())} / {n_samples}")


if __name__ == "__main__":
    main()
