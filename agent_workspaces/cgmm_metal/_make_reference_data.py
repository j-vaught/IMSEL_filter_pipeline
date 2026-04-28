"""Generate inputs.npz and expected.npz for the c-GMM Metal acceptance tests.

Builds a realistic per-pixel measurement set by running the fused
Metal front end (WVF + LF + orientation recovery) at 1024^2 with
N=4 inputs per pixel, then freezes the c-GMM K=3 hard-EM output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgecritic.pipeline import wvf_lf_recover_metal
from reference_impl import theta_M_to_phi_w, cgmm_fuse_two_pass


def main():
    img_path = (ROOT / "example_images/synthetic_nested_shapes/clean/1024"
                / "nested_star_square_oval_low_contrast_mixed_chroma_1024.png")
    sigma = 13.0
    r, d = 9, 3
    m_values = [40, 60, 80, 100]
    n_orient = 32
    K = 3

    print(f"image: {img_path.name}")
    rng = np.random.default_rng(0)
    rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32)
    H, W = rgb.shape[:2]
    rgb_n = np.clip(rgb + rng.normal(0.0, sigma, rgb.shape).astype(np.float32),
                    0.0, 255.0)
    L = (0.2126 * rgb_n[..., 0] + 0.7152 * rgb_n[..., 1]
         + 0.0722 * rgb_n[..., 2]).astype(np.float32)

    # Run fused pipeline N times (one per m), collect (theta, M, v) per pixel.
    print(f"running fused front end at r={r}, d={d}, "
          f"m in {m_values}, n_orient={n_orient} ...")
    th_p_list, M_p_list, th_s_list, M_s_list, v_list = [], [], [], [], []
    for m in m_values:
        t0 = time.perf_counter()
        th_p, M_p, th_s, M_s, v = wvf_lf_recover_metal(
            L, radius=r, degree=d, lf_half_length=m,
            n_orientations=n_orient,
            tau_sec_floor=0.40, tau_validity=0.10,
            dense_n=500, min_sep_frac=0.125, method="box")
        print(f"  m={m:>3}  v=1: {v.mean()*100:5.1f}%  "
              f"{(time.perf_counter()-t0)*1000:.0f} ms")
        th_p_list.append(th_p)
        M_p_list.append(M_p)
        th_s_list.append(th_s)
        M_s_list.append(M_s)
        v_list.append(v)

    # Build (P, N) measurement arrays.  N = len(m_values).
    P = H * W
    N = len(m_values)
    theta_p_arr = np.stack([np.degrees(a).reshape(P) for a in th_p_list], axis=1)
    M_p_arr     = np.stack([a.reshape(P)             for a in M_p_list], axis=1)
    theta_s_arr = np.stack([np.degrees(a).reshape(P) for a in th_s_list], axis=1)
    M_s_arr     = np.stack([a.reshape(P)             for a in M_s_list], axis=1)
    v_arr       = np.stack([a.reshape(P)             for a in v_list],   axis=1)

    # Gate by per-input v before turning into (phi, w).
    M_p_gated = M_p_arr * v_arr
    M_s_gated = M_s_arr * v_arr
    phi_p64, w_p64, _ = theta_M_to_phi_w(theta_p_arr, M_p_gated)
    phi_s64, w_s64, _ = theta_M_to_phi_w(theta_s_arr, M_s_gated)

    # Round-trip through float32 so the saved inputs and the gold are
    # both computed from the same float32-rounded (phi, w).  Otherwise
    # implementer tests that load the saved float32 inputs and cast back
    # to float64 see slightly different EM trajectories on the ~0.2% of
    # pixels with near-tied component masses.
    phi_p = phi_p64.astype(np.float32).astype(np.float64)
    w_p   = w_p64.astype(np.float32).astype(np.float64)
    phi_s = phi_s64.astype(np.float32).astype(np.float64)
    w_s   = w_s64.astype(np.float32).astype(np.float64)

    print(f"\nphi_p shape: {phi_p.shape}  dtype={phi_p.dtype}")
    print(f"running c-GMM K={K} hard-EM reference ...")
    t0 = time.perf_counter()
    out = cgmm_fuse_two_pass(phi_p, w_p, phi_s, w_s,
                             K=K, n_iters=30,
                             init_kappa=4.0,
                             hard_em=True,
                             tau_M_rel=0.05,
                             theta_min_deg=10.0)
    print(f"  reference c-GMM: {time.perf_counter()-t0:.1f} s")
    print(f"  v_fused = 1 fraction: {out['v_fused'].mean()*100:.1f}%")
    print(f"  M_primary range: [{np.nanmin(out['M_primary']):.3f}, "
          f"{np.nanmax(out['M_primary']):.3f}]")
    print(f"  secondary kept: "
          f"{int(out['keep_secondary_mask'].sum()):,} / {P:,}")

    out_dir = Path(__file__).resolve().parent / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs as float32 to keep file size sane.
    np.savez_compressed(out_dir / "inputs.npz",
                        phi_p=phi_p.astype(np.float32),
                        w_p=w_p.astype(np.float32),
                        phi_s=phi_s.astype(np.float32),
                        w_s=w_s.astype(np.float32),
                        config_K=np.int32(K),
                        config_n_iters=np.int32(30),
                        config_init_kappa=np.float64(4.0),
                        config_tau_M_rel=np.float64(0.05),
                        config_theta_min_deg=np.float64(10.0),
                        config_hard_em=np.uint8(1),
                        config_image=str(img_path.name),
                        config_sigma=np.float64(sigma),
                        config_r=np.int32(r),
                        config_d=np.int32(d),
                        config_m_values=np.array(m_values, dtype=np.int32),
                        config_n_orientations=np.int32(n_orient),
                        config_H=np.int32(H),
                        config_W=np.int32(W))

    np.savez_compressed(out_dir / "expected.npz",
                        theta_primary=out["theta_primary"].astype(np.float32),
                        M_primary=out["M_primary"].astype(np.float32),
                        theta_sec=out["theta_sec"].astype(np.float32),
                        M_sec=out["M_sec"].astype(np.float32),
                        v_fused=out["v_fused"].astype(np.uint8),
                        primary_pi=out["primary_pi"].astype(np.float32),
                        primary_mu=out["primary_mu"].astype(np.float32),
                        primary_kappa=out["primary_kappa"].astype(np.float32),
                        secondary_pi=out["secondary_pi"].astype(np.float32),
                        secondary_mu=out["secondary_mu"].astype(np.float32),
                        secondary_kappa=out["secondary_kappa"].astype(np.float32),
                        keep_secondary_mask=out["keep_secondary_mask"])

    print(f"\nwrote {out_dir / 'inputs.npz'}")
    print(f"wrote {out_dir / 'expected.npz'}")
    print(f"  P = {P:,}   N = {N}")


if __name__ == "__main__":
    main()
