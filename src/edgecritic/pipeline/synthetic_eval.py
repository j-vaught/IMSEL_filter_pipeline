"""Image-wide two-pass c-GMM signal-mode evaluation.

Pipeline per condition (clean / noisy):
  1. Build GT orientation map and smooth-edge mask from the layer spec.
  2. Subsample N_PIXELS edge pixels from the smooth mask.
  3. Compute WVF gradients per channel (Metal if available).
  4. Vectorized LF response per (channel, m, theta) at the subsample.
  5. §6.3 orientation recovery: spline peak finder with the two-stage
     sentinel (no-local-max + tau_sec_floor weak-floor), producing the
     primary and secondary per-config measurement streams.
  6. Two-pass c-GMM fusion: independent K=3 hard-EM fits on each stream,
     primary = argmax-pi component of the primary fit, secondary =
     argmax-pi of the secondary fit.  Suppression rule:
        keep_sec = (M_sec/M_primary > tau_M_rel)
                   AND (|theta_primary - theta_sec| > theta_min_deg)
  7. Unsigned angle error of theta_primary vs GT tangent.

Usage::

    python -m edgecritic.pipeline.synthetic_eval \\
        --clean-rgb  <clean-rgb-image> \\
        --noisy-dir  /path/to/cetz_figures/data/color_channels/4096_noisy \\
        --manifest   <manifest.json> \\
        --image-key  garnet_atlantic_grass --size 4096 \\
        --r 9 --d 3 --m-values 0,5,10,20,30,40,50,60,70,80 \\
        --n-orientations 64 --n-pixels 2000 --seed 0 \\
        --out outputs/pipeline_synthetic_eval/result.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from edgecritic.wvf.radius import (
    wvf_radius_gradients_cpu, build_wvf_radius_kernels)

from edgecritic.cgmm.reference import cgmm_fuse_two_pass, theta_M_to_phi_w
from edgecritic.orientation.reference import find_two_peaks as _orec_find_two_peaks

try:
    from edgecritic.lf.metal import (
        lf_response_metal_batch, metal_backend_available)
    _METAL_LF_OK = metal_backend_available()
except Exception:
    _METAL_LF_OK = False
    lf_response_metal_batch = None

try:
    from edgecritic.wvf.metal import (
        wvf_radius_gradients_metal, metal_backend_available)
    _METAL_OK = metal_backend_available()
except Exception:
    _METAL_OK = False
    wvf_radius_gradients_metal = None


# -------- GT geometry (lifted from run_synthetic_wvf_orientation_gt.py) --

def star_points(center, outer_radius, inner_radius, rotation_deg):
    cx, cy = center
    start = math.radians(rotation_deg - 90.0)
    pts = []
    for i in range(10):
        radius = outer_radius if i % 2 == 0 else inner_radius
        a = start + i * math.pi / 5.0
        pts.append((cx + radius * math.cos(a),
                    cy + radius * math.sin(a)))
    return np.asarray(pts, dtype=np.float64)


def draw_layer_mask(layer, size):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = float(layer["center_x"]), float(layer["center_y"])
    w, h  = float(layer["width"]),    float(layer["height"])
    if layer["shape"] == "star":
        pts = star_points((cx, cy), w / 2.0, w * 0.34,
                          float(layer["rotation_deg"]))
        draw.polygon([tuple(p) for p in pts], fill=255)
    elif layer["shape"] == "square":
        half = w / 2.0
        draw.rectangle((cx - half, cy - half, cx + half, cy + half),
                       fill=255)
    elif layer["shape"] == "oval":
        draw.ellipse((cx - w / 2.0, cy - h / 2.0,
                      cx + w / 2.0, cy + h / 2.0), fill=255)
    return np.asarray(mask, dtype=bool)


def polygon_normal_angles(x, y, points):
    best_dist = np.full(x.shape, np.inf, dtype=np.float64)
    best_angle = np.zeros(x.shape, dtype=np.float64)
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        denom = vx * vx + vy * vy
        t = np.clip(((x - p0[0]) * vx + (y - p0[1]) * vy) / denom, 0.0, 1.0)
        proj_x = p0[0] + t * vx
        proj_y = p0[1] + t * vy
        dist = (x - proj_x) ** 2 + (y - proj_y) ** 2
        update = dist < best_dist
        normal_angle = math.atan2(vx, -vy)
        best_angle = np.where(update, normal_angle, best_angle)
        best_dist = np.where(update, dist, best_dist)
    return best_angle


def square_normal_angles(x, y, layer):
    cx, cy = float(layer["center_x"]), float(layer["center_y"])
    half = float(layer["width"]) / 2.0
    left   = abs(x - (cx - half))
    right  = abs(x - (cx + half))
    top    = abs(y - (cy - half))
    bottom = abs(y - (cy + half))
    distances = np.stack([left, right, top, bottom], axis=0)
    side = np.argmin(distances, axis=0)
    angle = np.zeros(x.shape, dtype=np.float64)
    angle = np.where(side == 2, math.pi / 2.0, angle)
    angle = np.where(side == 3, math.pi / 2.0, angle)
    return angle


def oval_normal_angles(x, y, layer):
    cx, cy = float(layer["center_x"]), float(layer["center_y"])
    a = float(layer["width"]) / 2.0
    b = float(layer["height"]) / 2.0
    nx = (x - cx) / max(a * a, 1e-12)
    ny = (y - cy) / max(b * b, 1e-12)
    return np.arctan2(ny, nx)


def vertex_mask_for_layer(layer, size, exclude_px):
    if exclude_px <= 0:
        return np.zeros((size, size), dtype=bool)
    cx, cy = float(layer["center_x"]), float(layer["center_y"])
    w = float(layer["width"])
    if layer["shape"] == "star":
        pts = star_points((cx, cy), w / 2.0, w * 0.34,
                          float(layer["rotation_deg"]))
    elif layer["shape"] == "square":
        half = w / 2.0
        pts = np.asarray([(cx - half, cy - half), (cx + half, cy - half),
                          (cx + half, cy + half), (cx - half, cy + half)],
                         dtype=np.float64)
    else:
        return np.zeros((size, size), dtype=bool)

    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    r2 = float(exclude_px * exclude_px)
    for px, py in pts:
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= r2
    return mask


def layer_angles(layer, x, y):
    if layer["shape"] == "star":
        pts = star_points(
            (float(layer["center_x"]), float(layer["center_y"])),
            float(layer["width"]) / 2.0,
            float(layer["width"]) * 0.34,
            float(layer["rotation_deg"]))
        return polygon_normal_angles(x, y, pts)
    if layer["shape"] == "square":
        return square_normal_angles(x, y, layer)
    if layer["shape"] == "oval":
        return oval_normal_angles(x, y, layer)
    raise ValueError(f"unsupported shape: {layer['shape']}")


def build_gt_orientation(image_spec, edge_band_px, vertex_exclude_px):
    size = int(image_spec["size"])
    orientation = np.zeros((size, size), dtype=np.float32)
    all_mask    = np.zeros((size, size), dtype=bool)
    smooth_mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]
    structure = np.ones((3, 3), dtype=bool)
    for layer in image_spec["layers"]:
        m = draw_layer_mask(layer, size)
        inner = m & ~ndimage.binary_erosion(m, structure=structure,
                                            border_value=0)
        outer = ndimage.binary_dilation(m, structure=structure) & ~m
        boundary = inner | outer
        if edge_band_px > 0:
            boundary = ndimage.binary_dilation(boundary,
                                               iterations=edge_band_px)
        a = layer_angles(layer,
                         xx[boundary].astype(np.float64),
                         yy[boundary].astype(np.float64))
        orientation[boundary] = np.mod(a, math.pi).astype(np.float32)
        all_mask |= boundary
        v = vertex_mask_for_layer(layer, size, vertex_exclude_px)
        if edge_band_px > 0:
            v = ndimage.binary_dilation(v, iterations=edge_band_px)
        smooth_mask |= boundary & ~v
    return orientation, all_mask, smooth_mask


# -------------------- LF / WVF --------------------

def compute_wvf(img_f64, r, d):
    if _METAL_OK:
        kernels = build_wvf_radius_kernels(radius=r, order=d)
        return wvf_radius_gradients_metal(img_f64.astype(np.float32),
                                          kernels)
    return wvf_radius_gradients_cpu(img_f64, radius=r, order=d)


def lf_response_at_pixels(g_x, g_y, px, py, theta, m):
    """Vectorized LF response over a batch of pixels (px, py)."""
    H, W = g_x.shape
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    if m <= 0:
        gp = -sin_t * g_x[py, px] + cos_t * g_y[py, px]
        return np.abs(gp)
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0
    sigma = m / 2.0
    j_offsets = np.arange(-m, m + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)
    ix = np.round(j_offsets * step * cos_t).astype(np.int32)
    iy = np.round(j_offsets * step * sin_t).astype(np.int32)
    yy = py[:, None] + iy[None, :]
    xx = px[:, None] + ix[None, :]
    valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_safe = np.clip(yy, 0, H - 1)
    xx_safe = np.clip(xx, 0, W - 1)
    g = -sin_t * g_x[yy_safe, xx_safe] + cos_t * g_y[yy_safe, xx_safe]
    g = g * valid
    w_eff = (weights[None, :] * valid)
    num = (g * weights[None, :]).sum(axis=1)
    den = w_eff.sum(axis=1)
    return np.abs(num / np.maximum(den, 1e-12))


def unsigned_angle_error_deg(est_deg, gt_deg):
    diff = (est_deg - gt_deg + 90.0) % 180.0 - 90.0
    return np.abs(diff)


# -------------------- driver --------------------

def load_channels_clean(rgb_path):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    R = rgb[..., 0].astype(np.float64)
    G = rgb[..., 1].astype(np.float64)
    B = rgb[..., 2].astype(np.float64)
    L = (0.2126 * R + 0.7152 * G + 0.0722 * B)
    return {"L": L, "R": R, "G": G, "B": B}


def load_channels_noisy(noisy_dir):
    out = {}
    for ch in "LRGB":
        p = noisy_dir / f"channel_{ch}.png"
        out[ch] = np.asarray(Image.open(p).convert("L"),
                             dtype=np.float64)
    return out


def find_image_spec(manifest_path, image_key, size):
    m = json.loads(Path(manifest_path).read_text())
    for img in m["images"]:
        if int(img["size"]) == size and image_key in img.get("palette", ""):
            return img
    raise ValueError(f"no image with key '{image_key}' size {size} in manifest")


def evaluate(label, channels, sample_pixels, gt_tangent_at_samples,
             m_values, n_orientations, r, d,
             tau_sec_floor=0.40,
             spline_dense_n=500):
    angles = np.linspace(0, math.pi, n_orientations, endpoint=False)
    px = sample_pixels[:, 0].astype(np.int32)
    py = sample_pixels[:, 1].astype(np.int32)
    N = len(px)
    n_m = len(m_values)
    n_ch = len(channels)

    primary_t = np.zeros((N, n_ch * n_m), dtype=np.float64)
    primary_m = np.zeros((N, n_ch * n_m), dtype=np.float64)
    secondary_t = np.zeros((N, n_ch * n_m), dtype=np.float64)
    secondary_m = np.zeros((N, n_ch * n_m), dtype=np.float64)

    m_arr = np.asarray(m_values, dtype=np.int32)
    col = 0
    for ch_name, img in channels.items():
        t0 = time.perf_counter()
        g_x, g_y = compute_wvf(img, r, d)
        print(f"  [{label}] WVF channel {ch_name}: "
              f"{time.perf_counter()-t0:.1f}s")

        if _METAL_LF_OK:
            # Single batched Metal call: (T orientations, M m-values, P pixels).
            t1 = time.perf_counter()
            grid = lf_response_metal_batch(
                g_x.astype(np.float32), g_y.astype(np.float32),
                px, py, angles, m_arr)
            print(f"    LF metal batch (T={n_orientations}, M={n_m}): "
                  f"{time.perf_counter()-t1:.2f}s")
            for m_idx, m in enumerate(m_values):
                resp = grid[:, m_idx, :].T.astype(np.float64)  # (P, T)
                t_p, m_p, t_s, m_s, _ = _orec_find_two_peaks(
                    angles, resp, tau_sec_floor=tau_sec_floor,
                    dense_n=spline_dense_n)
                primary_t[:, col]   = np.degrees(t_p)
                primary_m[:, col]   = m_p
                secondary_t[:, col] = np.degrees(t_s)
                secondary_m[:, col] = m_s
                col += 1
        else:
            for m in m_values:
                t1 = time.perf_counter()
                resp = np.zeros((N, n_orientations), dtype=np.float64)
                for k, theta in enumerate(angles):
                    resp[:, k] = lf_response_at_pixels(g_x, g_y, px, py,
                                                       float(theta),
                                                       int(m))
                t_p, m_p, t_s, m_s, _ = _orec_find_two_peaks(
                    angles, resp, tau_sec_floor=tau_sec_floor,
                    dense_n=spline_dense_n)
                primary_t[:, col]   = np.degrees(t_p)
                primary_m[:, col]   = m_p
                secondary_t[:, col] = np.degrees(t_s)
                secondary_m[:, col] = m_s
                col += 1
                print(f"    m={m:>3}: {time.perf_counter()-t1:.1f}s")

    return primary_t, primary_m, secondary_t, secondary_m


def fit_cgmm_errors(primary_t, primary_m, secondary_t, secondary_m,
                    gt_tangent_at, K, label,
                    n_iters=30, init_kappa=4.0,
                    hard_seed=False, hard_em=True,
                    tau_M_rel=0.05, theta_min_deg=10.0):
    """Two-pass c-GMM fusion + primary-orientation error vs GT tangent."""
    phi_p, w_p, _ = theta_M_to_phi_w(primary_t,   primary_m)
    phi_s, w_s, _ = theta_M_to_phi_w(secondary_t, secondary_m)
    t0 = time.perf_counter()
    out = cgmm_fuse_two_pass(phi_p, w_p, phi_s, w_s,
                             K=K, n_iters=n_iters,
                             init_kappa=init_kappa,
                             hard_seed=hard_seed, hard_em=hard_em,
                             tau_M_rel=tau_M_rel,
                             theta_min_deg=theta_min_deg)
    elapsed = time.perf_counter() - t0
    theta_est_deg = np.degrees(out["theta_primary"]) % 180.0
    errs = unsigned_angle_error_deg(theta_est_deg, gt_tangent_at)
    print(f"  [{label}] two-pass c-GMM K={K} fit: {elapsed:.2f}s")
    return errs, elapsed


PCTS = (50, 90, 99, 99.9, 99.99, 99.999)


def summarise(name, errs):
    valid = errs[np.isfinite(errs)]
    out = dict(name=name, n=int(len(valid)),
               mean=float("nan"), max=float("nan"))
    for q in PCTS:
        out[f"p{q}"] = float("nan")
    if len(valid) == 0:
        return out
    out["mean"] = float(np.mean(valid))
    out["max"]  = float(np.max(valid))
    for q in PCTS:
        out[f"p{q}"] = float(np.percentile(valid, q))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-rgb",  required=True, type=Path)
    p.add_argument("--noisy-dir",  required=True, type=Path)
    p.add_argument("--manifest",   required=True, type=Path)
    p.add_argument("--image-key",  default="garnet_atlantic_grass")
    p.add_argument("--size",       type=int, default=4096)
    p.add_argument("--r", type=int, default=9)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--m-values", default="0,5,10,20,30,40,50,60,70,80")
    p.add_argument("--n-orientations", type=int, default=64)
    p.add_argument("--n-pixels", type=int, default=2000)
    p.add_argument("--edge-band-px", type=int, default=0)
    p.add_argument("--vertex-exclude-px", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--Ks", default="3",
                   help="comma list of component counts to evaluate")
    # §6.3 orientation-recovery sentinel
    p.add_argument("--cgmm-tau-sec-floor", "--vmm-tau-sec-floor",
                   dest="cgmm_tau_sec_floor", type=float, default=0.40,
                   help="orientation recovery (§6.3): per-config "
                        "secondary peaks with M_sec/M_hat below this "
                        "are zeroed before fusion (default 0.40)")
    # Two-pass fusion suppression rule
    p.add_argument("--cgmm-tau-M-rel", "--vmm-tau-M-rel",
                   dest="cgmm_tau_M_rel", type=float, default=0.05,
                   help="secondary suppression: M_sec/M_primary "
                        "must exceed this (default 0.05)")
    p.add_argument("--cgmm-theta-min-deg", "--vmm-theta-min-deg",
                   dest="cgmm_theta_min_deg", type=float, default=10.0,
                   help="secondary suppression: |theta_primary - theta_sec| "
                        "must exceed this (default 10 deg)")
    # c-GMM EM knobs
    p.add_argument("--cgmm-n-iters", "--vmm-n-iters",
                   dest="cgmm_n_iters", type=int, default=30)
    p.add_argument("--cgmm-init-kappa", "--vmm-init-kappa",
                   dest="cgmm_init_kappa", type=float, default=4.0,
                   help="initial vM concentration (used by soft EM only; "
                        "ignored under hard EM)")
    p.add_argument("--cgmm-hard-seed", "--vmm-hard-seed",
                   dest="cgmm_hard_seed", action="store_true",
                   help="use hard k-means assignment at iter 0 (soft EM "
                        "thereafter); diagnostic, not production")
    p.add_argument("--cgmm-soft-em", "--vmm-soft-em",
                   dest="cgmm_soft_em", action="store_true",
                   help="opt-in to soft-EM (hard-EM is production "
                        "default; soft has a ~0.15 deg p50 floor on tight "
                        "unimodal data due to prior-leakage)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    print(f"WVF backend: {'Metal (GPU)' if _METAL_OK else 'CPU'}")

    spec = find_image_spec(args.manifest, args.image_key, args.size)
    print(f"image spec: {spec['palette']}, size={spec['size']}, "
          f"{len(spec['layers'])} layers")

    t0 = time.perf_counter()
    gt_normal, all_mask, smooth_mask = build_gt_orientation(
        spec, args.edge_band_px, args.vertex_exclude_px)
    print(f"GT built: smooth_mask has {smooth_mask.sum():,} pixels "
          f"({time.perf_counter()-t0:.1f}s)")

    rng = np.random.default_rng(args.seed)
    ys, xs = np.where(smooth_mask)
    if len(ys) > args.n_pixels:
        idx = rng.choice(len(ys), size=args.n_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]
    sample_pixels = np.column_stack([xs, ys])

    # Convert GT normal -> tangent (LF reports edge tangent direction).
    gt_normal_at = gt_normal[ys, xs].astype(np.float64)
    gt_tangent_at = np.degrees((gt_normal_at + math.pi / 2.0) % math.pi)

    m_values = [int(s) for s in args.m_values.split(",") if s.strip()]
    Ks       = [int(s) for s in args.Ks.split(",")      if s.strip()]

    # ---- Clean ----
    print("\n========== CLEAN ==========")
    clean_channels = load_channels_clean(args.clean_rgb)
    pt_c, pm_c, st_c, sm_c = evaluate("clean", clean_channels,
                                      sample_pixels, gt_tangent_at,
                                      m_values, args.n_orientations,
                                      args.r, args.d,
                                      tau_sec_floor=args.cgmm_tau_sec_floor)

    # ---- Noisy ----
    print("\n========== NOISY ==========")
    noisy_channels = load_channels_noisy(args.noisy_dir)
    pt_n, pm_n, st_n, sm_n = evaluate("noisy", noisy_channels,
                                      sample_pixels, gt_tangent_at,
                                      m_values, args.n_orientations,
                                      args.r, args.d,
                                      tau_sec_floor=args.cgmm_tau_sec_floor)

    # ---- Two-pass c-GMM fit per K, per condition ----
    rows = []
    timings = {}
    for cond, pt, pm, st, sm in (
            ("clean", pt_c, pm_c, st_c, sm_c),
            ("noisy", pt_n, pm_n, st_n, sm_n),
    ):
        for K in Ks:
            tag = f"{cond} / c-GMM K={K}"
            errs, t = fit_cgmm_errors(pt, pm, st, sm, gt_tangent_at, K, tag,
                                      n_iters=args.cgmm_n_iters,
                                      init_kappa=args.cgmm_init_kappa,
                                      hard_seed=args.cgmm_hard_seed,
                                      hard_em=(not args.cgmm_soft_em),
                                      tau_M_rel=args.cgmm_tau_M_rel,
                                      theta_min_deg=args.cgmm_theta_min_deg)
            rows.append(summarise(tag, errs))
            timings[tag] = t

    print("\n=========================================================")
    print(f"Image-wide c-GMM evaluation @ smooth-edge pixels (n={len(ys)})")
    print(f"r={args.r}  d={args.d}  m={args.m_values}  "
          f"n_orientations={args.n_orientations}")
    print("=========================================================")
    hdr = (f"{'condition':<22} {'mean':>7} {'p50':>7} {'p90':>7} "
           f"{'p99':>7} {'p99.9':>7} {'p99.99':>8} {'p99.999':>8} "
           f"{'max':>7} {'n':>8} {'fit_s':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = timings.get(r["name"], float("nan"))
        print(f"{r['name']:<22} {r['mean']:>7.3f} {r['p50']:>7.3f} "
              f"{r['p90']:>7.3f} {r['p99']:>7.3f} "
              f"{r['p99.9']:>7.3f} {r['p99.99']:>8.3f} "
              f"{r['p99.999']:>8.3f} {r['max']:>7.3f} "
              f"{r['n']:>8d} {t:>7.2f}")
    print("(degrees, unsigned line-orientation error vs GT tangent; "
          "fit_s = wall-clock for the c-GMM step only)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "r": args.r, "d": args.d,
                "m_values": m_values,
                "n_orientations": args.n_orientations,
                "n_pixels": int(len(ys)),
                "vertex_exclude_px": args.vertex_exclude_px,
                "edge_band_px": args.edge_band_px,
                "seed": args.seed,
                "image_key": args.image_key,
                "size": args.size,
                "Ks": Ks,
                "cgmm_tau_sec_floor": args.cgmm_tau_sec_floor,
                "cgmm_tau_M_rel": args.cgmm_tau_M_rel,
                "cgmm_theta_min_deg": args.cgmm_theta_min_deg,
                "cgmm_n_iters": args.cgmm_n_iters,
                "architecture": "two-pass+sec-sentinel",
            },
            "results": rows,
            "timings_seconds": timings,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
