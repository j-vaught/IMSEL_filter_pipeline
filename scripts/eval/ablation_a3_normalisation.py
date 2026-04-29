"""A3 reviewer-pass-5 ablation: magnitude normalization before c-GMM fusion."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from cgmm_image_wide_eval import build_gt_orientation, find_image_spec
from cgmm_nms import enhanced_nms
from edgecritic.cgmm import cgmm_fuse_two_pass_metal
from edgecritic.evaluation.metrics import compute_ods_ois
from edgecritic.pipeline import wvf_lf_recover_metal


VARIANTS = (
    ("baseline", "baseline"),
    ("robust_p99", "A robust p99"),
    ("kernel_gain", "B unit gain"),
    ("noise_mad", "C noise MAD"),
    ("combo", "D unit gain + p99"),
)

ORIENTATION_COLORS = (
    "#73000A",
    "#466A9F",
    "#1F414D",
    "#65780B",
    "#A49137",
    "#CC2E40",
    "#5C5C5C",
    "#000000",
)


def _scaled_spec(spec: dict, size: int) -> dict:
    out = copy.deepcopy(spec)
    scale = float(size) / float(spec["size"])
    out["size"] = int(size)
    for layer in out["layers"]:
        for key in ("center_x", "center_y", "width", "height"):
            if key in layer:
                layer[key] = float(layer[key]) * scale
    return out


def _load_base_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _make_production_synthetic(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = rgb + rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0)


def _make_aquatic_surrogate(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    attenuation = np.array([0.50, 0.76, 0.95], dtype=np.float32)
    water_color = np.array([24.0, 78.0, 92.0], dtype=np.float32)
    haze = 0.25 + 0.35 * yy + 0.10 * np.sin(2.0 * math.pi * xx)
    low_contrast = 128.0 + 0.62 * (rgb - 128.0)
    degraded = low_contrast * attenuation[None, None, :] + haze * water_color
    texture = rng.normal(0.0, 4.5, (h, w, 1)).astype(np.float32)
    channel_noise = rng.normal(0.0, 8.0, rgb.shape).astype(np.float32)
    degraded = ndimage.gaussian_filter(degraded + texture + channel_noise, sigma=(0.45, 0.45, 0.0))
    return np.clip(degraded, 0.0, 255.0).astype(np.float32)


def _channels(rgb: np.ndarray) -> dict[str, np.ndarray]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    return {"L": luma, "R": r, "G": g, "B": b}


def _theta_error_deg(theta: np.ndarray, gt: np.ndarray) -> np.ndarray:
    delta = np.abs(theta - gt)
    return np.degrees(np.minimum(delta, math.pi - delta))


def _unit_step_gain(radius: int, degree: int, lf_half_length: int) -> float:
    size = 256
    coords = np.arange(size, dtype=np.float32)
    _, xx = np.meshgrid(coords, coords, indexing="ij")
    signed = xx - (size - 1) / 2.0
    image = 0.5 + 0.5 * np.tanh(signed / 1.25)
    _, mag, _, _, _ = wvf_lf_recover_metal(
        image.astype(np.float32),
        radius=radius,
        degree=degree,
        lf_half_length=lf_half_length,
        n_orientations=64,
        tau_sec_floor=0.40,
        tau_validity=0.0,
        dense_n=500,
        min_sep_frac=0.125,
        method="box",
    )
    edge = np.abs(signed) <= 0.6
    edge[: radius + lf_half_length + 4, :] = False
    edge[-(radius + lf_half_length + 4) :, :] = False
    edge[:, : radius + lf_half_length + 4] = False
    edge[:, -(radius + lf_half_length + 4) :] = False
    values = mag[edge]
    if values.size == 0:
        return 1.0
    gain = float(np.median(values))
    return max(gain, 1.0e-6)


def _build_configs(channels: list[str], radii: list[int], degrees: list[int], lf_half_lengths: list[int]) -> list[dict]:
    configs = []
    gain_cache: dict[tuple[int, int, int], float] = {}
    for channel in channels:
        for radius in radii:
            for degree in degrees:
                for m in lf_half_lengths:
                    key = (radius, degree, m)
                    if key not in gain_cache:
                        gain_cache[key] = _unit_step_gain(radius, degree, m)
                    configs.append(
                        {
                            "channel": channel,
                            "radius": radius,
                            "degree": degree,
                            "lf_half_length": m,
                            "unit_step_gain": gain_cache[key],
                        }
                    )
    return configs


def _front_end_stack(rgb: np.ndarray, configs: list[dict]) -> dict[str, np.ndarray]:
    h, w = rgb.shape[:2]
    p = h * w
    n = len(configs)
    chan = _channels(rgb)
    theta_p = np.empty((p, n), dtype=np.float32)
    mag_p = np.empty((p, n), dtype=np.float32)
    theta_s = np.empty((p, n), dtype=np.float32)
    mag_s = np.empty((p, n), dtype=np.float32)
    valid = np.empty((p, n), dtype=np.uint8)

    for j, cfg in enumerate(configs):
        t0 = time.perf_counter()
        thp, mp, ths, ms, v = wvf_lf_recover_metal(
            chan[cfg["channel"]],
            radius=cfg["radius"],
            degree=cfg["degree"],
            lf_half_length=cfg["lf_half_length"],
            n_orientations=64,
            tau_sec_floor=0.40,
            tau_validity=0.10,
            dense_n=500,
            min_sep_frac=0.125,
            method="box",
        )
        theta_p[:, j] = thp.reshape(p).astype(np.float32)
        mag_p[:, j] = mp.reshape(p).astype(np.float32)
        theta_s[:, j] = ths.reshape(p).astype(np.float32)
        mag_s[:, j] = ms.reshape(p).astype(np.float32)
        valid[:, j] = v.reshape(p)
        print(
            f"    {j + 1:02d}/{n} ch={cfg['channel']} r={cfg['radius']} "
            f"d={cfg['degree']} m={cfg['lf_half_length']}: "
            f"v={v.mean() * 100:5.1f}% {time.perf_counter() - t0:.2f}s"
        )

    return {
        "theta_p": theta_p,
        "mag_p": mag_p,
        "theta_s": theta_s,
        "mag_s": mag_s,
        "valid": valid,
    }


def _stack_stats(stack: dict[str, np.ndarray], configs: list[dict], flat_mask: np.ndarray) -> dict[str, np.ndarray]:
    mag_p = stack["mag_p"]
    valid = stack["valid"].astype(bool)
    n = mag_p.shape[1]
    robust = np.ones(n, dtype=np.float32)
    noise = np.ones(n, dtype=np.float32)
    gain = np.asarray([cfg["unit_step_gain"] for cfg in configs], dtype=np.float32)
    flat = flat_mask.reshape(-1)
    for j in range(n):
        valid_values = mag_p[valid[:, j], j]
        if valid_values.size:
            robust[j] = max(float(np.percentile(valid_values, 99.0)), 1.0e-6)
        flat_values = mag_p[flat, j]
        if flat_values.size:
            med = float(np.median(flat_values))
            mad = float(np.median(np.abs(flat_values - med))) * 1.4826
            noise[j] = max(mad, 1.0e-6)
    gain_adjusted = mag_p / gain[None, :]
    robust_after_gain = np.ones(n, dtype=np.float32)
    gain_valid = gain_adjusted > 0.10
    for j in range(n):
        values = gain_adjusted[gain_valid[:, j], j]
        if values.size:
            robust_after_gain[j] = max(float(np.percentile(values, 99.0)), 1.0e-6)
    return {
        "robust_p99": robust,
        "unit_step_gain": gain,
        "noise_mad": noise,
        "robust_after_gain": robust_after_gain,
    }


def _variant_inputs(stack: dict[str, np.ndarray], stats: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_p = stack["theta_p"]
    theta_s = stack["theta_s"]
    mag_p = stack["mag_p"]
    mag_s = stack["mag_s"]
    valid = stack["valid"].astype(np.float32)

    if variant == "baseline":
        mp = mag_p
        ms = mag_s
        v = valid
    elif variant == "robust_p99":
        scale = stats["robust_p99"][None, :]
        mp = mag_p / scale
        ms = mag_s / scale
        v = valid
    elif variant == "kernel_gain":
        scale = stats["unit_step_gain"][None, :]
        mp = mag_p / scale
        ms = mag_s / scale
        v = (mp > 0.10).astype(np.float32)
    elif variant == "noise_mad":
        scale = stats["noise_mad"][None, :]
        mp = mag_p / scale
        ms = mag_s / scale
        v = valid
    elif variant == "combo":
        gain = stats["unit_step_gain"][None, :]
        scale = stats["robust_after_gain"][None, :]
        mp = (mag_p / gain) / scale
        ms = (mag_s / gain) / scale
        v = ((mag_p / gain) > 0.10).astype(np.float32)
    else:
        raise ValueError(variant)

    finite_p = np.isfinite(theta_p) & np.isfinite(mp) & (mp > 0)
    finite_s = np.isfinite(theta_s) & np.isfinite(ms) & (ms > 0)
    two_pi = np.float32(2.0 * math.pi)
    phi_p = np.where(finite_p, (2.0 * theta_p) % two_pi, 0.0).astype(np.float32)
    w_p = (v * np.where(finite_p, np.maximum(mp, 0.0), 0.0)).astype(np.float32)
    phi_s = np.where(finite_s, (2.0 * theta_s) % two_pi, 0.0).astype(np.float32)
    w_s = (v * np.where(finite_s, np.maximum(ms, 0.0), 0.0)).astype(np.float32)
    return phi_p, w_p, phi_s, w_s


def _fuse_variant(stack: dict[str, np.ndarray], stats: dict[str, np.ndarray], variant: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    phi_p, w_p, phi_s, w_s = _variant_inputs(stack, stats, variant)
    t0 = time.perf_counter()
    out = cgmm_fuse_two_pass_metal(
        phi_p,
        w_p,
        phi_s,
        w_s,
        K=3,
        n_iters=30,
        init_kappa=4.0,
        hard_em=True,
        tau_M_rel=0.05,
        theta_min_deg=10.0,
    )
    print(f"      c-GMM {variant}: {time.perf_counter() - t0:.2f}s")
    h, w = shape
    return {
        "theta_primary": out["theta_primary"].reshape(h, w),
        "M_primary": out["M_primary"].reshape(h, w),
        "theta_sec": out["theta_sec"].reshape(h, w),
        "M_sec": out["M_sec"].reshape(h, w),
        "v_fused": out["v_fused"].reshape(h, w),
    }


def _metrics_for_output(out: dict[str, np.ndarray], gt_tangent: np.ndarray, all_mask: np.ndarray, smooth_mask: np.ndarray) -> dict:
    theta = out["theta_primary"]
    v = out["v_fused"] == 1
    finite = v & np.isfinite(theta)
    err = np.full(theta.shape, 90.0, dtype=np.float64)
    err[finite] = _theta_error_deg(theta[finite], gt_tangent[finite])

    smooth_err = err[smooth_mask]
    gt_deg = np.degrees(gt_tangent) % 180.0
    worst = 0.0
    for lo in range(0, 180, 15):
        bin_mask = smooth_mask & (gt_deg >= lo) & (gt_deg < lo + 15)
        if bin_mask.any():
            worst = max(worst, float(err[bin_mask].mean()))

    junction_mask = all_mask & ~smooth_mask
    if junction_mask.any():
        secondary = (
            (out["M_sec"] > 0)
            & np.isfinite(out["theta_sec"])
            & (out["v_fused"] == 1)
        )
        corner_recall = float(secondary[junction_mask].mean())
    else:
        corner_recall = 0.0

    nms = enhanced_nms(
        out["theta_primary"],
        out["M_primary"],
        out["theta_sec"],
        out["M_sec"],
        out["v_fused"],
        neighborhood=4,
        angular_fidelity="Acont",
        corner_method="or",
    )
    ods, _, thresholds, f_scores = compute_ods_ois(
        nms,
        all_mask.astype(np.float64),
        n_thresholds=80,
        match_radius=3,
    )
    best_idx = int(np.argmax(f_scores)) if f_scores.size else 0
    threshold = float(thresholds[best_idx]) if thresholds.size else 0.0
    return {
        "mean_error_deg": float(np.mean(smooth_err)) if smooth_err.size else 90.0,
        "worst_bin_error_deg": float(worst),
        "corner_recall": corner_recall,
        "nms_f_measure": float(ods),
        "n_smooth_pixels": int(smooth_mask.sum()),
        "n_junction_pixels": int(junction_mask.sum()),
        "nms_threshold": threshold,
        "valid_fraction": float(v.mean()),
    }


def _map_grid(out: dict[str, np.ndarray], edge_mask: np.ndarray, grid_size: int = 32) -> list[list[str]]:
    theta = out["theta_primary"]
    valid = (out["v_fused"] == 1) & np.isfinite(theta) & edge_mask
    h, w = theta.shape
    rows: list[list[str]] = []
    for gy in range(grid_size):
        row = []
        y0 = int(round(gy * h / grid_size))
        y1 = int(round((gy + 1) * h / grid_size))
        for gx in range(grid_size):
            x0 = int(round(gx * w / grid_size))
            x1 = int(round((gx + 1) * w / grid_size))
            mask = valid[y0:y1, x0:x1]
            if not mask.any():
                row.append("#ECECEC")
                continue
            vals = theta[y0:y1, x0:x1][mask]
            c = float(np.mean(np.cos(2.0 * vals)))
            s = float(np.mean(np.sin(2.0 * vals)))
            angle = (0.5 * math.atan2(s, c)) % math.pi
            idx = int(math.floor(angle / math.pi * len(ORIENTATION_COLORS))) % len(ORIENTATION_COLORS)
            row.append(ORIENTATION_COLORS[idx])
        rows.append(row)
    return rows


def _metric_winners(rows_by_variant: dict[str, dict], image_key: str) -> dict[str, int]:
    wins = {variant: 0 for variant in rows_by_variant}
    metrics = (
        ("mean_error_deg", "min"),
        ("corner_recall", "max"),
        ("nms_f_measure", "max"),
    )
    for metric, direction in metrics:
        values = {
            variant: rows_by_variant[variant][image_key][metric]
            for variant in rows_by_variant
        }
        best = min(values.values()) if direction == "min" else max(values.values())
        for variant, value in values.items():
            if abs(value - best) <= 1.0e-9:
                wins[variant] += 1
    return wins


def _choose_decision(summary_rows: list[dict], image_keys: tuple[str, str]) -> tuple[str, str]:
    by_variant = {row["variant"]: row["metrics"] for row in summary_rows}
    win_by_image = {image: _metric_winners(by_variant, image) for image in image_keys}
    for row in summary_rows:
        row["wins_by_image"] = {
            image: win_by_image[image][row["variant"]]
            for image in image_keys
        }
    qualifying = [
        row for row in summary_rows
        if all(row["wins_by_image"][image] >= 2 for image in image_keys)
    ]
    if qualifying:
        best = max(
            qualifying,
            key=lambda row: (
                sum(row["wins_by_image"].values()),
                sum(row["metrics"][image]["nms_f_measure"] for image in image_keys),
                -sum(row["metrics"][image]["mean_error_deg"] for image in image_keys),
            ),
        )
        return best["variant"], (
            f"{best['label']} wins at least two of the three decision metrics "
            "on both test images."
        )

    best = max(
        summary_rows,
        key=lambda row: (
            sum(row["wins_by_image"].values()),
            sum(row["metrics"][image]["nms_f_measure"] for image in image_keys),
            -sum(row["metrics"][image]["mean_error_deg"] for image in image_keys),
        ),
    )
    return best["variant"], (
        f"No variant wins two metrics on both images; {best['label']} has the "
        "best aggregate rank across the decision metrics."
    )


def run_ablation(
    output_path: Path,
    image_path: Path,
    manifest_path: Path,
    image_key: str,
    size: int,
    sigma: float,
    channels: list[str],
    radii: list[int],
    degrees: list[int],
    lf_half_lengths: list[int],
    map_grid_size: int,
) -> dict:
    rng = np.random.default_rng(2026)
    base_spec = find_image_spec(manifest_path, image_key, 4096)
    spec = _scaled_spec(base_spec, size)
    gt_normal, all_mask, smooth_mask = build_gt_orientation(
        spec,
        edge_band_px=0,
        vertex_exclude_px=max(3, int(round(24 * size / 4096))),
    )
    gt_tangent = (gt_normal.astype(np.float64) + math.pi / 2.0) % math.pi
    flat_mask = ~ndimage.binary_dilation(all_mask, iterations=max(2, size // 128))

    base_rgb = _load_base_rgb(image_path, size)
    images = {
        "synthetic": _make_production_synthetic(base_rgb, sigma, rng),
        "aquatic": _make_aquatic_surrogate(base_rgb, rng),
    }

    print("calibrating unit-step gains")
    configs = _build_configs(channels, radii, degrees, lf_half_lengths)
    summary = {variant: {"variant": variant, "label": label, "metrics": {}} for variant, label in VARIANTS}
    map_panels = []

    for image_name, rgb in images.items():
        print(f"\n========== {image_name} ==========")
        stack = _front_end_stack(rgb, configs)
        stats = _stack_stats(stack, configs, flat_mask)
        for variant, _ in VARIANTS:
            out = _fuse_variant(stack, stats, variant, (size, size))
            metrics = _metrics_for_output(out, gt_tangent, all_mask, smooth_mask)
            summary[variant]["metrics"][image_name] = metrics
            map_panels.append(
                {
                    "image": image_name,
                    "variant": variant,
                    "grid": _map_grid(out, all_mask, grid_size=map_grid_size),
                }
            )
            print(
                f"    {variant:>12}: mean={metrics['mean_error_deg']:.3f} deg "
                f"worst={metrics['worst_bin_error_deg']:.3f} deg "
                f"corner={metrics['corner_recall']:.3f} "
                f"F={metrics['nms_f_measure']:.3f}"
            )
        del stack

    summary_rows = [summary[variant] for variant, _ in VARIANTS]
    decision, decision_text = _choose_decision(summary_rows, ("synthetic", "aquatic"))
    output = {
        "ablation": "A3",
        "config": {
            "image": str(image_path),
            "image_key": image_key,
            "size": size,
            "sigma": sigma,
            "channels": channels,
            "radii": radii,
            "degrees": degrees,
            "lf_half_lengths": lf_half_lengths,
            "n_orientations": 64,
            "dense_n": 500,
            "n_configs": len(configs),
            "map_grid_size": map_grid_size,
            "test_images": {
                "synthetic": "production synthetic geometry with sigma-noise",
                "aquatic": "deterministic aquatic-style degradation of the same geometry",
            },
        },
        "config_scales": configs,
        "summary_rows": summary_rows,
        "maps": map_panels,
        "summary": {
            "decision": decision,
            "decision_text": decision_text,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT.parent / "New project")
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "example_images" / "synthetic_nested_shapes" / "clean" / "4096" / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "example_images" / "synthetic_nested_shapes" / "manifest.json",
    )
    parser.add_argument("--image-key", default="low_contrast_mixed_chroma")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=13.0)
    parser.add_argument("--channels", default="L,R,G,B")
    parser.add_argument("--radii", default="5,9")
    parser.add_argument("--degrees", default="1,3")
    parser.add_argument("--lf-half-lengths", default="40,60,80,100")
    parser.add_argument("--map-grid-size", type=int, default=32)
    args = parser.parse_args()

    out = args.paper_root / "cetz_figures" / "data" / "ablation_a3" / "results.json"
    result = run_ablation(
        out,
        args.image,
        args.manifest,
        args.image_key,
        args.size,
        args.sigma,
        _str_list(args.channels),
        _int_list(args.radii),
        _int_list(args.degrees),
        _int_list(args.lf_half_lengths),
        args.map_grid_size,
    )
    print(f"wrote {out}")
    print(f"A3 decision: {result['summary']['decision']}")
    for row in result["summary_rows"]:
        wins = row["wins_by_image"]
        syn = row["metrics"]["synthetic"]
        aq = row["metrics"]["aquatic"]
        print(
            f"  {row['variant']:>12}: wins synthetic/aquatic "
            f"{wins['synthetic']}/{wins['aquatic']}; "
            f"mean err {syn['mean_error_deg']:.3f}/{aq['mean_error_deg']:.3f}; "
            f"F {syn['nms_f_measure']:.3f}/{aq['nms_f_measure']:.3f}"
        )


if __name__ == "__main__":
    main()
