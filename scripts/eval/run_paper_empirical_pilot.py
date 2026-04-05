"""Pilot empirical evaluation for the anisotropic LF paper.

This script covers three execution modes:

1. ``backend_bench`` compares the original LF backend against the fused
   conv2d and Triton implementations on a small image subset.
2. ``clean_pilot`` runs a narrowed clean-data comparison across the filter
   families discussed in the manuscript.
3. ``noise_pilot`` repeats a smaller cross-family comparison under a chosen
   noise model and SNR.

All runs report dataset-level ODS/OIS together with simple timing and peak
VRAM measurements. Missing datasets are skipped.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image

from edgecritic.aniso_lf import aniso_lf_conv2d, aniso_lf_triton
from edgecritic.baselines.classical_gpu import gaussian_derivative
from edgecritic.core.taylor import compute_wvf_pseudoinverse
from edgecritic.evaluation.metrics import compute_ods_ois
from edgecritic.lf import lf_image
from edgecritic.wvf import wvf_image


ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "paper_empirical"
OUT.mkdir(parents=True, exist_ok=True)

MATCH_RADIUS = 3
N_THRESH = 1001


@dataclass(frozen=True)
class MethodConfig:
    label: str
    method: str
    params: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["backend_bench", "clean_pilot", "noise_pilot"],
        required=True,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BIPED_v1", "BIPED_v2", "BSDS500", "UDED"],
    )
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--speed-runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--noise-type", choices=["gaussian", "speckle"], default="gaussian")
    parser.add_argument("--snr", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def reset_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6


def load_biped(version: str, n: int | None = None):
    if version == "v1":
        base = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
    else:
        base = ROOT / "datasets" / "BIPED" / "BIPEDv2" / "BIPEDv2" / "BIPED" / "edges"
    img_dir = base / "imgs" / "test" / "rgbr"
    gt_dir = base / "edge_maps" / "test" / "rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        images.append(np.mean(np.array(Image.open(f)), axis=2).astype(np.float64))
        gts.append(np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128)
        names.append(f.stem)
        if n is not None and len(images) >= n:
            break
    return images, gts, names


def load_bsds500(n: int | None = None):
    base1 = ROOT / "datasets" / "BSDS500" / "BSDS500" / "data"
    base2 = ROOT / "datasets" / "BSDS500" / "BSDS500"
    base = base1 if (base1 / "images" / "test").exists() else base2
    img_dir = base / "images" / "test"
    gt_dir = base / "groundTruth" / "test"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2).astype(np.float64)
        gt_mat = sio.loadmat(str(gt_dir / f"{f.stem}.mat"))
        gt_cell = gt_mat["groundTruth"]
        gt_union = np.zeros(img.shape[:2], dtype=bool)
        for i in range(gt_cell.shape[1]):
            bdry = gt_cell[0, i]["Boundaries"][0, 0]
            bdry = bdry.toarray() if hasattr(bdry, "toarray") else np.asarray(bdry)
            gt_union |= bdry > 0
        images.append(img)
        gts.append(gt_union)
        names.append(f.stem)
        if n is not None and len(images) >= n:
            break
    return images, gts, names


def load_uded(n: int | None = None):
    img_dir = ROOT / "datasets" / "UDED" / "imgs"
    gt_dir = ROOT / "datasets" / "UDED" / "gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        images.append(np.mean(np.array(Image.open(f)), axis=2).astype(np.float64))
        gts.append(np.array(Image.open(gt_dir / f.name).convert("L")) > 128)
        names.append(f.stem)
        if n is not None and len(images) >= n:
            break
    return images, gts, names


def load_dataset(name: str, n: int):
    if name == "BIPED_v1":
        return load_biped("v1", n=n)
    if name == "BIPED_v2":
        return load_biped("v2", n=n)
    if name == "BSDS500":
        return load_bsds500(n=n)
    if name == "UDED":
        return load_uded(n=n)
    raise ValueError(f"Unknown dataset: {name}")


def add_gaussian(img: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    sigma = (img.max() - img.min()) / max(snr, 1e-6)
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0, 255)


def add_speckle(img: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    sigma = 1.0 / max(snr, 1e-6)
    return np.clip(img * (1.0 + rng.normal(0.0, sigma, img.shape)), 0, 255)


def apply_noise(images, noise_type: str, snr: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    fn = add_gaussian if noise_type == "gaussian" else add_speckle
    return [fn(img, snr, rng).astype(np.float64) for img in images]


def _to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def _conv_pair_mag(image: np.ndarray, kx: torch.Tensor, ky: torch.Tensor, device: str) -> np.ndarray:
    img = _to_tensor(image, device)
    pad_y = kx.shape[-2] // 2
    pad_x = kx.shape[-1] // 2
    gx = F.conv2d(img, kx, padding=(pad_y, pad_x))
    gy = F.conv2d(img, ky, padding=(pad_y, pad_x))
    return torch.sqrt(gx**2 + gy**2).squeeze().detach().cpu().numpy()


def _conv_bank_mag(image: np.ndarray, bank: torch.Tensor, device: str) -> np.ndarray:
    img = _to_tensor(image, device)
    pad = bank.shape[-1] // 2
    resp = F.conv2d(img, bank, padding=pad)
    return resp.abs().amax(dim=1).squeeze().detach().cpu().numpy()


def exact_disk_neighbors(radius: int) -> np.ndarray:
    coords = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius * radius:
                coords.append((dx, dy))
    return np.array(coords, dtype=np.float64)


def exact_square_neighbors(half_width: int) -> np.ndarray:
    coords = []
    for dy in range(-half_width, half_width + 1):
        for dx in range(-half_width, half_width + 1):
            if dx == 0 and dy == 0:
                continue
            coords.append((dx, dy))
    return np.array(coords, dtype=np.float64)


def dense_from_weights(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    min_x = int(coords[:, 0].min())
    max_x = int(coords[:, 0].max())
    min_y = int(coords[:, 1].min())
    max_y = int(coords[:, 1].max())
    kernel = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.float32)
    for (dx, dy), w in zip(coords.astype(int), weights):
        kernel[dy - min_y, dx - min_x] = float(w)
    return kernel


def build_polyfit_pair(kind: str, size_param: int, order: int, device: str):
    if kind == "circle":
        coords = exact_disk_neighbors(size_param)
    elif kind == "square":
        coords = exact_square_neighbors(size_param)
    else:
        raise ValueError(f"Unknown polyfit kind: {kind}")
    pinv, _ = compute_wvf_pseudoinverse(coords, order=order)
    kx = dense_from_weights(coords, pinv[1, :])
    ky = dense_from_weights(coords, pinv[2, :])
    kx_t = torch.from_numpy(kx).unsqueeze(0).unsqueeze(0).to(device)
    ky_t = torch.from_numpy(ky).unsqueeze(0).unsqueeze(0).to(device)
    return kx_t, ky_t


def oriented_grid(length: int):
    half = length // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    return x.astype(np.float32), y.astype(np.float32)


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = kernel - kernel.mean()
    scale = np.abs(kernel).sum()
    if scale > 0:
        kernel = kernel / scale
    return kernel.astype(np.float32)


def build_oriented_bank(
    method: str,
    n_orientations: int,
    half_along: float,
    half_across: float,
    sigma_along: float,
    sigma_across: float,
    device: str,
) -> torch.Tensor:
    radius = int(math.ceil(max(half_along, half_across, 3.0 * sigma_along, 3.0 * sigma_across)))
    length = 2 * radius + 1
    x, y = oriented_grid(length)
    thetas = np.linspace(0.0, np.pi, n_orientations, endpoint=False)
    kernels = []
    for theta in thetas:
        ct = math.cos(theta)
        st = math.sin(theta)
        u = x * ct + y * st
        v = -x * st + y * ct
        if method == "rectangular":
            mask = (np.abs(u) <= half_along) & (np.abs(v) <= half_across)
            gauss = np.exp(-0.5 * ((u / sigma_along) ** 2 + (v / sigma_across) ** 2)) * mask
        elif method == "elliptical":
            ellipse_arg = (u / sigma_along) ** 2 + (v / sigma_across) ** 2
            mask = ellipse_arg <= 9.0
            gauss = np.exp(-0.5 * ellipse_arg) * mask
        elif method == "aniso_gaussian":
            gauss = np.exp(-0.5 * ((u / sigma_along) ** 2 + (v / sigma_across) ** 2))
        else:
            raise ValueError(f"Unknown oriented-bank method: {method}")
        kernel = normalize_kernel(-v * gauss)
        kernels.append(torch.from_numpy(kernel))
    return torch.stack(kernels).unsqueeze(1).to(device)


def build_iso_gaussian_pair(sigma: float, device: str):
    ksize = int(6 * sigma + 1) | 1
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    dgauss = -ax / (sigma**2) * torch.exp(-0.5 * (ax / sigma) ** 2)
    dgauss = dgauss / (dgauss.abs().sum() / 2 + 1e-12)
    kx = (gauss.unsqueeze(1) * dgauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    ky = (dgauss.unsqueeze(1) * gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return kx, ky


def estimate_vram_budget(image_shape, half_width: int, np_count: int, total_vram_gb: float) -> float:
    h, w = image_shape
    n_pixels = (h - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2)) * (
        w - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2)
    )
    line_len = 2 * half_width + 1
    per_pixel = line_len * (np_count * 24 + 20)
    total_needed_gb = n_pixels * per_pixel / 1e9
    if total_needed_gb < total_vram_gb * 0.4:
        return total_vram_gb * 0.5
    return total_vram_gb * 0.35


def prepare_evaluator(cfg: MethodConfig, device: str):
    method = cfg.method
    params = dict(cfg.params)
    total_vram = None
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    if method == "wvf":
        def fn(image: np.ndarray) -> np.ndarray:
            return wvf_image(
                image,
                np_count=params["Np"],
                order=params["d"],
                n_orientations=params["Ns"],
                backend=device,
            ).gradient_mag
        return fn

    if method == "lf_ref":
        def fn(image: np.ndarray) -> np.ndarray:
            max_vram = None
            if total_vram is not None:
                max_vram = estimate_vram_budget(image.shape, params["m"], params["Np"], total_vram)
            return lf_image(
                image,
                half_width=params["m"],
                np_count=params["Np"],
                order=params["d"],
                n_orientations=params["Ns"],
                backend=device,
                max_vram_gb=max_vram,
            ).gradient_mag
        return fn

    if method == "lf_conv2d":
        def fn(image: np.ndarray) -> np.ndarray:
            mag, _, _ = aniso_lf_conv2d(
                image,
                half_width=params["m"],
                np_count=params["Np"],
                order=params["d"],
                n_orientations=params["Ns"],
                neighbor_type=params.get("neighbor_type", "circular"),
            )
            return mag
        return fn

    if method == "lf_triton":
        def fn(image: np.ndarray) -> np.ndarray:
            mag, _, _ = aniso_lf_triton(
                image,
                half_width=params["m"],
                np_count=params["Np"],
                order=params["d"],
                n_orientations=params["Ns"],
                neighbor_type=params.get("neighbor_type", "circular"),
            )
            return mag
        return fn

    if method in {"rectangular", "elliptical", "aniso_gaussian"}:
        bank = build_oriented_bank(
            method=method,
            n_orientations=params["Ns"],
            half_along=params["half_along"],
            half_across=params["half_across"],
            sigma_along=params["sigma_along"],
            sigma_across=params["sigma_across"],
            device=device,
        )

        def fn(image: np.ndarray) -> np.ndarray:
            return _conv_bank_mag(image, bank, device)
        return fn

    if method == "circle_sg":
        kx, ky = build_polyfit_pair("circle", params["radius"], params["d"], device)

        def fn(image: np.ndarray) -> np.ndarray:
            return _conv_pair_mag(image, kx, ky, device)
        return fn

    if method == "square_sg":
        kx, ky = build_polyfit_pair("square", params["half_width"], params["d"], device)

        def fn(image: np.ndarray) -> np.ndarray:
            return _conv_pair_mag(image, kx, ky, device)
        return fn

    if method == "iso_gaussian":
        kx, ky = build_iso_gaussian_pair(params["sigma"], device)

        def fn(image: np.ndarray) -> np.ndarray:
            return _conv_pair_mag(image, kx, ky, device)
        return fn

    if method == "iso_gaussian_ref":
        def fn(image: np.ndarray) -> np.ndarray:
            return gaussian_derivative(image, sigma=params["sigma"], device=device)
        return fn

    raise ValueError(f"Unknown method: {method}")


def max_abs_diff(mags_a, mags_b) -> float | None:
    diffs = []
    for a, b in zip(mags_a, mags_b):
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        diff = np.abs(a[:h, :w] - b[:h, :w])
        diffs.append(float(diff.max()))
    return max(diffs) if diffs else None


def evaluate_config(images, gts, cfg: MethodConfig, device: str, speed_runs: int):
    fn = prepare_evaluator(cfg, device)
    reset_gpu()
    _ = fn(images[0])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    mags_first = None
    vram_peak = 0.0
    for run_idx in range(speed_runs):
        reset_gpu()
        t0 = time.perf_counter()
        mags = [fn(img) for img in images]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        vram_peak = max(vram_peak, peak_vram_mb())
        if run_idx == 0:
            mags_first = mags

    ods, ois, _, _ = compute_ods_ois(
        mags_first,
        [gt.astype(np.float64) for gt in gts],
        n_thresholds=N_THRESH,
        match_radius=MATCH_RADIUS,
    )

    return {
        "label": cfg.label,
        "method": cfg.method,
        "params": cfg.params,
        "ods": float(ods),
        "ois": float(ois),
        "mean_total_s": float(np.mean(times)),
        "median_total_s": float(np.median(times)),
        "median_ms_per_image": float(np.median(times) * 1000.0 / len(images)),
        "peak_vram_mb": float(vram_peak),
        "mags": mags_first,
    }


def backend_bench_configs():
    return [
        MethodConfig("lf_ref_np100_ns18_d4_m1", "lf_ref", {"Np": 100, "Ns": 18, "d": 4, "m": 1}),
        MethodConfig("lf_conv2d_np100_ns18_d4_m1", "lf_conv2d", {"Np": 100, "Ns": 18, "d": 4, "m": 1}),
        MethodConfig("lf_triton_np100_ns18_d4_m1", "lf_triton", {"Np": 100, "Ns": 18, "d": 4, "m": 1}),
        MethodConfig("lf_ref_np100_ns18_d4_m7", "lf_ref", {"Np": 100, "Ns": 18, "d": 4, "m": 7}),
        MethodConfig("lf_conv2d_np100_ns18_d4_m7", "lf_conv2d", {"Np": 100, "Ns": 18, "d": 4, "m": 7}),
        MethodConfig("lf_triton_np100_ns18_d4_m7", "lf_triton", {"Np": 100, "Ns": 18, "d": 4, "m": 7}),
    ]


def clean_pilot_configs():
    return [
        MethodConfig("wvf_np50_ns6_d2", "wvf", {"Np": 50, "Ns": 6, "d": 2}),
        MethodConfig("wvf_np149_ns6_d2", "wvf", {"Np": 149, "Ns": 6, "d": 2}),
        MethodConfig("wvf_np149_ns18_d4", "wvf", {"Np": 149, "Ns": 18, "d": 4}),
        MethodConfig("lf_ref_np149_ns6_d2_m7", "lf_ref", {"Np": 149, "Ns": 6, "d": 2, "m": 7}),
        MethodConfig("lf_triton_np149_ns6_d2_m7", "lf_triton", {"Np": 149, "Ns": 6, "d": 2, "m": 7}),
        MethodConfig("lf_triton_np149_ns18_d4_m7", "lf_triton", {"Np": 149, "Ns": 18, "d": 4, "m": 7}),
        MethodConfig(
            "rect_ns6_a14_b7",
            "rectangular",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig(
            "ellipse_ns6_a14_b7",
            "elliptical",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig(
            "aniso_gaussian_ns6_a14_b7",
            "aniso_gaussian",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig("circle_sg_r7_d1", "circle_sg", {"radius": 7, "d": 1}),
        MethodConfig("circle_sg_r14_d1", "circle_sg", {"radius": 14, "d": 1}),
        MethodConfig("circle_sg_r14_d2", "circle_sg", {"radius": 14, "d": 2}),
        MethodConfig("square_sg_h7_d1", "square_sg", {"half_width": 7, "d": 1}),
        MethodConfig("square_sg_h14_d1", "square_sg", {"half_width": 14, "d": 1}),
        MethodConfig("iso_gaussian_sigma3p5", "iso_gaussian", {"sigma": 3.5}),
        MethodConfig("iso_gaussian_sigma7", "iso_gaussian", {"sigma": 7.0}),
    ]


def noise_pilot_configs():
    return [
        MethodConfig("wvf_np149_ns6_d2", "wvf", {"Np": 149, "Ns": 6, "d": 2}),
        MethodConfig("lf_triton_np149_ns6_d2_m7", "lf_triton", {"Np": 149, "Ns": 6, "d": 2, "m": 7}),
        MethodConfig(
            "rect_ns6_a14_b7",
            "rectangular",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig(
            "ellipse_ns6_a14_b7",
            "elliptical",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig(
            "aniso_gaussian_ns6_a14_b7",
            "aniso_gaussian",
            {"Ns": 6, "half_along": 14.0, "half_across": 7.0, "sigma_along": 14.0 / 3.0, "sigma_across": 7.0 / 3.0},
        ),
        MethodConfig("circle_sg_r14_d1", "circle_sg", {"radius": 14, "d": 1}),
        MethodConfig("circle_sg_r14_d2", "circle_sg", {"radius": 14, "d": 2}),
        MethodConfig("iso_gaussian_sigma7", "iso_gaussian", {"sigma": 7.0}),
    ]


def select_configs(mode: str):
    if mode == "backend_bench":
        return backend_bench_configs()
    if mode == "clean_pilot":
        return clean_pilot_configs()
    if mode == "noise_pilot":
        return noise_pilot_configs()
    raise ValueError(mode)


def summarize_backend(dataset_results):
    grouped = {}
    for result in dataset_results:
        suffix = None
        if "_m1" in result["label"]:
            suffix = "m1"
        elif "_m7" in result["label"]:
            suffix = "m7"
        if suffix is None:
            continue
        grouped.setdefault(suffix, {})[result["method"]] = result

    comparisons = []
    for suffix, family in grouped.items():
        ref = family.get("lf_ref")
        if ref is None:
            continue
        for candidate_name in ("lf_conv2d", "lf_triton"):
            cand = family.get(candidate_name)
            if cand is None:
                continue
            comparisons.append({
                "shape": suffix,
                "candidate": candidate_name,
                "speedup_vs_ref": ref["mean_total_s"] / cand["mean_total_s"],
                "max_abs_mag_diff": max_abs_diff(ref["mags"], cand["mags"]),
            })
    return comparisons


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    datasets = {}
    for dataset_name in args.datasets:
        try:
            images, gts, names = load_dataset(dataset_name, args.n_images)
        except FileNotFoundError:
            continue
        if not images:
            continue
        if args.mode == "noise_pilot":
            images = apply_noise(images, args.noise_type, args.snr, seed=0)
        datasets[dataset_name] = (images, gts, names)

    if not datasets:
        raise RuntimeError("No datasets loaded for the requested run")

    configs = select_configs(args.mode)
    output = {
        "mode": args.mode,
        "device": device,
        "datasets": {},
        "noise": None,
        "n_images": args.n_images,
        "speed_runs": args.speed_runs,
    }
    if args.mode == "noise_pilot":
        output["noise"] = {"type": args.noise_type, "snr": args.snr}

    for dataset_name, (images, gts, names) in datasets.items():
        print(f"\n=== {dataset_name} ({len(images)} images) ===")
        dataset_results = []
        for cfg in configs:
            print(f"  Running {cfg.label}...", flush=True)
            result = evaluate_config(images, gts, cfg, device, args.speed_runs)
            print(
                f"    ODS={result['ods']:.4f} "
                f"OIS={result['ois']:.4f} "
                f"median_ms_image={result['median_ms_per_image']:.2f} "
                f"VRAM={result['peak_vram_mb']:.0f}MB"
            )
            dataset_results.append(result)

        comparisons = None
        if args.mode == "backend_bench":
            comparisons = summarize_backend(dataset_results)
            for comp in comparisons:
                print(
                    f"    {comp['shape']} {comp['candidate']} "
                    f"speedup={comp['speedup_vs_ref']:.2f}x "
                    f"max_abs_diff={comp['max_abs_mag_diff']:.6f}"
                )

        output["datasets"][dataset_name] = {
            "image_names": names,
            "results": [{k: v for k, v in r.items() if k != "mags"} for r in dataset_results],
            "comparisons": comparisons,
        }

    out_path = Path(args.output) if args.output else OUT / f"{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
