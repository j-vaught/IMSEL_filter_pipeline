"""Back-to-back evaluation: Original LF vs Anisotropic LF.

Runs both implementations on the SAME GPU in the SAME process, using
a small sample of images from each dataset. Measures:
  - Wall-clock runtime (warmup + timed runs)
  - Peak VRAM usage
  - ODS/OIS scores
  - Numerical agreement between implementations

Usage:
    python3 -u scripts/eval_aniso_lf.py [--n-images 3] [--n-runs 5]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Parse args early (before slow imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n-images", type=int, default=3,
                    help="Images per dataset to evaluate")
parser.add_argument("--n-runs", type=int, default=5,
                    help="Timed runs per method (after 1 warmup)")
parser.add_argument("--output", type=str, default=None,
                    help="Output JSON path")
args = parser.parse_args()

print(f"=== Anisotropic LF Evaluation ===")
print(f"Images per dataset: {args.n_images}")
print(f"Timed runs: {args.n_runs}")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import scipy.io as sio

from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois
from edgecritic.aniso_lf import aniso_lf_conv2d

# Try importing triton-based variant
try:
    from edgecritic.aniso_lf import aniso_lf_triton
    HAS_TRITON = True
    print("Triton backend: available")
except (ImportError, RuntimeError):
    HAS_TRITON = False
    print("Triton backend: NOT available (will skip)")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "aniso_lf_eval"
OUT.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA required for this evaluation"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({total_vram:.1f} GB)")
print()

# ---------------------------------------------------------------------------
# Dataset loaders (same as run_full_dataset_ablation.py)
# ---------------------------------------------------------------------------
def load_biped_v1(n=None):
    base = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
    img_dir = base / "imgs" / "test" / "rgbr"
    gt_dir = base / "edge_maps" / "test" / "rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
        if n and len(images) >= n:
            break
    return images, gts, names


def load_bsds500(n=None):
    base1 = ROOT / "datasets" / "BSDS500" / "BSDS500" / "data"
    base2 = ROOT / "datasets" / "BSDS500" / "BSDS500"
    base = base1 if (base1 / "images" / "test").exists() else base2
    img_dir = base / "images" / "test"
    gt_dir = base / "groundTruth" / "test"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt_mat = sio.loadmat(str(gt_dir / f"{f.stem}.mat"))
        gt_cell = gt_mat["groundTruth"]
        gt_union = np.zeros(img.shape[:2], dtype=bool)
        for i in range(gt_cell.shape[1]):
            bdry = gt_cell[0, i]["Boundaries"][0, 0]
            bdry = bdry.toarray() if hasattr(bdry, "toarray") else np.asarray(bdry)
            gt_union |= (bdry > 0)
        images.append(img)
        gts.append(gt_union)
        names.append(f.stem)
        if n and len(images) >= n:
            break
    return images, gts, names


def load_uded(n=None):
    img_dir = ROOT / "datasets" / "UDED" / "imgs"
    gt_dir = ROOT / "datasets" / "UDED" / "gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f.name).convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
        if n and len(images) >= n:
            break
    return images, gts, names


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------
def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_vram_mb():
    return torch.cuda.max_memory_allocated() / 1e6


# ---------------------------------------------------------------------------
# LF configs to test (covering the interesting parameter space)
# ---------------------------------------------------------------------------
LF_CONFIGS = [
    # Best LF configs from ablation (m=1 and m=2 dominated)
    {"Np": 100, "Ns": 18, "d": 4, "m": 1, "label": "LF-best-biped"},
    {"Np": 250, "Ns": 18, "d": 4, "m": 2, "label": "LF-best-bsds"},
    # Higher m to stress-test the anisotropic advantage
    {"Np": 100, "Ns": 18, "d": 4, "m": 7, "label": "LF-m7"},
    {"Np": 100, "Ns": 36, "d": 4, "m": 14, "label": "LF-m14"},
    # Square neighborhood variant
    {"Np": 100, "Ns": 18, "d": 4, "m": 7, "label": "LF-m7-square",
     "neighbor_type": "square"},
]

# Also test FP16 for the conv2d path
FP16_CONFIGS = [
    {"Np": 100, "Ns": 18, "d": 4, "m": 7, "label": "LF-m7-fp16"},
]


def estimate_vram_budget(H, W, half_width, np_count, total_vram_gb):
    n_pixels = (H - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2)) * \
               (W - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2))
    L = 2 * half_width + 1
    per_pixel = L * (np_count * 24 + 20)
    total_needed_gb = n_pixels * per_pixel / 1e9
    if total_needed_gb < total_vram_gb * 0.4:
        return total_vram_gb * 0.5
    else:
        return total_vram_gb * 0.35


# ---------------------------------------------------------------------------
# Run a single method and collect timing + ODS
# ---------------------------------------------------------------------------
def run_original_lf(images, gts, cfg, n_runs):
    """Run original LF implementation."""
    m = cfg["m"]
    Np = cfg["Np"]
    Ns = cfg["Ns"]
    d = cfg["d"]

    # Warmup
    reset_gpu()
    H, W = images[0].shape
    vram_budget = estimate_vram_budget(H, W, m, Np, total_vram)
    _ = lf_image(images[0], half_width=m, np_count=Np, order=d,
                 n_orientations=Ns, backend="cuda", max_vram_gb=vram_budget)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    all_mags = []
    for run in range(n_runs):
        reset_gpu()
        t0 = time.perf_counter()
        mags = []
        for img in images:
            H, W = img.shape
            vram_budget = estimate_vram_budget(H, W, m, Np, total_vram)
            result = lf_image(img, half_width=m, np_count=Np, order=d,
                              n_orientations=Ns, backend="cuda",
                              max_vram_gb=vram_budget)
            mags.append(result.gradient_mag)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            all_mags = mags

    vram = peak_vram_mb()

    # Compute ODS
    ods_scores = []
    for mag, gt in zip(all_mags, gts):
        ods, _, _, _ = compute_ods_ois(mag, gt)
        ods_scores.append(ods)

    return {
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "peak_vram_mb": vram,
        "ods_per_image": ods_scores,
        "mean_ods": float(np.mean(ods_scores)),
        "mags": all_mags,  # for numerical comparison
    }


def run_aniso_conv2d(images, gts, cfg, n_runs, dtype=torch.float32):
    """Run anisotropic LF via conv2d."""
    m = cfg["m"]
    Np = cfg["Np"]
    Ns = cfg["Ns"]
    d = cfg["d"]
    neighbor_type = cfg.get("neighbor_type", "circular")

    # Warmup
    reset_gpu()
    _ = aniso_lf_conv2d(images[0], half_width=m, np_count=Np, order=d,
                        n_orientations=Ns, neighbor_type=neighbor_type,
                        dtype=dtype)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    all_mags = []
    stencil_stats = None
    for run in range(n_runs):
        reset_gpu()
        t0 = time.perf_counter()
        mags = []
        for img in images:
            mag, angle, timing = aniso_lf_conv2d(
                img, half_width=m, np_count=Np, order=d,
                n_orientations=Ns, neighbor_type=neighbor_type,
                dtype=dtype,
            )
            mags.append(mag)
            if stencil_stats is None:
                stencil_stats = timing.get("stencil_stats")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            all_mags = mags

    vram = peak_vram_mb()

    ods_scores = []
    for mag, gt in zip(all_mags, gts):
        ods, _, _, _ = compute_ods_ois(mag, gt)
        ods_scores.append(ods)

    return {
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "peak_vram_mb": vram,
        "ods_per_image": ods_scores,
        "mean_ods": float(np.mean(ods_scores)),
        "stencil_stats": stencil_stats,
        "mags": all_mags,
    }


def run_aniso_triton(images, gts, cfg, n_runs):
    """Run anisotropic LF via Triton kernel."""
    m = cfg["m"]
    Np = cfg["Np"]
    Ns = cfg["Ns"]
    d = cfg["d"]
    neighbor_type = cfg.get("neighbor_type", "circular")

    # Warmup (includes Triton JIT compilation)
    reset_gpu()
    _ = aniso_lf_triton(images[0], half_width=m, np_count=Np, order=d,
                        n_orientations=Ns, neighbor_type=neighbor_type)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    all_mags = []
    for run in range(n_runs):
        reset_gpu()
        t0 = time.perf_counter()
        mags = []
        for img in images:
            mag, angle, timing = aniso_lf_triton(
                img, half_width=m, np_count=Np, order=d,
                n_orientations=Ns, neighbor_type=neighbor_type,
            )
            mags.append(mag)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if run == 0:
            all_mags = mags

    vram = peak_vram_mb()

    ods_scores = []
    for mag, gt in zip(all_mags, gts):
        ods, _, _, _ = compute_ods_ois(mag, gt)
        ods_scores.append(ods)

    return {
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "peak_vram_mb": vram,
        "ods_per_image": ods_scores,
        "mean_ods": float(np.mean(ods_scores)),
        "mags": all_mags,
    }


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------
def compare_mags(mags_a, mags_b, label_a, label_b):
    """Compare gradient magnitude maps between two methods."""
    diffs = []
    for ma, mb in zip(mags_a, mags_b):
        # Both may have different border sizes, compare the overlap
        h = min(ma.shape[0], mb.shape[0])
        w = min(ma.shape[1], mb.shape[1])
        # Use interior where both should be nonzero
        a = ma[:h, :w]
        b = mb[:h, :w]
        mask = (a > 0) | (b > 0)
        if mask.sum() > 0:
            diff = np.abs(a[mask] - b[mask])
            diffs.append({
                "max_abs_diff": float(np.max(diff)),
                "mean_abs_diff": float(np.mean(diff)),
                "median_abs_diff": float(np.median(diff)),
                "n_pixels": int(mask.sum()),
            })
    return {
        f"{label_a}_vs_{label_b}": diffs,
        "max_across_images": float(max(d["max_abs_diff"] for d in diffs)) if diffs else None,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    n_img = args.n_images
    n_runs = args.n_runs

    # Load datasets
    print("Loading datasets...")
    datasets = {}
    for name, loader in [("BSDS500", load_bsds500), ("BIPED_v1", load_biped_v1),
                         ("UDED", load_uded)]:
        try:
            imgs, gts, names = loader(n=n_img)
            datasets[name] = (imgs, gts, names)
            print(f"  {name}: {len(imgs)} images, "
                  f"sizes: {[img.shape for img in imgs]}")
        except Exception as e:
            print(f"  {name}: FAILED to load ({e})")

    if not datasets:
        print("ERROR: No datasets loaded!")
        return

    # Results container
    all_results = {
        "gpu": gpu_name,
        "total_vram_gb": total_vram,
        "n_images_per_dataset": n_img,
        "n_runs": n_runs,
        "configs": {},
    }

    for cfg in LF_CONFIGS:
        label = cfg["label"]
        print(f"\n{'='*70}")
        print(f"Config: {label}")
        print(f"  Np={cfg['Np']} Ns={cfg['Ns']} d={cfg['d']} m={cfg['m']} "
              f"neighbor={cfg.get('neighbor_type', 'circular')}")
        print(f"{'='*70}")

        cfg_results = {"params": {k: v for k, v in cfg.items() if k != "label"}}

        for ds_name, (imgs, gts, names) in datasets.items():
            print(f"\n  Dataset: {ds_name} ({len(imgs)} images)")

            ds_result = {}

            # --- Original LF ---
            print(f"    Running original LF...", end=" ", flush=True)
            try:
                orig = run_original_lf(imgs, gts, cfg, n_runs)
                print(f"mean={orig['mean_time']:.3f}s "
                      f"ODS={orig['mean_ods']:.4f} "
                      f"VRAM={orig['peak_vram_mb']:.0f}MB")
                ds_result["original_lf"] = {
                    k: v for k, v in orig.items() if k != "mags"
                }
            except Exception as e:
                print(f"FAILED: {e}")
                orig = None
                ds_result["original_lf"] = {"error": str(e)}

            # --- Aniso conv2d ---
            print(f"    Running aniso conv2d...", end=" ", flush=True)
            try:
                conv = run_aniso_conv2d(imgs, gts, cfg, n_runs)
                print(f"mean={conv['mean_time']:.3f}s "
                      f"ODS={conv['mean_ods']:.4f} "
                      f"VRAM={conv['peak_vram_mb']:.0f}MB")
                ds_result["aniso_conv2d"] = {
                    k: v for k, v in conv.items() if k != "mags"
                }
            except Exception as e:
                print(f"FAILED: {e}")
                conv = None
                ds_result["aniso_conv2d"] = {"error": str(e)}

            # --- Aniso Triton ---
            if HAS_TRITON:
                print(f"    Running aniso triton...", end=" ", flush=True)
                try:
                    tri = run_aniso_triton(imgs, gts, cfg, n_runs)
                    print(f"mean={tri['mean_time']:.3f}s "
                          f"ODS={tri['mean_ods']:.4f} "
                          f"VRAM={tri['peak_vram_mb']:.0f}MB")
                    ds_result["aniso_triton"] = {
                        k: v for k, v in tri.items() if k != "mags"
                    }
                except Exception as e:
                    print(f"FAILED: {e}")
                    tri = None
                    ds_result["aniso_triton"] = {"error": str(e)}
            else:
                tri = None

            # --- Numerical comparison ---
            if orig and conv:
                comp = compare_mags(orig["mags"], conv["mags"],
                                    "original", "conv2d")
                ds_result["numerical_comparison_conv2d"] = comp
                print(f"    Max |diff| original vs conv2d: "
                      f"{comp['max_across_images']:.6f}")

            if orig and tri:
                comp = compare_mags(orig["mags"], tri["mags"],
                                    "original", "triton")
                ds_result["numerical_comparison_triton"] = comp
                print(f"    Max |diff| original vs triton: "
                      f"{comp['max_across_images']:.6f}")

            # --- Speedup ---
            if orig and conv and "mean_time" in orig and "mean_time" in conv:
                speedup = orig["mean_time"] / conv["mean_time"]
                ds_result["speedup_conv2d"] = speedup
                print(f"    Speedup conv2d: {speedup:.2f}x")
            if orig and tri and "mean_time" in orig and "mean_time" in tri:
                speedup = orig["mean_time"] / tri["mean_time"]
                ds_result["speedup_triton"] = speedup
                print(f"    Speedup triton: {speedup:.2f}x")

            cfg_results[ds_name] = ds_result

        all_results["configs"][label] = cfg_results

    # --- FP16 test (conv2d only) ---
    print(f"\n{'='*70}")
    print(f"FP16 Experiment")
    print(f"{'='*70}")
    for cfg in FP16_CONFIGS:
        label = cfg["label"]
        cfg_results = {"params": {k: v for k, v in cfg.items() if k != "label"}}

        for ds_name, (imgs, gts, names) in datasets.items():
            print(f"\n  {label} on {ds_name}...", end=" ", flush=True)
            try:
                fp16_result = run_aniso_conv2d(imgs, gts, cfg, n_runs,
                                               dtype=torch.float16)
                print(f"mean={fp16_result['mean_time']:.3f}s "
                      f"ODS={fp16_result['mean_ods']:.4f} "
                      f"VRAM={fp16_result['peak_vram_mb']:.0f}MB")
                cfg_results[ds_name] = {
                    "aniso_conv2d_fp16": {
                        k: v for k, v in fp16_result.items() if k != "mags"
                    }
                }
                # Compare to FP32 conv2d if we have it
                fp32_key = "LF-m7"
                if fp32_key in all_results["configs"]:
                    fp32_time = all_results["configs"][fp32_key].get(
                        ds_name, {}).get("aniso_conv2d", {}).get("mean_time")
                    if fp32_time:
                        print(f"    FP16 vs FP32 speedup: "
                              f"{fp32_time / fp16_result['mean_time']:.2f}x")
            except Exception as e:
                print(f"FAILED: {e}")
                cfg_results[ds_name] = {"error": str(e)}

        all_results["configs"][label] = cfg_results

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for cfg_label, cfg_data in all_results["configs"].items():
        print(f"\n{cfg_label}:")
        for ds_name in datasets:
            ds = cfg_data.get(ds_name, {})
            orig_t = ds.get("original_lf", {}).get("mean_time", "N/A")
            conv_t = ds.get("aniso_conv2d", {}).get("mean_time", "N/A")
            tri_t = ds.get("aniso_triton", {}).get("mean_time", "N/A")
            speedup = ds.get("speedup_conv2d", "N/A")
            print(f"  {ds_name}: orig={orig_t}s conv2d={conv_t}s "
                  f"triton={tri_t}s speedup={speedup}")

    # Save results
    out_path = args.output or str(OUT / "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
