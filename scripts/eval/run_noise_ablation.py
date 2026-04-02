"""Multi-noise-type ablation across 4 datasets, 5 images each.

Runs a comprehensive WVF/LF parameter grid at multiple SNR levels
with 5 noise types to determine how noise characteristics affect
optimal filter parameters.
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "noise_ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001
MATCH_RADIUS = 3

# ---- Grid ----
WVF_NP = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
WVF_NS = [3, 9, 18, 36, 72]
WVF_D = [2, 3, 4, 5]

LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

SNR_LEVELS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # inf handled as clean

wvf_configs = []
for d in WVF_D:
    min_c = (d + 1) * (d + 2) // 2
    for ns in WVF_NS:
        for np_val in WVF_NP:
            if np_val >= min_c:
                wvf_configs.append({"Np": np_val, "Ns": ns, "d": d})

lf_configs = []
for m in LF_M:
    for ns in LF_NS:
        for np_val in LF_NP:
            lf_configs.append({"Np": np_val, "Ns": ns, "m": m, "d": 4})

n_wvf = len(wvf_configs)
n_lf = len(lf_configs)
n_total = n_wvf + n_lf

# ---- Imports ----
import torch
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

HAS_CUDA = torch.cuda.is_available()
backend = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram:.0f} GB)")

print(f"WVF configs: {n_wvf}")
print(f"LF configs:  {n_lf}")
print(f"Total:       {n_total}")


# =========================================================================
# Noise functions
# =========================================================================
def add_gaussian(img, snr, rng):
    sigma = (img.max() - img.min()) / snr
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def add_salt_pepper(img, snr, rng):
    # Map SNR to density: lower SNR = more noise
    # density = 1 / (1 + snr) gives ~0.25 at SNR=3, ~0.5 at SNR=1, ~0.77 at SNR=0.3
    density = 1.0 / (1.0 + snr)
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < density / 2] = 0
    out[mask > 1 - density / 2] = 255
    return out


def add_poisson(img, snr, rng):
    # Scale controls noise level: lower scale = more noise
    scale = snr * snr  # quadratic scaling so SNR=1 gives moderate noise
    scaled = np.clip(img / 255.0 * scale, 0.001, None)
    noisy = rng.poisson(scaled).astype(np.float64) / scale * 255.0
    return np.clip(noisy, 0, 255)


def add_speckle(img, snr, rng):
    sigma = 1.0 / snr
    return np.clip(img * (1 + rng.normal(0, sigma, img.shape)), 0, 255)


def add_uniform(img, snr, rng):
    amplitude = (img.max() - img.min()) / snr
    return np.clip(img + rng.uniform(-amplitude, amplitude, img.shape), 0, 255)


NOISE_TYPES = {
    "gaussian": add_gaussian,
    "salt_pepper": add_salt_pepper,
    "poisson": add_poisson,
    "speckle": add_speckle,
    "uniform": add_uniform,
}


# =========================================================================
# Dataset loaders
# =========================================================================
def load_biped(version):
    if version == "v1":
        base = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
    else:
        base = ROOT / "datasets/BIPED/BIPEDv2/BIPEDv2/BIPED/edges"
    img_dir = base / "imgs/test/rgbr"
    gt_dir = base / "edge_maps/test/rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        images.append(np.mean(np.array(Image.open(f)), axis=2))
        gts.append(np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names


def load_bsds500():
    base1 = ROOT / "datasets/BSDS500/BSDS500/data"
    base2 = ROOT / "datasets/BSDS500/BSDS500"
    base = base1 if (base1 / "images/test").exists() else base2
    img_dir = base / "images/test"
    gt_dir = base / "groundTruth/test"
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
    return images, gts, names


def load_uded():
    img_dir = ROOT / "datasets/UDED/imgs"
    gt_dir = ROOT / "datasets/UDED/gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        images.append(np.mean(np.array(Image.open(f)), axis=2))
        gts.append(np.array(Image.open(gt_dir / f.name).convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names


# =========================================================================
# Select top 5 images per dataset
# =========================================================================
def select_top_images(images, gts, names, n=5):
    """Run one good WVF config on all images, pick top n by ODS."""
    scores = []
    for i, (img, gt) in enumerate(zip(images, gts)):
        mag = wvf_image(img, np_count=50, order=2, n_orientations=18,
                        backend=backend).gradient_mag
        ods, _, _, _ = compute_ods_ois(mag, gt.astype(np.float64),
                                        n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
        scores.append((float(ods), i))
    scores.sort(reverse=True)
    indices = [s[1] for s in scores[:n]]
    print(f"  Selected images: {[names[i] for i in indices]} (ODS: {[f'{scores[j][0]:.3f}' for j in range(n)]})")
    return [images[i] for i in indices], [gts[i] for i in indices], [names[i] for i in indices]


# =========================================================================
# Run ablation on one image at one SNR with one noise type
# =========================================================================
def estimate_vram(H, W, half_width, np_count):
    L = 2 * half_width + 1
    per_pixel = L * (np_count * 24 + 20)
    total_gb = H * W * per_pixel / 1e9
    if total_gb < total_vram * 0.4:
        return total_vram * 0.5
    else:
        return total_vram * 0.35


def run_single_ablation(img, gt_bool):
    """Run all WVF + LF configs on one image, return results list."""
    results = []

    for cfg in wvf_configs:
        try:
            mag = wvf_image(img, np_count=cfg["Np"], order=cfg["d"],
                            n_orientations=cfg["Ns"], backend=backend).gradient_mag
            ods, _, _, _ = compute_ods_ois(mag, gt_bool.astype(np.float64),
                                            n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
            results.append({"filter": "WVF", "Np": cfg["Np"], "Ns": cfg["Ns"],
                            "d": cfg["d"], "ods": float(ods)})
        except Exception:
            pass

    for cfg in lf_configs:
        if HAS_CUDA:
            torch.cuda.empty_cache()
        try:
            H, W = img.shape
            vram = estimate_vram(H, W, cfg["m"], cfg["Np"])
            mag = lf_image(img, half_width=cfg["m"], np_count=cfg["Np"],
                           order=4, n_orientations=cfg["Ns"], backend=backend,
                           max_vram_gb=vram).gradient_mag
            ods, _, _, _ = compute_ods_ois(mag, gt_bool.astype(np.float64),
                                            n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
            results.append({"filter": "LF", "Np": cfg["Np"], "Ns": cfg["Ns"],
                            "m": cfg["m"], "d": 4, "ods": float(ods)})
        except Exception:
            if HAS_CUDA:
                torch.cuda.empty_cache()

    return results


# =========================================================================
# Main
# =========================================================================
ALL_DATASETS = [
    ("UDED", load_uded),
    ("BIPED_v1", lambda: load_biped("v1")),
    ("BIPED_v2", lambda: load_biped("v2")),
    ("BSDS500", load_bsds500),
]

grand_t0 = time.perf_counter()

for ds_name, loader in ALL_DATASETS:
    ds_out = OUT / ds_name.lower()
    ds_out.mkdir(exist_ok=True)

    # Check if this dataset is already done
    done_file = ds_out / "DONE"
    if done_file.exists():
        print(f"\n=== {ds_name}: already done, skipping ===")
        continue

    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*70}")

    try:
        all_images, all_gts, all_names = loader()
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        continue

    print(f"  Loaded {len(all_images)} images")

    # Select top 5
    print("  Selecting top 5 images...")
    images, gts, names = select_top_images(all_images, all_gts, all_names, n=5)
    del all_images, all_gts, all_names
    gc.collect()

    # Save selection
    with open(ds_out / "selected_images.json", "w") as f:
        json.dump({"images": names}, f)

    # Run each image × SNR × noise type
    for img_idx, (img_clean, gt, img_name) in enumerate(zip(images, gts, names)):
        img_file = ds_out / f"{img_name}_results.json"
        if img_file.exists():
            print(f"\n  [{img_idx+1}/5] {img_name}: already done, skipping")
            continue

        print(f"\n  [{img_idx+1}/5] Image: {img_name} ({img_clean.shape})")
        img_results = {"image": img_name, "shape": list(img_clean.shape),
                       "n_gt": int(np.sum(gt)), "conditions": []}

        conditions = []
        # Clean (inf)
        conditions.append(("clean", "inf", img_clean))
        # Noisy
        for snr in SNR_LEVELS:
            for noise_name, noise_fn in NOISE_TYPES.items():
                rng = np.random.default_rng(seed=42)
                noisy = noise_fn(img_clean, snr, rng)
                conditions.append((noise_name, snr, noisy))

        for ci, (noise_name, snr, img) in enumerate(conditions):
            t0 = time.perf_counter()
            results = run_single_ablation(img, gt)
            elapsed = time.perf_counter() - t0

            best_wvf = max((r for r in results if r["filter"] == "WVF"),
                           key=lambda x: x["ods"], default=None)
            best_lf = max((r for r in results if r["filter"] == "LF"),
                          key=lambda x: x["ods"], default=None)

            snr_str = str(snr) if snr != "inf" else "inf"
            label = f"{noise_name}/SNR={snr_str}"
            bw = f"ODS={best_wvf['ods']:.3f} Np={best_wvf['Np']}" if best_wvf else "---"
            bl = f"ODS={best_lf['ods']:.3f} m={best_lf['m']}" if best_lf else "---"
            print(f"    [{ci+1}/{len(conditions)}] {label:25s} WVF:{bw}  LF:{bl}  ({elapsed:.0f}s)")

            img_results["conditions"].append({
                "noise_type": noise_name,
                "snr": snr_str,
                "n_configs": len(results),
                "time_s": round(elapsed, 1),
                "results": results,
            })

        with open(img_file, "w") as f:
            json.dump(img_results, f, indent=2)
        print(f"    Saved {img_file.name}")

    # Mark dataset done
    done_file.write_text("done")
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()

grand_time = time.perf_counter() - grand_t0

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'#'*70}")
print(f"NOISE ABLATION COMPLETE in {grand_time:.0f}s ({grand_time/3600:.1f}h)")
print(f"{'#'*70}")

# Print summary table
for ds_name, _ in ALL_DATASETS:
    ds_out = OUT / ds_name.lower()
    if not ds_out.exists():
        continue
    print(f"\n--- {ds_name} ---")
    sel_file = ds_out / "selected_images.json"
    if not sel_file.exists():
        continue
    with open(sel_file) as f:
        selected = json.load(f)

    for img_name in selected["images"]:
        res_file = ds_out / f"{img_name}_results.json"
        if not res_file.exists():
            continue
        with open(res_file) as f:
            data = json.load(f)

        print(f"  {img_name}:")
        # Show clean vs noisiest for each noise type
        for cond in data["conditions"]:
            if cond["snr"] == "inf" or cond["snr"] == "0.3":
                bw = max((r for r in cond["results"] if r["filter"] == "WVF"),
                         key=lambda x: x["ods"], default=None)
                bl = max((r for r in cond["results"] if r["filter"] == "LF"),
                         key=lambda x: x["ods"], default=None)
                label = f"{cond['noise_type']}/SNR={cond['snr']}"
                print(f"    {label:25s} WVF={bw['ods']:.3f}(Np={bw['Np']}) LF={bl['ods']:.3f}(m={bl['m']})" if bw and bl else "")
