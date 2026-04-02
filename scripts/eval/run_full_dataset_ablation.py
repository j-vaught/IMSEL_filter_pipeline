"""Full dataset-wide ODS ablation across BIPED v1, BIPED v2, BSDS500, and UDED.

Runs the complete WVF/LF parameter grid on every test image in each dataset,
computing dataset-wide ODS (single best threshold across all images).
No NMS, no Canny, no resizing. Matches Bagan's eval protocol (1001 thresholds, 3px match).
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
from PIL import Image

N_THRESH = 1001
MATCH_RADIUS = 3

# Full grids — same as single-image ablation
WVF_NP = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
WVF_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24, 36, 48, 72, 90, 120, 180]
WVF_D = [2, 3, 4, 5]

LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36, 72]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

# Build config lists
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

print(f"WVF configs: {len(wvf_configs)}")
print(f"LF configs:  {len(lf_configs)}")
print(f"Total:       {len(wvf_configs) + len(lf_configs)}")

# ---------------------------------------------------------------------------
# Imports (after config so we can print counts before slow imports)
# ---------------------------------------------------------------------------
import torch
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "full_dataset_ablation"
OUT.mkdir(parents=True, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
backend = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({total_vram:.0f} GB)")


def estimate_vram_budget(H, W, half_width, np_count, total_vram_gb):
    """Calculate safe VRAM budget based on image size and LF params.

    Returns max_vram_gb that should avoid OOM for lf_image_cuda.
    """
    n_pixels = (H - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2)) * \
               (W - 2 * (int(np.ceil(np.sqrt(np_count / np.pi))) + half_width + 2))
    L = 2 * half_width + 1
    # Peak memory per pixel: L * (Np * 24 + 20) bytes
    per_pixel = L * (np_count * 24 + 20)
    total_needed_gb = n_pixels * per_pixel / 1e9
    # If total needed fits in 50% of VRAM, no batching needed
    # Otherwise budget is 40% of VRAM (conservative for fragmentation)
    if total_needed_gb < total_vram_gb * 0.4:
        return total_vram_gb * 0.5
    else:
        return total_vram_gb * 0.35


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_biped_v1():
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
    return images, gts, names


def load_biped_v2():
    base = ROOT / "datasets" / "BIPED" / "BIPEDv2" / "BIPEDv2" / "BIPED" / "edges"
    img_dir = base / "imgs" / "test" / "rgbr"
    gt_dir = base / "edge_maps" / "test" / "rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
    return images, gts, names


def load_bsds500():
    # Try both common path layouts
    base1 = ROOT / "datasets" / "BSDS500" / "BSDS500" / "data"
    base2 = ROOT / "datasets" / "BSDS500" / "BSDS500"
    if (base1 / "images" / "test").exists():
        base = base1
    else:
        base = base2
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
    return images, gts, names


def load_uded():
    img_dir = ROOT / "datasets" / "UDED" / "imgs"
    gt_dir = ROOT / "datasets" / "UDED" / "gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        img = np.mean(np.array(Image.open(f)), axis=2)
        gt = np.array(Image.open(gt_dir / f.name).convert("L")) > 128
        images.append(img)
        gts.append(gt)
        names.append(f.stem)
    return images, gts, names


# ---------------------------------------------------------------------------
# Run ablation on one dataset
# ---------------------------------------------------------------------------
def run_dataset_ablation(dataset_name, images, gts, names):
    """Run full WVF+LF ablation, return dataset-wide ODS for each config."""
    n_images = len(images)
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"  Images: {n_images}, max size: {max_h}x{max_w}")
    print(f"  GT pixels (avg): {np.mean([np.sum(gt) for gt in gts]):.0f}")
    print(f"{'='*70}")

    results = {"wvf": [], "lf": []}

    # --- WVF ---
    print(f"\n  WVF: {len(wvf_configs)} configs × {n_images} images = {len(wvf_configs)*n_images} runs")
    t0 = time.perf_counter()
    for ci, cfg in enumerate(wvf_configs):
        mags = []
        for img in images:
            mag = wvf_image(img, np_count=cfg["Np"], order=cfg["d"],
                            n_orientations=cfg["Ns"], backend=backend).gradient_mag
            mags.append(mag)

        # Dataset-wide ODS
        ods, ois, _, _ = compute_ods_ois(
            mags, [gt.astype(np.float64) for gt in gts],
            n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)

        results["wvf"].append({
            "filter": "WVF", "Np": cfg["Np"], "Ns": cfg["Ns"], "d": cfg["d"],
            "ods": float(ods), "ois": float(ois),
        })

        if (ci + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (ci + 1) * (len(wvf_configs) - ci - 1)
            print(f"    WVF {ci+1}/{len(wvf_configs)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    wvf_time = time.perf_counter() - t0
    print(f"  WVF done: {len(results['wvf'])} configs in {wvf_time:.0f}s")

    # --- LF ---
    print(f"\n  LF: {len(lf_configs)} configs × {n_images} images = {len(lf_configs)*n_images} runs")
    t0 = time.perf_counter()
    for ci, cfg in enumerate(lf_configs):
        mags = []
        skip = False
        for img in images:
            H, W = img.shape
            vram_gb = estimate_vram_budget(H, W, cfg["m"], cfg["Np"], total_vram) if HAS_CUDA else None
            if HAS_CUDA:
                torch.cuda.empty_cache()
            try:
                mag = lf_image(img, half_width=cfg["m"], np_count=cfg["Np"],
                               order=4, n_orientations=cfg["Ns"],
                               backend=backend, max_vram_gb=vram_gb).gradient_mag
                mags.append(mag)
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM: LF Np={cfg['Np']} m={cfg['m']} Ns={cfg['Ns']} on {H}x{W}")
                torch.cuda.empty_cache()
                skip = True
                break

        if skip:
            results["lf"].append({
                "filter": "LF", "Np": cfg["Np"], "Ns": cfg["Ns"],
                "m": cfg["m"], "d": 4,
                "ods": -1.0, "ois": -1.0, "oom": True,
            })
            continue

        ods, ois, _, _ = compute_ods_ois(
            mags, [gt.astype(np.float64) for gt in gts],
            n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)

        results["lf"].append({
            "filter": "LF", "Np": cfg["Np"], "Ns": cfg["Ns"],
            "m": cfg["m"], "d": 4,
            "ods": float(ods), "ois": float(ois),
        })

        if (ci + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (ci + 1) * (len(lf_configs) - ci - 1)
            print(f"    LF {ci+1}/{len(lf_configs)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    lf_time = time.perf_counter() - t0
    print(f"  LF done: {len(results['lf'])} configs in {lf_time:.0f}s")

    # Summary
    valid_wvf = [r for r in results["wvf"] if r["ods"] >= 0]
    valid_lf = [r for r in results["lf"] if r["ods"] >= 0]
    oom_lf = [r for r in results["lf"] if r.get("oom")]

    best_wvf = max(valid_wvf, key=lambda x: x["ods"]) if valid_wvf else None
    best_lf = max(valid_lf, key=lambda x: x["ods"]) if valid_lf else None
    paper_wvf = next((r for r in valid_wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
    paper_lf = next((r for r in valid_lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

    print(f"\n  --- {dataset_name} Summary ---")
    if best_wvf:
        print(f"  Best WVF: ODS={best_wvf['ods']:.4f} OIS={best_wvf['ois']:.4f} Np={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}")
    if best_lf:
        print(f"  Best LF:  ODS={best_lf['ods']:.4f} OIS={best_lf['ois']:.4f} Np={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}")
    if paper_wvf:
        print(f"  Bagan WVF: ODS={paper_wvf['ods']:.4f} OIS={paper_wvf['ois']:.4f}")
    if paper_lf:
        print(f"  Bagan LF:  ODS={paper_lf['ods']:.4f} OIS={paper_lf['ois']:.4f}")
    if oom_lf:
        print(f"  OOM configs: {len(oom_lf)}")

    # Save per-dataset results
    output = {
        "dataset": dataset_name,
        "n_images": n_images,
        "image_names": names,
        "max_resolution": f"{max_h}x{max_w}",
        "evaluation": {
            "match_radius": MATCH_RADIUS,
            "n_thresholds": N_THRESH,
            "method": "distance_transform + maximum_filter + searchsorted",
            "post_processing": "none (raw magnitude)",
            "protocol": "dataset-wide ODS (single threshold across all images)",
        },
        "wvf_time_s": round(wvf_time, 1),
        "lf_time_s": round(lf_time, 1),
        "wvf": results["wvf"],
        "lf": results["lf"],
    }

    out_path = OUT / f"{dataset_name.lower().replace(' ', '_')}_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")

    # Free memory
    del mags
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\n{'#'*70}")
print(f"FULL DATASET ABLATION")
print(f"  WVF: {len(wvf_configs)} configs")
print(f"  LF:  {len(lf_configs)} configs")
print(f"  Eval: {N_THRESH} thresholds, {MATCH_RADIUS}px match radius")
print(f"  Post-processing: none")
print(f"{'#'*70}")

datasets = [
    ("UDED", load_uded),           # smallest first (~30 images, 339x510)
    ("BIPED_v1", load_biped_v1),    # 50 images, 720x1280
    ("BIPED_v2", load_biped_v2),    # 50 images, 720x1280
    ("BSDS500", load_bsds500),      # 200 images, 481x321
]

all_results = {}
total_t0 = time.perf_counter()

for name, loader in datasets:
    try:
        print(f"\nLoading {name}...")
        images, gts, names = loader()
        result = run_dataset_ablation(name, images, gts, names)
        all_results[name] = result
    except FileNotFoundError as e:
        print(f"  SKIPPED {name}: {e}")
    except Exception as e:
        print(f"  ERROR on {name}: {e}")
        import traceback
        traceback.print_exc()

total_time = time.perf_counter() - total_t0

# Save combined summary
summary = {
    "total_time_s": round(total_time, 1),
    "datasets": list(all_results.keys()),
    "wvf_configs": len(wvf_configs),
    "lf_configs": len(lf_configs),
    "evaluation": {
        "match_radius": MATCH_RADIUS,
        "n_thresholds": N_THRESH,
        "post_processing": "none",
    },
}

# Add best results per dataset
for name, result in all_results.items():
    valid_wvf = [r for r in result["wvf"] if r["ods"] >= 0]
    valid_lf = [r for r in result["lf"] if r["ods"] >= 0]
    best_wvf = max(valid_wvf, key=lambda x: x["ods"]) if valid_wvf else None
    best_lf = max(valid_lf, key=lambda x: x["ods"]) if valid_lf else None
    paper_wvf = next((r for r in valid_wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
    paper_lf = next((r for r in valid_lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)
    summary[name] = {
        "n_images": result["n_images"],
        "best_wvf": best_wvf,
        "best_lf": best_lf,
        "bagan_wvf": paper_wvf,
        "bagan_lf": paper_lf,
    }

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'#'*70}")
print(f"ALL DONE in {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"Results in {OUT}")
print(f"{'#'*70}")

# Final comparison table
print(f"\n{'Dataset':<12} {'Best WVF ODS':>13} {'Best LF ODS':>12} {'Bagan WVF':>10} {'Bagan LF':>10}")
print("-" * 60)
for name in all_results:
    s = summary[name]
    bw = f"{s['best_wvf']['ods']:.4f}" if s['best_wvf'] else "---"
    bl = f"{s['best_lf']['ods']:.4f}" if s['best_lf'] else "---"
    pw = f"{s['bagan_wvf']['ods']:.4f}" if s['bagan_wvf'] else "---"
    pl = f"{s['bagan_lf']['ods']:.4f}" if s['bagan_lf'] else "---"
    print(f"{name:<12} {bw:>13} {bl:>12} {pw:>10} {pl:>10}")
