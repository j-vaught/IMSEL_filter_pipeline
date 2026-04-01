"""Full dataset-wide ODS for all classical edge detectors across all 4 datasets.

Mirrors the protocol of run_full_dataset_ablation.py: dataset-wide ODS
(single best threshold across all images). Also includes best WVF/LF for
direct comparison.

Usage: python run_classical_full_dataset.py
Output: outputs/classical_full_dataset/
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import scipy.io as sio

from edgecritic.baselines.classical_gpu import run_all_classical
from edgecritic.evaluation.metrics import compute_ods_ois
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "classical_full_dataset"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001
MATCH_RADIUS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Best WVF/LF params per dataset from full_dataset_ablation summary
# (Np=25, Ns=4, d=2 wins on UDED/BIPED; Np=100 wins on BSDS500)
WVF_CONFIGS = [
    {"Np": 25,  "Ns": 4,  "d": 2, "label": "WVF(Np=25,d=2)"},
    {"Np": 100, "Ns": 4,  "d": 2, "label": "WVF(Np=100,d=2)"},
]
LF_CONFIGS = [
    {"Np": 25,  "Ns": 18, "m": 3, "d": 4, "label": "LF(Np=25,m=3)"},
    {"Np": 100, "Ns": 18, "m": 1, "d": 4, "label": "LF(Np=100,m=1)"},
]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_biped_v1():
    base = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
    img_dir = base / "imgs/test/rgbr"
    gt_dir  = base / "edge_maps/test/rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        images.append(np.mean(np.array(Image.open(f)), axis=2))
        gts.append(np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names

def load_biped_v2():
    base = ROOT / "datasets/BIPED/BIPEDv2/BIPEDv2/BIPED/edges"
    img_dir = base / "imgs/test/rgbr"
    gt_dir  = base / "edge_maps/test/rgbr"
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
    gt_dir  = base / "groundTruth/test"
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
    gt_dir  = ROOT / "datasets/UDED/gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        images.append(np.mean(np.array(Image.open(f)), axis=2))
        gts.append(np.array(Image.open(gt_dir / f.name).convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names

ALL_DATASETS = [
    ("UDED",     load_uded),
    ("BIPED_v1", load_biped_v1),
    ("BIPED_v2", load_biped_v2),
    ("BSDS500",  load_bsds500),
]


# ---------------------------------------------------------------------------
# Run one dataset
# ---------------------------------------------------------------------------
def run_dataset(ds_name, images, gts, names):
    n = len(images)
    gts_f = [gt.astype(np.float64) for gt in gts]
    print(f"\n{'='*60}\nDataset: {ds_name}  ({n} images)\n{'='*60}")

    result_file = OUT / f"{ds_name.lower()}_classical.json"
    if result_file.exists():
        print(f"  Already done — skipping.")
        with open(result_file) as f:
            return json.load(f)

    all_results = []

    # --- Classical methods ---
    print(f"\n  Running classical methods on all {n} images...")
    t0 = time.perf_counter()

    # Run on first image to get method list
    sample_results = run_all_classical(images[0], device=device)
    n_methods = len(sample_results)
    print(f"  {n_methods} classical configs found.")

    # Collect edge maps per method across all images
    # Structure: method_idx -> list of magnitude maps
    method_mags = [[] for _ in range(n_methods)]
    method_mags[0].append(sample_results[0]["magnitude"])  # already have image 0

    # Append remaining images for first result set
    for mi in range(1, n_methods):
        method_mags[mi].append(sample_results[mi]["magnitude"])

    for img_idx in range(1, n):
        img_results = run_all_classical(images[img_idx], device=device)
        for mi, r in enumerate(img_results):
            method_mags[mi].append(r["magnitude"])
        if (img_idx + 1) % 20 == 0:
            print(f"    {img_idx+1}/{n} images done ({time.perf_counter()-t0:.0f}s)")

    classical_time = time.perf_counter() - t0
    print(f"  Classical inference: {classical_time:.1f}s")

    # Evaluate dataset-wide ODS for each method
    print(f"  Evaluating dataset-wide ODS ({n_methods} methods × {N_THRESH} thresholds)...")
    t0 = time.perf_counter()
    for mi, r in enumerate(sample_results):
        ods, ois, _, _ = compute_ods_ois(
            method_mags[mi], gts_f, n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
        all_results.append({
            "method": r["method"],
            "params": r["params"],
            "ods": float(ods),
            "ois": float(ois),
        })
    eval_time = time.perf_counter() - t0
    print(f"  Evaluation: {eval_time:.1f}s")

    # Sort by ODS
    all_results.sort(key=lambda x: x["ods"], reverse=True)

    # Print top 10
    print(f"\n  Top 10 classical methods (dataset-wide ODS):")
    print(f"  {'Rank':>4s} {'Method':<20s} {'Params':<20s} {'ODS':>7s}")
    print("  " + "-" * 55)
    for i, r in enumerate(all_results[:10]):
        ps = ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "(default)"
        print(f"  {i+1:>4d} {r['method']:<20s} {ps:<20s} {r['ods']:>7.4f}")

    # --- WVF configs ---
    print(f"\n  Running WVF reference configs...")
    wvf_results = []
    for cfg in WVF_CONFIGS:
        mags = [wvf_image(img, np_count=cfg["Np"], order=cfg["d"],
                          n_orientations=cfg["Ns"], backend=device).gradient_mag
                for img in images]
        ods, ois, _, _ = compute_ods_ois(mags, gts_f, n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
        wvf_results.append({"label": cfg["label"], "ods": float(ods), "ois": float(ois), **cfg})
        print(f"    {cfg['label']}: ODS={ods:.4f}")
        torch.cuda.empty_cache()

    # --- LF configs ---
    print(f"  Running LF reference configs...")
    lf_results = []
    for cfg in LF_CONFIGS:
        try:
            mags = [lf_image(img, half_width=cfg["m"], np_count=cfg["Np"],
                             order=cfg["d"], n_orientations=cfg["Ns"],
                             backend=device).gradient_mag
                    for img in images]
            ods, ois, _, _ = compute_ods_ois(mags, gts_f, n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
            lf_results.append({"label": cfg["label"], "ods": float(ods), "ois": float(ois), **cfg})
            print(f"    {cfg['label']}: ODS={ods:.4f}")
        except Exception as e:
            print(f"    {cfg['label']}: FAILED ({e})")
        torch.cuda.empty_cache()

    output = {
        "dataset": ds_name,
        "n_images": n,
        "image_names": names,
        "evaluation": {
            "match_radius": MATCH_RADIUS,
            "n_thresholds": N_THRESH,
            "protocol": "dataset-wide ODS (single threshold across all images)",
        },
        "classical": all_results,
        "wvf": wvf_results,
        "lf": lf_results,
        "classical_inference_s": round(classical_time, 1),
        "eval_s": round(eval_time, 1),
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {result_file.name}")

    del method_mags
    gc.collect()
    torch.cuda.empty_cache()

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
grand_t0 = time.perf_counter()
all_outputs = {}

for ds_name, loader in ALL_DATASETS:
    try:
        images, gts, names = loader()
        out = run_dataset(ds_name, images, gts, names)
        all_outputs[ds_name] = out
        del images, gts
        gc.collect()
    except FileNotFoundError as e:
        print(f"  SKIPPED {ds_name}: {e}")
    except Exception as e:
        import traceback
        print(f"  ERROR {ds_name}: {e}")
        traceback.print_exc()

grand_total = time.perf_counter() - grand_t0

# Summary table
print(f"\n{'='*70}")
print(f"ALL DONE in {grand_total:.0f}s ({grand_total/3600:.1f}h)")
print(f"{'='*70}")
print(f"\n{'Dataset':<12} {'Best Classical':>15} {'Best WVF':>10} {'Best LF':>10}")
print("-" * 50)
for ds, out in all_outputs.items():
    best_cl  = out["classical"][0]["ods"] if out["classical"] else float("nan")
    best_wvf = max((r["ods"] for r in out["wvf"]), default=float("nan"))
    best_lf  = max((r["ods"] for r in out["lf"]),  default=float("nan"))
    print(f"{ds:<12} {best_cl:>15.4f} {best_wvf:>10.4f} {best_lf:>10.4f}")

# Save combined summary
summary = {
    "total_time_s": round(grand_total, 1),
    "datasets": {
        ds: {
            "best_classical": out["classical"][0] if out["classical"] else None,
            "top5_classical": out["classical"][:5],
            "wvf": out["wvf"],
            "lf":  out["lf"],
        }
        for ds, out in all_outputs.items()
    }
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {OUT / 'summary.json'}")
