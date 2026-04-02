"""Full BIPED v1 ablation — matches BSDS500 polar bear coverage exactly,
plus low-Ns configs. Uses pixel-batched LF to avoid OOM.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
IMG_PATH = BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"
GT_PATH = BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png"
OUT_DIR = ROOT / "outputs" / "biped_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(IMG_PATH))
img_gray = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(GT_PATH).convert("L")) > 128
print(f"  Image: {img_gray.shape}, GT: {np.sum(gt_bool)} edge px")

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"  Free VRAM: {free_gb:.1f} GB")
except ImportError:
    HAS_CUDA = False
backend = "cuda" if HAS_CUDA else "cpu"

# VRAM budget: use 70% of free memory, minimum 20 GB
VRAM_GB = max(free_gb * 0.7, 20) if HAS_CUDA else None
print(f"  VRAM budget: {VRAM_GB:.1f} GB" if VRAM_GB else "  CPU mode")


def evaluate_config(mag, gt_b, n_thresholds=500):
    ods, ois, thresholds, f_scores = compute_ods_ois(
        mag, gt_b.astype(np.float64), n_thresholds=n_thresholds, match_radius=3)
    best_idx = np.argmax(f_scores)
    best_t = thresholds[best_idx]
    pred = mag > best_t
    dist_to_gt = distance_transform_edt(~gt_b)
    near_gt = dist_to_gt <= 3
    tp = int(np.sum(pred & near_gt))
    fp = int(np.sum(pred & ~near_gt))
    mag_max_local = maximum_filter(mag, size=7)
    gt_local_max = mag_max_local[gt_b]
    fn = int(np.sum(gt_local_max <= best_t))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "ods": float(ods), "best_threshold": float(best_t),
        "precision": float(precision), "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": int(np.sum(pred)), "n_gt": int(np.sum(gt_b)),
    }


# ---------------------------------------------------------------------------
# Load existing results to avoid recomputing
# ---------------------------------------------------------------------------
json_path = OUT_DIR / "ablation_metrics.json"
if json_path.exists():
    with open(json_path) as f:
        existing = json.load(f)
    existing_wvf = existing.get("wvf", [])
    existing_lf = existing.get("lf", [])
    print(f"  Loaded {len(existing_wvf)} WVF + {len(existing_lf)} LF existing results")
else:
    existing_wvf = []
    existing_lf = []

existing_wvf_keys = {(r["Np"], r["Ns"], r["d"]) for r in existing_wvf}
existing_lf_keys = {(r["Np"], r["Ns"], r["m"]) for r in existing_lf}

# ---------------------------------------------------------------------------
# Full grids — matching BSDS500 + low-Ns extension
# ---------------------------------------------------------------------------
# WVF: BSDS had Np 5-500 (20 values), Ns 9-180 (10 values), d 2-5 (4 values)
# Plus low-Ns: 1,2,3,4,5,6,7,8
WVF_NP = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
WVF_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24, 36, 48, 72, 90, 120, 180]
WVF_D = [2, 3, 4, 5]

# LF: BSDS had Np 7 values, Ns 3 values, m 8 values
LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36, 72]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

# ---------------------------------------------------------------------------
# WVF sweep
# ---------------------------------------------------------------------------
wvf_todo = []
for d in WVF_D:
    min_c = (d + 1) * (d + 2) // 2
    for ns in WVF_NS:
        for np_val in WVF_NP:
            if np_val < min_c:
                continue
            if (np_val, ns, d) not in existing_wvf_keys:
                wvf_todo.append((np_val, ns, d))

print(f"\n=== WVF: {len(wvf_todo)} new configs to run ===")
new_wvf = []
t0 = time.perf_counter()
for idx, (np_val, ns, d) in enumerate(wvf_todo):
    try:
        mag = wvf_image(img_gray, np_count=np_val, order=d,
                        n_orientations=ns, backend=backend).gradient_mag
        metrics = evaluate_config(mag, gt_bool)
        metrics.update({"filter": "WVF", "Np": np_val, "Ns": ns, "d": d})
        new_wvf.append(metrics)
    except Exception as e:
        print(f"  SKIP WVF Np={np_val} Ns={ns} d={d}: {e}")
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(wvf_todo)} ({time.perf_counter()-t0:.1f}s)")
wvf_time = time.perf_counter() - t0
print(f"  WVF: {len(new_wvf)} new in {wvf_time:.1f}s")

# ---------------------------------------------------------------------------
# LF sweep
# ---------------------------------------------------------------------------
lf_todo = []
for m in LF_M:
    for ns in LF_NS:
        for np_val in LF_NP:
            if (np_val, ns, m) not in existing_lf_keys:
                lf_todo.append((np_val, ns, m))

print(f"\n=== LF: {len(lf_todo)} new configs to run ===")
new_lf = []
oom_configs = []
t0 = time.perf_counter()
for idx, (np_val, ns, m) in enumerate(lf_todo):
    if HAS_CUDA:
        torch.cuda.empty_cache()
    try:
        mag = lf_image(img_gray, half_width=m, np_count=np_val,
                       order=4, n_orientations=ns, backend=backend,
                       max_vram_gb=VRAM_GB).gradient_mag
        metrics = evaluate_config(mag, gt_bool)
        metrics.update({"filter": "LF", "Np": np_val, "Ns": ns, "m": m, "d": 4})
        new_lf.append(metrics)
    except Exception as e:
        if "out of memory" in str(e).lower():
            oom_configs.append({"Np": np_val, "Ns": ns, "m": m})
            print(f"  OOM LF Np={np_val} Ns={ns} m={m}")
            if HAS_CUDA:
                torch.cuda.empty_cache()
        else:
            print(f"  SKIP LF Np={np_val} Ns={ns} m={m}: {e}")
    if (idx + 1) % 20 == 0:
        print(f"  {idx+1}/{len(lf_todo)} ({time.perf_counter()-t0:.1f}s)")
lf_time = time.perf_counter() - t0
print(f"  LF: {len(new_lf)} new in {lf_time:.1f}s, {len(oom_configs)} OOM'd")

# ---------------------------------------------------------------------------
# Merge and save
# ---------------------------------------------------------------------------
all_wvf = existing_wvf + new_wvf
all_lf = existing_lf + new_lf

all_results = {
    "dataset": "BIPED_v1",
    "image": "RGB_008",
    "image_shape": list(img_gray.shape),
    "n_gt_pixels": int(np.sum(gt_bool)),
    "evaluation": {"match_radius": 3, "n_thresholds": 500,
                    "method": "distance_transform + maximum_filter + searchsorted"},
    "wvf_time_s": round(wvf_time, 2),
    "lf_time_s": round(lf_time, 2),
    "wvf": all_wvf,
    "lf": all_lf,
    "lf_oom": oom_configs,
}

with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"BIPED ABLATION COMPLETE")
print(f"  WVF: {len(all_wvf)} total ({len(new_wvf)} new)")
print(f"  LF:  {len(all_lf)} total ({len(new_lf)} new)")
print(f"  OOM: {len(oom_configs)}")
print(f"  Saved to {json_path}")

best_wvf = max(all_wvf, key=lambda x: x["ods"])
best_lf = max(all_lf, key=lambda x: x["ods"])
print(f"\nBest WVF: ODS={best_wvf['ods']:.4f} Np={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}")
print(f"Best LF:  ODS={best_lf['ods']:.4f} Np={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}")

paper_wvf = next((r for r in all_wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in all_lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)
if paper_wvf:
    print(f"Bagan WVF: ODS={paper_wvf['ods']:.4f}")
if paper_lf:
    print(f"Bagan LF:  ODS={paper_lf['ods']:.4f}")
print(f"{'='*60}")
