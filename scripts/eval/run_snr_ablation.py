"""Full single-image ablation at multiple SNR levels on BIPED v1 RGB_008.

Same parameter grid as the clean ablation, repeated at each SNR.
SNR=inf (clean) is skipped since we already have that data.
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001

# Full grids — matching the clean BIPED ablation
WVF_NP = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500]
WVF_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24, 36, 48, 72, 90, 120, 180]
WVF_D = [2, 3, 4, 5]

LF_NP = [15, 25, 50, 75, 100, 150, 250]
LF_NS = [18, 36, 72]
LF_M = [1, 2, 3, 5, 7, 10, 14, 20]

SNR_LEVELS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]

# ---------------------------------------------------------------------------
print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_clean = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
print(f"  {img_clean.shape}, {np.sum(gt_bool)} GT edge px")
print(f"  Signal amplitude: {signal_amplitude:.1f}")

import torch
HAS_CUDA = torch.cuda.is_available()
backend = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name} ({total_vram:.0f} GB)")


def estimate_vram_budget(H, W, half_width, np_count):
    L = 2 * half_width + 1
    per_pixel = L * (np_count * 24 + 20)
    n_pixels = H * W
    total_needed_gb = n_pixels * per_pixel / 1e9
    if total_needed_gb < total_vram * 0.4:
        return total_vram * 0.5
    else:
        return total_vram * 0.35


def evaluate_config(mag, gt_b):
    ods, ois, thresholds, f_scores = compute_ods_ois(
        mag, gt_b.astype(np.float64), n_thresholds=N_THRESH, match_radius=3)
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


# Count configs
n_wvf = sum(1 for d in WVF_D for ns in WVF_NS for np_val in WVF_NP
            if np_val >= (d + 1) * (d + 2) // 2)
n_lf = len(LF_NP) * len(LF_NS) * len(LF_M)
total_configs = n_wvf + n_lf
print(f"\nPer SNR: {n_wvf} WVF + {n_lf} LF = {total_configs} configs")
print(f"SNR levels: {SNR_LEVELS}")
print(f"Total runs: {total_configs * len(SNR_LEVELS)}")

grand_t0 = time.perf_counter()

for snr in SNR_LEVELS:
    # Check if this SNR was already completed
    snr_file = OUT / f"snr_{snr:.2f}_ablation.json"
    if snr_file.exists():
        print(f"\n=== SNR={snr} already done, skipping ===")
        continue

    # Generate noisy image
    noise_std = signal_amplitude / snr
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, img_clean.shape)
    img_noisy = np.clip(img_clean + noise, 0, 255)

    print(f"\n{'='*70}")
    print(f"SNR = {snr} (noise_std = {noise_std:.1f})")
    print(f"{'='*70}")

    # --- WVF ---
    wvf_results = []
    t0 = time.perf_counter()
    count = 0
    for d in WVF_D:
        min_c = (d + 1) * (d + 2) // 2
        for ns in WVF_NS:
            for np_val in WVF_NP:
                if np_val < min_c:
                    continue
                try:
                    mag = wvf_image(img_noisy, np_count=np_val, order=d,
                                    n_orientations=ns, backend=backend).gradient_mag
                    metrics = evaluate_config(mag, gt_bool)
                    metrics.update({"filter": "WVF", "Np": np_val, "Ns": ns, "d": d})
                    wvf_results.append(metrics)
                except Exception as e:
                    print(f"  SKIP WVF Np={np_val} Ns={ns} d={d}: {e}")
                count += 1
                if count % 200 == 0:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / count * (n_wvf - count)
                    print(f"  WVF {count}/{n_wvf} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    wvf_time = time.perf_counter() - t0
    print(f"  WVF: {len(wvf_results)} configs in {wvf_time:.0f}s")

    # --- LF ---
    lf_results = []
    oom_configs = []
    t0 = time.perf_counter()
    count = 0
    for m in LF_M:
        for ns in LF_NS:
            for np_val in LF_NP:
                if HAS_CUDA:
                    torch.cuda.empty_cache()
                try:
                    H, W = img_noisy.shape
                    vram_gb = estimate_vram_budget(H, W, m, np_val)
                    mag = lf_image(img_noisy, half_width=m, np_count=np_val,
                                   order=4, n_orientations=ns, backend=backend,
                                   max_vram_gb=vram_gb).gradient_mag
                    metrics = evaluate_config(mag, gt_bool)
                    metrics.update({"filter": "LF", "Np": np_val, "Ns": ns, "m": m, "d": 4})
                    lf_results.append(metrics)
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        oom_configs.append({"Np": np_val, "Ns": ns, "m": m})
                        print(f"  OOM LF Np={np_val} Ns={ns} m={m}")
                        if HAS_CUDA:
                            torch.cuda.empty_cache()
                    else:
                        print(f"  SKIP LF Np={np_val} Ns={ns} m={m}: {e}")
                count += 1
                if count % 30 == 0:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / count * (n_lf - count)
                    print(f"  LF {count}/{n_lf} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    lf_time = time.perf_counter() - t0
    print(f"  LF: {len(lf_results)} configs in {lf_time:.0f}s, {len(oom_configs)} OOM")

    # Summary
    best_wvf = max(wvf_results, key=lambda x: x["ods"]) if wvf_results else None
    best_lf = max(lf_results, key=lambda x: x["ods"]) if lf_results else None
    paper_wvf = next((r for r in wvf_results if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
    paper_lf = next((r for r in lf_results if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

    print(f"\n  --- SNR={snr} Summary ---")
    if best_wvf:
        print(f"  Best WVF: ODS={best_wvf['ods']:.4f} Np={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}")
    if best_lf:
        print(f"  Best LF:  ODS={best_lf['ods']:.4f} Np={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}")
    if paper_wvf:
        print(f"  Bagan WVF: ODS={paper_wvf['ods']:.4f}")
    if paper_lf:
        print(f"  Bagan LF:  ODS={paper_lf['ods']:.4f}")

    # Save per-SNR results
    output = {
        "dataset": "BIPED_v1",
        "image": "RGB_008",
        "snr": snr,
        "noise_std": round(noise_std, 2),
        "image_shape": list(img_clean.shape),
        "n_gt_pixels": int(np.sum(gt_bool)),
        "evaluation": {
            "match_radius": 3,
            "n_thresholds": N_THRESH,
            "method": "distance_transform + maximum_filter + searchsorted",
            "post_processing": "none",
        },
        "wvf_time_s": round(wvf_time, 1),
        "lf_time_s": round(lf_time, 1),
        "wvf": wvf_results,
        "lf": lf_results,
        "lf_oom": oom_configs,
    }

    with open(snr_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {snr_file}")

    # Free memory
    del mag, img_noisy
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()

grand_time = time.perf_counter() - grand_t0

# ---------------------------------------------------------------------------
# Final summary across all SNRs
# ---------------------------------------------------------------------------
print(f"\n{'#'*70}")
print(f"ALL SNR LEVELS DONE in {grand_time:.0f}s ({grand_time/3600:.1f}h)")
print(f"{'#'*70}")

print(f"\n{'SNR':>5s} {'Best WVF':>10s} {'(config)':>25s} {'Best LF':>10s} {'(config)':>25s} {'Bagan WVF':>10s} {'Bagan LF':>10s}")
print("-" * 100)

for snr in SNR_LEVELS:
    snr_file = OUT / f"snr_{snr:.2f}_ablation.json"
    if not snr_file.exists():
        continue
    with open(snr_file) as f:
        data = json.load(f)

    bw = max(data["wvf"], key=lambda x: x["ods"])
    bl = max(data["lf"], key=lambda x: x["ods"]) if data["lf"] else None
    pw = next((r for r in data["wvf"] if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
    pl = next((r for r in data["lf"] if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

    bw_cfg = f"Np={bw['Np']} d={bw['d']} Ns={bw['Ns']}"
    bl_cfg = f"Np={bl['Np']} m={bl['m']} Ns={bl['Ns']}" if bl else "---"
    bl_ods = f"{bl['ods']:.4f}" if bl else "---"
    pw_ods = f"{pw['ods']:.4f}" if pw else "---"
    pl_ods = f"{pl['ods']:.4f}" if pl else "---"

    print(f"{snr:>5.2f} {bw['ods']:>10.4f} {bw_cfg:>25s} {bl_ods:>10s} {bl_cfg:>25s} {pw_ods:>10s} {pl_ods:>10s}")

# Add clean data reference
print(f"{'inf':>5s} {'0.8440':>10s} {'Np=25 d=2 Ns=3':>25s} {'0.8370':>10s} {'Np=75 m=1 Ns=18':>25s} {'0.7662':>10s} {'0.6207':>10s}")
