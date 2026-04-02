"""Fill remaining OOM cells with batched LF, save edge map images, regenerate plots."""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "biped_ablation"
EDGE_DIR = OUT / "edge_maps"
EDGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load image + GT
# ---------------------------------------------------------------------------
print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_gray = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
print(f"  {img_gray.shape}, {np.sum(gt_bool)} GT edge px")

import torch
HAS_CUDA = torch.cuda.is_available()
backend = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Load existing results
# ---------------------------------------------------------------------------
with open(OUT / "ablation_metrics.json") as f:
    data = json.load(f)

wvf_results = data["wvf"]
lf_results = data["lf"]
oom_configs = data.get("lf_oom", [])
print(f"  Existing: {len(wvf_results)} WVF + {len(lf_results)} LF, {len(oom_configs)} OOM")


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
# Fill OOM configs with batched LF
# ---------------------------------------------------------------------------
if oom_configs:
    print(f"\n=== Filling {len(oom_configs)} OOM configs with batched LF ===")
    existing_lf_keys = {(r["Np"], r["Ns"], r["m"]) for r in lf_results}
    new_lf = []
    still_oom = []
    for cfg in oom_configs:
        key = (cfg["Np"], cfg["Ns"], cfg["m"])
        if key in existing_lf_keys:
            continue
        torch.cuda.empty_cache()
        try:
            mag = lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                           order=4, n_orientations=cfg["Ns"], backend="cuda",
                           max_vram_gb=25).gradient_mag
            metrics = evaluate_config(mag, gt_bool)
            metrics.update({"filter": "LF", "Np": cfg["Np"], "Ns": cfg["Ns"],
                            "m": cfg["m"], "d": 4})
            new_lf.append(metrics)
            print(f"  FILLED Np={cfg['Np']} m={cfg['m']} Ns={cfg['Ns']}: ODS={metrics['ods']:.4f}")
        except Exception as e:
            still_oom.append(cfg)
            print(f"  STILL OOM Np={cfg['Np']} m={cfg['m']} Ns={cfg['Ns']}: {e}")
            torch.cuda.empty_cache()

    lf_results.extend(new_lf)
    data["lf"] = lf_results
    data["lf_oom"] = still_oom
    with open(OUT / "ablation_metrics.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Filled {len(new_lf)}, {len(still_oom)} still OOM")

wvf = data["wvf"]
lf = data["lf"]
print(f"\nTotal: {len(wvf)} WVF + {len(lf)} LF = {len(wvf)+len(lf)} configs")

# ---------------------------------------------------------------------------
# Helper to run a config and get magnitude map
# ---------------------------------------------------------------------------
def run_config(cfg):
    if cfg["filter"] == "WVF":
        return wvf_image(img_gray, np_count=cfg["Np"], order=cfg["d"],
                         n_orientations=cfg["Ns"], backend=backend).gradient_mag
    else:
        return lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                        order=cfg["d"], n_orientations=cfg["Ns"],
                        backend=backend, max_vram_gb=25).gradient_mag


def save_edge_map(mag, gt_b, cfg, label):
    """Save a 4-panel image: magnitude, thresholded, GT, overlay."""
    best_t = cfg["best_threshold"]
    pred = mag > best_t

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(mag, cmap="gray")
    axes[0].set_title("Gradient Magnitude", fontsize=10)

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"Thresholded (t={best_t:.3f})", fontsize=10)

    axes[2].imshow(gt_b, cmap="gray")
    axes[2].set_title("Ground Truth", fontsize=10)

    # Overlay: green=TP, red=FP, blue=FN
    dist_to_gt = distance_transform_edt(~gt_b)
    near_gt = dist_to_gt <= 3
    mag_max_local = maximum_filter(mag, size=7)

    overlay = np.zeros((*mag.shape, 3), dtype=np.uint8)
    tp_mask = pred & near_gt
    fp_mask = pred & ~near_gt
    fn_mask = gt_b & (mag_max_local <= best_t)
    overlay[tp_mask] = [0, 200, 0]     # green = correct prediction
    overlay[fp_mask] = [200, 0, 0]     # red = false positive
    overlay[fn_mask] = [0, 80, 200]    # blue = missed GT edge
    axes[3].imshow(overlay)
    axes[3].set_title(f"TP(green) FP(red) FN(blue)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    title = f"{cfg['filter']} — {label}\n"
    if cfg["filter"] == "WVF":
        title += f"Np={cfg['Np']} Ns={cfg['Ns']} d={cfg['d']}"
    else:
        title += f"Np={cfg['Np']} Ns={cfg['Ns']} m={cfg['m']}"
    title += f" | ODS={cfg['ods']:.3f} P={cfg['precision']:.2f} R={cfg['recall']:.2f}"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    safe_label = label.lower().replace(" ", "_").replace("'", "")
    fname = EDGE_DIR / f"{safe_label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Save edge maps for key configs
# ---------------------------------------------------------------------------
print("\n=== Saving edge map images ===")

best_wvf = max(wvf, key=lambda x: x["ods"])
best_lf = max(lf, key=lambda x: x["ods"])
worst = min([c for c in wvf + lf if c["ods"] > 0], key=lambda x: x["ods"])
paper_wvf = next((r for r in wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

# Also pick some interesting intermediate configs
mid_wvf = next((r for r in wvf if r["Np"]==100 and r["d"]==4 and r["Ns"]==18), None)
small_wvf = next((r for r in wvf if r["Np"]==15 and r["d"]==4 and r["Ns"]==18), None)
ns1_wvf = next((r for r in wvf if r["Np"]==50 and r["d"]==4 and r["Ns"]==1), None)
ns3_wvf = next((r for r in wvf if r["Np"]==50 and r["d"]==4 and r["Ns"]==3), None)
lf_m3 = next((r for r in lf if r["Np"]==15 and r["m"]==3 and r["Ns"]==18), None)
lf_m14 = next((r for r in lf if r["Np"]==50 and r["m"]==14 and r["Ns"]==18), None)

configs_to_save = [
    (best_wvf, "best_wvf"),
    (best_lf, "best_lf"),
    (paper_wvf, "bagan_wvf"),
    (paper_lf, "bagan_lf"),
    (worst, "worst"),
    (mid_wvf, "wvf_np100_d4"),
    (small_wvf, "wvf_np15_d4"),
    (ns1_wvf, "wvf_np50_ns1"),
    (ns3_wvf, "wvf_np50_ns3"),
    (lf_m3, "lf_np15_m3"),
    (lf_m14, "lf_np50_m14"),
]

for cfg, label in configs_to_save:
    if cfg is None:
        continue
    mag = run_config(cfg)
    fname = save_edge_map(mag, gt_bool, cfg, label)
    print(f"  {fname.name}")

# Also save raw magnitude maps as numpy arrays for the key configs
for cfg, label in [(best_wvf, "best_wvf"), (best_lf, "best_lf"),
                    (paper_wvf, "bagan_wvf"), (paper_lf, "bagan_lf")]:
    if cfg is None:
        continue
    mag = run_config(cfg)
    np.save(EDGE_DIR / f"{label}_magnitude.npy", mag)
    # Also save thresholded version
    pred = (mag > cfg["best_threshold"]).astype(np.uint8) * 255
    Image.fromarray(pred).save(EDGE_DIR / f"{label}_edges.png")
    print(f"  {label}_magnitude.npy + {label}_edges.png")

# ---------------------------------------------------------------------------
# Generate all plots (same as generate_biped_plots.py but fresh data)
# ---------------------------------------------------------------------------
print("\n=== Generating plots ===")

# Plot 1: Ns cliff
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for ci, np_val in enumerate([15, 50, 100, 200, 500]):
    ns_vals, ods_vals = [], []
    for r in wvf:
        if r["Np"] == np_val and r["d"] == 4:
            ns_vals.append(r["Ns"]); ods_vals.append(r["ods"])
    if ns_vals:
        order = np.argsort(ns_vals)
        ax.plot(np.array(ns_vals)[order], np.array(ods_vals)[order], "o-",
                color=colors[ci], markersize=6, label=f"$N_p$={np_val}")
ax.axvline(x=3, color="red", linestyle=":", alpha=0.5, label="$N_s$=3")
ax.set_xlabel("$N_s$ (orientations)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("BIPED v1: Orientation Cliff", fontsize=13)
ax.legend(); ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 3, 4, 6, 9, 12, 18, 36, 72])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "plot_ns_cliff.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_ns_cliff.png")

# Plot 2: PR scatter
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([r["recall"] for r in wvf], [r["precision"] for r in wvf],
                c=[r["ods"] for r in wvf], cmap="viridis", s=20, alpha=0.6,
                marker="o", label="WVF", vmin=0.4, vmax=0.86)
ax.scatter([r["recall"] for r in lf], [r["precision"] for r in lf],
           c=[r["ods"] for r in lf], cmap="viridis", s=20, alpha=0.6,
           marker="^", label="LF", vmin=0.4, vmax=0.86)
for f_val in [0.5, 0.6, 0.7, 0.8]:
    r_range = np.linspace(0.01, 1, 200)
    p_iso = f_val * r_range / (2 * r_range - f_val)
    valid = (p_iso > 0) & (p_iso <= 1)
    ax.plot(r_range[valid], p_iso[valid], "--", color="gray", alpha=0.3, linewidth=0.8)
    idx = np.argmin(np.abs(r_range - 0.95))
    if valid[idx]:
        ax.text(0.96, p_iso[idx], f"F={f_val}", fontsize=7, color="gray", va="center")
ax.scatter([best_wvf["recall"]], [best_wvf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.scatter([best_lf["recall"]], [best_lf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.annotate(f"Best WVF\nODS={best_wvf['ods']:.3f}", (best_wvf["recall"], best_wvf["precision"]),
            textcoords="offset points", xytext=(10, 10), fontsize=8)
ax.annotate(f"Best LF\nODS={best_lf['ods']:.3f}", (best_lf["recall"], best_lf["precision"]),
            textcoords="offset points", xytext=(10, -15), fontsize=8)
if paper_wvf:
    ax.scatter([paper_wvf["recall"]], [paper_wvf["precision"]], c="orange", s=100,
               marker="D", zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Bagan WVF\nODS={paper_wvf['ods']:.3f}", (paper_wvf["recall"], paper_wvf["precision"]),
                textcoords="offset points", xytext=(-60, -20), fontsize=8, color="darkorange")
if paper_lf:
    ax.scatter([paper_lf["recall"]], [paper_lf["precision"]], c="orange", s=100,
               marker="D", zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Bagan LF\nODS={paper_lf['ods']:.3f}", (paper_lf["recall"], paper_lf["precision"]),
                textcoords="offset points", xytext=(-55, 10), fontsize=8, color="darkorange")
fig.colorbar(sc, ax=ax, label="ODS F-score")
ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
ax.set_title(f"BIPED v1: Precision-Recall ({len(wvf)+len(lf)} configs)", fontsize=13)
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0)
ax.legend(fontsize=10, loc="lower left"); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "plot_pr_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_pr_scatter.png")

# Plot 3: ODS vs Np — two panels
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: WVF by d
ax = axes[0]
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
all_d = sorted(set(r["d"] for r in wvf))
for d in all_d:
    xs, ys = [], []
    for r in wvf:
        if r["d"] == d and r["Ns"] == 18:
            xs.append(r["Np"]); ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=d_colors.get(d, "#666666"), markersize=5, label=f"d={d}")
if paper_wvf:
    ax.scatter([250], [paper_wvf["ods"]], c="orange", s=100, marker="D",
               zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate("Bagan", (250, paper_wvf["ods"]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color="darkorange")
ax.set_xlabel("$N_p$ (support size)", fontsize=12); ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("WVF: ODS vs $N_p$ ($N_s$=18)", fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Right: LF by m, with WVF d=4 as reference
ax = axes[1]
lf_ms_all = sorted(set(r["m"] for r in lf))
lf_colors = plt.cm.tab10(np.linspace(0, 1, len(lf_ms_all)))
for ci, m in enumerate(lf_ms_all):
    xs, ys = [], []
    for r in lf:
        if r["m"] == m and r["Ns"] == 18:
            xs.append(r["Np"]); ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=lf_colors[ci], markersize=5, label=f"m={m}")
# WVF d=4 reference line
xs, ys = [], []
for r in wvf:
    if r["d"] == 4 and r["Ns"] == 18:
        xs.append(r["Np"]); ys.append(r["ods"])
if xs:
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], "k--", markersize=4,
            linewidth=1.5, label="WVF (d=4)")
if paper_lf:
    ax.scatter([250], [paper_lf["ods"]], c="orange", s=100, marker="D",
               zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate("Bagan", (250, paper_lf["ods"]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color="darkorange")
ax.set_xlabel("$N_p$ (support size)", fontsize=12); ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("LF: ODS vs $N_p$ ($N_s$=18, varying $m$)", fontsize=13)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

fig.suptitle("BIPED v1 RGB_008", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_np_full.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_ods_vs_np_full.png")

# Plot 4: LF heatmap
fig, ax = plt.subplots(figsize=(8, 5))
lf_nps_all = sorted(set(r["Np"] for r in lf))
lf_ms = sorted(set(r["m"] for r in lf))
hm = np.full((len(lf_nps_all), len(lf_ms)), np.nan)
lookup = {(r["Np"], r["m"]): r["ods"] for r in lf if r["Ns"] == 18}
for i, np_v in enumerate(lf_nps_all):
    for j, m_v in enumerate(lf_ms):
        if (np_v, m_v) in lookup: hm[i, j] = lookup[(np_v, m_v)]
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.5, vmax=np.nanmax(hm)+0.01)
ax.set_xticks(range(len(lf_ms))); ax.set_xticklabels(lf_ms)
ax.set_yticks(range(len(lf_nps_all))); ax.set_yticklabels(lf_nps_all)
ax.set_xlabel("Line half-width $m$", fontsize=12); ax.set_ylabel("$N_p$", fontsize=12)
ax.set_title("BIPED v1: LF ODS Heatmap ($N_s$=18)", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS")
for i in range(len(lf_nps_all)):
    for j in range(len(lf_ms)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < np.nanmean(hm) else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=8, color=c)
fig.tight_layout()
fig.savefig(OUT / "plot_lf_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_lf_heatmap.png")

# Plot 5: LF vs m
fig, ax = plt.subplots(figsize=(8, 5))
tab_colors = plt.cm.tab10(range(len(lf_nps_all)))
for ci, np_val in enumerate(lf_nps_all):
    ms, ods_vals = [], []
    for r in lf:
        if r["Np"] == np_val and r["Ns"] == 18:
            ms.append(r["m"]); ods_vals.append(r["ods"])
    order = np.argsort(ms)
    ax.plot(np.array(ms)[order], np.array(ods_vals)[order], "o-",
            color=tab_colors[ci], markersize=5, label=f"$N_p$={np_val}")
ax.set_xlabel("Line half-width $m$", fontsize=12); ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("BIPED v1: LF vs Line Length ($N_s$=18)", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "plot_lf_vs_m.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_lf_vs_m.png")

# Plot 6: WVF heatmap (Np x Ns, d=4)
fig, ax = plt.subplots(figsize=(14, 7))
nps = sorted(set(r["Np"] for r in wvf if r["d"] == 4))
nss = sorted(set(r["Ns"] for r in wvf if r["d"] == 4))
hm = np.full((len(nps), len(nss)), np.nan)
lookup = {(r["Np"], r["Ns"]): r["ods"] for r in wvf if r["d"] == 4}
for i, np_v in enumerate(nps):
    for j, ns_v in enumerate(nss):
        if (np_v, ns_v) in lookup: hm[i, j] = lookup[(np_v, ns_v)]
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.45, vmax=np.nanmax(hm)+0.01)
ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss, fontsize=9)
ax.set_yticks(range(len(nps))); ax.set_yticklabels(nps, fontsize=9)
ax.set_xlabel("$N_s$", fontsize=12); ax.set_ylabel("$N_p$", fontsize=12)
ax.set_title("BIPED v1: WVF ODS Heatmap ($d$=4)", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS", shrink=0.8)
cliff_x = nss.index(3) - 0.5 if 3 in nss else None
if cliff_x is not None:
    ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < np.nanmean(hm) else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=5.5, color=c)
fig.tight_layout()
fig.savefig(OUT / "plot_wvf_heatmap_full.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_wvf_heatmap_full.png")

# Plot 7: Pipeline overview
panels = [
    ("Input (RGB)", img_color, {}),
    ("Grayscale", img_gray, {"cmap": "gray"}),
    (f"Ground Truth\n({np.sum(gt_bool)} edge px)", gt_bool.astype(float), {"cmap": "gray"}),
    (f"Best WVF: ODS={best_wvf['ods']:.3f}\nNp={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}",
     run_config(best_wvf), {"cmap": "gray"}),
    (f"Best LF: ODS={best_lf['ods']:.3f}\nNp={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}",
     run_config(best_lf), {"cmap": "gray"}),
]
if paper_wvf:
    panels.append((f"Bagan WVF: ODS={paper_wvf['ods']:.3f}\nNp=250 d=4", run_config(paper_wvf), {"cmap": "gray"}))
if paper_lf:
    panels.append((f"Bagan LF: ODS={paper_lf['ods']:.3f}\nNp=250 m=14", run_config(paper_lf), {"cmap": "gray"}))
n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
for ax, (title, dat, kwargs) in zip(axes, panels):
    ax.imshow(dat, **kwargs); ax.set_title(title, fontsize=9); ax.axis("off")
fig.suptitle("BIPED v1 RGB_008 — Ablation Overview", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_pipeline_overview.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_pipeline_overview.png")

# Plot 8: ODS distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0.4, 0.88, 40)
ax.hist([r["ods"] for r in wvf], bins=bins, alpha=0.6, label=f"WVF ({len(wvf)})", color="#377eb8")
ax.hist([r["ods"] for r in lf], bins=bins, alpha=0.6, label=f"LF ({len(lf)})", color="#e41a1c")
ax.axvline(max(r["ods"] for r in wvf), color="#377eb8", linestyle="--", linewidth=1.5)
ax.axvline(max(r["ods"] for r in lf), color="#e41a1c", linestyle="--", linewidth=1.5)
if paper_wvf: ax.axvline(paper_wvf["ods"], color="darkorange", linestyle="-.", linewidth=1.5, label="Bagan WVF")
if paper_lf: ax.axvline(paper_lf["ods"], color="darkorange", linestyle=":", linewidth=1.5, label="Bagan LF")
ax.set_xlabel("ODS F-score", fontsize=12); ax.set_ylabel("Configs", fontsize=12)
ax.set_title("BIPED v1: ODS Distribution", fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(OUT / "plot_ods_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  plot_ods_distribution.png")

print(f"\nAll done. Outputs in {OUT}")
print(f"Edge maps in {EDGE_DIR}")
