"""Generate comprehensive ablation visualization plots."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "single_image_ablation"

# Load data
with open(OUT / "ablation_metrics.json") as f:
    data = json.load(f)
with open(OUT / "low_ns_results.json") as f:
    low_ns = json.load(f)

wvf = data["wvf"]
lf = data["lf"]

# Load image + GT for edge map visualizations
img_color = np.array(Image.open(ROOT / "datasets/BSDS500/BSDS500/data/images/test/100007.jpg"))
img_gray = np.mean(img_color, axis=2)
gt_mat = sio.loadmat(str(ROOT / "datasets/BSDS500/BSDS500/data/groundTruth/test/100007.mat"))
gt_cell = gt_mat["groundTruth"]
gt_union = np.zeros(img_gray.shape, dtype=bool)
for i in range(gt_cell.shape[1]):
    gt_union |= (np.asarray(gt_cell[0, i]["Boundaries"][0, 0]) > 0)

# Also add Ns=1 data point (hardcoded from the run)
ns1_data = [
    {"Np": 15, "Ns": 1, "d": 4, "ods": 0.5819},
    {"Np": 50, "Ns": 1, "d": 4, "ods": 0.6183},
    {"Np": 100, "Ns": 1, "d": 4, "ods": 0.6100},
    {"Np": 250, "Ns": 1, "d": 4, "ods": 0.5969},
]
low_ns_all = ns1_data + low_ns

# =========================================================================
# Plot 1: The Ns cliff — ODS vs Ns at different Np (d=4)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for ci, np_val in enumerate([15, 50, 100, 150, 250]):
    ns_vals, ods_vals = [], []
    for r in low_ns_all:
        if r["Np"] == np_val and r["d"] == 4:
            ns_vals.append(r["Ns"])
            ods_vals.append(r["ods"])
    order = np.argsort(ns_vals)
    ns_arr = np.array(ns_vals)[order]
    ods_arr = np.array(ods_vals)[order]
    ax.plot(ns_arr, ods_arr, "o-", color=colors[ci], markersize=6,
            label=f"$N_p$={np_val}")
ax.axvline(x=3, color="red", linestyle=":", alpha=0.5, label="$N_s$=3 threshold")
ax.set_xlabel("$N_s$ (number of orientations)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("Orientation Count: Cliff at $N_s$=3, Flat Beyond", fontsize=13)
ax.legend(fontsize=10)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 18, 36])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_ns_cliff.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ns_cliff.png")

# =========================================================================
# Plot 2: Precision vs Recall scatter — all configs colored by filter type
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

wvf_p = [r["precision"] for r in wvf]
wvf_r = [r["recall"] for r in wvf]
lf_p = [r["precision"] for r in lf]
lf_r = [r["recall"] for r in lf]

ax.scatter(wvf_r, wvf_p, c=[r["ods"] for r in wvf], cmap="viridis",
           s=20, alpha=0.6, label="WVF", marker="o", vmin=0.5, vmax=0.9)
sc = ax.scatter(lf_r, lf_p, c=[r["ods"] for r in lf], cmap="viridis",
                s=20, alpha=0.6, label="LF", marker="^", vmin=0.5, vmax=0.9)

# F-score iso-lines
for f_val in [0.6, 0.7, 0.8, 0.85]:
    r_range = np.linspace(0.01, 1, 200)
    p_iso = f_val * r_range / (2 * r_range - f_val)
    valid = (p_iso > 0) & (p_iso <= 1)
    ax.plot(r_range[valid], p_iso[valid], "--", color="gray", alpha=0.3, linewidth=0.8)
    idx = np.argmin(np.abs(r_range - 0.95))
    if valid[idx]:
        ax.text(0.96, p_iso[idx], f"F={f_val}", fontsize=7, color="gray", va="center")

# Mark best configs
best_wvf = max(wvf, key=lambda x: x["ods"])
best_lf = max(lf, key=lambda x: x["ods"])
ax.scatter([best_wvf["recall"]], [best_wvf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.scatter([best_lf["recall"]], [best_lf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.annotate(f"Best WVF\nODS={best_wvf['ods']:.3f}",
            (best_wvf["recall"], best_wvf["precision"]),
            textcoords="offset points", xytext=(10, 10), fontsize=8)
ax.annotate(f"Best LF\nODS={best_lf['ods']:.3f}",
            (best_lf["recall"], best_lf["precision"]),
            textcoords="offset points", xytext=(10, -15), fontsize=8)

fig.colorbar(sc, ax=ax, label="ODS F-score")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision–Recall Space: All 838 Configurations", fontsize=13)
ax.set_xlim(0.3, 1.0)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_pr_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_pr_scatter.png")

# =========================================================================
# Plot 3: ODS vs Np — WVF at each d, with best LF overlaid
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

# WVF lines by d
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
for d in [2, 3, 4, 5]:
    xs, ys = [], []
    for r in wvf:
        if r["d"] == d and r["Ns"] == 18:
            xs.append(r["Np"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=d_colors[d], markersize=5, label=f"WVF d={d}")

# Best LF at each Np (envelope)
lf_nps = sorted(set(r["Np"] for r in lf))
lf_best_ods = []
lf_best_label = []
for np_val in lf_nps:
    best = max((r for r in lf if r["Np"] == np_val), key=lambda x: x["ods"])
    lf_best_ods.append(best["ods"])
    lf_best_label.append(f"m={best['m']}")
ax.plot(lf_nps, lf_best_ods, "s--", color="darkorange", markersize=7,
        linewidth=2, label="LF (best m per $N_p$)")

# Annotate LF best m values
for x, y, lbl in zip(lf_nps, lf_best_ods, lf_best_label):
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 8),
                fontsize=7, color="darkorange")

ax.set_xlabel("$N_p$ (support size)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("ODS vs Support Size: Polynomial Order and LF Comparison ($N_s$=18)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_np_full.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_vs_np_full.png")

# =========================================================================
# Plot 4: LF heatmap — Np vs m at Ns=18
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
lf_nps_all = sorted(set(r["Np"] for r in lf))
lf_ms = sorted(set(r["m"] for r in lf))
hm = np.full((len(lf_nps_all), len(lf_ms)), np.nan)
lookup = {(r["Np"], r["m"]): r["ods"] for r in lf if r["Ns"] == 18}
for i, np_v in enumerate(lf_nps_all):
    for j, m_v in enumerate(lf_ms):
        if (np_v, m_v) in lookup:
            hm[i, j] = lookup[(np_v, m_v)]

im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.6, vmax=0.86)
ax.set_xticks(range(len(lf_ms)))
ax.set_xticklabels(lf_ms)
ax.set_yticks(range(len(lf_nps_all)))
ax.set_yticklabels(lf_nps_all)
ax.set_xlabel("Line half-width $m$", fontsize=12)
ax.set_ylabel("$N_p$ (support size)", fontsize=12)
ax.set_title("LF ODS Heatmap ($N_s$=18, $d$=4)", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS F-score")
for i in range(len(lf_nps_all)):
    for j in range(len(lf_ms)):
        if not np.isnan(hm[i, j]):
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if hm[i, j] < 0.73 else "black")
fig.tight_layout()
fig.savefig(OUT / "plot_lf_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_lf_heatmap.png")

# =========================================================================
# Plot 5: LF ODS vs m at different Np — shows optimal line length
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(range(len(lf_nps_all)))
for ci, np_val in enumerate(lf_nps_all):
    ms, ods_vals = [], []
    for r in lf:
        if r["Np"] == np_val and r["Ns"] == 18:
            ms.append(r["m"])
            ods_vals.append(r["ods"])
    order = np.argsort(ms)
    ax.plot(np.array(ms)[order], np.array(ods_vals)[order], "o-",
            color=colors[ci], markersize=5, label=f"$N_p$={np_val}")
ax.set_xlabel("Line half-width $m$", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("LF: Effect of Line Length ($N_s$=18)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_lf_vs_m.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_lf_vs_m.png")

# =========================================================================
# Plot 6: Edge maps — best, median, worst, plus Np progression
# =========================================================================
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image

configs_to_show = [
    ("$N_p$=15, $N_s$=1\nODS=0.582", dict(filter="WVF", np_count=15, order=4, n_orientations=1)),
    ("$N_p$=15, $N_s$=3\nODS=0.737", dict(filter="WVF", np_count=15, order=4, n_orientations=3)),
    ("$N_p$=50, $N_s$=9, d=2\nODS=0.860 (best)", dict(filter="WVF", np_count=50, order=2, n_orientations=9)),
    ("$N_p$=100, $N_s$=18\nODS=0.854", dict(filter="WVF", np_count=100, order=4, n_orientations=18)),
    ("$N_p$=250, $N_s$=18\nODS=0.813", dict(filter="WVF", np_count=250, order=4, n_orientations=18)),
    ("LF $N_p$=15, m=3\nODS=0.846", dict(filter="LF", np_count=15, order=4, n_orientations=18, half_width=3)),
]

backend = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        backend = "cpu"
except ImportError:
    backend = "cpu"

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Row 0: input, GT, then edge maps
axes[0, 0].imshow(img_color)
axes[0, 0].set_title("Input", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(gt_union, cmap="gray")
axes[0, 1].set_title(f"Ground Truth\n({data['n_gt_pixels']} edge px)", fontsize=10)
axes[0, 1].axis("off")

for idx, (label, cfg) in enumerate(configs_to_show[:2]):
    if cfg["filter"] == "WVF":
        mag = wvf_image(img_gray, np_count=cfg["np_count"], order=cfg["order"],
                        n_orientations=cfg["n_orientations"], backend=backend).gradient_mag
    axes[0, idx + 2].imshow(mag, cmap="gray")
    axes[0, idx + 2].set_title(label, fontsize=9)
    axes[0, idx + 2].axis("off")

for idx, (label, cfg) in enumerate(configs_to_show[2:]):
    if cfg["filter"] == "WVF":
        mag = wvf_image(img_gray, np_count=cfg["np_count"], order=cfg["order"],
                        n_orientations=cfg["n_orientations"], backend=backend).gradient_mag
    else:
        mag = lf_image(img_gray, half_width=cfg["half_width"], np_count=cfg["np_count"],
                       order=cfg["order"], n_orientations=cfg["n_orientations"],
                       backend=backend).gradient_mag
    axes[1, idx].imshow(mag, cmap="gray")
    axes[1, idx].set_title(label, fontsize=9)
    axes[1, idx].axis("off")

fig.suptitle("BSDS500 #100007 — Edge Maps Across Ablation Configurations", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_edge_maps.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_edge_maps.png")

# =========================================================================
# Plot 7: Distribution of ODS scores — histogram
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
wvf_ods = [r["ods"] for r in wvf]
lf_ods = [r["ods"] for r in lf]
bins = np.linspace(0.55, 0.88, 40)
ax.hist(wvf_ods, bins=bins, alpha=0.6, label=f"WVF ({len(wvf_ods)} configs)", color="#377eb8")
ax.hist(lf_ods, bins=bins, alpha=0.6, label=f"LF ({len(lf_ods)} configs)", color="#e41a1c")
ax.axvline(max(wvf_ods), color="#377eb8", linestyle="--", linewidth=1.5)
ax.axvline(max(lf_ods), color="#e41a1c", linestyle="--", linewidth=1.5)
ax.set_xlabel("ODS F-score", fontsize=12)
ax.set_ylabel("Number of configurations", fontsize=12)
ax.set_title("Distribution of ODS Across All Ablation Configurations", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "plot_ods_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_distribution.png")

print(f"\nAll plots saved to {OUT}")
