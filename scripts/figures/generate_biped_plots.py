"""Generate ablation plots for BIPED v1 RGB_008."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "biped_ablation"

with open(OUT / "ablation_metrics.json") as f:
    data = json.load(f)

wvf = data["wvf"]
lf = data["lf"]

# Load image + GT
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_gray = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128

# =========================================================================
# Plot 1: Ns cliff
# =========================================================================
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
print("Saved plot_ns_cliff.png")

# =========================================================================
# Plot 2: Precision vs Recall scatter
# =========================================================================
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

best_wvf = max(wvf, key=lambda x: x["ods"])
best_lf = max(lf, key=lambda x: x["ods"])
ax.scatter([best_wvf["recall"]], [best_wvf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.scatter([best_lf["recall"]], [best_lf["precision"]], c="red", s=120,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.annotate(f"Best WVF\nODS={best_wvf['ods']:.3f}", (best_wvf["recall"], best_wvf["precision"]),
            textcoords="offset points", xytext=(10, 10), fontsize=8)
ax.annotate(f"Best LF\nODS={best_lf['ods']:.3f}", (best_lf["recall"], best_lf["precision"]),
            textcoords="offset points", xytext=(10, -15), fontsize=8)
# Mark Bagan's configs
paper_wvf = next((r for r in wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)
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
ax.set_title("BIPED v1: Precision–Recall Space (590 configs)", fontsize=13)
ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0)
ax.legend(fontsize=10, loc="lower left"); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_pr_scatter.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_pr_scatter.png")

# =========================================================================
# Plot 3: ODS vs Np — WVF by d + best LF envelope
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a"}
for d in [2, 3, 4]:
    xs, ys = [], []
    for r in wvf:
        if r["d"] == d and r["Ns"] == 18:
            xs.append(r["Np"]); ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=d_colors[d], markersize=5, label=f"WVF d={d}")
lf_nps = sorted(set(r["Np"] for r in lf))
lf_best, lf_labels = [], []
for np_val in lf_nps:
    best = max((r for r in lf if r["Np"] == np_val), key=lambda x: x["ods"])
    lf_best.append(best["ods"]); lf_labels.append(f"m={best['m']}")
ax.plot(lf_nps, lf_best, "s--", color="darkorange", markersize=7, linewidth=2,
        label="LF (best m per $N_p$)")
for x, y, lbl in zip(lf_nps, lf_best, lf_labels):
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 8), fontsize=7, color="darkorange")
# Mark Bagan's point
if paper_wvf:
    ax.scatter([250], [paper_wvf["ods"]], c="orange", s=100, marker="D",
               zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate("Bagan", (250, paper_wvf["ods"]), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color="darkorange")
ax.set_xlabel("$N_p$ (support size)", fontsize=12); ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("BIPED v1: ODS vs Support Size ($N_s$=18)", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_np_full.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_ods_vs_np_full.png")

# =========================================================================
# Plot 4: LF heatmap — Np vs m
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
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn",
               vmin=0.5, vmax=np.nanmax(hm) + 0.01)
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
print("Saved plot_lf_heatmap.png")

# =========================================================================
# Plot 5: LF ODS vs m
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(range(len(lf_nps_all)))
for ci, np_val in enumerate(lf_nps_all):
    ms, ods_vals = [], []
    for r in lf:
        if r["Np"] == np_val and r["Ns"] == 18:
            ms.append(r["m"]); ods_vals.append(r["ods"])
    order = np.argsort(ms)
    ax.plot(np.array(ms)[order], np.array(ods_vals)[order], "o-",
            color=colors[ci], markersize=5, label=f"$N_p$={np_val}")
ax.set_xlabel("Line half-width $m$", fontsize=12); ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("BIPED v1: LF Effect of Line Length ($N_s$=18)", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); fig.tight_layout()
fig.savefig(OUT / "plot_lf_vs_m.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_lf_vs_m.png")

# =========================================================================
# Plot 6: WVF heatmap (Np x Ns, d=4)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
nps = sorted(set(r["Np"] for r in wvf if r["d"] == 4))
nss = sorted(set(r["Ns"] for r in wvf if r["d"] == 4))
hm = np.full((len(nps), len(nss)), np.nan)
lookup = {(r["Np"], r["Ns"]): r["ods"] for r in wvf if r["d"] == 4}
for i, np_v in enumerate(nps):
    for j, ns_v in enumerate(nss):
        if (np_v, ns_v) in lookup:
            hm[i, j] = lookup[(np_v, ns_v)]
im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn",
               vmin=0.45, vmax=np.nanmax(hm) + 0.01)
ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss)
ax.set_yticks(range(len(nps))); ax.set_yticklabels(nps)
ax.set_xlabel("$N_s$", fontsize=12); ax.set_ylabel("$N_p$", fontsize=12)
ax.set_title("BIPED v1: WVF ODS Heatmap ($d$=4)", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS")
cliff_x = nss.index(3) - 0.5 if 3 in nss else None
if cliff_x is not None:
    ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < np.nanmean(hm) else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=5.5, color=c)
fig.tight_layout()
fig.savefig(OUT / "plot_wvf_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_wvf_heatmap.png")

# =========================================================================
# Plot 7: Pipeline overview with edge maps
# =========================================================================
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image

backend = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        backend = "cpu"
except ImportError:
    backend = "cpu"

def run_cfg(cfg):
    if cfg["filter"] == "WVF":
        return wvf_image(img_gray, np_count=cfg["Np"], order=cfg["d"],
                         n_orientations=cfg["Ns"], backend=backend).gradient_mag
    else:
        return lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                        order=cfg["d"], n_orientations=cfg["Ns"],
                        backend=backend, max_vram_gb=30).gradient_mag

mag_best_wvf = run_cfg(best_wvf)
mag_best_lf = run_cfg(best_lf)
mag_paper_wvf = run_cfg(paper_wvf) if paper_wvf else None
mag_paper_lf = run_cfg(paper_lf) if paper_lf else None

panels = [
    ("Input (RGB)", img_color, {}),
    ("Grayscale", img_gray, {"cmap": "gray"}),
    (f"Ground Truth\n({np.sum(gt_bool)} edge px)", gt_bool.astype(float), {"cmap": "gray"}),
    (f"Best WVF: ODS={best_wvf['ods']:.3f}\nNp={best_wvf['Np']} Ns={best_wvf['Ns']} d={best_wvf['d']}\nP={best_wvf['precision']:.2f} R={best_wvf['recall']:.2f}",
     mag_best_wvf, {"cmap": "gray"}),
    (f"Best LF: ODS={best_lf['ods']:.3f}\nNp={best_lf['Np']} Ns={best_lf['Ns']} m={best_lf['m']}\nP={best_lf['precision']:.2f} R={best_lf['recall']:.2f}",
     mag_best_lf, {"cmap": "gray"}),
]
if mag_paper_wvf is not None:
    panels.append((f"Bagan WVF: ODS={paper_wvf['ods']:.3f}\nNp=250 d=4 Ns=18\nP={paper_wvf['precision']:.2f} R={paper_wvf['recall']:.2f}",
                    mag_paper_wvf, {"cmap": "gray"}))
if mag_paper_lf is not None:
    panels.append((f"Bagan LF: ODS={paper_lf['ods']:.3f}\nNp=250 m=14 Ns=18\nP={paper_lf['precision']:.2f} R={paper_lf['recall']:.2f}",
                    mag_paper_lf, {"cmap": "gray"}))

n = len(panels)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
for ax, (title, dat, kwargs) in zip(axes, panels):
    ax.imshow(dat, **kwargs); ax.set_title(title, fontsize=9); ax.axis("off")
fig.suptitle("BIPED v1 RGB_008 — Single-Image Ablation", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_pipeline_overview.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_pipeline_overview.png")

# =========================================================================
# Plot 8: ODS distribution histogram
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0.4, 0.88, 40)
ax.hist([r["ods"] for r in wvf], bins=bins, alpha=0.6, label=f"WVF ({len(wvf)})", color="#377eb8")
ax.hist([r["ods"] for r in lf], bins=bins, alpha=0.6, label=f"LF ({len(lf)})", color="#e41a1c")
ax.axvline(max(r["ods"] for r in wvf), color="#377eb8", linestyle="--", linewidth=1.5)
ax.axvline(max(r["ods"] for r in lf), color="#e41a1c", linestyle="--", linewidth=1.5)
if paper_wvf:
    ax.axvline(paper_wvf["ods"], color="darkorange", linestyle="-.", linewidth=1.5, label="Bagan WVF")
if paper_lf:
    ax.axvline(paper_lf["ods"], color="darkorange", linestyle=":", linewidth=1.5, label="Bagan LF")
ax.set_xlabel("ODS F-score", fontsize=12); ax.set_ylabel("Configs", fontsize=12)
ax.set_title("BIPED v1: ODS Distribution", fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(OUT / "plot_ods_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("Saved plot_ods_distribution.png")

print(f"\nAll plots saved to {OUT}")
