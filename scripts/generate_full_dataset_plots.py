"""Generate plots for the full dataset ablation results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "full_dataset_ablation"

DATASETS = ["uded", "biped_v1", "biped_v2", "bsds500"]
LABELS = {"uded": "UDED", "biped_v1": "BIPED v1", "biped_v2": "BIPED v2", "bsds500": "BSDS500"}

# Load all results
data = {}
for ds in DATASETS:
    p = OUT / f"{ds}_ablation.json"
    if p.exists():
        with open(p) as f:
            data[ds] = json.load(f)
        print(f"Loaded {ds}: {len(data[ds]['wvf'])} WVF + {len(data[ds]['lf'])} LF")

# =========================================================================
# Plot 1: Cross-dataset comparison bar chart
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ds_names = [LABELS[ds] for ds in DATASETS if ds in data]
x = np.arange(len(ds_names))
width = 0.18

bars = []
for ds in DATASETS:
    if ds not in data:
        continue
    d = data[ds]
    best_wvf = max(d["wvf"], key=lambda r: r["ods"])
    best_lf = max((r for r in d["lf"] if r["ods"] >= 0), key=lambda r: r["ods"]) if d["lf"] else None
    paper_wvf = next((r for r in d["wvf"] if r["Np"] == 250 and r["d"] == 4 and r["Ns"] == 18), None)
    paper_lf = next((r for r in d["lf"] if r["Np"] == 250 and r.get("m") == 14 and r["Ns"] == 18 and r["ods"] >= 0), None)
    bars.append({
        "best_wvf": best_wvf["ods"],
        "best_lf": best_lf["ods"] if best_lf else 0,
        "bagan_wvf": paper_wvf["ods"] if paper_wvf else 0,
        "bagan_lf": paper_lf["ods"] if paper_lf else 0,
        "best_wvf_cfg": f"Np={best_wvf['Np']} d={best_wvf['d']}",
        "best_lf_cfg": f"Np={best_lf['Np']} m={best_lf['m']}" if best_lf else "",
    })

ax.bar(x - 1.5*width, [b["best_wvf"] for b in bars], width, label="Best WVF", color="#377eb8")
ax.bar(x - 0.5*width, [b["best_lf"] for b in bars], width, label="Best LF", color="#4daf4a")
ax.bar(x + 0.5*width, [b["bagan_wvf"] for b in bars], width, label="Bagan WVF", color="#ff7f00", alpha=0.7)
ax.bar(x + 1.5*width, [b["bagan_lf"] for b in bars], width, label="Bagan LF", color="#e41a1c", alpha=0.7)

# Value labels
for i, b in enumerate(bars):
    ax.text(i - 1.5*width, b["best_wvf"] + 0.005, f"{b['best_wvf']:.3f}", ha="center", fontsize=7)
    ax.text(i - 0.5*width, b["best_lf"] + 0.005, f"{b['best_lf']:.3f}", ha="center", fontsize=7)
    ax.text(i + 0.5*width, b["bagan_wvf"] + 0.005, f"{b['bagan_wvf']:.3f}", ha="center", fontsize=7)
    ax.text(i + 1.5*width, b["bagan_lf"] + 0.005, f"{b['bagan_lf']:.3f}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=12)
ax.set_ylabel("Dataset-wide ODS", fontsize=12)
ax.set_title("Full Dataset Ablation: Best Configs vs Bagan's Parameters", fontsize=14)
ax.legend(fontsize=10)
ax.set_ylim(0.5, 0.95)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "plot_cross_dataset_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_cross_dataset_comparison.png")

# =========================================================================
# Plot 2: ODS vs Np for each dataset (WVF d=2, Ns=18)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for di, ds in enumerate([d for d in DATASETS if d in data]):
    ax = axes[di]
    d = data[ds]
    d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
    for order in sorted(set(r["d"] for r in d["wvf"])):
        xs, ys = [], []
        for r in d["wvf"]:
            if r["d"] == order and r["Ns"] == 18:
                xs.append(r["Np"]); ys.append(r["ods"])
        if xs:
            idx = np.argsort(xs)
            ax.plot(np.array(xs)[idx], np.array(ys)[idx], "o-",
                    color=d_colors.get(order, "#666"), markersize=4, label=f"d={order}")

    paper_wvf = next((r for r in d["wvf"] if r["Np"] == 250 and r["d"] == 4 and r["Ns"] == 18), None)
    if paper_wvf:
        ax.scatter([250], [paper_wvf["ods"]], c="darkorange", s=80, marker="D",
                   zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate("Bagan", (250, paper_wvf["ods"]), textcoords="offset points",
                    xytext=(8, -10), fontsize=7, color="darkorange")

    ax.set_xlabel("$N_p$", fontsize=11)
    ax.set_ylabel("ODS", fontsize=11)
    ax.set_title(f"{LABELS[ds]} ({d['n_images']} images)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("WVF: ODS vs $N_p$ ($N_s$=18) — All Datasets", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_np_all_datasets.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_vs_np_all_datasets.png")

# =========================================================================
# Plot 3: LF ODS vs m for each dataset (Ns=18)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for di, ds in enumerate([d for d in DATASETS if d in data]):
    ax = axes[di]
    d = data[ds]
    lf_nps = sorted(set(r["Np"] for r in d["lf"]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(lf_nps)))
    for ci, np_val in enumerate(lf_nps):
        xs, ys = [], []
        for r in d["lf"]:
            if r["Np"] == np_val and r["Ns"] == 18 and r["ods"] >= 0:
                xs.append(r["m"]); ys.append(r["ods"])
        if xs:
            idx = np.argsort(xs)
            ax.plot(np.array(xs)[idx], np.array(ys)[idx], "o-",
                    color=colors[ci], markersize=4, label=f"$N_p$={np_val}")

    ax.set_xlabel("Line half-width $m$", fontsize=11)
    ax.set_ylabel("ODS", fontsize=11)
    ax.set_title(f"{LABELS[ds]} ({d['n_images']} images)", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle("LF: ODS vs $m$ ($N_s$=18) — All Datasets", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_m_all_datasets.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_vs_m_all_datasets.png")

# =========================================================================
# Plot 4: Ns cliff for each dataset
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for di, ds in enumerate([d for d in DATASETS if d in data]):
    ax = axes[di]
    d = data[ds]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    for ci, np_val in enumerate([25, 50, 100, 250]):
        ns_vals, ods_vals = [], []
        for r in d["wvf"]:
            if r["Np"] == np_val and r["d"] == 4:
                ns_vals.append(r["Ns"]); ods_vals.append(r["ods"])
        if ns_vals:
            idx = np.argsort(ns_vals)
            ax.plot(np.array(ns_vals)[idx], np.array(ods_vals)[idx], "o-",
                    color=colors[ci], markersize=5, label=f"$N_p$={np_val}")

    import matplotlib.ticker
    ax.axvline(x=3, color="red", linestyle=":", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 3, 4, 6, 9, 12, 18, 36, 72, 180])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("$N_s$", fontsize=11)
    ax.set_ylabel("ODS", fontsize=11)
    ax.set_title(f"{LABELS[ds]}", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("WVF ($d$=4): Orientation Cliff — All Datasets", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_ns_cliff_all_datasets.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ns_cliff_all_datasets.png")

# =========================================================================
# Plot 5: Best config summary table as figure
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

headers = ["Dataset", "Images", "Best WVF ODS", "Best WVF Config",
           "Best LF ODS", "Best LF Config", "Bagan WVF", "Bagan LF"]
rows = []
for ds in DATASETS:
    if ds not in data:
        continue
    d = data[ds]
    bw = max(d["wvf"], key=lambda r: r["ods"])
    bl = max((r for r in d["lf"] if r["ods"] >= 0), key=lambda r: r["ods"]) if d["lf"] else None
    pw = next((r for r in d["wvf"] if r["Np"] == 250 and r["d"] == 4 and r["Ns"] == 18), None)
    pl = next((r for r in d["lf"] if r["Np"] == 250 and r.get("m") == 14 and r["Ns"] == 18 and r["ods"] >= 0), None)
    rows.append([
        LABELS[ds],
        str(d["n_images"]),
        f"{bw['ods']:.4f}",
        f"Np={bw['Np']} d={bw['d']} Ns={bw['Ns']}",
        f"{bl['ods']:.4f}" if bl else "---",
        f"Np={bl['Np']} m={bl['m']} Ns={bl['Ns']}" if bl else "---",
        f"{pw['ods']:.4f}" if pw else "---",
        f"{pl['ods']:.4f}" if pl else "---",
    ])

table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Color code: green for best, orange for Bagan
for i in range(len(rows)):
    table[i+1, 2].set_facecolor("#d4edda")  # best WVF
    table[i+1, 4].set_facecolor("#d4edda")  # best LF
    table[i+1, 6].set_facecolor("#fff3cd")  # bagan WVF
    table[i+1, 7].set_facecolor("#f8d7da")  # bagan LF

fig.suptitle("Full Dataset Ablation — Summary", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "plot_summary_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_summary_table.png")

print(f"\nAll plots saved to {OUT}")
