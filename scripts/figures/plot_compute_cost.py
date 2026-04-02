"""Visualize computational cost vs performance for WVF/LF configurations."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "biped_ablation"

with open(OUT / "ablation_metrics.json") as f:
    data = json.load(f)

wvf = data["wvf"]
lf = data["lf"]
H, W = data["image_shape"]
n_pixels = H * W


def flops_per_pixel(cfg):
    """Multiply-adds per pixel for a config."""
    d = cfg["d"]
    M = (d + 1) * (d + 2) // 2  # monomial terms
    Np = cfg["Np"]
    Ns = cfg["Ns"]
    # Pseudoinverse precompute is negligible per-pixel
    # Per pixel: Ns orientations × dot product of Np values with M-row vector
    # = Ns × Np (gather) + Ns × Np (matmul with P_fx row)
    cost = Ns * Np * 2  # gather + multiply
    if cfg["filter"] == "LF":
        L = 2 * cfg["m"] + 1
        cost *= L  # repeated for each line point
        cost += Ns * L  # weighted sum
    return cost


def total_flops(cfg):
    return flops_per_pixel(cfg) * n_pixels


# Add cost to each config
for r in wvf + lf:
    r["flops_per_pixel"] = flops_per_pixel(r)
    r["total_gflops"] = total_flops(r) / 1e9

# =========================================================================
# Plot 1: ODS vs Compute — two panels, WVF and LF separate
# =========================================================================
best_wvf = max(wvf, key=lambda x: x["ods"])
best_lf = max(lf, key=lambda x: x["ods"])
paper_wvf = next((r for r in wvf if r["Np"]==250 and r["d"]==4 and r["Ns"]==18), None)
paper_lf = next((r for r in lf if r["Np"]==250 and r["m"]==14 and r["Ns"]==18), None)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: WVF colored by d
ax = axes[0]
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
for d in sorted(set(r["d"] for r in wvf)):
    subset = [r for r in wvf if r["d"] == d]
    ax.scatter([r["total_gflops"] for r in subset], [r["ods"] for r in subset],
               s=20, alpha=0.5, c=d_colors.get(d, "#666"), label=f"d={d}", zorder=2)

# WVF Pareto frontier
wvf_sorted = sorted(wvf, key=lambda r: r["total_gflops"])
px, py, best = [], [], 0
for r in wvf_sorted:
    if r["ods"] > best:
        best = r["ods"]; px.append(r["total_gflops"]); py.append(r["ods"])
ax.step(px, py, where="post", color="black", linewidth=2, label="Pareto frontier", zorder=3)

ax.scatter([best_wvf["total_gflops"]], [best_wvf["ods"]], c="red", s=150,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.annotate(f"Best: Np={best_wvf['Np']} d={best_wvf['d']} Ns={best_wvf['Ns']}\n{best_wvf['total_gflops']:.2f} GFLOP, ODS={best_wvf['ods']:.3f}",
            (best_wvf["total_gflops"], best_wvf["ods"]),
            textcoords="offset points", xytext=(15, -10), fontsize=7)

if paper_wvf:
    ax.scatter([paper_wvf["total_gflops"]], [paper_wvf["ods"]], c="darkorange", s=120,
               marker="D", zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Bagan: Np=250 d=4 Ns=18\n{paper_wvf['total_gflops']:.1f} GFLOP, ODS={paper_wvf['ods']:.3f}",
                (paper_wvf["total_gflops"], paper_wvf["ods"]),
                textcoords="offset points", xytext=(10, -15), fontsize=7, color="darkorange")

ax.set_xscale("log")
ax.set_xlabel("Total compute per image (GFLOP)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("WVF: Quality vs Cost", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right: LF colored by m
ax = axes[1]
lf_ms = sorted(set(r["m"] for r in lf))
lf_colors = plt.cm.tab10(np.linspace(0, 1, len(lf_ms)))
for ci, m in enumerate(lf_ms):
    subset = [r for r in lf if r["m"] == m]
    ax.scatter([r["total_gflops"] for r in subset], [r["ods"] for r in subset],
               s=20, alpha=0.6, c=[lf_colors[ci]], label=f"m={m}", zorder=2)

# LF Pareto frontier
lf_sorted = sorted(lf, key=lambda r: r["total_gflops"])
px, py, best = [], [], 0
for r in lf_sorted:
    if r["ods"] > best:
        best = r["ods"]; px.append(r["total_gflops"]); py.append(r["ods"])
ax.step(px, py, where="post", color="black", linewidth=2, label="Pareto frontier", zorder=3)

ax.scatter([best_lf["total_gflops"]], [best_lf["ods"]], c="red", s=150,
           marker="*", zorder=5, edgecolors="black", linewidths=0.5)
ax.annotate(f"Best: Np={best_lf['Np']} m={best_lf['m']} Ns={best_lf['Ns']}\n{best_lf['total_gflops']:.1f} GFLOP, ODS={best_lf['ods']:.3f}",
            (best_lf["total_gflops"], best_lf["ods"]),
            textcoords="offset points", xytext=(15, -10), fontsize=7)

if paper_lf:
    ax.scatter([paper_lf["total_gflops"]], [paper_lf["ods"]], c="darkorange", s=120,
               marker="D", zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(f"Bagan: Np=250 m=14 Ns=18\n{paper_lf['total_gflops']:.1f} GFLOP, ODS={paper_lf['ods']:.3f}",
                (paper_lf["total_gflops"], paper_lf["ods"]),
                textcoords="offset points", xytext=(-80, -20), fontsize=7, color="darkorange")

ax.set_xscale("log")
ax.set_xlabel("Total compute per image (GFLOP)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("LF: Quality vs Cost", fontsize=13)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

fig.suptitle(f"BIPED v1: Edge Quality vs Computational Cost ({H}×{W} image)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_compute.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_vs_compute.png")

# =========================================================================
# Plot 2: WVF compute heatmap — Np × Ns (d=4)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: FLOPS per pixel
ax = axes[0]
nps = sorted(set(r["Np"] for r in wvf if r["d"] == 4))
nss = sorted(set(r["Ns"] for r in wvf if r["d"] == 4))
hm = np.full((len(nps), len(nss)), np.nan)
for r in wvf:
    if r["d"] == 4 and r["Np"] in nps and r["Ns"] in nss:
        i = nps.index(r["Np"])
        j = nss.index(r["Ns"])
        hm[i, j] = r["flops_per_pixel"]

im = ax.imshow(hm, aspect="auto", origin="lower", cmap="YlOrRd",
               norm=LogNorm(vmin=np.nanmin(hm), vmax=np.nanmax(hm)))
ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss, fontsize=8)
ax.set_yticks(range(len(nps))); ax.set_yticklabels(nps, fontsize=8)
ax.set_xlabel("$N_s$ (orientations)", fontsize=11)
ax.set_ylabel("$N_p$ (support size)", fontsize=11)
ax.set_title("FLOPs per pixel (log scale)", fontsize=12)
fig.colorbar(im, ax=ax, label="FLOPs/pixel", shrink=0.8)
for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm[i, j]):
            val = hm[i, j]
            label = f"{val:.0f}" if val < 10000 else f"{val/1000:.0f}k"
            c = "white" if val > np.nanmedian(hm) else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=5, color=c)

# Right: ODS (same grid for comparison)
ax = axes[1]
hm_ods = np.full((len(nps), len(nss)), np.nan)
for r in wvf:
    if r["d"] == 4 and r["Np"] in nps and r["Ns"] in nss:
        i = nps.index(r["Np"])
        j = nss.index(r["Ns"])
        hm_ods[i, j] = r["ods"]

im2 = ax.imshow(hm_ods, aspect="auto", origin="lower", cmap="RdYlGn",
                vmin=0.45, vmax=np.nanmax(hm_ods) + 0.01)
ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss, fontsize=8)
ax.set_yticks(range(len(nps))); ax.set_yticklabels(nps, fontsize=8)
ax.set_xlabel("$N_s$ (orientations)", fontsize=11)
ax.set_ylabel("$N_p$ (support size)", fontsize=11)
ax.set_title("ODS F-score", fontsize=12)
fig.colorbar(im2, ax=ax, label="ODS", shrink=0.8)
cliff_x = nss.index(3) - 0.5 if 3 in nss else None
if cliff_x is not None:
    ax.axvline(x=cliff_x, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
for i in range(len(nps)):
    for j in range(len(nss)):
        if not np.isnan(hm_ods[i, j]):
            c = "white" if hm_ods[i, j] < np.nanmean(hm_ods) else "black"
            ax.text(j, i, f"{hm_ods[i, j]:.3f}", ha="center", va="center", fontsize=5, color=c)

fig.suptitle(f"WVF ($d$=4): Compute Cost vs Quality ({H}×{W} image)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_compute_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_compute_heatmap.png")

# =========================================================================
# Plot 3: Efficiency — ODS per GFLOP
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

# WVF by d at Ns=18
d_colors = {2: "#e41a1c", 3: "#377eb8", 4: "#4daf4a", 5: "#984ea3"}
for d in sorted(set(r["d"] for r in wvf)):
    xs, ys = [], []
    for r in wvf:
        if r["d"] == d and r["Ns"] == 18:
            xs.append(r["total_gflops"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=d_colors.get(d, "#666"), markersize=5, label=f"WVF d={d}")

# Best LF envelope
lf_nps = sorted(set(r["Np"] for r in lf))
for np_val in lf_nps:
    best = max((r for r in lf if r["Np"] == np_val and r["Ns"] == 18), key=lambda x: x["ods"], default=None)
    if best:
        ax.scatter([best["total_gflops"]], [best["ods"]], c="darkorange", s=40,
                   marker="s", zorder=4)
ax.scatter([], [], c="darkorange", s=40, marker="s", label="LF (best m, $N_s$=18)")

if paper_wvf:
    ax.scatter([paper_wvf["total_gflops"]], [paper_wvf["ods"]], c="darkorange", s=100,
               marker="D", zorder=5, edgecolors="black")
    ax.annotate("Bagan WVF", (paper_wvf["total_gflops"], paper_wvf["ods"]),
                textcoords="offset points", xytext=(8, -12), fontsize=8, color="darkorange")
if paper_lf:
    ax.scatter([paper_lf["total_gflops"]], [paper_lf["ods"]], c="darkorange", s=100,
               marker="D", zorder=5, edgecolors="black")
    ax.annotate("Bagan LF", (paper_lf["total_gflops"], paper_lf["ods"]),
                textcoords="offset points", xytext=(8, 5), fontsize=8, color="darkorange")

ax.set_xscale("log")
ax.set_xlabel("Total compute per image (GFLOP)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title(f"BIPED v1: Quality vs Cost ($N_s$=18, {H}×{W})", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_cost_by_d.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved plot_ods_vs_cost_by_d.png")

# =========================================================================
# Print summary table
# =========================================================================
print("\n=== Cost Summary ===")
print(f"{'Config':<35} {'GFLOP':>8} {'ODS':>6} {'ODS/GFLOP':>10}")
print("-" * 65)
for label, cfg in [("Best WVF", best_wvf), ("Best LF", best_lf),
                    ("Bagan WVF (Np=250 d=4)", paper_wvf),
                    ("Bagan LF (Np=250 m=14)", paper_lf)]:
    if cfg:
        eff = cfg["ods"] / cfg["total_gflops"]
        print(f"{label:<35} {cfg['total_gflops']:>8.1f} {cfg['ods']:>6.4f} {eff:>10.4f}")

# Cheapest config that beats Bagan's best
bagan_best_ods = max(paper_wvf["ods"] if paper_wvf else 0,
                      paper_lf["ods"] if paper_lf else 0)
cheaper_and_better = [r for r in all_configs
                       if r["ods"] >= bagan_best_ods and r["total_gflops"] < (paper_wvf["total_gflops"] if paper_wvf else 1e9)]
if cheaper_and_better:
    cheapest = min(cheaper_and_better, key=lambda r: r["total_gflops"])
    print(f"\nCheapest config that beats Bagan's best ODS ({bagan_best_ods:.4f}):")
    print(f"  {cheapest['filter']} Np={cheapest['Np']} Ns={cheapest['Ns']} d={cheapest.get('d')} m={cheapest.get('m','-')}")
    print(f"  ODS={cheapest['ods']:.4f}, {cheapest['total_gflops']:.1f} GFLOP")
    print(f"  {paper_wvf['total_gflops']/cheapest['total_gflops']:.0f}x cheaper than Bagan's WVF")
