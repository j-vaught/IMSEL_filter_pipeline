#!/usr/bin/env python3
"""Generate all figures for Report 4: Statistical Analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "outputs" / "statistical_analysis" / "results.json"
OUT = ROOT / "papers" / "report4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    results = json.load(f)

# ---------- style constants ----------
COLORS = ["black", "#444444", "#888888", "#BBBBBB"]
HATCH_PATTERNS = ["/", "\\", "|", "x"]
MARKERS = ["o", "s", "D", "^"]
LINESTYLES = ["-", "--", ":", "-."]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


def save(fig, name, dpi=300):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  saved {name}")


# ============================================================
# 1. Spearman heatmap  (4x4)
# ============================================================
def fig_spearman_heatmap():
    datasets = ["UDED", "BIPED_v1", "BIPED_v2", "BSDS500"]
    n = len(datasets)
    mat = np.eye(n)
    pairs = results["ranking_consistency"]["pairwise_spearman"]
    idx = {d: i for i, d in enumerate(datasets)}
    for p in pairs:
        i, j = idx[p["ds1"]], idx[p["ds2"]]
        mat[i, j] = mat[j, i] = p["rho"]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    from matplotlib.colors import LinearSegmentedColormap
    cmap_light = LinearSegmentedColormap.from_list("light_gray", ["#FFFFFF", "#666666"])
    im = ax.imshow(mat, cmap=cmap_light, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    labels = ["UDED", "BIPED v1", "BIPED v2", "BSDS500"]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Pairwise Spearman $\\rho$", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="$\\rho$")
    save(fig, "fig_spearman_heatmap.png")


# ============================================================
# 2. Optimal vs Bagan gaps  (grouped bars)
# ============================================================
def fig_optimal_vs_bagan_gaps():
    datasets = ["UDED", "BIPED_v1", "BIPED_v2", "BSDS500"]
    labels = ["UDED", "BIPED v1", "BIPED v2", "BSDS500"]
    ovb = results["optimal_vs_bagan"]
    wvf_gaps = [ovb[d]["wvf_gap"] for d in datasets]
    lf_gaps = [ovb[d]["lf_gap"] for d in datasets]

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars1 = ax.bar(x - w / 2, wvf_gaps, w, color="#888888", edgecolor="black",
                   hatch="/", linewidth=0.5, label="WVF gap")
    bars2 = ax.bar(x + w / 2, lf_gaps, w, color="#CCCCCC", edgecolor="black",
                   hatch="\\\\", linewidth=0.5, label="LF gap")

    ax.set_ylabel("ODS gap (best $-$ Bagan)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_ylim(0, max(lf_gaps) * 1.15)
    # annotate values
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6)
    save(fig, "fig_optimal_vs_bagan_gaps.png")


# ============================================================
# 3. Parameter sensitivity (horizontal bars of eta^2)
# ============================================================
def fig_parameter_sensitivity():
    ps = results["parameter_sensitivity"]
    params = ["LF_m", "Np", "Ns", "d"]
    labels = ["LF $m$", "$N_p$", "$N_s$", "$d$"]
    eta2 = [ps[p]["eta_squared"] for p in params]

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    y = np.arange(len(params))
    for i, (val, lab) in enumerate(zip(eta2, labels)):
        ax.barh(i, val, color=COLORS[i], edgecolor="black",
                hatch=HATCH_PATTERNS[i], linewidth=0.5)
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("$\\eta^2$ (effect size)")
    ax.set_title("Kruskal-Wallis effect sizes", fontsize=9)
    ax.set_xlim(0, max(eta2) * 1.25)
    save(fig, "fig_parameter_sensitivity.png")


# ============================================================
# 4. WVF vs DL table-heatmap
# ============================================================
def fig_wvf_vs_dl_table():
    wvd = results["wvf_vs_dl"]
    datasets = ["UDED", "BIPED_v1", "BIPED_v2", "BSDS500"]
    ds_labels = ["UDED", "BIPED v1", "BIPED v2", "BSDS500"]
    methods = ["best_wvf", "dexined", "teed", "pidinet", "nbed", "diffusionedge"]
    method_labels = ["WVF (best)", "DexiNed", "TEED", "PiDiNet", "NBED", "DiffusionEdge"]

    n_ds = len(datasets)
    n_m = len(methods)
    mat = np.full((n_ds, n_m), np.nan)
    for i, ds in enumerate(datasets):
        mat[i, 0] = wvd[ds]["best_wvf"]
        dl = wvd[ds]["dl"]
        for j, m in enumerate(methods[1:], 1):
            if m in dl:
                mat[i, j] = dl[m]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # mask NaN for display
    masked = np.ma.masked_invalid(mat)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_light = LinearSegmentedColormap.from_list("light_gray", ["#FFFFFF", "#666666"])
    cmap_light.set_bad(color="#EEEEEE")
    im = ax.imshow(masked, cmap=cmap_light, vmin=0.3, vmax=1.0, aspect="auto")

    for i in range(n_ds):
        wvf_val = mat[i, 0]
        for j in range(n_m):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=6,
                        color="black", style="italic")
            else:
                txt = f"{mat[i, j]:.3f}"
                if j > 0 and not np.isnan(mat[i, j]):
                    if mat[i, j] > wvf_val:
                        txt += "*"  # DL wins
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="black")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(method_labels, fontsize=6, rotation=35, ha="right")
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels(ds_labels, fontsize=7)
    ax.set_title("ODS comparison (* = DL wins)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ODS")
    save(fig, "fig_wvf_vs_dl_table.png")


# ============================================================
# 5. Median ODS by parameter (2x2 grid)
# ============================================================
def fig_median_ods_by_param():
    ps = results["parameter_sensitivity"]
    params = ["Np", "Ns", "d", "LF_m"]
    titles = ["$N_p$", "$N_s$", "$d$", "LF $m$"]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
    axes = axes.ravel()
    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]
        levels = ps[param]["median_ods_per_level"]
        xs = [int(k) for k in levels.keys()]
        ys = [levels[k] for k in levels.keys()]
        ax.plot(xs, ys, marker="o", color="black", linestyle="-",
                markersize=4, linewidth=0.8)
        ax.set_xlabel(title, fontsize=8)
        ax.set_ylabel("Median ODS", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Sensitivity to {title}", fontsize=9)
        # use log scale for Np and Ns
        if param in ("Np", "Ns"):
            ax.set_xscale("log")
    fig.tight_layout()
    save(fig, "fig_median_ods_by_param.png")


# ============================================================
# 6. Crossover heatmap
# ============================================================
def fig_crossover_heatmap():
    nc = results["noise_crossover"]
    noise_types = ["gaussian", "salt_pepper", "poisson", "speckle", "uniform"]
    noise_labels = ["Gaussian", "Salt & Pepper", "Poisson", "Speckle", "Uniform"]
    models = ["dexined", "teed", "pidinet", "nbed", "diffusionedge"]
    model_labels = ["DexiNed", "TEED", "PiDiNet", "NBED", "DiffusionEdge"]

    mat = np.zeros((len(noise_types), len(models)))
    for i, nt in enumerate(noise_types):
        for j, m in enumerate(models):
            mat[i, j] = nc[nt][m]["median_crossover_snr"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap_light = LinearSegmentedColormap.from_list("light_gray", ["#FFFFFF", "#666666"])
    im = ax.imshow(mat, cmap=cmap_light, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(noise_types)):
        for j in range(len(models)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=6, rotation=30, ha="right")
    ax.set_yticks(range(len(noise_types)))
    ax.set_yticklabels(noise_labels, fontsize=7)
    ax.set_title("Median crossover SNR", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="SNR")
    save(fig, "fig_crossover_heatmap.png")


# ============================================================
if __name__ == "__main__":
    print("Generating Report 4 figures...")
    fig_spearman_heatmap()
    fig_optimal_vs_bagan_gaps()
    fig_parameter_sensitivity()
    fig_wvf_vs_dl_table()
    fig_median_ods_by_param()
    fig_crossover_heatmap()
    print("Done.")
