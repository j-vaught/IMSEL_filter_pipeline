#!/usr/bin/env python3
"""Generate all figures for Report 1: Parameter Optimality of the Wide View
and Line Filters.

Reads ablation data from outputs/ and saves publication-quality PNG figures
to papers/report1/figures/.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSDS_FULL = os.path.join(ROOT, "outputs", "full_dataset_ablation", "bsds500_ablation.json")
SUMMARY   = os.path.join(ROOT, "outputs", "full_dataset_ablation", "summary.json")
CANNY_PP  = os.path.join(ROOT, "outputs", "biped_ablation", "canny_postprocess_test.json")
FIG_DIR   = os.path.join(ROOT, "papers", "report1", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Global style
# --------------------------------------------------------------------------- #
IEEE_COL = 3.5   # single-column width in inches
IEEE_FULL = 7.16  # two-column width
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# --------------------------------------------------------------------------- #
# Consistent style mappings (grayscale-friendly)
# --------------------------------------------------------------------------- #
GRAY_PALETTE = {
    "black":      "#000000",
    "dark_gray":  "#444444",
    "med_gray":   "#888888",
    "light_gray": "#BBBBBB",
}

# Polynomial order d mappings
D_STYLE = {
    2: {"ls": "-",  "marker": "o", "color": GRAY_PALETTE["black"],     "label": "$d = 2$"},
    3: {"ls": "--", "marker": "s", "color": GRAY_PALETTE["dark_gray"], "label": "$d = 3$"},
    4: {"ls": ":",  "marker": "D", "color": GRAY_PALETTE["med_gray"],  "label": "$d = 4$"},
    5: {"ls": "-.", "marker": "^", "color": GRAY_PALETTE["light_gray"],"label": "$d = 5$"},
}

# Dataset mappings
DS_STYLE = {
    "BSDS500":  {"ls": "-",  "marker": "o"},
    "BIPED_v1": {"ls": "--", "marker": "s"},
    "BIPED_v2": {"ls": ":",  "marker": "D"},
    "UDED":     {"ls": "-.", "marker": "^"},
}

# Filter hatching for bar charts
FILTER_HATCH = {
    "best_wvf":  "/",
    "best_lf":   "\\",
    "bagan_wvf": "|",
    "bagan_lf":  "x",
}

MARKER_SIZE = 4.0
LINE_WIDTH  = 1.2

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
with open(BSDS_FULL) as f:
    bsds = json.load(f)
with open(SUMMARY) as f:
    summary = json.load(f)
with open(CANNY_PP) as f:
    canny_pp = json.load(f)

wvf_all = bsds["wvf"]
lf_all  = bsds["lf"]

# --------------------------------------------------------------------------- #
# Helper: index WVF results
# --------------------------------------------------------------------------- #
def wvf_lookup():
    """Return dict keyed by (Np, Ns, d) -> ods."""
    d = {}
    for r in wvf_all:
        d[(r["Np"], r["Ns"], r["d"])] = r["ods"]
    return d

wvf_idx = wvf_lookup()

# Unique sorted parameter values
np_vals = sorted(set(r["Np"] for r in wvf_all))
ns_vals = sorted(set(r["Ns"] for r in wvf_all))
d_vals  = sorted(set(r["d"]  for r in wvf_all))

# =========================================================================== #
# Figure 1: Heatmap of WVF ODS vs Np and Ns, fixed d=2
# =========================================================================== #
def fig_wvf_heatmap():
    d_fixed = 2
    # Build matrix: rows = Np (y-axis), cols = Ns (x-axis)
    mat = np.full((len(np_vals), len(ns_vals)), np.nan)
    for i, np_v in enumerate(np_vals):
        for j, ns_v in enumerate(ns_vals):
            key = (np_v, ns_v, d_fixed)
            if key in wvf_idx:
                mat[i, j] = wvf_idx[key]

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap_light = LinearSegmentedColormap.from_list("light_gray", ["#FFFFFF", "#666666"])
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap=cmap_light, interpolation="nearest")
    ax.set_xticks(range(len(ns_vals)))
    ax.set_xticklabels(ns_vals, rotation=45, ha="right")
    ax.set_yticks(range(len(np_vals)))
    ax.set_yticklabels(np_vals)
    ax.set_xlabel(r"$N_s$ (sectors)")
    ax.set_ylabel(r"$N_p$ (profile points)")
    ax.set_title(r"WVF ODS on BSDS500 ($d\!=\!2$)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("ODS F-measure")
    fig.savefig(os.path.join(FIG_DIR, "fig_wvf_heatmap_np_ns.png"))
    plt.close(fig)
    print("  saved fig_wvf_heatmap_np_ns.png")

# =========================================================================== #
# Figure 2: ODS vs Np for multiple d values (Ns=18)
# =========================================================================== #
def fig_wvf_ods_vs_np():
    ns_fixed = 18
    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.4))
    for d_v in d_vals:
        sty = D_STYLE.get(d_v, D_STYLE[2])
        xs, ys = [], []
        for np_v in np_vals:
            key = (np_v, ns_fixed, d_v)
            if key in wvf_idx:
                xs.append(np_v)
                ys.append(wvf_idx[key])
        ax.plot(xs, ys, marker=sty["marker"], markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, linestyle=sty["ls"],
                color=sty["color"], label=sty["label"])
    ax.set_xlabel(r"$N_p$ (profile points)")
    ax.set_ylabel("ODS F-measure")
    ax.set_title(r"WVF ODS vs $N_p$ on BSDS500 ($N_s\!=\!18$)")
    ax.set_xscale("log")
    ax.legend(framealpha=0.8)
    fig.savefig(os.path.join(FIG_DIR, "fig_wvf_ods_vs_np.png"))
    plt.close(fig)
    print("  saved fig_wvf_ods_vs_np.png")

# =========================================================================== #
# Figure 3: LF ODS vs half-width m for different Np values
# =========================================================================== #
def fig_lf_ods_vs_m():
    lf_idx = defaultdict(dict)
    for r in lf_all:
        lf_idx[r["Np"]][r["m"]] = r["ods"]

    lf_np_vals = sorted(set(r["Np"] for r in lf_all))
    m_vals     = sorted(set(r["m"]  for r in lf_all))

    # Use grayscale palette cycling for multiple Np values
    gray_cycle = [
        ("-",  "o", GRAY_PALETTE["black"]),
        ("--", "s", GRAY_PALETTE["dark_gray"]),
        (":",  "D", GRAY_PALETTE["med_gray"]),
        ("-.", "^", GRAY_PALETTE["light_gray"]),
        ("-",  "v", GRAY_PALETTE["dark_gray"]),
        ("--", "P", GRAY_PALETTE["med_gray"]),
        (":",  "X", GRAY_PALETTE["black"]),
    ]

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.4))
    for idx, np_v in enumerate(lf_np_vals):
        ls, mk, col = gray_cycle[idx % len(gray_cycle)]
        xs, ys = [], []
        for m_v in m_vals:
            if m_v in lf_idx[np_v]:
                xs.append(m_v)
                ys.append(lf_idx[np_v][m_v])
        ax.plot(xs, ys, marker=mk, markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, linestyle=ls, color=col,
                label=f"$N_p = {np_v}$")
    ax.set_xlabel("Half-width $m$")
    ax.set_ylabel("ODS F-measure")
    ax.set_title("LF ODS vs half-width $m$ on BSDS500")
    ax.legend(framealpha=0.8, ncol=2)
    fig.savefig(os.path.join(FIG_DIR, "fig_lf_ods_vs_m.png"))
    plt.close(fig)
    print("  saved fig_lf_ods_vs_m.png")

# =========================================================================== #
# Figure 4: Cross-dataset grouped bar chart
# =========================================================================== #
def fig_cross_dataset_best():
    datasets = ["UDED", "BIPED_v1", "BIPED_v2", "BSDS500"]
    labels   = ["UDED", "BIPED v1", "BIPED v2", "BSDS500"]
    categories = ["best_wvf", "best_lf", "bagan_wvf", "bagan_lf"]
    cat_labels = ["Best WVF", "Best LF", "Bagan WVF", "Bagan LF"]

    bar_grays   = [GRAY_PALETTE["black"], GRAY_PALETTE["dark_gray"],
                   GRAY_PALETTE["med_gray"], GRAY_PALETTE["light_gray"]]
    bar_hatches = [FILTER_HATCH["best_wvf"], FILTER_HATCH["best_lf"],
                   FILTER_HATCH["bagan_wvf"], FILTER_HATCH["bagan_lf"]]

    n_ds = len(datasets)
    n_cat = len(categories)
    x = np.arange(n_ds)
    width = 0.18

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.6))
    for i, (cat, clabel) in enumerate(zip(categories, cat_labels)):
        vals = [summary[ds][cat]["ods"] for ds in datasets]
        offset = (i - (n_cat - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=clabel,
                      color=bar_grays[i], hatch=bar_hatches[i],
                      edgecolor="white", linewidth=0.3)
        # Add value labels on top of bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ODS F-measure")
    ax.set_title("Optimal vs Bagan Parameters Across Datasets")
    ax.set_ylim(0.55, 0.96)
    ax.legend(framealpha=0.8, loc="upper right", fontsize=6)
    fig.savefig(os.path.join(FIG_DIR, "fig_cross_dataset_best.png"))
    plt.close(fig)
    print("  saved fig_cross_dataset_best.png")

# =========================================================================== #
# Figure 5: Ns saturation plot
# =========================================================================== #
def fig_ns_saturation():
    d_fixed = 2
    best_np_candidates = [25, 50, 100]

    # Cycle through grayscale styles for the three Np candidates
    np_styles = [
        ("-",  "o", GRAY_PALETTE["black"]),
        ("--", "s", GRAY_PALETTE["dark_gray"]),
        (":",  "D", GRAY_PALETTE["med_gray"]),
    ]

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.4))
    for idx, np_v in enumerate(best_np_candidates):
        ls, mk, col = np_styles[idx]
        xs, ys = [], []
        for ns_v in ns_vals:
            key = (np_v, ns_v, d_fixed)
            if key in wvf_idx:
                xs.append(ns_v)
                ys.append(wvf_idx[key])
        ax.plot(xs, ys, marker=mk, markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, linestyle=ls, color=col,
                label=f"$N_p = {np_v}$")

    ax.set_xlabel(r"$N_s$ (sectors)")
    ax.set_ylabel("ODS F-measure")
    ax.set_title(r"$N_s$ Saturation on BSDS500 ($d\!=\!2$)")
    ax.set_xscale("log")
    ax.axvline(x=6, color=GRAY_PALETTE["med_gray"], linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.text(6.5, ax.get_ylim()[0] + 0.001, "$N_s=6$", fontsize=6,
            color=GRAY_PALETTE["med_gray"])
    ax.legend(framealpha=0.8)
    fig.savefig(os.path.join(FIG_DIR, "fig_ns_saturation.png"))
    plt.close(fig)
    print("  saved fig_ns_saturation.png")

# =========================================================================== #
# Figure 6: Post-processing comparison (NMS bar chart)
# =========================================================================== #
def fig_postprocessing():
    target_configs = [
        "Best WVF (Np=25 d=2 Ns=3)",
        "Bagan WVF (Np=250 d=4)",
        "Best LF (Np=75 m=1)",
        "Bagan LF (Np=250 m=14)",
    ]
    short_names = ["Best WVF", "Bagan WVF", "Best LF", "Bagan LF"]
    nms_types = ["No NMS (raw)", "NMS 4-dir", "NMS 8-dir"]
    nms_short = ["Raw", "NMS-4", "NMS-8"]

    nms_grays   = [GRAY_PALETTE["black"], GRAY_PALETTE["med_gray"],
                   GRAY_PALETTE["light_gray"]]
    nms_hatches = ["/", "\\", "+"]

    # Build lookup
    pp_data = {}
    for r in canny_pp:
        pp_data[(r["config"], r["canny"])] = r["ods"]

    n_cfg = len(target_configs)
    n_nms = len(nms_types)
    x = np.arange(n_cfg)
    width = 0.25

    fig, ax = plt.subplots(figsize=(IEEE_COL, 2.6))
    for i, (nms_t, nms_s) in enumerate(zip(nms_types, nms_short)):
        vals = [pp_data.get((cfg, nms_t), 0) for cfg in target_configs]
        offset = (i - (n_nms - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=nms_s,
                      color=nms_grays[i], hatch=nms_hatches[i],
                      edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7)
    ax.set_ylabel("ODS F-measure")
    ax.set_title("Effect of Non-Maximum Suppression (BIPED)")
    ax.set_ylim(0.45, 0.92)
    ax.legend(framealpha=0.8)
    fig.savefig(os.path.join(FIG_DIR, "fig_postprocessing.png"))
    plt.close(fig)
    print("  saved fig_postprocessing.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Generating Report 1 figures...")
    fig_wvf_heatmap()
    fig_wvf_ods_vs_np()
    fig_lf_ods_vs_m()
    fig_cross_dataset_best()
    fig_ns_saturation()
    fig_postprocessing()
    print("Done. All figures saved to", FIG_DIR)
