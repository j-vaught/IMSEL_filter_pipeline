#!/usr/bin/env python3
"""Generate all figures for Report 3: Noise Robustness of Edge Detection Methods.

Uses a grayscale-friendly style: primary distinction via linestyle + marker,
with a grayscale palette as secondary cue.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
WVF_DIR = ROOT / "outputs" / "noise_ablation"
DL_DIR = ROOT / "outputs" / "dl_noise_ablation"
FIG_DIR = ROOT / "papers" / "report3" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["bsds500", "biped_v1", "biped_v2", "uded"]
DATASET_LABELS = {"bsds500": "BSDS500", "biped_v1": "BIPED-v1",
                  "biped_v2": "BIPED-v2", "uded": "UDED"}
DL_MODELS = ["dexined", "diffusionedge", "nbed", "pidinet", "teed"]
DL_MODEL_LABELS = {"dexined": "DexiNed", "diffusionedge": "DiffusionEdge",
                   "nbed": "NBED", "pidinet": "PiDiNet", "teed": "TEED"}
NOISE_TYPES = ["gaussian", "salt_pepper", "poisson", "speckle", "uniform"]
NOISE_LABELS = {"gaussian": "Gaussian", "salt_pepper": "Salt & Pepper",
                "poisson": "Poisson", "speckle": "Speckle", "uniform": "Uniform"}

# SNR levels present in data (string keys)
SNR_LEVELS = ["0.3", "0.5", "1.0", "1.5", "2.0", "3.0", "4.0"]
SNR_NUMERIC = [float(s) for s in SNR_LEVELS]

# ── Grayscale palette & style mappings ─────────────────────────────────────────
GRAY_BLACK   = "#000000"
GRAY_DARK    = "#444444"
GRAY_MEDIUM  = "#888888"
GRAY_LIGHT   = "#BBBBBB"
GRAY_VLIGHT  = "#DDDDDD"

# Dataset mappings (consistent across both scripts)
DS_STYLE = {
    "bsds500":   {"ls": "solid",   "marker": "o", "color": GRAY_BLACK},
    "biped_v1":  {"ls": "dashed",  "marker": "s", "color": GRAY_DARK},
    "biped_v2":  {"ls": "dotted",  "marker": "D", "color": GRAY_MEDIUM},
    "uded":      {"ls": "dashdot", "marker": "^", "color": GRAY_LIGHT},
}

# DL model mappings (consistent across both scripts)
DL_STYLE = {
    "dexined":       {"ls": "solid",       "marker": "o", "color": GRAY_BLACK},
    "teed":          {"ls": "dashed",      "marker": "s", "color": GRAY_DARK},
    "pidinet":       {"ls": "dotted",      "marker": "D", "color": GRAY_MEDIUM},
    "nbed":          {"ls": "dashdot",     "marker": "^", "color": GRAY_LIGHT},
    "diffusionedge": {"ls": (0, (5, 2)),   "marker": "v", "color": GRAY_VLIGHT},
}

# Noise type mappings (consistent across both scripts)
NOISE_STYLE = {
    "gaussian":    {"ls": "solid",         "marker": "o", "color": GRAY_BLACK},
    "salt_pepper": {"ls": "dashed",        "marker": "s", "color": GRAY_DARK},
    "poisson":     {"ls": "dotted",        "marker": "D", "color": GRAY_MEDIUM},
    "speckle":     {"ls": "dashdot",       "marker": "^", "color": GRAY_LIGHT},
    "uniform":     {"ls": (0, (3, 1, 1, 1)), "marker": "v", "color": GRAY_VLIGHT},
}

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

IEEE_COL_W = 3.5  # inches


# ── Data loading helpers ───────────────────────────────────────────────────────
def load_wvf_results(dataset):
    """Load all WVF/LF noise ablation results for a dataset.
    Returns list of per-image dicts.
    """
    ds_dir = WVF_DIR / dataset
    sel_path = ds_dir / "selected_images.json"
    with open(sel_path) as f:
        images = json.load(f)["images"]
    all_results = []
    for img in images:
        fpath = ds_dir / f"{img}_results.json"
        if fpath.exists():
            with open(fpath) as f:
                all_results.append(json.load(f))
    return all_results


def load_dl_results(model, dataset):
    """Load all DL noise ablation results for a model/dataset.
    Returns list of per-image dicts.
    """
    ds_dir = DL_DIR / model / dataset
    sel_path = ds_dir / "selected_images.json"
    if not sel_path.exists():
        return []
    with open(sel_path) as f:
        images = json.load(f)["images"]
    all_results = []
    for img in images:
        fpath = ds_dir / f"{img}_results.json"
        if fpath.exists():
            with open(fpath) as f:
                all_results.append(json.load(f))
    return all_results


def best_wvf_ods(conditions, noise_type, snr, filt="WVF"):
    """Get best ODS for given filter/noise/snr from a single image's conditions."""
    for cond in conditions:
        match_noise = (cond["noise_type"] == noise_type) if noise_type != "clean" else (cond["noise_type"] == "clean")
        match_snr = (str(cond["snr"]) == str(snr))
        if match_noise and match_snr:
            return max(r["ods"] for r in cond["results"] if r["filter"] == filt)
    return None


def best_wvf_config(conditions, noise_type, snr, filt="WVF"):
    """Get the best config (Np, Ns, d) for given filter/noise/snr."""
    for cond in conditions:
        if cond["noise_type"] == noise_type and str(cond["snr"]) == str(snr):
            best = max((r for r in cond["results"] if r["filter"] == filt),
                       key=lambda r: r["ods"])
            return best
    return None


def dl_ods(conditions, noise_type, snr):
    """Get ODS from DL conditions for given noise_type and snr."""
    for cond in conditions:
        if cond["noise_type"] == noise_type and str(cond["snr"]) == str(snr):
            return cond["ods"]
    return None


# ── Figure 1: WVF ODS vs SNR (Gaussian), per dataset ──────────────────────────
def fig_wvf_ods_vs_snr():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    all_snrs = ["inf"] + SNR_LEVELS[::-1]  # high SNR to low
    x_labels = [r"$\infty$"] + [s for s in SNR_LEVELS[::-1]]
    x_pos = list(range(len(all_snrs)))

    for ds in DATASETS:
        st = DS_STYLE[ds]
        results = load_wvf_results(ds)
        means = []
        stds = []
        for snr in all_snrs:
            ods_vals = []
            for img_data in results:
                if snr == "inf":
                    val = best_wvf_ods(img_data["conditions"], "clean", "inf")
                else:
                    val = best_wvf_ods(img_data["conditions"], "gaussian", snr)
                if val is not None:
                    ods_vals.append(val)
            means.append(np.mean(ods_vals) if ods_vals else np.nan)
            stds.append(np.std(ods_vals) if ods_vals else 0)

        ax.errorbar(x_pos, means, yerr=stds, label=DATASET_LABELS[ds],
                    marker=st["marker"], color=st["color"], linestyle=st["ls"],
                    capsize=2, capthick=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("ODS (best WVF config)")
    ax.set_title("WVF Edge Detection Under Gaussian Noise")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)

    out = FIG_DIR / "fig_wvf_ods_vs_snr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Optimal Np shift with SNR ───────────────────────────────────────
def fig_optimal_params_shift():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    all_snrs = ["inf"] + SNR_LEVELS[::-1]
    x_labels = [r"$\infty$"] + [s for s in SNR_LEVELS[::-1]]
    x_pos = list(range(len(all_snrs)))

    for ds in DATASETS:
        st = DS_STYLE[ds]
        results = load_wvf_results(ds)
        mean_nps = []
        for snr in all_snrs:
            nps = []
            for img_data in results:
                if snr == "inf":
                    cfg = best_wvf_config(img_data["conditions"], "clean", "inf")
                else:
                    cfg = best_wvf_config(img_data["conditions"], "gaussian", snr)
                if cfg is not None:
                    nps.append(cfg["Np"])
            mean_nps.append(np.mean(nps) if nps else np.nan)

        ax.plot(x_pos, mean_nps, label=DATASET_LABELS[ds],
                marker=st["marker"], color=st["color"], linestyle=st["ls"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Optimal $N_p$ (WVF)")
    ax.set_title("Optimal Filter Support Shifts Under Gaussian Noise")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    out = FIG_DIR / "fig_optimal_params_shift.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: DL degradation vs WVF ──────────────────────────────────────────
def fig_dl_degradation():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.6))

    all_snrs = ["inf"] + SNR_LEVELS[::-1]
    x_labels = [r"$\infty$"] + [s for s in SNR_LEVELS[::-1]]
    x_pos = list(range(len(all_snrs)))

    # DL models
    for model in DL_MODELS:
        st = DL_STYLE[model]
        means = []
        for snr in all_snrs:
            ods_vals = []
            for ds in DATASETS:
                img_results = load_dl_results(model, ds)
                for img_data in img_results:
                    if snr == "inf":
                        val = dl_ods(img_data["conditions"], "clean", "inf")
                    else:
                        val = dl_ods(img_data["conditions"], "gaussian", snr)
                    if val is not None:
                        ods_vals.append(val)
            means.append(np.mean(ods_vals) if ods_vals else np.nan)
        ax.plot(x_pos, means, label=DL_MODEL_LABELS[model],
                marker=st["marker"], color=st["color"], linestyle=st["ls"])

    # WVF best (averaged across datasets) — solid black, thick (1.5pt)
    wvf_means = []
    for snr in all_snrs:
        ods_vals = []
        for ds in DATASETS:
            results = load_wvf_results(ds)
            for img_data in results:
                if snr == "inf":
                    val = best_wvf_ods(img_data["conditions"], "clean", "inf")
                else:
                    val = best_wvf_ods(img_data["conditions"], "gaussian", snr)
                if val is not None:
                    ods_vals.append(val)
        wvf_means.append(np.mean(ods_vals) if ods_vals else np.nan)
    ax.plot(x_pos, wvf_means, label="WVF (best)", marker='*',
            color=GRAY_BLACK, linewidth=1.5, linestyle='solid')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("ODS")
    ax.set_title("DL Models vs. WVF Under Gaussian Noise")
    ax.legend(loc="lower left", fontsize=6, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)

    out = FIG_DIR / "fig_dl_degradation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Crossover by noise type ────────────────────────────────────────
def fig_crossover_by_noise_type():
    """For each noise type, find the SNR at which best-WVF overtakes each DL model.
    Show as a heatmap: rows = DL models, columns = noise types.
    """
    crossover_matrix = np.full((len(DL_MODELS), len(NOISE_TYPES)), np.nan)

    for ni, noise_type in enumerate(NOISE_TYPES):
        # Get WVF best ODS per SNR (averaged across all datasets and images)
        wvf_by_snr = {}
        for snr in SNR_LEVELS:
            vals = []
            for ds in DATASETS:
                for img_data in load_wvf_results(ds):
                    v = best_wvf_ods(img_data["conditions"], noise_type, snr)
                    if v is not None:
                        vals.append(v)
            wvf_by_snr[snr] = np.mean(vals) if vals else np.nan

        for mi, model in enumerate(DL_MODELS):
            dl_by_snr = {}
            for snr in SNR_LEVELS:
                vals = []
                for ds in DATASETS:
                    for img_data in load_dl_results(model, ds):
                        v = dl_ods(img_data["conditions"], noise_type, snr)
                        if v is not None:
                            vals.append(v)
                dl_by_snr[snr] = np.mean(vals) if vals else np.nan

            # Find highest SNR where WVF >= DL (i.e., crossover point)
            crossover_snr = np.nan
            for snr in reversed(SNR_LEVELS):
                wvf_val = wvf_by_snr.get(snr, np.nan)
                dl_val = dl_by_snr.get(snr, np.nan)
                if not np.isnan(wvf_val) and not np.isnan(dl_val):
                    if wvf_val >= dl_val:
                        crossover_snr = float(snr)
                        break
            crossover_matrix[mi, ni] = crossover_snr

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.2))
    from matplotlib.colors import LinearSegmentedColormap
    masked = np.ma.masked_invalid(crossover_matrix)
    cmap = LinearSegmentedColormap.from_list("light_gray", ["#FFFFFF", "#666666"])
    cmap.set_bad(color='white')
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=4.5)

    ax.set_xticks(range(len(NOISE_TYPES)))
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_TYPES], rotation=30, ha='right')
    ax.set_yticks(range(len(DL_MODELS)))
    ax.set_yticklabels([DL_MODEL_LABELS[m] for m in DL_MODELS])

    # Annotate cells
    for mi in range(len(DL_MODELS)):
        for ni in range(len(NOISE_TYPES)):
            val = crossover_matrix[mi, ni]
            if np.isnan(val):
                ax.text(ni, mi, "N/A", ha='center', va='center', fontsize=6,
                        color=GRAY_MEDIUM)
            else:
                # Use white text on dark cells, black on light cells
                txt_color = GRAY_BLACK
                ax.text(ni, mi, f"{val:.1f}", ha='center', va='center', fontsize=6,
                        color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Crossover SNR", fontsize=7)
    ax.set_title("SNR at Which WVF Overtakes DL Models")

    out = FIG_DIR / "fig_crossover_by_noise_type.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 5: Noise type comparison for WVF ──────────────────────────────────
def fig_noise_type_comparison():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    all_snrs = ["inf"] + SNR_LEVELS[::-1]
    x_labels = [r"$\infty$"] + [s for s in SNR_LEVELS[::-1]]
    x_pos = list(range(len(all_snrs)))

    for noise_type in NOISE_TYPES:
        st = NOISE_STYLE[noise_type]
        means = []
        for snr in all_snrs:
            vals = []
            for ds in DATASETS:
                for img_data in load_wvf_results(ds):
                    if snr == "inf":
                        v = best_wvf_ods(img_data["conditions"], "clean", "inf")
                    else:
                        v = best_wvf_ods(img_data["conditions"], noise_type, snr)
                    if v is not None:
                        vals.append(v)
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(x_pos, means, label=NOISE_LABELS[noise_type],
                marker=st["marker"], color=st["color"], linestyle=st["ls"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("ODS (best WVF config)")
    ax.set_title("WVF Performance Across Noise Types")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)

    out = FIG_DIR / "fig_noise_type_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 6: LF vs WVF gap under noise ─────────────────────────────────────
def fig_lf_vs_wvf_noise():
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    all_snrs = ["inf"] + SNR_LEVELS[::-1]
    x_labels = [r"$\infty$"] + [s for s in SNR_LEVELS[::-1]]
    x_pos = list(range(len(all_snrs)))

    for ds in DATASETS:
        st = DS_STYLE[ds]
        results = load_wvf_results(ds)
        gaps = []
        for snr in all_snrs:
            wvf_vals = []
            lf_vals = []
            for img_data in results:
                if snr == "inf":
                    w = best_wvf_ods(img_data["conditions"], "clean", "inf", filt="WVF")
                    l = best_wvf_ods(img_data["conditions"], "clean", "inf", filt="LF")
                else:
                    w = best_wvf_ods(img_data["conditions"], "gaussian", snr, filt="WVF")
                    l = best_wvf_ods(img_data["conditions"], "gaussian", snr, filt="LF")
                if w is not None and l is not None:
                    wvf_vals.append(w)
                    lf_vals.append(l)
            if wvf_vals and lf_vals:
                gap = np.mean(wvf_vals) - np.mean(lf_vals)
            else:
                gap = np.nan
            gaps.append(gap)

        ax.plot(x_pos, gaps, label=DATASET_LABELS[ds],
                marker=st["marker"], color=st["color"], linestyle=st["ls"])

    ax.axhline(0, color=GRAY_MEDIUM, linewidth=0.5, linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR")
    ax.set_ylabel("ODS Gap (WVF $-$ LF)")
    ax.set_title("WVF vs. LF Performance Gap Under Gaussian Noise")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    out = FIG_DIR / "fig_lf_vs_wvf_noise.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Report 3 figures...")
    print()
    print("[1/6] WVF ODS vs SNR")
    fig_wvf_ods_vs_snr()
    print("[2/6] Optimal Np shift")
    fig_optimal_params_shift()
    print("[3/6] DL degradation")
    fig_dl_degradation()
    print("[4/6] Crossover by noise type")
    fig_crossover_by_noise_type()
    print("[5/6] Noise type comparison")
    fig_noise_type_comparison()
    print("[6/6] LF vs WVF gap")
    fig_lf_vs_wvf_noise()
    print()
    print("All figures saved to", FIG_DIR)
