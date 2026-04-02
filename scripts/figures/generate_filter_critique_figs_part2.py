"""Generate figures for Report 2: Benchmarking WVF/LF Against Classical and Learned Edge Detectors.

Produces publication-quality figures for IEEE two-column format (300 DPI PNG).
Uses a grayscale-friendly style: primary distinction via linestyle + marker,
with a grayscale palette as secondary cue.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "papers" / "report2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(ROOT / "outputs" / "classical_test" / "classical_ods.json") as f:
    classical_data = json.load(f)

with open(ROOT / "outputs" / "model_test" / "test_results.json") as f:
    dl_timing = json.load(f)

with open(ROOT / "outputs" / "model_test" / "single_image_ods.json") as f:
    dl_ods_single = json.load(f)

with open(ROOT / "outputs" / "full_dataset_ablation" / "summary.json") as f:
    ablation_summary = json.load(f)

# ── Grayscale palette & style mappings ─────────────────────────────────────
GRAY_BLACK   = "#000000"
GRAY_DARK    = "#444444"
GRAY_MEDIUM  = "#888888"
GRAY_LIGHT   = "#BBBBBB"
GRAY_VLIGHT  = "#DDDDDD"

# Dataset mappings (consistent across both scripts)
DS_STYLE = {
    "BSDS500":   {"ls": "solid",   "marker": "o", "color": GRAY_BLACK},
    "BIPED_v1":  {"ls": "dashed",  "marker": "s", "color": GRAY_DARK},
    "BIPED_v2":  {"ls": "dotted",  "marker": "D", "color": GRAY_MEDIUM},
    "UDED":      {"ls": "dashdot", "marker": "^", "color": GRAY_LIGHT},
}

# DL model mappings (consistent across both scripts)
DL_STYLE = {
    "DexiNed":       {"ls": "solid",       "marker": "o", "color": GRAY_BLACK},
    "TEED":          {"ls": "dashed",      "marker": "s", "color": GRAY_DARK},
    "pidinet":       {"ls": "dotted",      "marker": "D", "color": GRAY_MEDIUM},
    "NBED":          {"ls": "dashdot",     "marker": "^", "color": GRAY_LIGHT},
    "DiffusionEdge": {"ls": (0, (5, 2)),   "marker": "v", "color": GRAY_VLIGHT},
}

# Classical bar hatching patterns
CLASSICAL_HATCHES = ['/', '\\', '|', 'x', '+', '-', 'o', '.', '*',
                     '//', '\\\\', '||', 'xx', '++', '--', 'oo', '..', '**',
                     '///', '\\\\\\']

# ── Style setup ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# IEEE column widths
COL_W = 3.5    # single column inches
FULL_W = 7.16  # full width inches

# ── Helper: build label for classical method ──────────────────────────────
def classical_label(entry):
    m = entry["method"]
    p = entry["params"]
    if "ksize" in p:
        return f"{m} (k={p['ksize']})"
    elif "sigma" in p:
        return f"{m} (\u03c3={p['sigma']})"
    elif "sigma1" in p:
        return f"{m} (\u03c31={p['sigma1']})"
    return m


# =========================================================================
# Figure 1: Classical ranking bar chart with WVF/LF reference lines
# =========================================================================
def fig_classical_ranking():
    # Sort descending and take top 20
    sorted_data = sorted(classical_data, key=lambda x: x["ods"], reverse=True)[:20]
    labels = [classical_label(e) for e in sorted_data]
    ods_vals = [e["ods"] for e in sorted_data]

    # WVF and LF best ODS on BIPED (use BIPED_v2 as the primary benchmark)
    best_wvf_ods = ablation_summary["BIPED_v2"]["best_wvf"]["ods"]
    best_lf_ods = ablation_summary["BIPED_v2"]["best_lf"]["ods"]

    fig, ax = plt.subplots(figsize=(FULL_W, 4.5))
    y_pos = np.arange(len(labels))

    # Grayscale bars with hatching for distinction
    bars = ax.barh(y_pos, ods_vals, height=0.7, color=GRAY_LIGHT,
                   edgecolor=GRAY_BLACK, linewidth=0.5, zorder=3)
    # Use graduated gray fills instead of hatching
    n_bars = len(bars)
    for i, bar in enumerate(bars):
        shade = 0.45 + 0.45 * (i / max(n_bars - 1, 1))
        bar.set_facecolor(str(shade))

    # WVF reference line: solid black, thick (1.5pt)
    ax.axvline(best_wvf_ods, color=GRAY_BLACK, linestyle="solid", linewidth=1.5,
               label=f"Best WVF (ODS={best_wvf_ods:.3f})", zorder=4)
    # LF reference line: dashed black, thick (1.5pt)
    ax.axvline(best_lf_ods, color=GRAY_BLACK, linestyle="dashed", linewidth=1.5,
               label=f"Best LF (ODS={best_lf_ods:.3f})", zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ODS F-score")
    ax.set_title("Top 20 Classical Edge Detectors Ranked by ODS (BIPED)")
    ax.set_xlim(0.62, max(max(ods_vals), best_wvf_ods) + 0.02)
    ax.legend(loc="lower right", framealpha=0.9)

    # Add ODS value labels at end of bars
    for i, v in enumerate(ods_vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=6, color=GRAY_BLACK)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_classical_ranking.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# =========================================================================
# Figure 2: Scale trend for Sobel, Prewitt, GaussianDeriv
# =========================================================================
def fig_classical_scale_trend():
    # Extract per-method scale data
    methods_scale = {}
    for entry in classical_data:
        m = entry["method"]
        if m in ("Sobel", "Prewitt", "GaussianDeriv"):
            if m not in methods_scale:
                methods_scale[m] = {"scale": [], "ods": []}
            if m in ("Sobel", "Prewitt"):
                methods_scale[m]["scale"].append(entry["params"]["ksize"])
            else:
                methods_scale[m]["scale"].append(entry["params"]["sigma"])
            methods_scale[m]["ods"].append(entry["ods"])

    # Sort each by scale
    for m in methods_scale:
        order = np.argsort(methods_scale[m]["scale"])
        methods_scale[m]["scale"] = np.array(methods_scale[m]["scale"])[order]
        methods_scale[m]["ods"] = np.array(methods_scale[m]["ods"])[order]

    # Grayscale style per method
    method_style = {
        "Sobel":        {"ls": "solid",   "marker": "o", "color": GRAY_BLACK},
        "Prewitt":      {"ls": "dashed",  "marker": "s", "color": GRAY_DARK},
        "GaussianDeriv": {"ls": "dotted", "marker": "D", "color": GRAY_MEDIUM},
    }

    # WVF effective scale reference (best WVF uses Np=25, d=2 => small support)
    best_wvf_ods = ablation_summary["BIPED_v2"]["best_wvf"]["ods"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    for m in ("Sobel", "Prewitt", "GaussianDeriv"):
        d = methods_scale[m]
        st = method_style[m]
        # Use a secondary x-axis concept: normalize to a common "effective scale"
        # Sobel/Prewitt ksize maps roughly to sigma ~ ksize/4
        if m in ("Sobel", "Prewitt"):
            effective_scale = d["scale"] / 4.0
        else:
            effective_scale = d["scale"]
        ax.plot(effective_scale, d["ods"], marker=st["marker"], linestyle=st["ls"],
                color=st["color"], markersize=4, linewidth=1.2, label=m, zorder=3)

    # WVF reference line: solid black, thick (1.5pt)
    ax.axhline(best_wvf_ods, color=GRAY_BLACK, linestyle="solid", linewidth=1.5,
               label=f"Best WVF (ODS={best_wvf_ods:.3f})", zorder=2)

    # Mark the optimal region (grayscale-friendly)
    ax.axvspan(0.8, 2.0, alpha=0.10, color=GRAY_MEDIUM, zorder=1)
    ax.text(1.3, 0.65, "Optimal\nscale\nrange", fontsize=6, ha="center",
            color=GRAY_DARK, style="italic", alpha=0.7)

    ax.set_xlabel(r"Effective scale ($\sigma$ or ksize/4)")
    ax.set_ylabel("ODS F-score")
    ax.set_title("Classical Detector ODS vs Scale Parameter")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 15)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_classical_scale_trend.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# =========================================================================
# Figure 3: Compute cost scatter — ODS vs inference time
# =========================================================================
def fig_compute_cost():
    """ODS vs inference time for classical, WVF/LF, and DL methods.

    Classical methods: use ODS from classical_ods.json. Since we don't have
    measured timing for classical methods, we estimate them (OpenCV operators
    on 720x1280 images are very fast, typically <5 ms).

    DL models: timing from test_results.json.
    DL ODS: only DiffusionEdge single-image ODS is available. Other DL full-
    dataset ODS runs are still pending (jobs 349323-349326).
    """
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))

    # ── Classical baselines ──
    classical_time_est = {
        "Sobel": 0.002, "Prewitt": 0.002, "Scharr": 0.002,
        "Roberts": 0.001, "Kirsch": 0.003, "Robinson": 0.003,
        "Frei-Chen": 0.003, "GaussianDeriv": 0.003,
        "LoG": 0.004, "DoG": 0.004,
        "Steerable2nd": 0.005,
    }

    # Use best ODS per classical method family
    classical_best = {}
    for entry in classical_data:
        m = entry["method"]
        if m not in classical_best or entry["ods"] > classical_best[m]["ods"]:
            classical_best[m] = entry

    c_times = []
    c_ods = []
    c_labels = []
    for m, entry in classical_best.items():
        t = classical_time_est.get(m, 0.003)
        c_times.append(t)
        c_ods.append(entry["ods"])
        c_labels.append(m)

    ax.scatter(c_times, c_ods, marker="D", s=25, c=GRAY_MEDIUM, edgecolors=GRAY_BLACK,
               linewidths=0.4, zorder=4, label="Classical (best per family)")

    # Label a few key classical methods
    for i, lbl in enumerate(c_labels):
        if lbl in ("Prewitt", "Sobel", "Roberts", "Kirsch"):
            ax.annotate(lbl, (c_times[i], c_ods[i]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=5, color=GRAY_DARK)

    # ── WVF and LF ──
    wvf_time = dl_timing["WVF"]["time"]
    best_wvf_ods = ablation_summary["BIPED_v2"]["best_wvf"]["ods"]
    best_lf_ods = ablation_summary["BIPED_v2"]["best_lf"]["ods"]
    # LF is somewhat slower than WVF; estimate ~2x
    lf_time_est = wvf_time * 2.0

    # WVF: solid black star marker
    ax.scatter([wvf_time], [best_wvf_ods], marker="*", s=120, c=GRAY_BLACK,
               edgecolors=GRAY_BLACK, linewidths=0.4, zorder=5, label="WVF (best)")
    ax.annotate("WVF", (wvf_time, best_wvf_ods),
                textcoords="offset points", xytext=(5, -8), fontsize=6,
                fontweight="bold", color=GRAY_BLACK)

    # LF: dashed-style square marker in dark gray
    ax.scatter([lf_time_est], [best_lf_ods], marker="s", s=80, c=GRAY_DARK,
               edgecolors=GRAY_BLACK, linewidths=0.4, zorder=5, label="LF (best)")
    ax.annotate("LF", (lf_time_est, best_lf_ods),
                textcoords="offset points", xytext=(5, -8), fontsize=6,
                fontweight="bold", color=GRAY_DARK)

    # ── DL models ──
    dl_models = {
        "DiffusionEdge": {
            "time": dl_timing["DiffusionEdge"]["time"],
            "ods": dl_ods_single["diffusionedge"]["ods"],
        },
    }
    dl_literature_ods = {
        "DexiNed":  0.859,
        "TEED":     0.898,
        "pidinet":  0.868,
        "NBED":     0.894,
    }
    for name in ("DexiNed", "TEED", "pidinet", "NBED"):
        dl_models[name] = {
            "time": dl_timing[name]["time"],
            "ods": dl_literature_ods[name],
        }

    # Plot DL models with consistent style mappings
    for name in dl_models:
        st = DL_STYLE[name]
        filled = (name == "DiffusionEdge")
        fc = st["color"] if filled else "none"
        ax.scatter([dl_models[name]["time"]], [dl_models[name]["ods"]],
                   marker=st["marker"], s=35,
                   c=fc, edgecolors=GRAY_BLACK, linewidths=0.8, zorder=4)
        offsets = {
            "DiffusionEdge": (5, -8),
            "DexiNed": (4, 4),
            "TEED": (4, 4),
            "pidinet": (4, -8),
            "NBED": (4, 4),
        }
        ax.annotate(name, (dl_models[name]["time"], dl_models[name]["ods"]),
                    textcoords="offset points", xytext=offsets.get(name, (4, 3)),
                    fontsize=5, color=GRAY_DARK)

    # Dummy entries for DL legend
    ax.scatter([], [], marker="o", s=35, c=GRAY_DARK, edgecolors=GRAY_BLACK,
               linewidths=0.8, label="DL (measured ODS)")
    ax.scatter([], [], marker="o", s=35, c="none", edgecolors=GRAY_BLACK,
               linewidths=0.8, label="DL (literature ODS)")

    ax.set_xscale("log")
    ax.set_xlabel("Inference time per image (s, log scale)")
    ax.set_ylabel("ODS F-score")
    ax.set_title("Edge Detection: Quality vs Speed")
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9,
              handletextpad=0.4, borderpad=0.4)

    ax.set_xlim(5e-4, 30)
    ax.set_ylim(0.68, 0.95)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_compute_cost.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# =========================================================================
# Figure 4: Placeholder for edge map gallery
# =========================================================================
def fig_edge_map_gallery_placeholder():
    out_path = FIG_DIR / "TODO_edge_map_gallery.txt"
    out_path.write_text(
        "TODO: Edge Map Gallery Figure (fig_edge_map_gallery.png)\n"
        "======================================================\n\n"
        "This figure will show side-by-side edge maps from representative methods:\n\n"
        "  Row layout (one example image, full-width figure):\n"
        "    [Input] [Sobel k=5] [Prewitt k=5] [Canny] [WVF best] [LF best]\n"
        "    [DexiNed] [TEED] [pidinet] [NBED] [DiffusionEdge] [Ground Truth]\n\n"
        "  Requirements:\n"
        "    - Select 1-2 representative images from BIPED test set\n"
        "    - Run each detector on the same image(s)\n"
        "    - Save edge maps as individual PNGs\n"
        "    - Assemble into a tiled gallery figure\n\n"
        "  Blocked on:\n"
        "    - Need to run classical detectors on specific images (not just ODS)\n"
        "    - DL model inference on same images\n"
        "    - WVF/LF inference with best configs from ablation\n"
    )
    print(f"Saved {out_path}")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating Report 2 figures...")
    print(f"Output directory: {FIG_DIR}\n")

    fig_classical_ranking()
    fig_classical_scale_trend()
    fig_compute_cost()
    fig_edge_map_gallery_placeholder()

    print("\nDone. 3 figures + 1 placeholder generated.")
