"""Plot DL model noise ablation results — ODS vs SNR for each model."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DL_OUT = ROOT / "outputs" / "dl_noise_ablation"
WVF_OUT = ROOT / "outputs" / "noise_ablation"
PLOT_OUT = ROOT / "outputs" / "dl_noise_ablation" / "plots"
PLOT_OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {"dexined": "DexiNed", "teed": "TEED", "pidinet": "pidinet",
               "nbed": "NBED", "diffusionedge": "DiffusionEdge"}
DATASETS = ["uded", "biped_v1", "biped_v2", "bsds500"]
DS_LABELS = {"uded": "UDED", "biped_v1": "BIPED v1", "biped_v2": "BIPED v2", "bsds500": "BSDS500"}
# Only plot models that have completed all datasets
ALL_MODELS = ["dexined", "teed", "pidinet", "nbed", "diffusionedge"]
MODELS = [m for m in ALL_MODELS if all(
    (DL_OUT / m / ds / "DONE").exists() for ds in DATASETS)]
print(f"Plotting models: {MODELS} (completed all datasets)")
NOISE_TYPES = ["gaussian", "salt_pepper", "poisson", "speckle", "uniform"]
SNR_LEVELS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def load_model_results(model, dataset):
    """Load all results for a model on a dataset, return {(noise,snr): avg_ods}."""
    ds_dir = DL_OUT / model / dataset
    if not ds_dir.exists():
        return {}
    results = {}
    for f in ds_dir.glob("*_results.json"):
        with open(f) as fh:
            data = json.load(fh)
        for cond in data["conditions"]:
            key = (cond["noise_type"], cond["snr"])
            if cond["ods"] >= 0:
                results.setdefault(key, []).append(cond["ods"])
    # Average over images
    return {k: np.mean(v) for k, v in results.items()}


def load_wvf_results(dataset):
    """Load WVF best ODS per noise/SNR from the WVF noise ablation."""
    ds_dir = WVF_OUT / dataset
    if not ds_dir.exists():
        return {}
    results = {}
    for f in ds_dir.glob("*_results.json"):
        with open(f) as fh:
            data = json.load(fh)
        for cond in data["conditions"]:
            key = (cond["noise_type"], str(cond["snr"]))
            wvf_configs = [r for r in cond["results"] if r["filter"] == "WVF"]
            if wvf_configs:
                best = max(wvf_configs, key=lambda x: x["ods"])
                results.setdefault(key, []).append(best["ods"])
    return {k: np.mean(v) for k, v in results.items()}


# =========================================================================
# Plot 1: ODS vs SNR per dataset (all models + WVF best, Gaussian noise)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

model_colors = {"dexined": "#e41a1c", "teed": "#ff7f00", "pidinet": "#377eb8",
                "nbed": "#4daf4a", "diffusionedge": "#984ea3"}

for di, ds in enumerate(DATASETS):
    ax = axes[di]
    snr_plot = SNR_LEVELS + [100]  # 100 = clean
    snr_labels = [str(s) for s in SNR_LEVELS] + ["clean"]

    for model in MODELS:
        data = load_model_results(model, ds)
        if not data:
            continue
        ods_vals = []
        for snr in SNR_LEVELS:
            key = ("gaussian", str(snr))
            ods_vals.append(data.get(key, np.nan))
        # Clean
        clean_key = ("clean", "inf")
        ods_vals.append(data.get(clean_key, np.nan))

        ax.plot(snr_plot, ods_vals, "o-", color=model_colors[model],
                markersize=5, label=MODEL_NAMES[model])

    # WVF best
    wvf_data = load_wvf_results(ds)
    if wvf_data:
        ods_vals = []
        for snr in SNR_LEVELS:
            key = ("gaussian", str(snr))
            ods_vals.append(wvf_data.get(key, np.nan))
        clean_key = ("clean", "inf")
        ods_vals.append(wvf_data.get(clean_key, np.nan))
        ax.plot(snr_plot, ods_vals, "s--", color="black", markersize=5,
                linewidth=2, label="WVF (best)")

    ax.set_xscale("log")
    ax.set_xticks(snr_plot)
    ax.set_xticklabels(snr_labels, fontsize=8)
    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("ODS", fontsize=11)
    ax.set_title(DS_LABELS[ds], fontsize=12)
    ax.grid(True, alpha=0.3)
    if di == 0:
        ax.legend(fontsize=8)

fig.suptitle("Gaussian Noise: ODS vs SNR — DL Models vs WVF", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_OUT / "ods_vs_snr_gaussian.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved ods_vs_snr_gaussian.png")

# =========================================================================
# Plot 2: ODS vs SNR by noise type (UDED only, all models)
# =========================================================================
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
noise_labels = {"gaussian": "Gaussian", "salt_pepper": "Salt & Pepper",
                "poisson": "Poisson", "speckle": "Speckle", "uniform": "Uniform"}

for ni, noise in enumerate(NOISE_TYPES):
    ax = axes[ni]
    snr_plot = SNR_LEVELS + [100]

    for model in MODELS:
        data = load_model_results(model, "uded")
        if not data:
            continue
        ods_vals = []
        for snr in SNR_LEVELS:
            ods_vals.append(data.get((noise, str(snr)), np.nan))
        ods_vals.append(data.get(("clean", "inf"), np.nan))
        ax.plot(snr_plot, ods_vals, "o-", color=model_colors[model],
                markersize=4, label=MODEL_NAMES[model])

    wvf_data = load_wvf_results("uded")
    if wvf_data:
        ods_vals = []
        for snr in SNR_LEVELS:
            ods_vals.append(wvf_data.get((noise, str(snr)), np.nan))
        ods_vals.append(wvf_data.get(("clean", "inf"), np.nan))
        ax.plot(snr_plot, ods_vals, "s--", color="black", markersize=4,
                linewidth=2, label="WVF (best)")

    ax.set_xscale("log")
    ax.set_xticks(snr_plot)
    ax.set_xticklabels([str(s) for s in SNR_LEVELS] + ["∞"], fontsize=7)
    ax.set_xlabel("SNR", fontsize=10)
    ax.set_ylabel("ODS", fontsize=10)
    ax.set_title(noise_labels[noise], fontsize=11)
    ax.grid(True, alpha=0.3)
    if ni == 0:
        ax.legend(fontsize=7)

fig.suptitle("UDED: ODS vs SNR by Noise Type", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOT_OUT / "ods_vs_snr_by_noise_uded.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved ods_vs_snr_by_noise_uded.png")

# =========================================================================
# Plot 3: Bar chart — all models at SNR=0.3 and clean, per dataset
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for si, (snr_key, title) in enumerate([
    (("gaussian", "0.3"), "Gaussian SNR=0.3"),
    (("clean", "inf"), "Clean"),
]):
    ax = axes[si]
    x = np.arange(len(DATASETS))
    width = 0.13
    offsets = np.arange(len(MODELS) + 1) - (len(MODELS)) / 2

    for mi, model in enumerate(MODELS):
        vals = []
        for ds in DATASETS:
            data = load_model_results(model, ds)
            vals.append(data.get(snr_key, 0))
        ax.bar(x + offsets[mi] * width, vals, width, label=MODEL_NAMES[model],
               color=model_colors[model], alpha=0.8)

    # WVF
    wvf_vals = []
    for ds in DATASETS:
        data = load_wvf_results(ds)
        wvf_vals.append(data.get(snr_key, 0))
    ax.bar(x + offsets[-1] * width, wvf_vals, width, label="WVF (best)",
           color="black", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS])
    ax.set_ylabel("ODS", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("DL Models vs WVF: Clean vs Heavy Noise", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOT_OUT / "clean_vs_noisy_bars.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved clean_vs_noisy_bars.png")

# =========================================================================
# Print summary table
# =========================================================================
print("\n=== UDED Gaussian: ODS by model and SNR ===")
header = f"{'Model':<15s}" + "".join(f"{'SNR='+str(s):>10s}" for s in SNR_LEVELS) + f"{'clean':>10s}"
print(header)
print("-" * len(header))
for model in MODELS:
    data = load_model_results(model, "uded")
    row = f"{MODEL_NAMES[model]:<15s}"
    for snr in SNR_LEVELS:
        v = data.get(("gaussian", str(snr)), -1)
        row += f"{v:>10.4f}" if v >= 0 else f"{'---':>10s}"
    v = data.get(("clean", "inf"), -1)
    row += f"{v:>10.4f}" if v >= 0 else f"{'---':>10s}"
    print(row)
# WVF
data = load_wvf_results("uded")
row = f"{'WVF (best)':<15s}"
for snr in SNR_LEVELS:
    v = data.get(("gaussian", str(snr)), -1)
    row += f"{v:>10.4f}" if v >= 0 else f"{'---':>10s}"
v = data.get(("clean", "inf"), -1)
row += f"{v:>10.4f}" if v >= 0 else f"{'---':>10s}"
print(row)

# =========================================================================
# Plot 4: Per noise type, per dataset (4x5 grid)
# =========================================================================
fig, axes = plt.subplots(len(DATASETS), len(NOISE_TYPES), figsize=(5 * len(NOISE_TYPES), 5 * len(DATASETS)))
noise_labels = {"gaussian": "Gaussian", "salt_pepper": "Salt & Pepper",
                "poisson": "Poisson", "speckle": "Speckle", "uniform": "Uniform"}

for di, ds in enumerate(DATASETS):
    for ni, noise in enumerate(NOISE_TYPES):
        ax = axes[di, ni]
        snr_plot = SNR_LEVELS + [100]

        for model in MODELS:
            data = load_model_results(model, ds)
            if not data:
                continue
            ods_vals = []
            for snr in SNR_LEVELS:
                ods_vals.append(data.get((noise, str(snr)), np.nan))
            ods_vals.append(data.get(("clean", "inf"), np.nan))
            ax.plot(snr_plot, ods_vals, "o-", color=model_colors[model],
                    markersize=4, label=MODEL_NAMES[model] if di == 0 and ni == 0 else None)

        wvf_data = load_wvf_results(ds)
        if wvf_data:
            ods_vals = []
            for snr in SNR_LEVELS:
                ods_vals.append(wvf_data.get((noise, str(snr)), np.nan))
            ods_vals.append(wvf_data.get(("clean", "inf"), np.nan))
            ax.plot(snr_plot, ods_vals, "s--", color="black", markersize=4,
                    linewidth=2, label="WVF (best)" if di == 0 and ni == 0 else None)

        ax.set_xscale("log")
        ax.set_xticks(snr_plot)
        ax.set_xticklabels([str(s) for s in SNR_LEVELS] + ["∞"], fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.0)
        if di == 0:
            ax.set_title(noise_labels[noise], fontsize=11)
        if ni == 0:
            ax.set_ylabel(DS_LABELS[ds], fontsize=11)
        if di == len(DATASETS) - 1:
            ax.set_xlabel("SNR", fontsize=9)

# Single legend at top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(MODELS) + 1, fontsize=9,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("ODS vs SNR: All Models × All Noise Types × All Datasets", fontsize=14, y=1.05)
fig.tight_layout()
fig.savefig(PLOT_OUT / "full_grid_all_noise_all_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved full_grid_all_noise_all_datasets.png")

# =========================================================================
# Plot 5: Crossover SNR table — at what SNR does WVF beat each DL model?
# =========================================================================
print("\n=== Crossover SNR (Gaussian): SNR where WVF first beats DL model ===")
print(f"{'Model':<15s}" + "".join(f"{DS_LABELS[ds]:>12s}" for ds in DATASETS))
print("-" * (15 + 12 * len(DATASETS)))
for model in MODELS:
    row = f"{MODEL_NAMES[model]:<15s}"
    for ds in DATASETS:
        dl_data = load_model_results(model, ds)
        wvf_data = load_wvf_results(ds)
        if not dl_data or not wvf_data:
            row += f"{'N/A':>12s}"
            continue
        crossover = "never"
        for snr in reversed(SNR_LEVELS + [100]):
            snr_key = "inf" if snr == 100 else str(snr)
            dl_ods = dl_data.get(("gaussian", snr_key), 0)
            wvf_ods = wvf_data.get(("gaussian", snr_key), 0)
            if wvf_ods > dl_ods:
                crossover = "clean" if snr == 100 else f"SNR={snr}"
        row += f"{crossover:>12s}"
    print(row)

print(f"\nAll plots saved to {PLOT_OUT}")
