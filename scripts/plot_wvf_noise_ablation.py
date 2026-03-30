"""Plot WVF/LF noise ablation results across all datasets and noise types."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WVF_OUT = ROOT / "outputs" / "noise_ablation"
PLOT_OUT = WVF_OUT / "plots"
PLOT_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["uded", "biped_v1", "biped_v2", "bsds500"]
DS_LABELS = {"uded": "UDED", "biped_v1": "BIPED v1", "biped_v2": "BIPED v2", "bsds500": "BSDS500"}
NOISE_TYPES = ["gaussian", "salt_pepper", "poisson", "speckle", "uniform"]
NOISE_LABELS = {"gaussian": "Gaussian", "salt_pepper": "Salt & Pepper",
                "poisson": "Poisson", "speckle": "Speckle", "uniform": "Uniform"}
SNR_LEVELS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def load_dataset_results(dataset):
    """Load all WVF/LF results for a dataset. Returns {(noise,snr): {best_wvf_ods, best_lf_ods, best_wvf_np, best_lf_m, ...}}"""
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
            lf_configs = [r for r in cond["results"] if r["filter"] == "LF"]

            if wvf_configs:
                best_wvf = max(wvf_configs, key=lambda x: x["ods"])
                results.setdefault(key, {"wvf_ods": [], "lf_ods": [], "wvf_np": [], "lf_m": []})
                results[key]["wvf_ods"].append(best_wvf["ods"])
                results[key]["wvf_np"].append(best_wvf["Np"])
            if lf_configs:
                best_lf = max(lf_configs, key=lambda x: x["ods"])
                results[key]["lf_ods"].append(best_lf["ods"])
                results[key]["lf_m"].append(best_lf["m"])

    # Average over images
    avg = {}
    for k, v in results.items():
        avg[k] = {
            "wvf_ods": np.mean(v["wvf_ods"]) if v["wvf_ods"] else 0,
            "lf_ods": np.mean(v["lf_ods"]) if v["lf_ods"] else 0,
            "wvf_np": int(np.median(v["wvf_np"])) if v["wvf_np"] else 0,
            "lf_m": int(np.median(v["lf_m"])) if v["lf_m"] else 0,
        }
    return avg


# Load all data
all_data = {}
for ds in DATASETS:
    all_data[ds] = load_dataset_results(ds)
    print(f"{ds}: {len(all_data[ds])} conditions loaded")

# =========================================================================
# Plot 1: WVF best ODS vs SNR — all noise types, per dataset (4x5 grid)
# =========================================================================
fig, axes = plt.subplots(len(DATASETS), len(NOISE_TYPES), figsize=(5 * len(NOISE_TYPES), 5 * len(DATASETS)))
snr_plot = SNR_LEVELS + [100]
snr_labels = [str(s) for s in SNR_LEVELS] + ["clean"]

for di, ds in enumerate(DATASETS):
    data = all_data[ds]
    for ni, noise in enumerate(NOISE_TYPES):
        ax = axes[di, ni]
        wvf_ods = [data.get((noise, str(s)), {}).get("wvf_ods", np.nan) for s in SNR_LEVELS]
        wvf_ods.append(data.get(("clean", "inf"), {}).get("wvf_ods", np.nan))
        lf_ods = [data.get((noise, str(s)), {}).get("lf_ods", np.nan) for s in SNR_LEVELS]
        lf_ods.append(data.get(("clean", "inf"), {}).get("lf_ods", np.nan))

        ax.plot(snr_plot, wvf_ods, "o-", color="#377eb8", markersize=5, label="WVF (best)")
        ax.plot(snr_plot, lf_ods, "s--", color="#e41a1c", markersize=5, label="LF (best)")

        ax.set_xscale("log")
        ax.set_xticks(snr_plot)
        ax.set_xticklabels(snr_labels, fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.0)
        if di == 0:
            ax.set_title(NOISE_LABELS[noise], fontsize=11)
        if ni == 0:
            ax.set_ylabel(DS_LABELS[ds], fontsize=11)
        if di == len(DATASETS) - 1:
            ax.set_xlabel("SNR", fontsize=9)
        if di == 0 and ni == 0:
            ax.legend(fontsize=7)

fig.suptitle("WVF vs LF: Best ODS vs SNR — All Noise Types × All Datasets", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOT_OUT / "wvf_lf_ods_vs_snr_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved wvf_lf_ods_vs_snr_all.png")

# =========================================================================
# Plot 2: Optimal Np vs SNR per noise type, per dataset
# =========================================================================
fig, axes = plt.subplots(len(DATASETS), len(NOISE_TYPES), figsize=(5 * len(NOISE_TYPES), 5 * len(DATASETS)))

for di, ds in enumerate(DATASETS):
    data = all_data[ds]
    for ni, noise in enumerate(NOISE_TYPES):
        ax = axes[di, ni]
        wvf_np = [data.get((noise, str(s)), {}).get("wvf_np", np.nan) for s in SNR_LEVELS]
        wvf_np.append(data.get(("clean", "inf"), {}).get("wvf_np", np.nan))
        lf_m = [data.get((noise, str(s)), {}).get("lf_m", np.nan) for s in SNR_LEVELS]
        lf_m.append(data.get(("clean", "inf"), {}).get("lf_m", np.nan))

        ax.plot(snr_plot, wvf_np, "o-", color="#377eb8", markersize=5, label="WVF best $N_p$")
        ax2 = ax.twinx()
        ax2.plot(snr_plot, lf_m, "s--", color="#e41a1c", markersize=5, label="LF best $m$")

        ax.set_xscale("log")
        ax.set_xticks(snr_plot)
        ax.set_xticklabels(snr_labels, fontsize=6)
        ax.grid(True, alpha=0.3)
        if di == 0:
            ax.set_title(NOISE_LABELS[noise], fontsize=11)
        if ni == 0:
            ax.set_ylabel(f"{DS_LABELS[ds]}\n$N_p$", fontsize=10, color="#377eb8")
        if ni == len(NOISE_TYPES) - 1:
            ax2.set_ylabel("$m$", fontsize=10, color="#e41a1c")
        if di == len(DATASETS) - 1:
            ax.set_xlabel("SNR", fontsize=9)

fig.suptitle("Optimal Parameters vs SNR — All Noise Types × All Datasets", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOT_OUT / "optimal_params_vs_snr_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved optimal_params_vs_snr_all.png")

# =========================================================================
# Plot 3: WVF ODS comparison across noise types (one panel per dataset)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

noise_colors = {"gaussian": "#e41a1c", "salt_pepper": "#ff7f00", "poisson": "#377eb8",
                "speckle": "#4daf4a", "uniform": "#984ea3"}

for di, ds in enumerate(DATASETS):
    ax = axes[di]
    data = all_data[ds]
    for noise in NOISE_TYPES:
        wvf_ods = [data.get((noise, str(s)), {}).get("wvf_ods", np.nan) for s in SNR_LEVELS]
        wvf_ods.append(data.get(("clean", "inf"), {}).get("wvf_ods", np.nan))
        ax.plot(snr_plot, wvf_ods, "o-", color=noise_colors[noise],
                markersize=5, label=NOISE_LABELS[noise])

    ax.set_xscale("log")
    ax.set_xticks(snr_plot)
    ax.set_xticklabels(snr_labels, fontsize=8)
    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("WVF Best ODS", fontsize=11)
    ax.set_title(DS_LABELS[ds], fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    if di == 0:
        ax.legend(fontsize=9)

fig.suptitle("WVF: How Different Noise Types Affect Performance", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_OUT / "wvf_noise_type_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved wvf_noise_type_comparison.png")

# =========================================================================
# Plot 4: LF advantage over WVF per noise type
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for di, ds in enumerate(DATASETS):
    ax = axes[di]
    data = all_data[ds]
    for noise in NOISE_TYPES:
        gaps = []
        for snr in SNR_LEVELS:
            wvf = data.get((noise, str(snr)), {}).get("wvf_ods", 0)
            lf = data.get((noise, str(snr)), {}).get("lf_ods", 0)
            gaps.append(lf - wvf)
        # Clean
        wvf = data.get(("clean", "inf"), {}).get("wvf_ods", 0)
        lf = data.get(("clean", "inf"), {}).get("lf_ods", 0)
        gaps.append(lf - wvf)
        ax.plot(snr_plot, gaps, "o-", color=noise_colors[noise],
                markersize=5, label=NOISE_LABELS[noise])

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xticks(snr_plot)
    ax.set_xticklabels(snr_labels, fontsize=8)
    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("LF - WVF (ODS)", fontsize=11)
    ax.set_title(DS_LABELS[ds], fontsize=12)
    ax.grid(True, alpha=0.3)
    if di == 0:
        ax.legend(fontsize=9)

fig.suptitle("LF Advantage Over WVF by Noise Type (positive = LF wins)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(PLOT_OUT / "lf_vs_wvf_gap_by_noise.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved lf_vs_wvf_gap_by_noise.png")

# =========================================================================
# Plot 5: Summary heatmap — best WVF ODS by (dataset, noise, SNR)
# =========================================================================
fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))

for di, ds in enumerate(DATASETS):
    ax = axes[di]
    data = all_data[ds]
    hm = np.full((len(NOISE_TYPES), len(SNR_LEVELS) + 1), np.nan)
    for ni, noise in enumerate(NOISE_TYPES):
        for si, snr in enumerate(SNR_LEVELS):
            hm[ni, si] = data.get((noise, str(snr)), {}).get("wvf_ods", np.nan)
        hm[ni, -1] = data.get(("clean", "inf"), {}).get("wvf_ods", np.nan)

    im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(SNR_LEVELS) + 1))
    ax.set_xticklabels([str(s) for s in SNR_LEVELS] + ["clean"], fontsize=8)
    ax.set_yticks(range(len(NOISE_TYPES)))
    ax.set_yticklabels([NOISE_LABELS[n] for n in NOISE_TYPES], fontsize=9)
    ax.set_xlabel("SNR", fontsize=10)
    ax.set_title(DS_LABELS[ds], fontsize=12)

    for i in range(len(NOISE_TYPES)):
        for j in range(len(SNR_LEVELS) + 1):
            if not np.isnan(hm[i, j]):
                c = "white" if hm[i, j] < 0.6 else "black"
                ax.text(j, i, f"{hm[i, j]:.2f}", ha="center", va="center", fontsize=7, color=c)

fig.colorbar(im, ax=axes, label="WVF Best ODS", shrink=0.6)
fig.suptitle("WVF Best ODS: Noise Type × SNR × Dataset", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOT_OUT / "wvf_heatmap_noise_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved wvf_heatmap_noise_snr.png")

# =========================================================================
# Print summary table
# =========================================================================
print("\n=== Best WVF ODS by dataset and noise type at key SNRs ===")
for ds in DATASETS:
    data = all_data[ds]
    print(f"\n{DS_LABELS[ds]}:")
    print(f"  {'Noise':<15s} {'SNR=0.3':>8s} {'SNR=1.0':>8s} {'SNR=2.0':>8s} {'SNR=4.0':>8s} {'clean':>8s}")
    print(f"  {'-'*55}")
    for noise in NOISE_TYPES:
        vals = []
        for snr in [0.3, 1.0, 2.0, 4.0]:
            v = data.get((noise, str(snr)), {}).get("wvf_ods", 0)
            vals.append(f"{v:.4f}" if v > 0 else "  ---")
        clean = data.get(("clean", "inf"), {}).get("wvf_ods", 0)
        vals.append(f"{clean:.4f}" if clean > 0 else "  ---")
        print(f"  {NOISE_LABELS[noise]:<15s} {'  '.join(vals)}")

print(f"\nAll plots saved to {PLOT_OUT}")
