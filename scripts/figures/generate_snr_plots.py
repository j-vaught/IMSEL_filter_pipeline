"""Generate plots and edge map examples for the SNR ablation."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path
from PIL import Image

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_ablation"

import torch
backend = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load clean image + GT
img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())

SNR_LEVELS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]

# Load all SNR results
all_data = {}
for snr in SNR_LEVELS:
    p = OUT / f"snr_{snr:.2f}_ablation.json"
    if p.exists():
        with open(p) as f:
            all_data[snr] = json.load(f)

# Also load clean data
clean_path = ROOT / "outputs" / "biped_ablation" / "ablation_metrics.json"
if clean_path.exists():
    with open(clean_path) as f:
        all_data["inf"] = json.load(f)

ALL_SNRS = SNR_LEVELS + ["inf"]


def make_noisy(snr):
    if snr == "inf":
        return img_clean.copy()
    np.random.seed(42)
    noise = np.random.normal(0, signal_amplitude / snr, img_clean.shape)
    return np.clip(img_clean + noise, 0, 255)


# =========================================================================
# Plot 1: Edge map examples at each SNR — best WVF, best LF, Bagan LF
# =========================================================================
print("Generating edge map examples...")

n_snrs = len(ALL_SNRS)
fig, axes = plt.subplots(4, n_snrs, figsize=(3.5 * n_snrs, 14))

# Row labels
row_labels = ["Noisy Input", "Best WVF", "Best LF", "Bagan LF (Np=250 m=14)"]

for si, snr in enumerate(ALL_SNRS):
    img = make_noisy(snr)
    snr_label = "clean" if snr == "inf" else f"SNR={snr}"

    if snr in all_data:
        data = all_data[snr]
        best_wvf = max(data["wvf"], key=lambda x: x["ods"])
        best_lf = max((r for r in data["lf"] if r["ods"] >= 0), key=lambda x: x["ods"]) if data["lf"] else None
        bagan_lf = next((r for r in data["lf"] if r["Np"] == 250 and r.get("m") == 14 and r["Ns"] == 18), None)
    else:
        best_wvf = best_lf = bagan_lf = None

    # Row 0: noisy input
    axes[0, si].imshow(img, cmap="gray")
    axes[0, si].set_title(snr_label, fontsize=10, fontweight="bold")
    axes[0, si].axis("off")

    # Row 1: best WVF
    if best_wvf:
        mag = wvf_image(img, np_count=best_wvf["Np"], order=best_wvf["d"],
                        n_orientations=best_wvf["Ns"], backend=backend).gradient_mag
        axes[1, si].imshow(mag, cmap="gray")
        axes[1, si].set_title(f"Np={best_wvf['Np']} d={best_wvf['d']}\nODS={best_wvf['ods']:.3f}", fontsize=8)
    axes[1, si].axis("off")

    # Row 2: best LF
    if best_lf:
        mag = lf_image(img, half_width=best_lf["m"], np_count=best_lf["Np"],
                       order=4, n_orientations=best_lf["Ns"],
                       backend=backend, max_vram_gb=30).gradient_mag
        axes[2, si].imshow(mag, cmap="gray")
        axes[2, si].set_title(f"Np={best_lf['Np']} m={best_lf['m']}\nODS={best_lf['ods']:.3f}", fontsize=8)
    axes[2, si].axis("off")

    # Row 3: Bagan LF
    if bagan_lf and bagan_lf["ods"] >= 0:
        mag = lf_image(img, half_width=14, np_count=250, order=4,
                       n_orientations=18, backend=backend, max_vram_gb=30).gradient_mag
        axes[3, si].imshow(mag, cmap="gray")
        axes[3, si].set_title(f"ODS={bagan_lf['ods']:.3f}", fontsize=8)
    axes[3, si].axis("off")

# Row labels on left
for ri, label in enumerate(row_labels):
    axes[ri, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=100, va="center")

fig.suptitle("BIPED v1 RGB_008: Edge Detection Across SNR Levels", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_edge_maps_by_snr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  plot_edge_maps_by_snr.png")

# =========================================================================
# Plot 2: Optimal Np and m vs SNR
# =========================================================================
print("Generating parameter shift plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

snr_vals, best_np_wvf, best_d_wvf, best_np_lf, best_m_lf = [], [], [], [], []
best_ods_wvf, best_ods_lf, bagan_ods_wvf, bagan_ods_lf = [], [], [], []

for snr in ALL_SNRS:
    if snr not in all_data:
        continue
    data = all_data[snr]
    bw = max(data["wvf"], key=lambda x: x["ods"])
    bl = max((r for r in data["lf"] if r["ods"] >= 0), key=lambda x: x["ods"]) if data["lf"] else None
    pw = next((r for r in data["wvf"] if r["Np"] == 250 and r["d"] == 4 and r["Ns"] == 18), None)
    pl = next((r for r in data["lf"] if r["Np"] == 250 and r.get("m") == 14 and r["Ns"] == 18), None)

    snr_num = 100 if snr == "inf" else snr
    snr_vals.append(snr_num)
    best_np_wvf.append(bw["Np"])
    best_d_wvf.append(bw["d"])
    best_np_lf.append(bl["Np"] if bl else 0)
    best_m_lf.append(bl["m"] if bl else 0)
    best_ods_wvf.append(bw["ods"])
    best_ods_lf.append(bl["ods"] if bl else 0)
    bagan_ods_wvf.append(pw["ods"] if pw else 0)
    bagan_ods_lf.append(pl["ods"] if pl and pl["ods"] >= 0 else 0)

snr_labels = [f"{s}" if s != 100 else "clean" for s in snr_vals]

# Panel 1: Optimal Np vs SNR
ax = axes[0]
ax.plot(snr_vals, best_np_wvf, "o-", color="#377eb8", markersize=8, label="Best WVF $N_p$")
ax.plot(snr_vals, best_np_lf, "s-", color="#e41a1c", markersize=8, label="Best LF $N_p$")
ax.axhline(y=250, color="darkorange", linestyle="--", alpha=0.5, label="Bagan $N_p$=250")
ax.set_xscale("log")
ax.set_xlabel("SNR", fontsize=12)
ax.set_ylabel("Optimal $N_p$", fontsize=12)
ax.set_title("Optimal Support Size vs SNR", fontsize=13)
ax.set_xticks(snr_vals)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Optimal m vs SNR
ax = axes[1]
ax.plot(snr_vals, best_m_lf, "s-", color="#e41a1c", markersize=8, label="Best LF $m$")
ax.axhline(y=14, color="darkorange", linestyle="--", alpha=0.5, label="Bagan $m$=14")
ax.set_xscale("log")
ax.set_xlabel("SNR", fontsize=12)
ax.set_ylabel("Optimal $m$ (LF half-width)", fontsize=12)
ax.set_title("Optimal Line Length vs SNR", fontsize=13)
ax.set_xticks(snr_vals)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Best ODS vs SNR
ax = axes[2]
ax.plot(snr_vals, best_ods_wvf, "o-", color="#377eb8", markersize=8, label="Best WVF")
ax.plot(snr_vals, best_ods_lf, "s-", color="#e41a1c", markersize=8, label="Best LF")
ax.plot(snr_vals, bagan_ods_wvf, "D--", color="darkorange", markersize=6, label="Bagan WVF")
ax.plot(snr_vals, bagan_ods_lf, "D:", color="darkorange", markersize=6, label="Bagan LF")
ax.set_xscale("log")
ax.set_xlabel("SNR", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("Best ODS vs SNR", fontsize=13)
ax.set_xticks(snr_vals)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("BIPED v1: How Optimal Parameters Shift with Noise", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_optimal_params_vs_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_optimal_params_vs_snr.png")

# =========================================================================
# Plot 3: ODS vs Np at each SNR (WVF d=2, Ns=18)
# =========================================================================
print("Generating ODS vs Np by SNR...")

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(ALL_SNRS)))

for si, snr in enumerate(ALL_SNRS):
    if snr not in all_data:
        continue
    data = all_data[snr]
    xs, ys = [], []
    for r in data["wvf"]:
        if r["d"] == 2 and r["Ns"] == 18:
            xs.append(r["Np"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        label = "clean" if snr == "inf" else f"SNR={snr}"
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=colors[si], markersize=4, label=label)

ax.set_xlabel("$N_p$ (support size)", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("WVF ($d$=2, $N_s$=18): ODS vs $N_p$ at Each SNR", fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_np_by_snr_wvf.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_ods_vs_np_by_snr_wvf.png")

# =========================================================================
# Plot 4: ODS vs m at each SNR (LF Np=100, Ns=18)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for si, snr in enumerate(ALL_SNRS):
    if snr not in all_data:
        continue
    data = all_data[snr]
    xs, ys = [], []
    for r in data["lf"]:
        if r["Np"] == 100 and r["Ns"] == 18 and r["ods"] >= 0:
            xs.append(r["m"])
            ys.append(r["ods"])
    if xs:
        order = np.argsort(xs)
        label = "clean" if snr == "inf" else f"SNR={snr}"
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=colors[si], markersize=5, label=label)

ax.set_xlabel("Line half-width $m$", fontsize=12)
ax.set_ylabel("ODS F-score", fontsize=12)
ax.set_title("LF ($N_p$=100, $N_s$=18): ODS vs $m$ at Each SNR", fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plot_ods_vs_m_by_snr_lf.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_ods_vs_m_by_snr_lf.png")

# =========================================================================
# Plot 5: WVF vs LF gap across SNR
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

gap_best = [bl - bw for bw, bl in zip(best_ods_wvf, best_ods_lf)]
gap_bagan = [bl - bw for bw, bl in zip(bagan_ods_wvf, bagan_ods_lf)]

ax.bar(np.arange(len(snr_vals)) - 0.15, gap_best, 0.3, label="Best LF - Best WVF", color="#377eb8", alpha=0.7)
ax.bar(np.arange(len(snr_vals)) + 0.15, gap_bagan, 0.3, label="Bagan LF - Bagan WVF", color="darkorange", alpha=0.7)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(range(len(snr_vals)))
ax.set_xticklabels(snr_labels, fontsize=9)
ax.set_xlabel("SNR", fontsize=12)
ax.set_ylabel("ODS gap (LF - WVF)", fontsize=12)
ax.set_title("LF Advantage Over WVF vs SNR", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "plot_lf_vs_wvf_gap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_lf_vs_wvf_gap.png")

# =========================================================================
# Plot 6: Heatmap of best ODS by (SNR, Np) for WVF d=2
# =========================================================================
print("Generating SNR heatmaps...")

nps = [15, 25, 50, 100, 160, 250, 500]
fig, ax = plt.subplots(figsize=(10, 5))
hm = np.full((len(ALL_SNRS), len(nps)), np.nan)

for si, snr in enumerate(ALL_SNRS):
    if snr not in all_data:
        continue
    data = all_data[snr]
    for ni, np_val in enumerate(nps):
        r = next((x for x in data["wvf"] if x["Np"] == np_val and x["d"] == 2 and x["Ns"] == 18), None)
        if r:
            hm[si, ni] = r["ods"]

im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.2, vmax=0.85)
ax.set_xticks(range(len(nps)))
ax.set_xticklabels(nps)
ax.set_yticks(range(len(ALL_SNRS)))
ax.set_yticklabels(["clean" if s == "inf" else f"{s}" for s in ALL_SNRS])
ax.set_xlabel("$N_p$ (support size)", fontsize=12)
ax.set_ylabel("SNR", fontsize=12)
ax.set_title("WVF ($d$=2, $N_s$=18): ODS by SNR and $N_p$", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS")
for i in range(len(ALL_SNRS)):
    for j in range(len(nps)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < 0.5 else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=8, color=c)
fig.tight_layout()
fig.savefig(OUT / "plot_heatmap_snr_np_wvf.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_heatmap_snr_np_wvf.png")

# =========================================================================
# Plot 7: Heatmap of best ODS by (SNR, m) for LF Np=100
# =========================================================================
ms = [1, 2, 3, 5, 7, 10, 14, 20]
fig, ax = plt.subplots(figsize=(10, 5))
hm = np.full((len(ALL_SNRS), len(ms)), np.nan)

for si, snr in enumerate(ALL_SNRS):
    if snr not in all_data:
        continue
    data = all_data[snr]
    for mi, m_val in enumerate(ms):
        r = next((x for x in data["lf"] if x["Np"] == 100 and x["m"] == m_val and x["Ns"] == 18 and x["ods"] >= 0), None)
        if r:
            hm[si, mi] = r["ods"]

im = ax.imshow(hm, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0.2, vmax=0.85)
ax.set_xticks(range(len(ms)))
ax.set_xticklabels(ms)
ax.set_yticks(range(len(ALL_SNRS)))
ax.set_yticklabels(["clean" if s == "inf" else f"{s}" for s in ALL_SNRS])
ax.set_xlabel("Line half-width $m$", fontsize=12)
ax.set_ylabel("SNR", fontsize=12)
ax.set_title("LF ($N_p$=100, $N_s$=18): ODS by SNR and $m$", fontsize=13)
fig.colorbar(im, ax=ax, label="ODS")
for i in range(len(ALL_SNRS)):
    for j in range(len(ms)):
        if not np.isnan(hm[i, j]):
            c = "white" if hm[i, j] < 0.5 else "black"
            ax.text(j, i, f"{hm[i, j]:.3f}", ha="center", va="center", fontsize=8, color=c)
fig.tight_layout()
fig.savefig(OUT / "plot_heatmap_snr_m_lf.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  plot_heatmap_snr_m_lf.png")

print(f"\nAll plots saved to {OUT}")
