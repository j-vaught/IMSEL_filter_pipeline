"""Side-by-side comparison: WideFieldNet vs best WVF vs best LF vs Bagan at each SNR."""

import sys
sys.path.insert(0, "/work/jvaught/edge-cnn")

import json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.models.wide_field_net import WideFieldNet
from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image

ROOT = Path("/home/jvaught/edge-detection-filter-critique")
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_ablation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")

# Load WideFieldNet
with open("/work/jvaught/edge-cnn/configs/base.yaml") as f:
    config = yaml.safe_load(f)
mc = config["model"]
model = WideFieldNet(
    encoder_channels=mc["encoder_channels"],
    bottleneck_dilations=mc["bottleneck_dilations"],
    decoder_channels=mc["decoder_channels"],
    dropout=0.0,
)
state = torch.load("/work/jvaught/edge-cnn/checkpoints/best.pt", map_location=DEVICE)
model.load_state_dict(state["model_state_dict"])
model.to(DEVICE)
model.eval()
print("WideFieldNet loaded")

# Load image + GT
img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
img_clean_norm = (img_clean / 255.0).astype(np.float32)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
H, W = img_clean.shape


def run_cnn(img_norm):
    t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, _, edge_prob = model.predict(t)
    return edge_prob[0, 0].cpu().numpy()


# Load SNR ablation results to find best configs at each SNR
SNR_SHOW = [0.5, 1.0, 2.0, 5.0, "inf"]

# Load WFN results
wfn_results = {}
wfn_path = OUT / "widefieldnet_fullres_results.json"
if wfn_path.exists():
    with open(wfn_path) as f:
        for r in json.load(f)["results"]:
            wfn_results[r["snr"]] = r

print("Generating comparison grid...")

# 6 rows: Input, GT, CNN, Best WVF, Best LF, Bagan LF
# N columns: one per SNR
n_cols = len(SNR_SHOW)
fig, axes = plt.subplots(6, n_cols, figsize=(4.5 * n_cols, 24))

row_labels = ["Input", "Ground Truth", "WideFieldNet", "Best WVF", "Best LF", "Bagan LF (Np=250 m=14)"]

for si, snr in enumerate(SNR_SHOW):
    snr_label = "clean" if snr == "inf" else f"SNR={snr}"

    # Make noisy image
    if snr == "inf":
        img_noisy = img_clean.copy()
        img_norm = img_clean_norm.copy()
    else:
        np.random.seed(42)
        noise = np.random.normal(0, signal_amplitude / snr, img_clean.shape)
        img_noisy = np.clip(img_clean + noise, 0, 255)
        img_norm = np.clip(img_clean_norm + noise / 255.0, 0, 1).astype(np.float32)

    # Load best configs from ablation
    if snr == "inf":
        abl_path = ROOT / "outputs" / "biped_ablation" / "ablation_metrics.json"
    else:
        abl_path = OUT / f"snr_{float(snr):.2f}_ablation.json"

    best_wvf_cfg = best_lf_cfg = bagan_lf_cfg = None
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        best_wvf_cfg = max(abl["wvf"], key=lambda x: x["ods"])
        valid_lf = [r for r in abl["lf"] if r["ods"] >= 0]
        best_lf_cfg = max(valid_lf, key=lambda x: x["ods"]) if valid_lf else None
        bagan_lf_cfg = next((r for r in abl["lf"] if r["Np"] == 250 and r.get("m") == 14 and r["Ns"] == 18), None)

    # Row 0: Input
    axes[0, si].imshow(img_noisy, cmap="gray")
    axes[0, si].set_title(snr_label, fontsize=12, fontweight="bold")
    axes[0, si].axis("off")

    # Row 1: GT
    axes[1, si].imshow(gt_bool, cmap="gray")
    axes[1, si].axis("off")

    # Row 2: CNN
    cnn_out = run_cnn(img_norm)
    wfn_ods = wfn_results.get(snr if snr != "inf" else "inf", {}).get("ods_edge", 0)
    axes[2, si].imshow(cnn_out, cmap="gray", vmin=0, vmax=1)
    axes[2, si].set_title(f"ODS={wfn_ods:.3f}", fontsize=9)
    axes[2, si].axis("off")

    # Row 3: Best WVF
    if best_wvf_cfg:
        mag = wvf_image(img_noisy, np_count=best_wvf_cfg["Np"], order=best_wvf_cfg["d"],
                        n_orientations=best_wvf_cfg["Ns"], backend=backend).gradient_mag
        axes[3, si].imshow(mag, cmap="gray")
        axes[3, si].set_title(f"Np={best_wvf_cfg['Np']} d={best_wvf_cfg['d']}\nODS={best_wvf_cfg['ods']:.3f}", fontsize=8)
    axes[3, si].axis("off")

    # Row 4: Best LF
    if best_lf_cfg:
        mag = lf_image(img_noisy, half_width=best_lf_cfg["m"], np_count=best_lf_cfg["Np"],
                       order=4, n_orientations=best_lf_cfg["Ns"],
                       backend=backend, max_vram_gb=30).gradient_mag
        axes[4, si].imshow(mag, cmap="gray")
        axes[4, si].set_title(f"Np={best_lf_cfg['Np']} m={best_lf_cfg['m']}\nODS={best_lf_cfg['ods']:.3f}", fontsize=8)
    axes[4, si].axis("off")

    # Row 5: Bagan LF
    if bagan_lf_cfg and bagan_lf_cfg["ods"] >= 0:
        mag = lf_image(img_noisy, half_width=14, np_count=250, order=4,
                       n_orientations=18, backend=backend, max_vram_gb=30).gradient_mag
        axes[5, si].imshow(mag, cmap="gray")
        axes[5, si].set_title(f"ODS={bagan_lf_cfg['ods']:.3f}", fontsize=8)
    axes[5, si].axis("off")

# Row labels
for ri, label in enumerate(row_labels):
    axes[ri, 0].set_ylabel(label, fontsize=11, rotation=0, labelpad=120, va="center")

fig.suptitle("BIPED v1 RGB_008: WideFieldNet vs WVF/LF Across SNR", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_cnn_vs_wvf_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved plot_cnn_vs_wvf_comparison.png")
