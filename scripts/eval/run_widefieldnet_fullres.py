"""Run WideFieldNet on full-resolution BIPED image (no tiling, no resizing)
at each SNR level. Compare against WVF/LF ablation results.
"""

import sys
import os
sys.path.insert(0, "/work/jvaught/edge-cnn")

import json
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
from src.models.wide_field_net import WideFieldNet
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path("/home/jvaught/edge-detection-filter-critique")
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_ablation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_THRESH = 1001

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# Load model
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
n_params = sum(p.numel() for p in model.parameters())
print(f"WideFieldNet loaded ({n_params:,} params)")

# Load image + GT
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_clean = np.mean(img_color, axis=2)
img_clean_norm = (img_clean / 255.0).astype(np.float32)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
H, W = img_clean.shape
print(f"Image: {H}x{W}, GT: {np.sum(gt_bool)} edge px")


def fullres_inference(model, img_norm):
    """Run model on full image — no tiling."""
    t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
    with torch.no_grad():
        mag, angle, edge_prob = model.predict(t)
    return (
        mag[0, 0].cpu().numpy(),
        angle[0, 0].cpu().numpy(),
        edge_prob[0, 0].cpu().numpy(),
    )


SNR_LEVELS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, float("inf")]

results = []
edge_maps = {}

for snr in SNR_LEVELS:
    if snr == float("inf"):
        img_norm = img_clean_norm.copy()
        snr_label = "clean"
    else:
        np.random.seed(42)
        noise_std = signal_amplitude / snr
        noise = np.random.normal(0, noise_std / 255.0, img_clean_norm.shape).astype(np.float32)
        img_norm = np.clip(img_clean_norm + noise, 0, 1)
        snr_label = f"SNR={snr}"

    print(f"\n{snr_label}...")

    pred_mag, pred_angle, pred_edges = fullres_inference(model, img_norm)
    edge_maps[snr] = pred_edges

    # ODS using edge probability map
    ods_edge, ois_edge, _, _ = compute_ods_ois(
        pred_edges, gt_bool.astype(np.float64),
        n_thresholds=N_THRESH, match_radius=3)

    # ODS using gradient magnitude map
    ods_mag, ois_mag, _, _ = compute_ods_ois(
        pred_mag, gt_bool.astype(np.float64),
        n_thresholds=N_THRESH, match_radius=3)

    results.append({
        "snr": snr if snr != float("inf") else "inf",
        "ods_edge": float(ods_edge),
        "ois_edge": float(ois_edge),
        "ods_mag": float(ods_mag),
        "ois_mag": float(ois_mag),
    })
    print(f"  Edge prob ODS={ods_edge:.4f}  Grad mag ODS={ods_mag:.4f}")

# Save results
with open(OUT / "widefieldnet_fullres_results.json", "w") as f:
    json.dump({"model": "WideFieldNet", "inference": "full_resolution",
               "image_shape": [H, W], "n_params": n_params, "results": results}, f, indent=2)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"{'SNR':>6s} {'WFN edge':>10s} {'WFN mag':>10s} {'Best WVF':>10s} {'Best LF':>10s} {'Bagan WVF':>10s}")
print("-" * 80)

for r in results:
    snr = r["snr"]
    snr_label = "clean" if snr == "inf" else f"{snr}"

    if snr == "inf":
        abl_path = ROOT / "outputs" / "biped_ablation" / "ablation_metrics.json"
    else:
        abl_path = OUT / f"snr_{float(snr):.2f}_ablation.json"

    bw_ods = bl_ods = pw_ods = "---"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        bw = max(abl["wvf"], key=lambda x: x["ods"])
        bl = max((x for x in abl["lf"] if x["ods"] >= 0), key=lambda x: x["ods"]) if abl["lf"] else None
        pw = next((x for x in abl["wvf"] if x["Np"] == 250 and x["d"] == 4 and x["Ns"] == 18), None)
        bw_ods = f"{bw['ods']:.4f}"
        bl_ods = f"{bl['ods']:.4f}" if bl else "---"
        pw_ods = f"{pw['ods']:.4f}" if pw else "---"

    print(f"{snr_label:>6s} {r['ods_edge']:>10.4f} {r['ods_mag']:>10.4f} {bw_ods:>10s} {bl_ods:>10s} {pw_ods:>10s}")
print(f"{'='*80}")

# ---------------------------------------------------------------------------
# Edge map visualization
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("\nmatplotlib not available, skipping plots")

if not HAS_MPL:
    sys.exit(0)

print("\nGenerating edge map visualization...")

fig, axes = plt.subplots(3, len(SNR_LEVELS), figsize=(3.5 * len(SNR_LEVELS), 10))

for si, snr in enumerate(SNR_LEVELS):
    snr_label = "clean" if snr == float("inf") else f"SNR={snr}"

    # Row 0: noisy input
    if snr == float("inf"):
        img_show = img_clean_norm
    else:
        np.random.seed(42)
        noise = np.random.normal(0, signal_amplitude / snr / 255.0, img_clean_norm.shape).astype(np.float32)
        img_show = np.clip(img_clean_norm + noise, 0, 1)

    axes[0, si].imshow(img_show, cmap="gray", vmin=0, vmax=1)
    axes[0, si].set_title(snr_label, fontsize=10, fontweight="bold")
    axes[0, si].axis("off")

    # Row 1: edge probability
    axes[1, si].imshow(edge_maps[snr], cmap="gray", vmin=0, vmax=1)
    r = next(x for x in results if x["snr"] == (snr if snr != float("inf") else "inf"))
    axes[1, si].set_title(f"ODS={r['ods_edge']:.3f}", fontsize=9)
    axes[1, si].axis("off")

    # Row 2: thresholded at 0.5
    axes[2, si].imshow(edge_maps[snr] > 0.5, cmap="gray")
    axes[2, si].set_title("threshold=0.5", fontsize=9)
    axes[2, si].axis("off")

axes[0, 0].set_ylabel("Input", fontsize=11, rotation=0, labelpad=50, va="center")
axes[1, 0].set_ylabel("Edge Prob", fontsize=11, rotation=0, labelpad=50, va="center")
axes[2, 0].set_ylabel("Edges >0.5", fontsize=11, rotation=0, labelpad=50, va="center")

fig.suptitle(f"WideFieldNet Full-Res ({H}×{W}) — Edge Detection Across SNR", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "plot_widefieldnet_fullres_snr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved plot_widefieldnet_fullres_snr.png")
