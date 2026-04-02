"""Run WideFieldNet (edge-cnn) on BIPED v1 RGB_008 at each SNR level.

Uses tiled 64x64 inference and computes ODS for comparison with WVF/LF.
"""

import sys
import os
sys.path.insert(0, "/work/jvaught/edge-cnn")

import json
import math
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
PATCH = 64
N_THRESH = 1001

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
print(f"WideFieldNet loaded ({sum(p.numel() for p in model.parameters())} params)")

# Load image + GT
img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
# Normalize to [0, 1] for the model
img_clean_norm = img_clean / 255.0
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
print(f"Image: {img_clean.shape}, GT: {np.sum(gt_bool)} edge px")


def tile_inference(model, img_norm):
    """Run model on image by tiling into 64x64 patches."""
    H, W = img_norm.shape
    pad_h = (PATCH - H % PATCH) % PATCH
    pad_w = (PATCH - W % PATCH) % PATCH
    padded = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape

    n_rows = pH // PATCH
    n_cols = pW // PATCH

    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = padded[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            patches.append(patch)

    batch = torch.from_numpy(np.array(patches, dtype=np.float32)).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        _, _, edges_out = model.predict(batch)
    edges_np = edges_out[:, 0].cpu().numpy()

    edge_map = np.zeros((pH, pW), dtype=np.float32)
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            edge_map[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = edges_np[idx]
            idx += 1

    return edge_map[:H, :W]


# SNR levels
SNR_LEVELS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, float("inf")]

results = []

for snr in SNR_LEVELS:
    if snr == float("inf"):
        img_noisy = img_clean.copy()
        img_norm = img_clean_norm.copy()
        snr_label = "clean"
    else:
        np.random.seed(42)
        noise_std = signal_amplitude / snr
        noise = np.random.normal(0, noise_std, img_clean.shape)
        img_noisy = np.clip(img_clean + noise, 0, 255)
        img_norm = np.clip(img_clean_norm + noise / 255.0, 0, 1).astype(np.float32)
        snr_label = f"SNR={snr}"

    print(f"\n{snr_label}...")

    # Run WideFieldNet
    pred_edges = tile_inference(model, img_norm)

    # Compute ODS
    ods, ois, _, _ = compute_ods_ois(
        pred_edges, gt_bool.astype(np.float64),
        n_thresholds=N_THRESH, match_radius=3)

    results.append({
        "snr": snr if snr != float("inf") else "inf",
        "ods": float(ods),
        "ois": float(ois),
    })
    print(f"  WideFieldNet ODS={ods:.4f} OIS={ois:.4f}")

# Save
with open(OUT / "widefieldnet_snr_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary table with WVF/LF comparison
print(f"\n{'='*70}")
print(f"{'SNR':>6s} {'WideFieldNet':>13s} {'Best WVF':>10s} {'Best LF':>10s} {'Bagan WVF':>10s} {'Bagan LF':>10s}")
print("-" * 70)

for r in results:
    snr = r["snr"]
    snr_label = "clean" if snr == "inf" else f"{snr}"

    # Load corresponding ablation data for comparison
    if snr == "inf":
        abl_path = ROOT / "outputs" / "biped_ablation" / "ablation_metrics.json"
    else:
        abl_path = OUT / f"snr_{float(snr):.2f}_ablation.json"

    bw_ods = bl_ods = pw_ods = pl_ods = "---"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        bw = max(abl["wvf"], key=lambda x: x["ods"])
        bl = max((x for x in abl["lf"] if x["ods"] >= 0), key=lambda x: x["ods"]) if abl["lf"] else None
        pw = next((x for x in abl["wvf"] if x["Np"] == 250 and x["d"] == 4 and x["Ns"] == 18), None)
        pl = next((x for x in abl["lf"] if x["Np"] == 250 and x.get("m") == 14 and x["Ns"] == 18), None)
        bw_ods = f"{bw['ods']:.4f}"
        bl_ods = f"{bl['ods']:.4f}" if bl else "---"
        pw_ods = f"{pw['ods']:.4f}" if pw else "---"
        pl_ods = f"{pl['ods']:.4f}" if pl and pl["ods"] >= 0 else "---"

    print(f"{snr_label:>6s} {r['ods']:>13.4f} {bw_ods:>10s} {bl_ods:>10s} {pw_ods:>10s} {pl_ods:>10s}")

print(f"{'='*70}")
