"""Run WideFieldNet with 50% overlapping sliding window on BIPED v1 RGB_008.

Each pixel gets predictions from up to 4 overlapping patches.
Predictions are averaged in overlap regions to eliminate seam artifacts.
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
PATCH = 64
STRIDE = 32  # 50% overlap
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
n_params = sum(p.numel() for p in model.parameters())
print(f"WideFieldNet loaded ({n_params:,} params)")

# Load image + GT
img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
img_clean_norm = (img_clean / 255.0).astype(np.float32)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
H, W = img_clean.shape
print(f"Image: {H}x{W}, GT: {np.sum(gt_bool)} edge px")
print(f"Patch: {PATCH}x{PATCH}, Stride: {STRIDE} (50% overlap)")


def sliding_inference(model, img_norm):
    """Run model with 50% overlapping sliding window, average predictions."""
    H, W = img_norm.shape

    # Pad to ensure we cover the full image
    pad_h = (PATCH - H % STRIDE) % STRIDE
    pad_w = (PATCH - W % STRIDE) % STRIDE
    # Extra padding so last patches don't go out of bounds
    pad_h = max(pad_h, PATCH - (H % PATCH) if H % PATCH != 0 else 0)
    pad_w = max(pad_w, PATCH - (W % PATCH) if W % PATCH != 0 else 0)
    padded = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape

    # Collect all patch positions
    positions = []
    for y in range(0, pH - PATCH + 1, STRIDE):
        for x in range(0, pW - PATCH + 1, STRIDE):
            positions.append((y, x))

    print(f"    {len(positions)} patches ({pH}x{pW} padded)")

    # Extract and batch all patches
    patches = []
    for y, x in positions:
        patches.append(padded[y:y + PATCH, x:x + PATCH])

    # Run inference in batches to avoid OOM
    batch_size = 256
    all_edges = []
    all_mags = []
    for i in range(0, len(patches), batch_size):
        batch = torch.from_numpy(np.array(patches[i:i + batch_size], dtype=np.float32))
        batch = batch.unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            mag, _, edge_prob = model.predict(batch)
        all_edges.append(edge_prob[:, 0].cpu().numpy())
        all_mags.append(mag[:, 0].cpu().numpy())

    all_edges = np.concatenate(all_edges, axis=0)
    all_mags = np.concatenate(all_mags, axis=0)

    # Stitch with averaging
    edge_sum = np.zeros((pH, pW), dtype=np.float64)
    mag_sum = np.zeros((pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)

    for idx, (y, x) in enumerate(positions):
        edge_sum[y:y + PATCH, x:x + PATCH] += all_edges[idx]
        mag_sum[y:y + PATCH, x:x + PATCH] += all_mags[idx]
        count[y:y + PATCH, x:x + PATCH] += 1

    count = np.maximum(count, 1)
    edge_avg = (edge_sum / count)[:H, :W].astype(np.float32)
    mag_avg = (mag_sum / count)[:H, :W].astype(np.float32)

    return mag_avg, edge_avg


SNR_LEVELS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, float("inf")]

results = []

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

    pred_mag, pred_edges = sliding_inference(model, img_norm)

    ods_edge, ois_edge, _, _ = compute_ods_ois(
        pred_edges, gt_bool.astype(np.float64),
        n_thresholds=N_THRESH, match_radius=3)

    ods_mag, ois_mag, _, _ = compute_ods_ois(
        pred_mag, gt_bool.astype(np.float64),
        n_thresholds=N_THRESH, match_radius=3)

    results.append({
        "snr": snr if snr != float("inf") else "inf",
        "ods_edge": float(ods_edge),
        "ods_mag": float(ods_mag),
    })
    print(f"  Edge prob ODS={ods_edge:.4f}  Grad mag ODS={ods_mag:.4f}")

# Save
with open(OUT / "widefieldnet_sliding_results.json", "w") as f:
    json.dump({"model": "WideFieldNet", "inference": "sliding_window",
               "patch": PATCH, "stride": STRIDE, "overlap": "50%",
               "image_shape": [H, W], "n_params": n_params,
               "results": results}, f, indent=2)

# Load previous results for comparison
tiled = {}
fullres = {}
tiled_path = OUT / "widefieldnet_snr_results.json"
fullres_path = OUT / "widefieldnet_fullres_results.json"
if tiled_path.exists():
    with open(tiled_path) as f:
        for r in json.load(f):
            tiled[r["snr"]] = r["ods"]
if fullres_path.exists():
    with open(fullres_path) as f:
        for r in json.load(f)["results"]:
            fullres[r["snr"]] = r["ods_edge"]

# Summary
print(f"\n{'='*80}")
print(f"{'SNR':>6s} {'Sliding':>10s} {'Tiled':>10s} {'FullRes':>10s} {'Best WVF':>10s} {'Best LF':>10s}")
print("-" * 80)

for r in results:
    snr = r["snr"]
    snr_label = "clean" if snr == "inf" else f"{snr}"

    t_ods = f"{tiled.get(snr, tiled.get(float(snr) if snr != 'inf' else 'x', 0)):.4f}" if snr in tiled else "---"
    f_ods = f"{fullres.get(snr, 0):.4f}" if snr in fullres else "---"

    if snr == "inf":
        abl_path = ROOT / "outputs" / "biped_ablation" / "ablation_metrics.json"
    else:
        abl_path = OUT / f"snr_{float(snr):.2f}_ablation.json"

    bw_ods = bl_ods = "---"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        bw = max(abl["wvf"], key=lambda x: x["ods"])
        bl = max((x for x in abl["lf"] if x["ods"] >= 0), key=lambda x: x["ods"]) if abl["lf"] else None
        bw_ods = f"{bw['ods']:.4f}"
        bl_ods = f"{bl['ods']:.4f}" if bl else "---"

    print(f"{snr_label:>6s} {r['ods_edge']:>10.4f} {t_ods:>10s} {f_ods:>10s} {bw_ods:>10s} {bl_ods:>10s}")
print(f"{'='*80}")
