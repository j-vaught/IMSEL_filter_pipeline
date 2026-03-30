"""Compute ODS for all model edge maps on BIPED v1 RGB_008."""

import numpy as np
from PIL import Image
from pathlib import Path
from edgecritic.evaluation.metrics import compute_ods_ois
from edgecritic.wvf import wvf_image

ROOT = Path(__file__).resolve().parent.parent
gt = np.array(Image.open(ROOT / "datasets/BIPED/BIPED/BIPED/edges/edge_maps/test/rgbr/RGB_008.png").convert("L")) > 128

models = {
    "DexiNed": ROOT / "outputs/model_test/DexiNed_edge.png",
    "TEED": ROOT / "outputs/model_test/TEED_edge.png",
    "pidinet": ROOT / "outputs/model_test/pidinet_edge.png",
    "NBED": ROOT / "outputs/model_test/NBED_edge.png",
    "DiffusionEdge": ROOT / "outputs/model_test/DiffusionEdge_edge.png",
}

print(f"{'Model':<20s} {'ODS':>7s}")
print("-" * 28)

for name, path in models.items():
    if not path.exists():
        print(f"{name:<20s}     N/A")
        continue
    edge = np.array(Image.open(path).convert("L")).astype(np.float64) / 255.0
    ods, _, _, _ = compute_ods_ois(edge, gt.astype(np.float64), n_thresholds=1001, match_radius=3)
    print(f"{name:<20s} {ods:>7.4f}")

# WVF
img = np.mean(np.array(Image.open(ROOT / "datasets/BIPED/BIPED/BIPED/edges/imgs/test/rgbr/RGB_008.jpg")), axis=2)

import torch
backend = "cuda" if torch.cuda.is_available() else "cpu"

wvf_mag = wvf_image(img, np_count=25, order=2, n_orientations=18, backend=backend).gradient_mag
ods, _, _, _ = compute_ods_ois(wvf_mag, gt.astype(np.float64), n_thresholds=1001, match_radius=3)
print(f"{'WVF (Np=25 d=2)':<20s} {ods:>7.4f}")

# Bagan WVF
wvf_bagan = wvf_image(img, np_count=250, order=4, n_orientations=18, backend=backend).gradient_mag
ods, _, _, _ = compute_ods_ois(wvf_bagan, gt.astype(np.float64), n_thresholds=1001, match_radius=3)
print(f"{'WVF (Bagan)':<20s} {ods:>7.4f}")
