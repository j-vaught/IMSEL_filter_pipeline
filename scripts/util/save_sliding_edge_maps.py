"""Save sliding window, tiled, and full-res CNN edge maps side by side."""

import sys
sys.path.insert(0, "/work/jvaught/edge-cnn")

import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image

from src.models.wide_field_net import WideFieldNet

ROOT = Path("/home/jvaught/edge-detection-filter-critique")
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "snr_ablation" / "cnn_edge_maps"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH = 64

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
print(f"WideFieldNet loaded on {DEVICE}")

img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
img_clean_norm = (img_clean / 255.0).astype(np.float32)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
signal_amplitude = float(img_clean.max() - img_clean.min())
H, W = img_clean.shape
print(f"Image: {H}x{W}")


def tiled_inference(model, img_norm):
    H, W = img_norm.shape
    pad_h = (PATCH - H % PATCH) % PATCH
    pad_w = (PATCH - W % PATCH) % PATCH
    padded = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape
    patches = []
    positions = []
    for r in range(0, pH, PATCH):
        for c in range(0, pW, PATCH):
            patches.append(padded[r:r+PATCH, c:c+PATCH])
            positions.append((r, c))
    batch = torch.from_numpy(np.array(patches, dtype=np.float32)).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        _, _, edges_out = model.predict(batch)
    edges_np = edges_out[:, 0].cpu().numpy()
    out = np.zeros((pH, pW), dtype=np.float32)
    for idx, (r, c) in enumerate(positions):
        out[r:r+PATCH, c:c+PATCH] = edges_np[idx]
    return out[:H, :W]


def sliding_inference(model, img_norm, stride=32):
    H, W = img_norm.shape
    pad_h = max((PATCH - H % stride) % stride, PATCH - (H % PATCH) if H % PATCH else 0)
    pad_w = max((PATCH - W % stride) % stride, PATCH - (W % PATCH) if W % PATCH else 0)
    padded = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape
    positions = []
    for y in range(0, pH - PATCH + 1, stride):
        for x in range(0, pW - PATCH + 1, stride):
            positions.append((y, x))
    patches = [padded[y:y+PATCH, x:x+PATCH] for y, x in positions]
    edge_sum = np.zeros((pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)
    batch_size = 256
    all_edges = []
    for i in range(0, len(patches), batch_size):
        batch = torch.from_numpy(np.array(patches[i:i+batch_size], dtype=np.float32)).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            _, _, edges_out = model.predict(batch)
        all_edges.append(edges_out[:, 0].cpu().numpy())
    all_edges = np.concatenate(all_edges)
    for idx, (y, x) in enumerate(positions):
        edge_sum[y:y+PATCH, x:x+PATCH] += all_edges[idx]
        count[y:y+PATCH, x:x+PATCH] += 1
    count = np.maximum(count, 1)
    return (edge_sum / count)[:H, :W].astype(np.float32)


def fullres_inference(model, img_norm):
    t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, _, edge_prob = model.predict(t)
    return edge_prob[0, 0].cpu().numpy()


SNR_LEVELS = [0.5, 1.0, 2.0, 5.0, "inf"]

for snr in SNR_LEVELS:
    snr_str = "inf" if snr == "inf" else f"{snr:.2f}"
    if snr == "inf":
        img_norm = img_clean_norm.copy()
    else:
        np.random.seed(42)
        noise = np.random.normal(0, signal_amplitude / snr / 255.0, img_clean_norm.shape).astype(np.float32)
        img_norm = np.clip(img_clean_norm + noise, 0, 1)

    print(f"\nSNR={snr_str}...")
    tiled = tiled_inference(model, img_norm)
    sliding = sliding_inference(model, img_norm, stride=32)
    fullres = fullres_inference(model, img_norm)

    np.save(OUT / f"tiled_snr_{snr_str}.npy", tiled)
    np.save(OUT / f"sliding_snr_{snr_str}.npy", sliding)
    np.save(OUT / f"fullres_snr_{snr_str}.npy", fullres)
    print(f"  tiled: [{tiled.min():.3f}, {tiled.max():.3f}]")
    print(f"  sliding: [{sliding.min():.3f}, {sliding.max():.3f}]")
    print(f"  fullres: [{fullres.min():.3f}, {fullres.max():.3f}]")

print("\nDone")
