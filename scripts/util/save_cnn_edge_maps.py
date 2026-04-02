"""Save WideFieldNet edge maps at each SNR as .npy for later plotting."""

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

# Load image
img_clean = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
img_clean_norm = (img_clean / 255.0).astype(np.float32)
signal_amplitude = float(img_clean.max() - img_clean.min())
print(f"Image: {img_clean.shape}")

SNR_LEVELS = [0.5, 1.0, 2.0, 5.0, "inf"]

for snr in SNR_LEVELS:
    if snr == "inf":
        img_norm = img_clean_norm.copy()
    else:
        np.random.seed(42)
        noise = np.random.normal(0, signal_amplitude / snr / 255.0, img_clean_norm.shape).astype(np.float32)
        img_norm = np.clip(img_clean_norm + noise, 0, 1)

    t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mag, angle, edge_prob = model.predict(t)

    edge_np = edge_prob[0, 0].cpu().numpy()
    snr_str = "inf" if snr == "inf" else f"{snr:.2f}"
    np.save(OUT / f"cnn_edges_snr_{snr_str}.npy", edge_np)
    print(f"  SNR={snr_str}: saved, range=[{edge_np.min():.4f}, {edge_np.max():.4f}]")

# Also save the noisy images for plotting
for snr in SNR_LEVELS:
    if snr == "inf":
        img = img_clean.copy()
    else:
        np.random.seed(42)
        noise = np.random.normal(0, signal_amplitude / snr, img_clean.shape)
        img = np.clip(img_clean + noise, 0, 255)
    snr_str = "inf" if snr == "inf" else f"{snr:.2f}"
    np.save(OUT / f"noisy_snr_{snr_str}.npy", img)

print("Done")
