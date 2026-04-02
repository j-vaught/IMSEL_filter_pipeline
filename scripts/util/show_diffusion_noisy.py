"""Show DiffusionEdge output on a noisy BIPED image at SNR=0.3."""

import numpy as np
import torch
import subprocess
import sys
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
OUT = ROOT / "outputs/model_test"
OUT.mkdir(exist_ok=True)

img_rgb = np.array(Image.open(BIPED / "imgs/test/rgbr/RGB_008.jpg").convert("RGB"))
gt = np.array(Image.open(BIPED / "edge_maps/test/rgbr/RGB_008.png").convert("L"))
img_gray = np.mean(img_rgb, axis=2)

# Add Gaussian noise at SNR=0.3
snr = 0.3
rng = np.random.default_rng(42)
sigma = (img_gray.max() - img_gray.min()) / snr
noisy_gray = np.clip(img_gray + rng.normal(0, sigma, img_gray.shape), 0, 255)
noisy_rgb = np.stack([noisy_gray]*3, axis=-1).astype(np.uint8)

# Run DiffusionEdge
de_dir = ROOT / "models/DiffusionEdge"
tmp_in = OUT / "de_tmp_in"
tmp_out = OUT / "de_tmp_out"
tmp_in.mkdir(exist_ok=True)
tmp_out.mkdir(exist_ok=True)
for f in tmp_out.glob("*"):
    f.unlink()

h, w = noisy_rgb.shape[:2]
new_h = max(((h + 15) // 16) * 16, 320)
new_w = max(((w + 15) // 16) * 16, 320)
Image.fromarray(noisy_rgb).resize((new_w, new_h), Image.BILINEAR).save(tmp_in / "tmp.jpg")

print(f"Running DiffusionEdge on {h}x{w} image at SNR={snr}...")
result = subprocess.run(
    [sys.executable, str(de_dir / "demo.py"),
     "--cfg", str(de_dir / "configs/BIPED_sample.yaml"),
     "--input_dir", str(tmp_in),
     "--pre_weight", str(de_dir / "weights/biped.pt"),
     "--out_dir", str(tmp_out),
     "--sampling_timesteps", "1", "--bs", "1"],
    capture_output=True, timeout=300, cwd=str(de_dir),
)
if result.returncode != 0:
    print("FAILED:", result.stderr.decode("utf-8", errors="replace")[-500:])
    sys.exit(1)

out_files = list(tmp_out.glob("*.png"))
edge = np.array(Image.open(out_files[0]).convert("L")).astype(np.float64) / 255.0
if edge.shape != (h, w):
    edge = np.array(Image.fromarray((edge * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0

# Also run WVF best clean and best noisy configs
from edgecritic.wvf import wvf_image
backend = "cuda" if torch.cuda.is_available() else "cpu"
wvf_clean_cfg = wvf_image(noisy_gray, np_count=25, order=2, n_orientations=18, backend=backend).gradient_mag
wvf_noisy_cfg = wvf_image(noisy_gray, np_count=500, order=2, n_orientations=18, backend=backend).gradient_mag

# Save comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0,0].imshow(img_rgb); axes[0,0].set_title("Clean Input", fontsize=12); axes[0,0].axis("off")
axes[0,1].imshow(noisy_gray, cmap="gray"); axes[0,1].set_title(f"Noisy (SNR={snr})", fontsize=12); axes[0,1].axis("off")
axes[0,2].imshow(gt, cmap="gray"); axes[0,2].set_title("Ground Truth", fontsize=12); axes[0,2].axis("off")

axes[1,0].imshow(edge, cmap="gray"); axes[1,0].set_title("DiffusionEdge", fontsize=12); axes[1,0].axis("off")
axes[1,1].imshow(wvf_clean_cfg / (wvf_clean_cfg.max()+1e-8), cmap="gray")
axes[1,1].set_title("WVF (Np=25 d=2, clean-optimal)", fontsize=12); axes[1,1].axis("off")
axes[1,2].imshow(wvf_noisy_cfg / (wvf_noisy_cfg.max()+1e-8), cmap="gray")
axes[1,2].set_title("WVF (Np=500 d=2, noise-optimal)", fontsize=12); axes[1,2].axis("off")

fig.suptitle(f"BIPED RGB_008 at SNR={snr}: DiffusionEdge vs WVF", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "diffusion_vs_wvf_snr03.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'diffusion_vs_wvf_snr03.png'}")
