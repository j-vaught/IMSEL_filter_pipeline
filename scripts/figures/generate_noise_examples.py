"""Generate noise type example images for the proposal document."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "noise_ablation"
OUT.mkdir(exist_ok=True)

img = np.mean(np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg")), axis=2)
# Crop a 300x400 region for cleaner visualization
crop = img[200:500, 400:800]
amplitude = crop.max() - crop.min()

rng = np.random.default_rng(42)

def add_gaussian(img, snr): return np.clip(img + rng.normal(0, amplitude/snr, img.shape), 0, 255)
def add_salt_pepper(img, snr):
    d = 1/(1+snr); out = img.copy(); m = rng.random(img.shape)
    out[m < d/2] = 0; out[m > 1-d/2] = 255; return out
def add_poisson(img, snr):
    s = snr*snr; return np.clip(rng.poisson(np.clip(img/255*s, 0.001, None))/s*255, 0, 255)
def add_speckle(img, snr): return np.clip(img * (1 + rng.normal(0, 1/snr, img.shape)), 0, 255)
def add_uniform(img, snr): return np.clip(img + rng.uniform(-amplitude/snr, amplitude/snr, img.shape), 0, 255)

noise_fns = [
    ("Clean", None),
    ("Gaussian", add_gaussian),
    ("Salt & Pepper", add_salt_pepper),
    ("Poisson", add_poisson),
    ("Speckle", add_speckle),
    ("Uniform", add_uniform),
]

snr_vals = [0.5, 1.0, 2.0]

# ---- Figure 1: All noise types at SNR=1.0 ----
fig, axes = plt.subplots(1, 6, figsize=(24, 4))
for i, (name, fn) in enumerate(noise_fns):
    rng = np.random.default_rng(42)
    if fn is None:
        axes[i].imshow(crop, cmap="gray", vmin=0, vmax=255)
    else:
        axes[i].imshow(fn(crop, 1.0), cmap="gray", vmin=0, vmax=255)
    axes[i].set_title(name, fontsize=12)
    axes[i].axis("off")
fig.suptitle("Noise Types at SNR = 1.0 (BIPED v1 RGB_008 crop)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_noise_types_snr1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_noise_types_snr1.png")

# ---- Figure 2: Each noise type across SNR levels ----
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
noise_only = noise_fns[1:]  # skip clean

for ri, (name, fn) in enumerate(noise_only):
    # Clean reference
    axes[ri, 0].imshow(crop, cmap="gray", vmin=0, vmax=255)
    axes[ri, 0].set_title("Clean" if ri == 0 else "", fontsize=10)
    axes[ri, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=80, va="center")
    axes[ri, 0].axis("off")

    for ci, snr in enumerate(snr_vals):
        rng = np.random.default_rng(42)
        noisy = fn(crop, snr)
        axes[ri, ci+1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
        axes[ri, ci+1].set_title(f"SNR={snr}" if ri == 0 else "", fontsize=10)
        axes[ri, ci+1].axis("off")

fig.suptitle("Noise Degradation Across SNR Levels", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig_noise_snr_grid.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_noise_snr_grid.png")

# ---- Figure 3: Pixel intensity histograms for each noise type at SNR=1.0 ----
fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
for i, (name, fn) in enumerate(noise_only):
    rng = np.random.default_rng(42)
    noisy = fn(crop, 1.0)
    diff = noisy - crop
    axes[i].hist(diff.ravel(), bins=100, density=True, alpha=0.7, color="steelblue")
    axes[i].set_title(f"{name}\n$\\mu$={diff.mean():.1f}, $\\sigma$={diff.std():.1f}", fontsize=10)
    axes[i].set_xlabel("Pixel difference", fontsize=9)
    axes[i].set_xlim(-300, 300)
    if i == 0:
        axes[i].set_ylabel("Density", fontsize=10)
fig.suptitle("Noise Distribution (noisy $-$ clean) at SNR = 1.0", fontsize=13, y=1.05)
fig.tight_layout()
fig.savefig(OUT / "fig_noise_distributions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_noise_distributions.png")

print("Done")
