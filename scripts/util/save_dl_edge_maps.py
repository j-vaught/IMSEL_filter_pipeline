"""Save edge map images from each DL model at key SNR levels.

Usage: python save_dl_edge_maps.py --model dexined
       python save_dl_edge_maps.py --model all
"""

import sys
import os
import argparse
import numpy as np
import torch
import cv2
import subprocess
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
IMG_PATH = BIPED / "imgs/test/rgbr/RGB_008.jpg"
GT_PATH = BIPED / "edge_maps/test/rgbr/RGB_008.png"
OUT = ROOT / "outputs" / "dl_edge_map_samples"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_rgb = np.array(Image.open(IMG_PATH).convert("RGB"))
gt = np.array(Image.open(GT_PATH).convert("L"))
img_gray = np.mean(img_rgb, axis=2)
signal_amp = img_gray.max() - img_gray.min()

SNR_SHOW = [0.3, 1.0, 2.0, 4.0, "inf"]


def make_noisy(snr):
    if snr == "inf":
        return img_rgb.copy(), img_gray.copy()
    rng = np.random.default_rng(42)
    noisy_gray = np.clip(img_gray + rng.normal(0, signal_amp / snr, img_gray.shape), 0, 255)
    noisy_rgb = np.stack([noisy_gray] * 3, axis=-1).astype(np.uint8)
    return noisy_rgb, noisy_gray


# --- Model runners (same as run_dl_noise_ablation.py) ---
def run_dexined(img_rgb):
    sys.path.insert(0, str(ROOT / "models/DexiNed"))
    from model import DexiNed
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(ROOT / "models/DexiNed/checkpoints/BIPED/10/10_model.pth", map_location=device))
    model.eval()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    hp, wp = (16 - h % 16) % 16, (16 - w % 16) % 16
    t = torch.from_numpy(np.pad(img_bgr, ((0, hp), (0, wp), (0, 0)), mode="reflect").transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(t)[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge

def run_teed(img_rgb):
    sys.path.insert(0, str(ROOT / "models/TEED"))
    from ted import TED
    model = TED().to(device)
    model.load_state_dict(torch.load(ROOT / "models/TEED/checkpoints/BIPED/7/7_model.pth", map_location=device))
    model.eval()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    hp, wp = (8 - h % 8) % 8, (8 - w % 8) % 8
    t = torch.from_numpy(np.pad(img_bgr, ((0, hp), (0, wp), (0, 0)), mode="reflect").transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(t)[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge

def run_pidinet(img_rgb):
    sys.path.insert(0, str(ROOT / "models/pidinet"))
    import models as pm
    import torchvision.transforms as T
    class A:
        config = "carv4"; sa = True; dil = True
    model = pm.pidinet(A()).to(device)
    s = torch.load(ROOT / "models/pidinet/trained_models/table5_pidinet.pth", map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in s.get("state_dict", s).items()})
    model.eval()
    t = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(T.ToTensor()(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(t)[-1]).squeeze().cpu().numpy()
    sys.path.pop(0)
    return edge

def run_nbed(img_rgb):
    nbed_dir = ROOT / "models/NBED"
    old = Path.cwd(); os.chdir(nbed_dir)
    sys.path.insert(0, str(nbed_dir))
    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model."): del sys.modules[k]
    from model.basemodel import Basemodel
    import torchvision.transforms as T
    model = Basemodel(encoder_name="DUL-M36", decoder_name="UNETP", head_name="default").to(device)
    s = torch.load(nbed_dir / "weights/nbed_biped.pth", map_location="cpu")
    s = {k.replace("module.", ""): v for k, v in s.get("state_dict", s).items()}
    if "encoder.conv2.1.weight" in s:
        s["encoder.conv2.0.weight"] = s.pop("encoder.conv2.1.weight")
    if "encoder.conv2.1.bias" in s:
        s["encoder.conv2.0.bias"] = s.pop("encoder.conv2.1.bias")
    for k in list(s.keys()):
        if k.startswith("decoder.final"):
            s[k.replace("decoder.final", "head.final")] = s.pop(k)
    model.load_state_dict(s, strict=False); model.eval()
    t = T.ToTensor()(img_rgb).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        edge = np.clip(model(t).squeeze().cpu().numpy(), 0, 1)
    os.chdir(old); sys.path.pop(0)
    return edge

def run_diffusionedge(img_rgb):
    de_dir = ROOT / "models/DiffusionEdge"
    tmp_in = OUT / "de_in"; tmp_out = OUT / "de_out"
    tmp_in.mkdir(exist_ok=True); tmp_out.mkdir(exist_ok=True)
    for f in tmp_out.glob("*"): f.unlink()
    h, w = img_rgb.shape[:2]
    new_h, new_w = max(((h+15)//16)*16, 320), max(((w+15)//16)*16, 320)
    Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR).save(tmp_in / "tmp.jpg")
    result = subprocess.run(
        [sys.executable, str(de_dir / "demo.py"),
         "--cfg", str(de_dir / "configs/BIPED_sample.yaml"),
         "--input_dir", str(tmp_in), "--pre_weight", str(de_dir / "weights/biped.pt"),
         "--out_dir", str(tmp_out), "--sampling_timesteps", "1", "--bs", "1"],
        capture_output=True, timeout=300, cwd=str(de_dir))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-300:])
    edge = np.array(Image.open(list(tmp_out.glob("*.png"))[0]).convert("L")).astype(np.float64) / 255.0
    if edge.shape != (h, w):
        edge = np.array(Image.fromarray((edge*255).astype(np.uint8)).resize((w,h), Image.BILINEAR)) / 255.0
    return edge

def run_wvf(img_gray_input):
    from edgecritic.wvf import wvf_image
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    mag = wvf_image(img_gray_input, np_count=50, order=2, n_orientations=18, backend=backend).gradient_mag
    return mag / (mag.max() + 1e-8)


RUNNERS = {
    "dexined": ("DexiNed", lambda rgb, gray: run_dexined(rgb)),
    "teed": ("TEED", lambda rgb, gray: run_teed(rgb)),
    "pidinet": ("pidinet", lambda rgb, gray: run_pidinet(rgb)),
    "nbed": ("NBED", lambda rgb, gray: run_nbed(rgb)),
    "diffusionedge": ("DiffusionEdge", lambda rgb, gray: run_diffusionedge(rgb)),
    "wvf": ("WVF (Np=50 d=2)", lambda rgb, gray: run_wvf(gray)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    args = parser.parse_args()

    if args.model == "all":
        models = list(RUNNERS.keys())
    else:
        models = [args.model]

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Models: {models}")
    print(f"SNRs: {SNR_SHOW}")

    # Run each model at each SNR, save individual edge maps
    all_edges = {}  # {model: {snr: edge_map}}
    for model_key in models:
        name, fn = RUNNERS[model_key]
        print(f"\n=== {name} ===")
        all_edges[model_key] = {}
        for snr in SNR_SHOW:
            noisy_rgb, noisy_gray = make_noisy(snr)
            snr_label = "clean" if snr == "inf" else f"snr{snr}"
            try:
                edge = fn(noisy_rgb, noisy_gray)
                all_edges[model_key][snr] = edge
                # Save individual PNG
                edge_uint8 = (np.clip(edge, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(edge_uint8).save(OUT / f"{model_key}_{snr_label}.png")
                print(f"  SNR={snr}: range [{edge.min():.3f}, {edge.max():.3f}]")
            except Exception as e:
                print(f"  SNR={snr}: FAIL ({e})")

    # Generate big comparison grid: rows=models, cols=SNR levels
    print("\n=== Generating comparison grid ===")
    n_models = len(models) + 2  # +2 for input row and GT
    n_snrs = len(SNR_SHOW)

    fig, axes = plt.subplots(n_models, n_snrs, figsize=(5 * n_snrs, 4 * n_models))

    # Row 0: noisy inputs
    for si, snr in enumerate(SNR_SHOW):
        noisy_rgb, _ = make_noisy(snr)
        axes[0, si].imshow(noisy_rgb)
        snr_label = "Clean" if snr == "inf" else f"SNR={snr}"
        axes[0, si].set_title(snr_label, fontsize=12, fontweight="bold")
        axes[0, si].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=12, rotation=0, labelpad=60, va="center")

    # Row 1: GT (same for all)
    for si in range(n_snrs):
        axes[1, si].imshow(gt, cmap="gray")
        axes[1, si].axis("off")
    axes[1, 0].set_ylabel("GT", fontsize=12, rotation=0, labelpad=60, va="center")

    # Remaining rows: model outputs
    for mi, model_key in enumerate(models):
        name = RUNNERS[model_key][0]
        for si, snr in enumerate(SNR_SHOW):
            ax = axes[mi + 2, si]
            if model_key in all_edges and snr in all_edges[model_key]:
                ax.imshow(all_edges[model_key][snr], cmap="gray")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        axes[mi + 2, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=80, va="center")

    fig.suptitle("BIPED v1 RGB_008: Edge Detection Across Noise Levels", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "all_models_all_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all_models_all_snr.png")


if __name__ == "__main__":
    main()
