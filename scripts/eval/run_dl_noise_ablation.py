"""Run a single DL model across 5 images per dataset × 8 SNRs × 5 noise types.

Usage: python run_dl_noise_ablation.py --model dexined
"""

import sys
import os
import json
import time
import argparse
import subprocess
import gc
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import scipy.io as sio

from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "dl_noise_ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001
MATCH_RADIUS = 3
SNR_LEVELS = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# Noise functions
# =========================================================================
def add_gaussian(img, snr, rng):
    sigma = (img.max() - img.min()) / snr
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)

def add_salt_pepper(img, snr, rng):
    d = 1.0 / (1.0 + snr); out = img.copy(); m = rng.random(img.shape)
    out[m < d / 2] = 0; out[m > 1 - d / 2] = 255; return out

def add_poisson(img, snr, rng):
    s = snr * snr
    return np.clip(rng.poisson(np.clip(img / 255 * s, 0.001, None)) / s * 255, 0, 255)

def add_speckle(img, snr, rng):
    return np.clip(img * (1 + rng.normal(0, 1 / snr, img.shape)), 0, 255)

def add_uniform(img, snr, rng):
    a = (img.max() - img.min()) / snr
    return np.clip(img + rng.uniform(-a, a, img.shape), 0, 255)

NOISE_TYPES = {
    "gaussian": add_gaussian, "salt_pepper": add_salt_pepper,
    "poisson": add_poisson, "speckle": add_speckle, "uniform": add_uniform,
}


# =========================================================================
# Dataset loaders (return ALL images, selection happens after)
# =========================================================================
def load_biped(version):
    if version == "v1":
        base = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
    else:
        base = ROOT / "datasets/BIPED/BIPEDv2/BIPEDv2/BIPED/edges"
    img_dir = base / "imgs/test/rgbr"
    gt_dir = base / "edge_maps/test/rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        images.append(np.array(Image.open(f).convert("RGB")))
        gts.append(np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names

def load_bsds500():
    base1 = ROOT / "datasets/BSDS500/BSDS500/data"
    base2 = ROOT / "datasets/BSDS500/BSDS500"
    base = base1 if (base1 / "images/test").exists() else base2
    img_dir = base / "images/test"
    gt_dir = base / "groundTruth/test"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        img = np.array(Image.open(f).convert("RGB"))
        gt_mat = sio.loadmat(str(gt_dir / f"{f.stem}.mat"))
        gt_cell = gt_mat["groundTruth"]
        gt_union = np.zeros(img.shape[:2], dtype=bool)
        for i in range(gt_cell.shape[1]):
            bdry = gt_cell[0, i]["Boundaries"][0, 0]
            bdry = bdry.toarray() if hasattr(bdry, "toarray") else np.asarray(bdry)
            gt_union |= (bdry > 0)
        images.append(img)
        gts.append(gt_union)
        names.append(f.stem)
    return images, gts, names

def load_uded():
    img_dir = ROOT / "datasets/UDED/imgs"
    gt_dir = ROOT / "datasets/UDED/gt"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.png")):
        images.append(np.array(Image.open(f).convert("RGB")))
        gts.append(np.array(Image.open(gt_dir / f.name).convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names


def select_top_images(images, gts, names, n=5):
    """Quick selection: pick top n by DexiNed ODS (or just first n if DexiNed unavailable)."""
    # Try to reuse selections from WVF noise ablation
    return images[:n], gts[:n], names[:n]


def load_wvf_selections(ds_name):
    """Load image selections from the WVF noise ablation if available."""
    sel_file = ROOT / "outputs" / "noise_ablation" / ds_name.lower() / "selected_images.json"
    if sel_file.exists():
        with open(sel_file) as f:
            return json.load(f)["images"]
    return None


# =========================================================================
# Model runners (take RGB numpy array, return edge map)
# =========================================================================
def run_dexined(img_rgb):
    sys.path.insert(0, str(ROOT / "models" / "DexiNed"))
    from model import DexiNed
    model = DexiNed().to(device)
    ckpt = torch.load(ROOT / "models/DexiNed/checkpoints/BIPED/10/10_model.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    hp, wp = (16 - h % 16) % 16, (16 - w % 16) % 16
    img_p = np.pad(img_bgr, ((0, hp), (0, wp), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(img_p.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(tensor)[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge

def run_teed(img_rgb):
    sys.path.insert(0, str(ROOT / "models" / "TEED"))
    from ted import TED
    model = TED().to(device)
    ckpt = torch.load(ROOT / "models/TEED/checkpoints/BIPED/7/7_model.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    hp, wp = (8 - h % 8) % 8, (8 - w % 8) % 8
    img_p = np.pad(img_bgr, ((0, hp), (0, wp), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(img_p.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(tensor)[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge

def run_pidinet(img_rgb):
    sys.path.insert(0, str(ROOT / "models" / "pidinet"))
    import models as pidi_models
    import torchvision.transforms as T
    class Args:
        config = "carv4"; sa = True; dil = True
    model = pidi_models.pidinet(Args()).to(device)
    state = torch.load(ROOT / "models/pidinet/trained_models/table5_pidinet.pth", map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state)
    model.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = normalize(T.ToTensor()(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        edge = torch.sigmoid(model(tensor)[-1]).squeeze().cpu().numpy()
    sys.path.pop(0)
    return edge

def run_nbed(img_rgb):
    nbed_dir = ROOT / "models" / "NBED"
    old_cwd = Path.cwd()
    os.chdir(nbed_dir)
    sys.path.insert(0, str(nbed_dir))
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]
    from model.basemodel import Basemodel
    import torchvision.transforms as T
    model = Basemodel(encoder_name="DUL-M36", decoder_name="UNETP", head_name="default").to(device)
    state = torch.load(nbed_dir / "weights/nbed_biped.pth", map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.get("state_dict", state).items()}
    if "encoder.conv2.1.weight" in state:
        state["encoder.conv2.0.weight"] = state.pop("encoder.conv2.1.weight")
    if "encoder.conv2.1.bias" in state:
        state["encoder.conv2.0.bias"] = state.pop("encoder.conv2.1.bias")
    for k in list(state.keys()):
        if k.startswith("decoder.final"):
            state[k.replace("decoder.final", "head.final")] = state.pop(k)
    model.load_state_dict(state, strict=False)
    model.eval()
    tensor = T.ToTensor()(img_rgb).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        edge = np.clip(model(tensor).squeeze().cpu().numpy(), 0, 1)
    os.chdir(old_cwd)
    sys.path.pop(0)
    return edge

def run_diffusionedge(img_rgb):
    de_dir = ROOT / "models" / "DiffusionEdge"
    tmp_in = OUT / "de_tmp_in"
    tmp_out = OUT / "de_tmp_out"
    tmp_in.mkdir(exist_ok=True)
    tmp_out.mkdir(exist_ok=True)
    for f in tmp_out.glob("*"):
        f.unlink()
    # Pad to at least 320x320 and multiple of 16 for FFT compatibility
    h, w = img_rgb.shape[:2]
    new_h = max(((h + 15) // 16) * 16, 320)
    new_w = max(((w + 15) // 16) * 16, 320)
    if new_h != h or new_w != w:
        img_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
    else:
        img_pil = Image.fromarray(img_rgb)
    img_pil.save(tmp_in / "tmp.jpg")
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
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-300:])
    out_files = list(tmp_out.glob("*.png"))
    edge = np.array(Image.open(out_files[0]).convert("L")).astype(np.float64) / 255.0
    # Resize back to original
    if edge.shape != (h, w):
        edge = np.array(Image.fromarray((edge * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    return edge


MODEL_RUNNERS = {
    "dexined": ("DexiNed", run_dexined),
    "teed": ("TEED", run_teed),
    "pidinet": ("pidinet", run_pidinet),
    "nbed": ("NBED", run_nbed),
    "diffusionedge": ("DiffusionEdge", run_diffusionedge),
}


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_RUNNERS.keys()))
    args = parser.parse_args()

    model_key = args.model
    model_name, model_fn = MODEL_RUNNERS[model_key]
    model_out = OUT / model_key
    model_out.mkdir(exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ALL_DATASETS = [
        ("UDED", load_uded),
        ("BIPED_v1", lambda: load_biped("v1")),
        ("BIPED_v2", lambda: load_biped("v2")),
        ("BSDS500", load_bsds500),
    ]

    grand_t0 = time.perf_counter()

    for ds_name, loader in ALL_DATASETS:
        ds_out = model_out / ds_name.lower()
        ds_out.mkdir(exist_ok=True)

        done_file = ds_out / "DONE"
        if done_file.exists():
            print(f"\n=== {ds_name}: already done, skipping ===")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            all_images, all_gts, all_names = loader()
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        # Use same selections as WVF noise ablation if available
        selected = load_wvf_selections(ds_name)
        if selected:
            indices = [all_names.index(n) for n in selected if n in all_names]
            images = [all_images[i] for i in indices]
            gts = [all_gts[i] for i in indices]
            names = [all_names[i] for i in indices]
            print(f"  Using WVF selections: {names}")
        else:
            images, gts, names = all_images[:5], all_gts[:5], all_names[:5]
            print(f"  Using first 5: {names}")

        with open(ds_out / "selected_images.json", "w") as f:
            json.dump({"images": names}, f)

        for img_idx, (img_rgb, gt, img_name) in enumerate(zip(images, gts, names)):
            img_file = ds_out / f"{img_name}_results.json"
            if img_file.exists():
                print(f"\n  [{img_idx+1}/5] {img_name}: already done")
                continue

            print(f"\n  [{img_idx+1}/5] {img_name} ({img_rgb.shape[:2]})")
            img_gray = np.mean(img_rgb, axis=2)
            img_results = {"image": img_name, "shape": list(img_rgb.shape[:2]),
                           "model": model_name, "conditions": []}

            # Build conditions: clean + noisy
            conditions = [("clean", "inf", img_rgb)]
            for snr in SNR_LEVELS:
                for noise_name, noise_fn in NOISE_TYPES.items():
                    rng = np.random.default_rng(seed=42)
                    # Apply noise to grayscale then reconstruct RGB
                    noisy_gray = noise_fn(img_gray, snr, rng)
                    # Stack to 3-channel for DL models
                    noisy_rgb = np.stack([noisy_gray]*3, axis=-1).astype(np.uint8)
                    conditions.append((noise_name, snr, noisy_rgb))

            for ci, (noise_name, snr, img) in enumerate(conditions):
                t0 = time.perf_counter()
                try:
                    edge = model_fn(img)
                    elapsed = time.perf_counter() - t0
                    ods, _, _, _ = compute_ods_ois(
                        edge, gt.astype(np.float64),
                        n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)

                    snr_str = str(snr) if snr != "inf" else "inf"
                    label = f"{noise_name}/SNR={snr_str}"
                    print(f"    [{ci+1}/{len(conditions)}] {label:25s} ODS={ods:.4f} ({elapsed:.1f}s)")

                    img_results["conditions"].append({
                        "noise_type": noise_name,
                        "snr": snr_str,
                        "ods": float(ods),
                        "time_s": round(elapsed, 2),
                    })
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    print(f"    [{ci+1}/{len(conditions)}] {noise_name}/SNR={snr}: FAIL ({e})")
                    img_results["conditions"].append({
                        "noise_type": noise_name,
                        "snr": str(snr),
                        "ods": -1, "error": str(e),
                    })

                torch.cuda.empty_cache()

            with open(img_file, "w") as f:
                json.dump(img_results, f, indent=2)

        done_file.write_text("done")
        del all_images, all_gts
        gc.collect()
        torch.cuda.empty_cache()

    total = time.perf_counter() - grand_t0
    print(f"\n{'='*60}")
    print(f"{model_name} DONE in {total:.0f}s ({total/3600:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
