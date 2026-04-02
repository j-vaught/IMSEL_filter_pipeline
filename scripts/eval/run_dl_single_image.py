"""Run a single DL model on one BIPED image, compute ODS.

Usage: python run_dl_single_image.py --model dexined
       python run_dl_single_image.py --model all
"""

import sys
import time
import json
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
IMG_PATH = BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"
GT_PATH = BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png"
OUT = ROOT / "outputs" / "model_test"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from edgecritic.evaluation.metrics import compute_ods_ois


def load_image():
    img_pil = Image.open(IMG_PATH).convert("RGB")
    img_np = np.array(img_pil)
    gt = np.array(Image.open(GT_PATH).convert("L")) > 128
    return img_pil, img_np, gt


def run_dexined(img_np):
    sys.path.insert(0, str(ROOT / "models" / "DexiNed"))
    from model import DexiNed

    model = DexiNed().to(device)
    ckpt = torch.load(ROOT / "models/DexiNed/checkpoints/BIPED/10/10_model.pth",
                       map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16
    img_padded = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge


def run_teed(img_np):
    sys.path.insert(0, str(ROOT / "models" / "TEED"))
    from ted import TED

    model = TED().to(device)
    ckpt = torch.load(ROOT / "models/TEED/checkpoints/BIPED/7/7_model.pth",
                       map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
    img_bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    h, w = img_bgr.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    img_padded = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()[:h, :w]
    sys.path.pop(0)
    return edge


def run_pidinet(img_np, img_pil):
    sys.path.insert(0, str(ROOT / "models" / "pidinet"))
    import models as pidi_models
    import torchvision.transforms as T

    class Args:
        config = "carv4"
        sa = True
        dil = True

    model = pidi_models.pidinet(Args()).to(device)
    ckpt = torch.load(ROOT / "models/pidinet/trained_models/table5_pidinet.pth",
                       map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt.get("state_dict", ckpt).items()}
    model.load_state_dict(state)
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = normalize(T.ToTensor()(img_pil)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()
    sys.path.pop(0)
    return edge


def run_nbed(img_pil):
    import os
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
    ckpt = torch.load(nbed_dir / "weights/nbed_biped.pth", map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt.get("state_dict", ckpt).items()}
    if "encoder.conv2.1.weight" in state:
        state["encoder.conv2.0.weight"] = state.pop("encoder.conv2.1.weight")
    if "encoder.conv2.1.bias" in state:
        state["encoder.conv2.0.bias"] = state.pop("encoder.conv2.1.bias")
    for k in list(state.keys()):
        if k.startswith("decoder.final"):
            state[k.replace("decoder.final", "head.final")] = state.pop(k)
    model.load_state_dict(state, strict=False)
    model.eval()

    tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    tensor = tensor * 2 - 1

    with torch.no_grad():
        edge = model(tensor)
    edge = np.clip(edge.squeeze().cpu().numpy(), 0, 1)
    os.chdir(old_cwd)
    sys.path.pop(0)
    return edge


def run_diffusionedge(img_pil):
    import os, subprocess
    de_dir = ROOT / "models" / "DiffusionEdge"
    input_dir = OUT / "de_input"
    output_dir = OUT / "de_output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    # Clear old outputs
    for f in output_dir.glob("*"):
        f.unlink()

    img_pil.save(input_dir / "RGB_008.jpg")
    cfg_path = de_dir / "configs" / "BIPED_sample.yaml"
    weight_path = de_dir / "weights" / "biped.pt"

    result = subprocess.run(
        [sys.executable, str(de_dir / "demo.py"),
         "--cfg", str(cfg_path),
         "--input_dir", str(input_dir),
         "--pre_weight", str(weight_path),
         "--out_dir", str(output_dir),
         "--sampling_timesteps", "1",
         "--bs", "1"],
        capture_output=True, timeout=120, cwd=str(de_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-300:])

    out_files = list(output_dir.glob("*.png"))
    edge = np.array(Image.open(out_files[0]).convert("L")).astype(np.float64) / 255.0
    h, w = np.array(img_pil).shape[:2]
    if edge.shape != (h, w):
        edge = np.array(Image.fromarray((edge * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR)) / 255.0
    return edge


def run_wvf(img_np):
    from edgecritic.wvf import wvf_image
    img_gray = np.mean(img_np, axis=2)
    mag = wvf_image(img_gray, np_count=25, order=2, n_orientations=18,
                    backend="cuda" if torch.cuda.is_available() else "cpu").gradient_mag
    return mag


MODELS = {
    "dexined": ("DexiNed", lambda np, pil: run_dexined(np)),
    "teed": ("TEED", lambda np, pil: run_teed(np)),
    "pidinet": ("pidinet", lambda np, pil: run_pidinet(np, pil)),
    "nbed": ("NBED", lambda np, pil: run_nbed(pil)),
    "diffusionedge": ("DiffusionEdge", lambda np, pil: run_diffusionedge(pil)),
    "wvf": ("WVF (Np=25 d=2)", lambda np, pil: run_wvf(np)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or 'all'. Options: " + ", ".join(MODELS.keys()))
    args = parser.parse_args()

    img_pil, img_np, gt = load_image()
    print(f"Image: {img_np.shape}, GT: {np.sum(gt)} edge px")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model == "all":
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = [args.model]

    results = {}
    print(f"\n{'Model':<20s} {'ODS':>7s} {'Time':>8s} {'Status':>8s}")
    print("-" * 46)

    for key in models_to_run:
        if key not in MODELS:
            print(f"Unknown model: {key}")
            continue
        name, fn = MODELS[key]
        try:
            t0 = time.perf_counter()
            edge = fn(img_np, img_pil)
            elapsed = time.perf_counter() - t0

            ods, _, _, _ = compute_ods_ois(edge, gt.astype(np.float64),
                                            n_thresholds=1001, match_radius=3)

            # Save edge map
            edge_uint8 = (np.clip(edge / (edge.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(edge_uint8).save(OUT / f"{key}_edge.png")

            results[key] = {"name": name, "ods": float(ods), "time": elapsed}
            print(f"{name:<20s} {ods:>7.4f} {elapsed:>7.2f}s {'OK':>8s}")

        except Exception as e:
            results[key] = {"name": name, "error": str(e)}
            print(f"{name:<20s} {'---':>7s} {'---':>8s} {'FAIL':>8s}")
            print(f"  Error: {str(e)[:200]}")

    with open(OUT / "single_image_ods.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT / 'single_image_ods.json'}")


if __name__ == "__main__":
    main()
