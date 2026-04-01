"""Full dataset-wide ODS evaluation for DL edge detection models.

Runs a single DL model on every test image in each dataset, then computes
dataset-wide ODS (single best threshold across all images) — same protocol
as run_full_dataset_ablation.py uses for WVF/LF.

Usage:
    python run_full_dataset_dl.py --model dexined
    python run_full_dataset_dl.py --model diffusionedge --datasets bsds500 uded
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
OUT = ROOT / "outputs" / "full_dataset_dl"
OUT.mkdir(parents=True, exist_ok=True)

N_THRESH = 1001
MATCH_RADIUS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# Dataset loaders (return RGB arrays for DL models + bool GT)
# =========================================================================
def load_biped_v1():
    base = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
    img_dir = base / "imgs/test/rgbr"
    gt_dir = base / "edge_maps/test/rgbr"
    images, gts, names = [], [], []
    for f in sorted(img_dir.glob("*.jpg")):
        images.append(np.array(Image.open(f).convert("RGB")))
        gts.append(np.array(Image.open(gt_dir / f"{f.stem}.png").convert("L")) > 128)
        names.append(f.stem)
    return images, gts, names


def load_biped_v2():
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


ALL_DATASETS = {
    "uded": ("UDED", load_uded),
    "biped_v1": ("BIPED_v1", load_biped_v1),
    "biped_v2": ("BIPED_v2", load_biped_v2),
    "bsds500": ("BSDS500", load_bsds500),
}


# =========================================================================
# Model runners (take RGB numpy array, return float [0,1] edge map)
# =========================================================================
def run_dexined(img_rgb):
    sys.path.insert(0, str(ROOT / "models" / "DexiNed"))
    from model import DexiNed
    model = DexiNed().to(device)
    ckpt = torch.load(ROOT / "models/DexiNed/checkpoints/BIPED/10/10_model.pth",
                       map_location=device)
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
    ckpt = torch.load(ROOT / "models/TEED/checkpoints/BIPED/7/7_model.pth",
                       map_location=device)
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
    state = torch.load(ROOT / "models/pidinet/trained_models/table5_pidinet.pth",
                        map_location=device)
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
    model = Basemodel(encoder_name="DUL-M36", decoder_name="UNETP",
                      head_name="default").to(device)
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
    pid = os.getpid()
    tmp_in = OUT / f"de_tmp_in_{pid}"
    tmp_out = OUT / f"de_tmp_out_{pid}"
    tmp_in.mkdir(exist_ok=True)
    tmp_out.mkdir(exist_ok=True)
    for f in tmp_out.glob("*"):
        f.unlink()
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
    if edge.shape != (h, w):
        edge = np.array(
            Image.fromarray((edge * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ) / 255.0
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
    parser.add_argument("--datasets", nargs="*", default=list(ALL_DATASETS.keys()),
                        choices=list(ALL_DATASETS.keys()),
                        help="Which datasets to run (default: all)")
    args = parser.parse_args()

    model_key = args.model
    model_name, model_fn = MODEL_RUNNERS[model_key]

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Eval: {N_THRESH} thresholds, {MATCH_RADIUS}px match radius")
    print()

    grand_t0 = time.perf_counter()
    all_results = {}

    for ds_key in args.datasets:
        ds_name, loader = ALL_DATASETS[ds_key]

        # Check for existing results
        result_file = OUT / f"{model_key}_{ds_key}.json"
        if result_file.exists():
            print(f"=== {ds_name}: already done ({result_file.name}), skipping ===\n")
            with open(result_file) as f:
                all_results[ds_name] = json.load(f)
            continue

        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            images, gts, names = loader()
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}\n")
            continue

        n_images = len(images)
        print(f"  Images: {n_images}")

        # Run model on every image, collecting edge maps
        edges = []
        times = []
        for i, (img_rgb, gt, name) in enumerate(zip(images, gts, names)):
            t0 = time.perf_counter()
            try:
                edge = model_fn(img_rgb)
                elapsed = time.perf_counter() - t0
                edges.append(edge)
                times.append(elapsed)
                print(f"  [{i+1}/{n_images}] {name} ({img_rgb.shape[1]}x{img_rgb.shape[0]}) "
                      f"{elapsed:.2f}s")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  [{i+1}/{n_images}] {name}: FAILED ({e})")
                # Use zeros so indexing stays aligned
                edges.append(np.zeros(img_rgb.shape[:2], dtype=np.float64))
                times.append(elapsed)

            torch.cuda.empty_cache()

        # Dataset-wide ODS (single threshold across all images)
        print(f"\n  Computing dataset-wide ODS ({n_images} images, {N_THRESH} thresholds)...")
        t0 = time.perf_counter()
        ods, ois, _, _ = compute_ods_ois(
            edges, [gt.astype(np.float64) for gt in gts],
            n_thresholds=N_THRESH, match_radius=MATCH_RADIUS)
        eval_time = time.perf_counter() - t0

        total_infer = sum(times)
        result = {
            "model": model_name,
            "dataset": ds_name,
            "n_images": n_images,
            "image_names": names,
            "ods": float(ods),
            "ois": float(ois),
            "total_inference_s": round(total_infer, 2),
            "mean_inference_s": round(total_infer / n_images, 4),
            "eval_time_s": round(eval_time, 2),
            "evaluation": {
                "match_radius": MATCH_RADIUS,
                "n_thresholds": N_THRESH,
                "post_processing": "none",
                "protocol": "dataset-wide ODS (single threshold across all images)",
            },
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  {ds_name} — ODS={ods:.4f}  OIS={ois:.4f}")
        print(f"  Inference: {total_infer:.1f}s total, {total_infer/n_images:.3f}s/img")
        print(f"  Evaluation: {eval_time:.1f}s")
        print(f"  Saved to {result_file.name}\n")

        all_results[ds_name] = result
        del images, gts, edges
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined summary
    grand_total = time.perf_counter() - grand_t0
    summary = {
        "model": model_name,
        "total_time_s": round(grand_total, 1),
        "datasets": {name: {"ods": r["ods"], "ois": r["ois"]}
                     for name, r in all_results.items()},
    }
    summary_file = OUT / f"{model_key}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*60}")
    print(f"{model_name} COMPLETE in {grand_total:.0f}s ({grand_total/3600:.1f}h)")
    print(f"{'='*60}")
    print(f"\n{'Dataset':<12} {'ODS':>8} {'OIS':>8}")
    print("-" * 30)
    for name, r in all_results.items():
        print(f"{name:<12} {r['ods']:>8.4f} {r['ois']:>8.4f}")
    print()


if __name__ == "__main__":
    main()
