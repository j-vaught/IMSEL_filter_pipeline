"""Quick smoke test: run each model on BIPED RGB_008, save output edge maps."""

import sys
import time
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping comparison figure")

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
IMG_PATH = BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"
GT_PATH = BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png"
OUT = ROOT / "outputs" / "model_test"
OUT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

img_pil = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(img_pil)
gt = np.array(Image.open(GT_PATH).convert("L"))
print(f"Image: {img_np.shape}")

results = {}


# =========================================================================
# 1. DexiNed
# =========================================================================
def test_dexined():
    print("\n=== DexiNed ===")
    sys.path.insert(0, str(ROOT / "models" / "DexiNed"))
    from model import DexiNed
    import cv2

    model = DexiNed().to(device)
    ckpt = torch.load(ROOT / "models/DexiNed/checkpoints/BIPED/10/10_model.pth",
                       map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # DexiNed expects BGR, float32, mean-subtracted
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_bgr -= mean

    # Pad to multiple of 16
    h, w = img_bgr.shape[:2]
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16
    img_padded = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

    tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()[:h, :w]
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.3f}s, shape: {edge.shape}, range: [{edge.min():.3f}, {edge.max():.3f}]")
    sys.path.pop(0)
    return edge, elapsed


# =========================================================================
# 2. TEED
# =========================================================================
def test_teed():
    print("\n=== TEED ===")
    sys.path.insert(0, str(ROOT / "models" / "TEED"))
    from ted import TED
    import cv2

    model = TED().to(device)
    ckpt = torch.load(ROOT / "models/TEED/checkpoints/BIPED/7/7_model.pth",
                       map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # TEED expects BGR, float32, mean-subtracted (same as DexiNed)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_bgr -= mean_bgr

    # Pad to multiple of 8
    h, w = img_bgr.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    img_padded = np.pad(img_bgr, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

    tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()[:h, :w]
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.3f}s, shape: {edge.shape}, range: [{edge.min():.3f}, {edge.max():.3f}]")
    sys.path.pop(0)
    return edge, elapsed


# =========================================================================
# 3. pidinet
# =========================================================================
def test_pidinet():
    print("\n=== pidinet ===")
    sys.path.insert(0, str(ROOT / "models" / "pidinet"))
    import models as pidi_models

    class Args:
        config = "carv4"
        sa = True
        dil = True

    model = pidi_models.pidinet(Args()).to(device)
    ckpt = torch.load(ROOT / "models/pidinet/trained_models/table5_pidinet.pth",
                       map_location=device)
    # Handle state dict keys
    state = ckpt.get("state_dict", ckpt)
    # Remove 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # ImageNet normalization
    import torchvision.transforms as T
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = normalize(T.ToTensor()(img_pil)).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
    edge = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.3f}s, shape: {edge.shape}, range: [{edge.min():.3f}, {edge.max():.3f}]")
    sys.path.pop(0)
    return edge, elapsed


# =========================================================================
# 4. NBED
# =========================================================================
def test_nbed():
    print("\n=== NBED ===")
    import importlib
    nbed_dir = ROOT / "models" / "NBED"
    # Need to run from NBED dir to avoid 'model' name collision
    old_cwd = Path.cwd()
    import os
    os.chdir(nbed_dir)
    sys.path.insert(0, str(nbed_dir))

    # Clear any cached 'model' module
    for key in list(sys.modules.keys()):
        if key == "model" or key.startswith("model."):
            del sys.modules[key]

    from model.basemodel import Basemodel

    model_obj = Basemodel(encoder_name="DUL-M36", decoder_name="UNETP", head_name="default").to(device)
    # BSDS weights are corrupted; use BIPED weights
    weight_file = nbed_dir / "weights/nbed_biped.pth"
    if not weight_file.exists():
        weight_file = nbed_dir / "weights/nbed_bsds.pth"
    ckpt = torch.load(weight_file, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # Fix key renames
    if "encoder.conv2.1.weight" in state:
        state["encoder.conv2.0.weight"] = state.pop("encoder.conv2.1.weight")
    if "encoder.conv2.1.bias" in state:
        state["encoder.conv2.0.bias"] = state.pop("encoder.conv2.1.bias")
    # decoder.final -> head.final
    for k in list(state.keys()):
        if k.startswith("decoder.final"):
            new_k = k.replace("decoder.final", "head.final")
            state[new_k] = state.pop(k)
    model_obj.load_state_dict(state, strict=False)
    model_obj.eval()

    import torchvision.transforms as T
    tensor = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    tensor = tensor * 2 - 1

    t0 = time.perf_counter()
    with torch.no_grad():
        edge = model_obj(tensor)
    edge = edge.squeeze().cpu().numpy()
    elapsed = time.perf_counter() - t0

    edge = np.clip(edge, 0, 1)
    print(f"  Time: {elapsed:.3f}s, shape: {edge.shape}, range: [{edge.min():.3f}, {edge.max():.3f}]")
    os.chdir(old_cwd)
    sys.path.pop(0)
    return edge, elapsed


# =========================================================================
# 5. DiffusionEdge
# =========================================================================
def test_diffusionedge():
    print("\n=== DiffusionEdge ===")
    import os, subprocess

    de_dir = ROOT / "models" / "DiffusionEdge"
    input_dir = OUT / "de_input"
    output_dir = OUT / "de_output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Copy test image
    img_pil.save(input_dir / "RGB_008.jpg")

    # Find config
    cfg_files = list((de_dir / "configs").glob("*BIPED*.yaml"))
    if not cfg_files:
        cfg_files = list((de_dir / "configs").glob("*.yaml"))
    if not cfg_files:
        print("  No config found, skipping")
        return None, 0

    cfg_path = cfg_files[0]
    weight_path = de_dir / "weights" / "biped.pt"
    print(f"  Config: {cfg_path.name}, weights: biped.pt")

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(de_dir / "demo.py"),
         "--cfg", str(cfg_path),
         "--input_dir", str(input_dir),
         "--pre_weight", str(weight_path),
         "--out_dir", str(output_dir),
         "--sampling_timesteps", "1",
         "--bs", "1"],
        capture_output=True, timeout=120,
        cwd=str(de_dir),
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr.decode('utf-8', errors='replace')[-500:]}")
        return None, 0

    # Find output
    out_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    if not out_files:
        print(f"  No output files found")
        return None, 0

    edge = np.array(Image.open(out_files[0]).convert("L")).astype(np.float64) / 255.0
    # Resize if needed
    if edge.shape != (img_np.shape[0], img_np.shape[1]):
        edge = np.array(Image.fromarray((edge * 255).astype(np.uint8)).resize(
            (img_np.shape[1], img_np.shape[0]), Image.BILINEAR)) / 255.0

    print(f"  Time: {elapsed:.3f}s, shape: {edge.shape}, range: [{edge.min():.3f}, {edge.max():.3f}]")
    return edge, elapsed


# =========================================================================
# Run all models
# =========================================================================
models_to_test = [
    ("DexiNed", test_dexined),
    ("TEED", test_teed),
    ("pidinet", test_pidinet),
    ("NBED", test_nbed),
    ("DiffusionEdge", test_diffusionedge),
]

edge_maps = {}
for name, fn in models_to_test:
    try:
        edge, elapsed = fn()
        if edge is not None:
            edge_maps[name] = edge
            results[name] = {"time": elapsed, "shape": list(edge.shape),
                             "min": float(edge.min()), "max": float(edge.max())}
            # Save individual edge map
            Image.fromarray((np.clip(edge, 0, 1) * 255).astype(np.uint8)).save(OUT / f"{name}_edge.png")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[name] = {"error": str(e)}

# =========================================================================
# Also run WVF for comparison
# =========================================================================
print("\n=== WVF (best config) ===")
from edgecritic.wvf import wvf_image
img_gray = np.mean(img_np, axis=2)
t0 = time.perf_counter()
wvf_result = wvf_image(img_gray, np_count=25, order=2, n_orientations=18, backend="cuda")
wvf_edge = wvf_result.gradient_mag
# Normalize to [0, 1]
wvf_edge = wvf_edge / (wvf_edge.max() + 1e-8)
elapsed = time.perf_counter() - t0
edge_maps["WVF (Np=25 d=2)"] = wvf_edge
results["WVF"] = {"time": elapsed}
print(f"  Time: {elapsed:.3f}s")

# Save results JSON
with open(OUT / "test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved test_results.json")

# Generate comparison figure if matplotlib available
if HAS_MPL and edge_maps:
    print("\n=== Generating comparison figure ===")
    n = len(edge_maps) + 2
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 9))
    axes = axes.flatten()
    axes[0].imshow(img_np); axes[0].set_title("Input (RGB)", fontsize=10); axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth", fontsize=10); axes[1].axis("off")
    for i, (name, edge) in enumerate(edge_maps.items()):
        ax = axes[i + 2]
        ax.imshow(edge, cmap="gray")
        t = results.get(name.split(" ")[0], {}).get("time", 0)
        ax.set_title(f"{name}\n({t:.3f}s)", fontsize=9)
        ax.axis("off")
    for i in range(len(edge_maps) + 2, len(axes)):
        axes[i].axis("off")
    fig.suptitle("BIPED v1 RGB_008 — All Models (Clean Image)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "all_models_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all_models_comparison.png")

# Print summary
print(f"\n{'='*50}")
print(f"{'Model':<20s} {'Time':>8s} {'Status':>10s}")
print(f"{'-'*50}")
for name, r in results.items():
    if "error" in r:
        print(f"{name:<20s} {'---':>8s} {'FAILED':>10s}")
    else:
        print(f"{name:<20s} {r['time']:>7.3f}s {'OK':>10s}")
print(f"{'='*50}")
