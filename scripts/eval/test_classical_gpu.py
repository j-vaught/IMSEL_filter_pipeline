"""Test all GPU-accelerated classical edge detectors on BIPED RGB_008.
Shows ODS for each config and saves edge map comparison images.
"""

import time
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from edgecritic.baselines.classical_gpu import run_all_classical
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets/BIPED/BIPED/BIPED/edges"
OUT = ROOT / "outputs" / "classical_test"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load image and GT
img_gray = np.mean(np.array(Image.open(BIPED / "imgs/test/rgbr/RGB_008.jpg")), axis=2)
gt = np.array(Image.open(BIPED / "edge_maps/test/rgbr/RGB_008.png").convert("L")) > 128
print(f"Image: {img_gray.shape}, GT: {np.sum(gt)} edge px")

# Run all classical methods
print(f"\nRunning all classical edge detectors...")
t0 = time.perf_counter()
results = run_all_classical(img_gray, device=device)
filter_time = time.perf_counter() - t0
print(f"Filters: {len(results)} configs in {filter_time:.2f}s")

# Evaluate each
print(f"\nEvaluating ODS (1001 thresholds, 3px match)...")
t0 = time.perf_counter()
for r in results:
    ods, _, _, _ = compute_ods_ois(r["magnitude"], gt.astype(np.float64),
                                    n_thresholds=1001, match_radius=3)
    r["ods"] = float(ods)
eval_time = time.perf_counter() - t0
print(f"Evaluation: {eval_time:.2f}s")

# Sort by ODS
results.sort(key=lambda x: x["ods"], reverse=True)

# Print results table
print(f"\n{'Rank':>4s} {'Method':<20s} {'Params':<25s} {'ODS':>7s}")
print("-" * 60)
for i, r in enumerate(results):
    params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "(default)"
    print(f"{i+1:>4d} {r['method']:<20s} {params_str:<25s} {r['ods']:>7.4f}")

# Save results JSON (without magnitude arrays)
json_results = [{"method": r["method"], "params": r["params"], "ods": r["ods"]} for r in results]
with open(OUT / "classical_ods.json", "w") as f:
    json.dump(json_results, f, indent=2)

# Also add WVF for comparison
from edgecritic.wvf import wvf_image
wvf_mag = wvf_image(img_gray, np_count=25, order=2, n_orientations=18, backend=device).gradient_mag
wvf_ods, _, _, _ = compute_ods_ois(wvf_mag, gt.astype(np.float64), n_thresholds=1001, match_radius=3)
print(f"\nWVF (Np=25 d=2): ODS={wvf_ods:.4f}")

# =========================================================================
# Save edge map images
# =========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Group by method for organized display
methods_order = ["Sobel", "Prewitt", "Scharr", "Roberts", "Kirsch", "Robinson",
                 "Frei-Chen", "GaussianDeriv", "LoG", "DoG", "Steerable2nd"]

# Pick best config per method for the comparison grid
best_per_method = {}
for r in results:
    m = r["method"]
    if m not in best_per_method or r["ods"] > best_per_method[m]["ods"]:
        best_per_method[m] = r

# Save individual edge maps for top configs
for r in results[:5]:  # top 5
    name = f"{r['method']}_{'_'.join(f'{k}{v}' for k,v in r['params'].items())}" if r["params"] else r["method"]
    name = name.replace(".", "p")
    edge_uint8 = (np.clip(r["magnitude"] / (r["magnitude"].max() + 1e-8), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(edge_uint8).save(OUT / f"{name}_edge.png")

# =========================================================================
# Comparison grid: best per method + GT + WVF
# =========================================================================
n_methods = len(best_per_method) + 2  # +input, +GT
ncols = 4
nrows = (n_methods + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
axes = axes.flatten()

# Input and GT
axes[0].imshow(np.array(Image.open(BIPED / "imgs/test/rgbr/RGB_008.jpg")))
axes[0].set_title("Input (RGB)", fontsize=10)
axes[0].axis("off")

axes[1].imshow(gt, cmap="gray")
axes[1].set_title("Ground Truth", fontsize=10)
axes[1].axis("off")

# Best per method
idx = 2
for method in methods_order:
    if method not in best_per_method:
        continue
    r = best_per_method[method]
    ax = axes[idx]
    mag = r["magnitude"]
    ax.imshow(mag / (mag.max() + 1e-8), cmap="gray")
    params = ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else ""
    ax.set_title(f"{r['method']} {params}\nODS={r['ods']:.3f}", fontsize=9)
    ax.axis("off")
    idx += 1

# WVF
ax = axes[idx]
ax.imshow(wvf_mag / (wvf_mag.max() + 1e-8), cmap="gray")
ax.set_title(f"WVF (Np=25 d=2)\nODS={wvf_ods:.3f}", fontsize=9)
ax.axis("off")
idx += 1

# Hide unused
for i in range(idx, len(axes)):
    axes[i].axis("off")

fig.suptitle("BIPED v1 RGB_008 — Best Classical Config per Method (Clean)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "classical_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved classical_comparison.png")

# =========================================================================
# Scale sweep plot: ODS vs kernel size / sigma
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: kernel-based methods
ax = axes[0]
for method, color in [("Sobel", "#e41a1c"), ("Prewitt", "#377eb8")]:
    xs = [r["params"]["ksize"] for r in results if r["method"] == method]
    ys = [r["ods"] for r in results if r["method"] == method]
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=color,
                markersize=5, label=method)
# Mark fixed-size methods
for method, marker, color in [("Scharr", "^", "#4daf4a"), ("Roberts", "v", "#984ea3"),
                                ("Kirsch", "D", "#ff7f00"), ("Robinson", "s", "#a65628"),
                                ("Frei-Chen", "p", "#f781bf")]:
    r = next((x for x in results if x["method"] == method), None)
    if r:
        ax.scatter([3], [r["ods"]], marker=marker, s=80, color=color, label=f"{method} ({r['ods']:.3f})", zorder=5)

ax.axhline(y=wvf_ods, color="black", linestyle="--", label=f"WVF ({wvf_ods:.3f})")
ax.set_xlabel("Kernel size", fontsize=12)
ax.set_ylabel("ODS", fontsize=12)
ax.set_title("Kernel-Based Methods", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: sigma-based methods
ax = axes[1]
for method, color in [("GaussianDeriv", "#e41a1c"), ("LoG", "#377eb8"),
                       ("DoG", "#4daf4a"), ("Steerable2nd", "#984ea3")]:
    xs = [r["params"].get("sigma", r["params"].get("sigma1")) for r in results if r["method"] == method]
    ys = [r["ods"] for r in results if r["method"] == method]
    if xs:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=color,
                markersize=5, label=method)

ax.axhline(y=wvf_ods, color="black", linestyle="--", label=f"WVF ({wvf_ods:.3f})")
ax.set_xlabel("$\\sigma$", fontsize=12)
ax.set_ylabel("ODS", fontsize=12)
ax.set_title("Scale-Space Methods", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("BIPED v1 RGB_008 — ODS vs Filter Scale (Clean)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "classical_ods_vs_scale.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved classical_ods_vs_scale.png")

print(f"\nTotal time: {filter_time + eval_time:.1f}s")
