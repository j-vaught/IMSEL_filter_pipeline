"""Test whether Canny post-processing affects WVF/LF ODS scores.

Runs ~10 representative WVF/LF configs with and without Canny NMS+hysteresis,
sweeping Canny parameters to find if/how it matters.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from edgecritic.wvf import wvf_image
from edgecritic.lf import lf_image
from edgecritic.evaluation.metrics import compute_ods_ois

ROOT = Path(__file__).resolve().parent.parent
BIPED = ROOT / "datasets" / "BIPED" / "BIPED" / "BIPED" / "edges"
OUT = ROOT / "outputs" / "biped_ablation"

N_THRESH = 1001

print("Loading BIPED v1 RGB_008...")
img_color = np.array(Image.open(BIPED / "imgs" / "test" / "rgbr" / "RGB_008.jpg"))
img_gray = np.mean(img_color, axis=2)
gt_bool = np.array(Image.open(BIPED / "edge_maps" / "test" / "rgbr" / "RGB_008.png").convert("L")) > 128
print(f"  {img_gray.shape}, {np.sum(gt_bool)} GT edge px")

import torch
backend = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# NMS implementations
# ---------------------------------------------------------------------------
def nms_4dir(mag, angle):
    """Standard 4-direction non-maximum suppression."""
    H, W = mag.shape
    out = np.zeros_like(mag)
    angle_deg = np.degrees(angle) % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            a = angle_deg[y, x]
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            else:
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            if mag[y, x] >= n1 and mag[y, x] >= n2:
                out[y, x] = mag[y, x]
    return out


def nms_8dir(mag, angle):
    """Bagan's enhanced 8-direction NMS."""
    H, W = mag.shape
    out = np.zeros_like(mag)
    angle_deg = np.degrees(angle) % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            a = angle_deg[y, x]
            # 8 directions: 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5
            if (0 <= a < 11.25) or (168.75 <= a < 180):
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif 11.25 <= a < 33.75:
                # 22.5 degree: interpolate between horizontal and diagonal
                n1 = 0.5 * (mag[y, x + 1] + mag[y - 1, x + 1])
                n2 = 0.5 * (mag[y, x - 1] + mag[y + 1, x - 1])
            elif 33.75 <= a < 56.25:
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            elif 56.25 <= a < 78.75:
                n1 = 0.5 * (mag[y - 1, x] + mag[y - 1, x + 1])
                n2 = 0.5 * (mag[y + 1, x] + mag[y + 1, x - 1])
            elif 78.75 <= a < 101.25:
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            elif 101.25 <= a < 123.75:
                n1 = 0.5 * (mag[y - 1, x] + mag[y - 1, x - 1])
                n2 = 0.5 * (mag[y + 1, x] + mag[y + 1, x + 1])
            elif 123.75 <= a < 146.25:
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            else:  # 146.25 to 168.75
                n1 = 0.5 * (mag[y, x - 1] + mag[y - 1, x - 1])
                n2 = 0.5 * (mag[y, x + 1] + mag[y + 1, x + 1])

            if mag[y, x] >= n1 and mag[y, x] >= n2:
                out[y, x] = mag[y, x]
    return out


def nms_vectorized(mag, angle, n_dirs=4):
    """Fast vectorized NMS (approximate but much faster than pixel loops)."""
    H, W = mag.shape
    angle_deg = np.degrees(angle) % 180

    # Shifted arrays for neighbor comparisons
    pad = np.pad(mag, 1, mode='constant')

    if n_dirs == 4:
        bins = [
            ((0, 22.5, 157.5, 180), pad[1:-1, :-2], pad[1:-1, 2:]),    # 0: horizontal
            ((22.5, 67.5),          pad[:-2, 2:],   pad[2:, :-2]),      # 45: diagonal
            ((67.5, 112.5),         pad[:-2, 1:-1], pad[2:, 1:-1]),     # 90: vertical
            ((112.5, 157.5),        pad[:-2, :-2],  pad[2:, 2:]),       # 135: anti-diagonal
        ]
    else:  # 8 directions - use 4-dir as approximation (exact 8-dir needs interpolation)
        bins = [
            ((0, 11.25, 168.75, 180), pad[1:-1, :-2], pad[1:-1, 2:]),
            ((11.25, 33.75), 0.5*(pad[1:-1, 2:]+pad[:-2, 2:]), 0.5*(pad[1:-1, :-2]+pad[2:, :-2])),
            ((33.75, 56.25), pad[:-2, 2:], pad[2:, :-2]),
            ((56.25, 78.75), 0.5*(pad[:-2, 1:-1]+pad[:-2, 2:]), 0.5*(pad[2:, 1:-1]+pad[2:, :-2])),
            ((78.75, 101.25), pad[:-2, 1:-1], pad[2:, 1:-1]),
            ((101.25, 123.75), 0.5*(pad[:-2, 1:-1]+pad[:-2, :-2]), 0.5*(pad[2:, 1:-1]+pad[2:, 2:])),
            ((123.75, 146.25), pad[:-2, :-2], pad[2:, 2:]),
            ((146.25, 168.75), 0.5*(pad[1:-1, :-2]+pad[:-2, :-2]), 0.5*(pad[1:-1, 2:]+pad[2:, 2:])),
        ]

    out = np.zeros_like(mag)
    for angles_range, n1, n2 in bins:
        if len(angles_range) == 4:
            # Wrapping case (e.g., 0-22.5 and 157.5-180)
            mask = ((angle_deg >= angles_range[0]) & (angle_deg < angles_range[1])) | \
                   ((angle_deg >= angles_range[2]) & (angle_deg < angles_range[3]))
        else:
            mask = (angle_deg >= angles_range[0]) & (angle_deg < angles_range[1])
        local_max = (mag >= n1) & (mag >= n2) & mask
        out[local_max] = mag[local_max]

    return out


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------
test_configs = [
    {"label": "Best WVF (Np=25 d=2 Ns=3)",    "filter": "WVF", "Np": 25,  "d": 2, "Ns": 3},
    {"label": "Best WVF d=4 (Np=50 Ns=18)",    "filter": "WVF", "Np": 50,  "d": 4, "Ns": 18},
    {"label": "WVF Np=100 d=4",                "filter": "WVF", "Np": 100, "d": 4, "Ns": 18},
    {"label": "Bagan WVF (Np=250 d=4)",        "filter": "WVF", "Np": 250, "d": 4, "Ns": 18},
    {"label": "WVF Np=500 d=4",                "filter": "WVF", "Np": 500, "d": 4, "Ns": 18},
    {"label": "Best LF (Np=75 m=1)",           "filter": "LF",  "Np": 75,  "d": 4, "Ns": 18, "m": 1},
    {"label": "LF Np=50 m=3",                  "filter": "LF",  "Np": 50,  "d": 4, "Ns": 18, "m": 3},
    {"label": "LF Np=15 m=3",                  "filter": "LF",  "Np": 15,  "d": 4, "Ns": 18, "m": 3},
    {"label": "Bagan LF (Np=250 m=14)",        "filter": "LF",  "Np": 250, "d": 4, "Ns": 18, "m": 14},
    {"label": "LF Np=100 m=7",                 "filter": "LF",  "Np": 100, "d": 4, "Ns": 18, "m": 7},
]

# Canny settings to test
canny_settings = [
    {"label": "No NMS (raw)",        "nms": None},
    {"label": "NMS 4-dir",           "nms": "4dir"},
    {"label": "NMS 8-dir",           "nms": "8dir"},
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"\n=== Testing {len(test_configs)} configs × {len(canny_settings)} Canny settings ===\n")

results = []

for ci, cfg in enumerate(test_configs):
    print(f"[{ci+1}/{len(test_configs)}] {cfg['label']}...")

    # Run filter
    if cfg["filter"] == "WVF":
        result = wvf_image(img_gray, np_count=cfg["Np"], order=cfg["d"],
                           n_orientations=cfg["Ns"], backend=backend)
    else:
        result = lf_image(img_gray, half_width=cfg["m"], np_count=cfg["Np"],
                          order=cfg["d"], n_orientations=cfg["Ns"],
                          backend=backend, max_vram_gb=100)

    mag = result.gradient_mag
    angle = result.gradient_angle

    for cs in canny_settings:
        if cs["nms"] is None:
            mag_processed = mag
        elif cs["nms"] == "4dir":
            mag_processed = nms_vectorized(mag, angle, n_dirs=4)
        elif cs["nms"] == "8dir":
            mag_processed = nms_vectorized(mag, angle, n_dirs=8)

        ods, _, _, _ = compute_ods_ois(
            mag_processed, gt_bool.astype(np.float64),
            n_thresholds=N_THRESH, match_radius=3)

        row = {
            "config": cfg["label"],
            "filter": cfg["filter"],
            "Np": cfg["Np"],
            "Ns": cfg["Ns"],
            "d": cfg["d"],
            "canny": cs["label"],
            "ods": float(ods),
        }
        if "m" in cfg:
            row["m"] = cfg["m"]
        results.append(row)
        print(f"  {cs['label']:20s} ODS={ods:.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
with open(OUT / "canny_postprocess_test.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*75}")
print(f"{'Config':<35} {'Raw':>7} {'NMS-4':>7} {'NMS-8':>7} {'4-Raw':>7} {'8-Raw':>7}")
print(f"{'-'*75}")

configs_seen = []
for cfg in test_configs:
    label = cfg["label"]
    if label in configs_seen:
        continue
    configs_seen.append(label)

    raw = next((r["ods"] for r in results if r["config"]==label and r["canny"]=="No NMS (raw)"), None)
    nms4 = next((r["ods"] for r in results if r["config"]==label and r["canny"]=="NMS 4-dir"), None)
    nms8 = next((r["ods"] for r in results if r["config"]==label and r["canny"]=="NMS 8-dir"), None)

    d4 = nms4 - raw if (nms4 and raw) else 0
    d8 = nms8 - raw if (nms8 and raw) else 0
    print(f"{label:<35} {raw:>7.4f} {nms4:>7.4f} {nms8:>7.4f} {d4:>+7.4f} {d8:>+7.4f}")

print(f"{'='*75}")
