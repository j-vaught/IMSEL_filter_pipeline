"""A5 reviewer-pass-5 ablation: validity reference statistic.

This script compares four choices for the reference statistic in
R(x, y) > tau * R_ref on a clean synthetic image and a deterministic
glint-contaminated variant of the same image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, maximum_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edgecritic.lf._metal import lf_stack
from edgecritic.wvf._metal import wvf_radius_gradients_metal
from edgecritic.wvf._radius_kernels import build_wvf_radius_kernels


VARIANTS = (
    ("max", "image-wide max"),
    ("p99", "99th percentile"),
    ("mad", "median + 6 MAD"),
    ("local", "local 256 px max"),
)


def _load_rgb(size: int) -> np.ndarray:
    path = (
        ROOT
        / "example_images"
        / "synthetic_nested_shapes"
        / "clean"
        / "4096"
        / "nested_star_square_oval_low_contrast_mixed_chroma_4096.png"
    )
    resample = Image.Resampling.BILINEAR
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), resample))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return (
        0.2126 * arr[..., 0]
        + 0.7152 * arr[..., 1]
        + 0.0722 * arr[..., 2]
    ).astype(np.float32)


def _edge_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16)
    hdiff = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).max(axis=2) > 6
    vdiff = np.abs(arr[1:, :, :] - arr[:-1, :, :]).max(axis=2) > 6
    edge = np.zeros(rgb.shape[:2], dtype=bool)
    edge[:, 1:] |= hdiff
    edge[:, :-1] |= hdiff
    edge[1:, :] |= vdiff
    edge[:-1, :] |= vdiff
    edge = binary_dilation(edge, iterations=1)
    border = 48
    edge[:border, :] = False
    edge[-border:, :] = False
    edge[:, :border] = False
    edge[:, -border:] = False
    return edge


def _edge_dense_glint_window(edge: np.ndarray, patch: int) -> tuple[slice, slice]:
    density = maximum_filter(edge.astype(np.float32), size=patch, mode="constant")
    border = patch
    density[:border, :] = -1.0
    density[-border:, :] = -1.0
    density[:, :border] = -1.0
    density[:, -border:] = -1.0
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    y0 = int(np.clip(cy - patch // 2, border, edge.shape[0] - border - patch))
    x0 = int(np.clip(cx - patch // 2, border, edge.shape[1] - border - patch))
    return slice(y0, y0 + patch), slice(x0, x0 + patch)


def _add_glint(rgb: np.ndarray, edge: np.ndarray, seed: int = 29) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    patch = max(64, rgb.shape[0] // 8)
    ys, xs = _edge_dense_glint_window(edge, patch)
    glint_mask = np.zeros(rgb.shape[:2], dtype=bool)
    yy, xx = np.mgrid[ys, xs]
    cy = (ys.start + ys.stop - 1) / 2.0
    cx = (xs.start + xs.stop - 1) / 2.0
    sigma = patch / 5.0
    cluster = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    sparkle = (cluster > 0.24) | ((cluster > 0.05) & (rng.random(cluster.shape) < 0.13))
    glint_mask[ys, xs] = sparkle
    out[glint_mask] = 255
    halo = binary_dilation(glint_mask, iterations=4)
    out[halo & ~glint_mask] = np.maximum(out[halo & ~glint_mask], 232)
    return out, binary_dilation(glint_mask, iterations=6)


def _range_image(image: np.ndarray, radius: int, degree: int, m: int) -> np.ndarray:
    kernels = build_wvf_radius_kernels(radius=radius, order=degree)
    gx, gy = wvf_radius_gradients_metal(image, kernels, output_dtype=np.float32)
    response = lf_stack(
        gx,
        gy,
        lf_half_length=m,
        n_orientations=64,
        output_dtype=np.float32,
        method="box",
    )
    return (response.max(axis=0) - response.min(axis=0)).astype(np.float32)


def _validity_mask(R: np.ndarray, variant: str, tau: float, window: int) -> np.ndarray:
    if variant == "max":
        ref = float(R.max())
    elif variant == "p99":
        ref = float(np.percentile(R, 99.0))
    elif variant == "mad":
        med = float(np.median(R))
        mad = float(np.median(np.abs(R - med)))
        ref = med + 6.0 * mad
    elif variant == "local":
        ref = maximum_filter(R, size=window, mode="nearest")
    else:
        raise ValueError(f"unknown validity variant {variant!r}")
    return R > tau * ref


def _metrics_for(
    R: np.ndarray,
    edge: np.ndarray,
    glint_region: np.ndarray,
    tau: float,
    window: int,
    image_class: str,
) -> list[dict]:
    rows = []
    for variant, label in VARIANTS:
        valid = _validity_mask(R, variant, tau, window)
        clean_edges = edge & ~glint_region
        glint_edges = edge & glint_region
        glint_non_edges = (~edge) & glint_region
        clean_recall = float(valid[clean_edges].mean()) if clean_edges.any() else 0.0
        glint_recall = float(valid[glint_edges].mean()) if glint_edges.any() else 0.0
        glint_fpr = float(valid[glint_non_edges].mean()) if glint_non_edges.any() else 0.0
        rows.append(
            {
                "image_class": image_class,
                "variant": variant,
                "label": label,
                "clean_recall": clean_recall,
                "glint_recall": glint_recall,
                "glint_false_positive_rate": glint_fpr,
                "n_clean_edge": int(clean_edges.sum()),
                "n_glint_edge": int(glint_edges.sum()),
                "n_glint_non_edge": int(glint_non_edges.sum()),
            }
        )
    return rows


def run_ablation(output_path: Path, size: int = 1024) -> dict:
    radius = 5
    degree = 3
    m = 40
    tau = 0.10
    local_window = 256
    rgb = _load_rgb(size)
    edge = _edge_mask(rgb)
    empty_glint = np.zeros(edge.shape, dtype=bool)
    glint_rgb, glint_region = _add_glint(rgb, edge)

    clean_R = _range_image(_luminance(rgb), radius, degree, m)
    glint_R = _range_image(_luminance(glint_rgb), radius, degree, m)
    rows = []
    rows.extend(_metrics_for(clean_R, edge, empty_glint, tau, local_window, "clean"))
    rows.extend(_metrics_for(glint_R, edge, glint_region, tau, local_window, "glint"))

    clean_rows = {row["variant"]: row for row in rows if row["image_class"] == "clean"}
    glint_rows = {row["variant"]: row for row in rows if row["image_class"] == "glint"}
    local_glint = glint_rows["local"]["glint_recall"]
    eligible = [
        variant
        for variant, _ in VARIANTS
        if glint_rows[variant]["glint_recall"] >= local_glint - 0.05
    ]
    winner = max(
        eligible,
        key=lambda variant: (
            clean_rows[variant]["clean_recall"],
            -glint_rows[variant]["glint_false_positive_rate"],
            glint_rows[variant]["glint_recall"],
        ),
    )
    decision_text = (
        f"Pick {dict(VARIANTS)[winner]}; it maximises clean recall among "
        "variants within 5 percentage points of the local-window glint recall."
    )

    output = {
        "ablation": "A5",
        "config": {
            "image_size": size,
            "radius": radius,
            "degree": degree,
            "lf_half_length": m,
            "tau": tau,
            "local_window": local_window,
            "mad_multiplier": 6.0,
        },
        "rows": rows,
        "summary": {
            "winner": winner,
            "winner_label": dict(VARIANTS)[winner],
            "local_glint_recall": local_glint,
            "decision_text": decision_text,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT.parent / "New project",
        help="paper repository root",
    )
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    out = (
        args.paper_root
        / "cetz_figures"
        / "data"
        / "ablation_a5"
        / "results.json"
    )
    result = run_ablation(out, size=args.size)
    print(f"wrote {out}")
    print(
        "A5 decision: "
        f"{result['summary']['winner']} "
        f"({result['summary']['winner_label']})"
    )
    for row in result["rows"]:
        if row["image_class"] == "glint":
            print(
                f"  {row['variant']:>5}: clean-edge recall in glint image "
                f"{row['clean_recall']:.3f}, glint-edge recall "
                f"{row['glint_recall']:.3f}, glint FPR "
                f"{row['glint_false_positive_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
