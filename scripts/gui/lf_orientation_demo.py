"""Interactive LF-orientation demo. Streamlit UI.

Run with::

    PYTHONPATH=src streamlit run scripts/gui/lf_orientation_demo.py

Lets the user pick an image (or one of the four channels of the
synthetic color image), a pixel, and the four LF parameters
(N, r, d, m). For a chosen demo orientation, the support footprint
of the LF (the 2m+1 WVF disks placed along the snapped line) is
overlaid on a 256x256 crop centered on the pixel. Below the image,
the |L_theta| response curve over N orientations is plotted, with
the demo angle marked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from matplotlib.patches import Circle, Rectangle

from edgecritic.wvf._radius_kernels import (
    wvf_radius_gradients_cpu, build_wvf_radius_kernels)

# Try to load the Metal GPU backend (Apple Silicon). Fall back to CPU
# silently if the Rust dylib hasn't been built or Metal isn't
# available.
try:
    from edgecritic.wvf._metal import (
        wvf_radius_gradients_metal, metal_backend_available)
    _METAL_OK = metal_backend_available()
except Exception:
    wvf_radius_gradients_metal = None
    _METAL_OK = False


# ----- Image registry --------------------------------------------------
DEFAULT_IMAGE_DIR = Path(
    "/Users/user/Documents/New project/cetz_figures/data/color_channels")
IMAGE_OPTIONS = {
    "1024 garnet · L":   DEFAULT_IMAGE_DIR / "1024" / "channel_L.png",
    "1024 garnet · R":   DEFAULT_IMAGE_DIR / "1024" / "channel_R.png",
    "1024 garnet · G":   DEFAULT_IMAGE_DIR / "1024" / "channel_G.png",
    "1024 garnet · B":   DEFAULT_IMAGE_DIR / "1024" / "channel_B.png",
    "1024 garnet · color (RGB->L)": DEFAULT_IMAGE_DIR / "1024" / "original.png",
    "4096 lowcon · L":   DEFAULT_IMAGE_DIR / "4096_lowcon" / "channel_L.png",
    "4096 garnet · L":   DEFAULT_IMAGE_DIR / "4096" / "channel_L.png",
}


# ----- LF computation --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_image(path_str: str) -> np.ndarray:
    return np.asarray(Image.open(path_str).convert("L"), dtype=np.float64)


@st.cache_data(show_spinner=False)
def compute_wvf(path_str: str, r: int, d: int,
                use_gpu: bool = True) -> tuple[np.ndarray, np.ndarray]:
    img = load_image(path_str)
    if use_gpu and _METAL_OK:
        kernels = build_wvf_radius_kernels(radius=r, order=d)
        return wvf_radius_gradients_metal(img, kernels)
    return wvf_radius_gradients_cpu(img, radius=r, order=d)


def lf_offsets(theta_rad: float, m: int) -> tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """Return (ix, iy, w) arrays for the 2m+1 line samples."""
    cos_t = float(np.cos(theta_rad))
    sin_t = float(np.sin(theta_rad))
    if m <= 0:
        return np.array([0]), np.array([0]), np.array([1.0])
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0
    j_off = np.arange(-m, m + 1)
    sigma = m / 2.0
    weights = np.exp(-0.5 * (j_off / sigma) ** 2)
    ix = np.round(j_off * step * cos_t).astype(int)
    iy = np.round(j_off * step * sin_t).astype(int)
    return ix, iy, weights


def lf_at_pixel(g_x, g_y, px: int, py: int,
                theta_rad: float, m: int) -> float:
    H, W = g_x.shape
    cos_t = float(np.cos(theta_rad))
    sin_t = float(np.sin(theta_rad))
    if m <= 0:
        return abs(-sin_t * g_x[py, px] + cos_t * g_y[py, px])
    ix, iy, w = lf_offsets(theta_rad, m)
    acc = 0.0
    wsum = 0.0
    for ixi, iyi, wi in zip(ix, iy, w):
        yy, xx = py + iyi, px + ixi
        if 0 <= yy < H and 0 <= xx < W:
            g_perp = -sin_t * g_x[yy, xx] + cos_t * g_y[yy, xx]
            acc += wi * g_perp
            wsum += wi
    return abs(acc / wsum) if wsum > 0 else 0.0


def lf_curve_at_pixel(g_x, g_y, px, py, n_orient, m):
    angles = np.linspace(0, np.pi, n_orient, endpoint=False)
    return angles, np.array([lf_at_pixel(g_x, g_y, px, py, t, m)
                              for t in angles])


# ----- Streamlit UI ----------------------------------------------------
st.set_page_config(page_title="LF orientation demo", layout="wide")
st.title("LF orientation demo")
if _METAL_OK:
    st.caption("WVF backend: **Metal (GPU)**")
else:
    st.caption("WVF backend: CPU "
                "(Metal dylib unavailable - fallback)")

with st.sidebar:
    st.header("Inputs")
    image_label = st.selectbox("Image / channel",
                                list(IMAGE_OPTIONS.keys()),
                                index=0)
    image_path = IMAGE_OPTIONS[image_label]
    if not image_path.exists():
        st.error(f"missing: {image_path}")
        st.stop()

    img = load_image(str(image_path))
    H, W = img.shape

    st.divider()
    st.subheader("Pixel")
    px = st.number_input("x", min_value=0, max_value=W - 1,
                          value=min(601, W - 1), step=1)
    py = st.number_input("y", min_value=0, max_value=H - 1,
                          value=min(512, H - 1), step=1)

    st.divider()
    st.subheader("Filter")
    r = st.number_input("WVF radius r",
                         min_value=1, value=3, step=1)
    d = st.number_input("WVF polynomial degree d",
                         min_value=1, value=5, step=1)
    m = st.number_input("LF half-length m",
                         min_value=0, value=2, step=1)
    n_orient = st.number_input("N orientations",
                                min_value=4, value=64, step=1)
    demo_orientation = st.number_input("demo_orientation (deg)",
                                        min_value=0, max_value=179,
                                        value=45, step=1)

# Compute
with st.spinner(
        f"computing WVF gradients "
        f"({'GPU/Metal' if _METAL_OK else 'CPU'}) …"):
    g_x, g_y = compute_wvf(str(image_path), int(r), int(d))

theta_rad_demo = np.radians(demo_orientation)
angles, response = lf_curve_at_pixel(g_x, g_y, px, py, n_orient, m)
demo_response = lf_at_pixel(g_x, g_y, px, py, theta_rad_demo, m)
ix, iy, weights = lf_offsets(theta_rad_demo, m)

# ----- Layout -----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Image with LF support")
    # Adaptive crop so the full LF support is always in view, even at
    # large m / r. Step factor 1/max(|cos|, |sin|) is at most sqrt(2).
    crop_half = max(80, int(m * 1.5 + r + 12))
    y0 = max(0, py - crop_half)
    x0 = max(0, px - crop_half)
    y1 = min(H, py + crop_half + 1)
    x1 = min(W, px + crop_half + 1)
    crop = img[y0:y1, x0:x1]
    fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
    ax1.imshow(crop, cmap="gray", vmin=0, vmax=255,
                extent=[x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5],
                interpolation="nearest")

    # Pixel marker
    ax1.plot([px], [py], marker="x", markersize=10,
              color="#CC2E40", markeredgewidth=2)

    # WVF disk at each line offset (LF support)
    for ixi, iyi in zip(ix, iy):
        cx = px + int(ixi)
        cy = py + int(iyi)
        ax1.add_patch(Circle((cx, cy), radius=r, fill=False,
                              linewidth=1.0, edgecolor="#466A9F",
                              alpha=0.7))

    # Line through virtual pixels
    if len(ix) > 1:
        ax1.plot([px + int(ix[0]), px + int(ix[-1])],
                  [py + int(iy[0]), py + int(iy[-1])],
                  color="#CC2E40", linewidth=1.5, alpha=0.9)

    ax1.set_xlim(x0 - 0.5, x1 - 0.5)
    ax1.set_ylim(y1 - 0.5, y0 - 0.5)  # image y down
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(
        f"({px}, {py})  ·  r={r}, d={d}, m={m}  ·  "
        f"theta={demo_orientation}°")
    st.pyplot(fig1)

with col2:
    st.subheader(f"Response curve  (N = {n_orient})")
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    ax2.plot(np.degrees(angles), response, marker="o",
              markersize=3.5, color="#363636", linewidth=1.2)
    ax2.axvline(demo_orientation, color="#CC2E40",
                 linestyle="--", linewidth=1.2,
                 label=f"demo θ = {demo_orientation}°  "
                       f"(|L| = {demo_response:.2f})")
    ax2.set_xlim(0, 180)
    ax2.set_xticks([0, 45, 90, 135, 180])
    ax2.set_xlabel("θ  (deg)")
    ax2.set_ylabel("|L_θ|")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    st.pyplot(fig2)

st.caption(
    "WVF disks (blue circles) show the support of each of the 2m+1 "
    "samples along the steered line; the rose × marks the target "
    "pixel and the rose line is the LF aggregation axis at the demo "
    "orientation. The response curve is the |L_θ| value at the same "
    "pixel for N evenly-spaced orientations on [0°, 180°)."
)
