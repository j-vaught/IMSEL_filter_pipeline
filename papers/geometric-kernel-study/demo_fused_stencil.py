"""
Anisotropic oriented-filter edge detection demo.

Implements a simplified version of the fused stencil filter:
  R_k = sum_l alpha_{k,l} * f(X0 + dx_l, Y0 + dy_l)
  k* = argmax_k |R_k|

Uses Gaussian-weighted oriented line kernels at N_s orientations,
applies them via convolution, and picks the strongest response per pixel.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_oriented_kernel(theta, length=15, sigma=2.0):
    """Build a single oriented edge-detection stencil for angle theta.

    The kernel is a Gaussian-weighted first-derivative profile
    perpendicular to the line direction, sampled on a discrete grid.
    """
    half = length // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    # Rotate coordinates into the kernel frame
    ct, st = np.cos(theta), np.sin(theta)
    u =  x * ct + y * st   # along the line
    v = -x * st + y * ct   # perpendicular

    # Gaussian envelope along the line direction
    g_along = np.exp(-u**2 / (2 * sigma**2))
    # First derivative of Gaussian across the line (edge profile)
    g_across = -v * np.exp(-v**2 / (2 * (sigma * 0.6)**2))

    kernel = g_along * g_across
    kernel -= kernel.mean()          # zero-DC
    kernel /= np.abs(kernel).sum()   # normalise
    return kernel


def anisotropic_edge_detect(gray, n_orientations=36, length=15, sigma=2.0):
    """Run the oriented filter bank and pick the max-response orientation."""
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.zeros((n_orientations, *gray.shape), dtype=np.float64)

    for k, theta in enumerate(thetas):
        kernel = build_oriented_kernel(theta, length, sigma)
        responses[k] = convolve(gray, kernel, mode="reflect")

    abs_resp = np.abs(responses)
    k_star = np.argmax(abs_resp, axis=0)                  # winning orientation
    magnitude = np.take_along_axis(abs_resp, k_star[None], axis=0)[0]
    angle = thetas[k_star]

    return magnitude, angle, k_star


def main():
    img = Image.open("test_image.jpg").convert("L").resize((640, 427))
    gray = np.asarray(img, dtype=np.float64) / 255.0

    magnitude, angle, _ = anisotropic_edge_detect(gray, n_orientations=36)

    # Normalise magnitude to [0, 1]
    mag_norm = magnitude / (magnitude.max() + 1e-12)

    # --- Figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_axis_off()
    axes[0].set_title("Input", fontsize=11)

    axes[1].imshow(mag_norm, cmap="gray")
    axes[1].set_axis_off()
    axes[1].set_title("Gradient magnitude  $|R_{k^*}|$", fontsize=11)

    hsv = np.zeros((*gray.shape, 3))
    hsv[..., 0] = angle / np.pi                       # hue = orientation
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm                             # brightness = strength
    from matplotlib.colors import hsv_to_rgb
    axes[2].imshow(hsv_to_rgb(hsv))
    axes[2].set_axis_off()
    axes[2].set_title("Edge orientation  $\\theta_{k^*}$", fontsize=11)

    plt.tight_layout()
    plt.savefig("edge_result.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save magnitude as standalone image for Typst inclusion
    mag_img = Image.fromarray((mag_norm * 255).astype(np.uint8), mode="L")
    mag_img.save("edge_magnitude.png")

    # Save input grayscale
    img.save("input_gray.png")

    print("Saved: edge_result.png, edge_magnitude.png, input_gray.png")


if __name__ == "__main__":
    main()
