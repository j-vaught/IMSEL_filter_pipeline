"""
Side-by-side comparison of the three oriented filter approaches:
  1. Original anisotropic (virtual-pixel fused stencil)
  2. Elliptical Gaussian kernel
  3. Rectangular Gaussian kernel

All use the same image, orientation count, and matching aspect ratio (~1.67).
"""

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from aniso_edge_demo import anisotropic_edge_detect as detect_original
from elliptical_edge_demo import anisotropic_edge_detect as detect_elliptical
from rectangular_edge_demo import anisotropic_edge_detect as detect_rectangular


def main():
    img = Image.open("test_image.jpg").convert("L").resize((640, 427))
    gray = np.asarray(img, dtype=np.float64) / 255.0

    # Run all three detectors
    mag_orig, ang_orig, _ = detect_original(gray, n_orientations=36,
                                            length=15, sigma=2.0)
    mag_ell, ang_ell, _ = detect_elliptical(gray, n_orientations=36,
                                            sigma_along=2.0,
                                            sigma_across=1.2, length=15)
    mag_rect, ang_rect, _ = detect_rectangular(gray, n_orientations=36,
                                               half_along=6.0,
                                               half_across=3.6,
                                               sigma_along=2.0,
                                               sigma_across=1.2, length=15)

    # Normalise magnitudes independently
    mag_orig_n = mag_orig / (mag_orig.max() + 1e-12)
    mag_ell_n = mag_ell / (mag_ell.max() + 1e-12)
    mag_rect_n = mag_rect / (mag_rect.max() + 1e-12)

    # Build HSV orientation images
    def make_hsv(angle, mag_n):
        hsv = np.zeros((*gray.shape, 3))
        hsv[..., 0] = angle / np.pi
        hsv[..., 1] = 1.0
        hsv[..., 2] = mag_n
        return hsv_to_rgb(hsv)

    hsv_orig = make_hsv(ang_orig, mag_orig_n)
    hsv_ell = make_hsv(ang_ell, mag_ell_n)
    hsv_rect = make_hsv(ang_rect, mag_rect_n)

    # --- Comparison figure: 3 rows x 3 cols --------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    labels = ["Original (fused stencil)", "Elliptical Gaussian",
              "Rectangular Gaussian"]
    mags = [mag_orig_n, mag_ell_n, mag_rect_n]
    hsvs = [hsv_orig, hsv_ell, hsv_rect]

    for row, (label, mag_n, hsv_img) in enumerate(zip(labels, mags, hsvs)):
        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_axis_off()
        if row == 0:
            axes[row, 0].set_title("Input", fontsize=11)

        axes[row, 1].imshow(mag_n, cmap="gray")
        axes[row, 1].set_axis_off()
        if row == 0:
            axes[row, 1].set_title("Gradient magnitude", fontsize=11)

        axes[row, 2].imshow(hsv_img)
        axes[row, 2].set_axis_off()
        if row == 0:
            axes[row, 2].set_title("Edge orientation", fontsize=11)

        axes[row, 0].set_ylabel(label, fontsize=11, rotation=90,
                                labelpad=15)

    plt.tight_layout()
    plt.savefig("filter_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: filter_comparison.png")


if __name__ == "__main__":
    main()
