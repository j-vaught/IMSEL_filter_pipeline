"""
Elliptical Gaussian oriented-filter edge detection demo.

Defines a rotatable elliptical region with a 2D Gaussian distribution,
multiplied by a first-derivative profile in the normal direction.
No virtual pixels or polynomial fitting -- each pixel gets a single,
clean weight based on its geometric position within the ellipse.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_elliptical_kernel(theta, sigma_along=2.0, sigma_across=1.2,
                            length=15):
    """Build an oriented edge kernel using an elliptical Gaussian envelope.

    Parameters
    ----------
    theta : float
        Orientation angle in radians.
    sigma_along : float
        Gaussian std along the edge direction (major axis).
    sigma_across : float
        Gaussian std perpendicular to the edge (minor axis).
    length : int
        Kernel side length (odd). Determines the discrete grid size.
    """
    half = length // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    # Rotate coordinates into the kernel frame
    ct, st = np.cos(theta), np.sin(theta)
    u =  x * ct + y * st   # along the edge
    v = -x * st + y * ct   # perpendicular to the edge (normal)

    # Elliptical Gaussian envelope
    ellipse_arg = (u**2 / sigma_along**2) + (v**2 / sigma_across**2)

    # Mask: only include pixels inside the ~3-sigma ellipse
    mask = ellipse_arg <= 9.0  # 3-sigma boundary

    gauss = np.exp(-0.5 * ellipse_arg) * mask

    # First derivative of Gaussian in the normal direction (edge profile)
    kernel = -v * gauss

    # Zero-DC and normalise
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-12
    return kernel


def anisotropic_edge_detect(gray, n_orientations=36, sigma_along=2.0,
                            sigma_across=1.2, length=15):
    """Run the oriented elliptical filter bank and pick max response."""
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.zeros((n_orientations, *gray.shape), dtype=np.float64)

    for k, theta in enumerate(thetas):
        kernel = build_elliptical_kernel(theta, sigma_along, sigma_across,
                                         length)
        responses[k] = convolve(gray, kernel, mode="reflect")

    abs_resp = np.abs(responses)
    k_star = np.argmax(abs_resp, axis=0)
    magnitude = np.take_along_axis(abs_resp, k_star[None], axis=0)[0]
    angle = thetas[k_star]

    return magnitude, angle, k_star


def main():
    img = Image.open("test_image.jpg").convert("L").resize((640, 427))
    gray = np.asarray(img, dtype=np.float64) / 255.0

    # Aspect ratio ~1.67 to match original filter (sigma 2.0 / 1.2)
    sigma_along = 2.0
    sigma_across = 1.2
    aspect_ratio = sigma_along / sigma_across

    magnitude, angle, _ = anisotropic_edge_detect(
        gray, n_orientations=36,
        sigma_along=sigma_along, sigma_across=sigma_across, length=15
    )

    mag_norm = magnitude / (magnitude.max() + 1e-12)

    # --- Figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_axis_off()

    axes[1].imshow(mag_norm, cmap="gray")
    axes[1].set_axis_off()
    axes[1].set_ylabel("Gradient magnitude", fontsize=11)

    hsv = np.zeros((*gray.shape, 3))
    hsv[..., 0] = angle / np.pi
    hsv[..., 1] = 1.0
    hsv[..., 2] = mag_norm
    from matplotlib.colors import hsv_to_rgb
    axes[2].imshow(hsv_to_rgb(hsv))
    axes[2].set_axis_off()

    fig.suptitle(
        f"Elliptical Gaussian kernel  "
        f"(aspect ratio {aspect_ratio:.2f})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig("edge_elliptical.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save standalone magnitude
    mag_img = Image.fromarray((mag_norm * 255).astype(np.uint8), mode="L")
    mag_img.save("edge_magnitude_elliptical.png")

    print(f"Elliptical kernel: sigma_along={sigma_along}, "
          f"sigma_across={sigma_across}, aspect={aspect_ratio:.2f}")
    print("Saved: edge_elliptical.png, edge_magnitude_elliptical.png")


if __name__ == "__main__":
    main()
