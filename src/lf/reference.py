"""CPU implementation of the Line Filter."""

import numpy as np

from core.taylor import get_circular_neighbors
from wvf.reference import wvf_single_pixel


def line_filter_single_pixel(image, X0, Y0, half_width=7, np_count=15,
                              order=4, theta=0.0, use_weights=True):
    """Apply Line Filter at a single pixel.

    Chains (2m+1) WVF applications along a line centered on (X0, Y0)
    at orientation theta, combined with Gaussian weighting.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    X0, Y0 : int
        Target pixel coordinates.
    half_width : int
        Half-width m of the line filter. Total length = 2m+1.
    np_count : int
        Neighbor count per WVF application.
    order : int
        Taylor expansion order.
    theta : float
        Line orientation in radians.
    use_weights : bool
        Whether to use Gaussian distance weighting.

    Returns
    -------
    f_x : float
        Weighted normal derivative.
    f_y : float
        Weighted tangential derivative.
    """
    H, W = image.shape
    img = image.astype(np.float64)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    total_fx = 0.0
    total_fy = 0.0
    total_weight = 0.0

    sigma = half_width / 2.0 if use_weights else 1.0

    for j in range(-half_width, half_width + 1):
        vx = X0 + j * cos_t
        vy = Y0 + j * sin_t

        vx_int = int(round(vx))
        vy_int = int(round(vy))

        if vx_int < 0 or vx_int >= W or vy_int < 0 or vy_int >= H:
            continue

        if use_weights:
            w = np.exp(-0.5 * (j / sigma) ** 2)
        else:
            w = 1.0

        fx, fy, _ = wvf_single_pixel(img, vx_int, vy_int,
                                       np_count=np_count, order=order,
                                       theta=theta)
        total_fx += w * fx
        total_fy += w * fy
        total_weight += w

    if total_weight > 0:
        total_fx /= total_weight
        total_fy /= total_weight

    return total_fx, total_fy


def lf_image(image, half_width=7, np_count=15, order=4, n_orientations=36,
             use_weights=True, subsample=1):
    """Apply Line Filter to entire image at multiple orientations (CPU).

    Note: sweeps angles over [0, pi), unlike WVF which sweeps [0, 2*pi).

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    half_width : int
        LF half-width (m).
    np_count : int
        Neighbor count per WVF.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations.
    use_weights : bool
        Gaussian weighting of virtual pixels.
    subsample : int
        Process every Nth pixel (for speed during testing).

    Returns
    -------
    gradient_mag : ndarray, shape (H, W)
    gradient_angle : ndarray, shape (H, W)
    all_gradients : ndarray, shape (H, W, n_orientations)
    """
    H, W = image.shape
    img = image.astype(np.float64)

    gradient_mag = np.zeros((H, W))
    gradient_angle = np.zeros((H, W))
    all_gradients = np.zeros((H, W, n_orientations))

    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)

    border = half_width + int(np.ceil(np.sqrt(np_count / np.pi))) + 2

    for y in range(border, H - border, subsample):
        for x in range(border, W - border, subsample):
            max_mag = 0.0
            best_angle = 0.0

            for ai, theta in enumerate(angles):
                fx, fy = line_filter_single_pixel(
                    img, x, y, half_width=half_width,
                    np_count=np_count, order=order,
                    theta=theta, use_weights=use_weights
                )
                mag = abs(fx)
                all_gradients[y, x, ai] = mag

                if mag > max_mag:
                    max_mag = mag
                    best_angle = theta

            gradient_mag[y, x] = max_mag
            gradient_angle[y, x] = best_angle

    return gradient_mag, gradient_angle, all_gradients
