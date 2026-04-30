"""CPU implementation of the Wide View Filter."""

import numpy as np

from edgecritic.core.taylor import (
    build_taylor_matrix,
    compute_wvf_pseudoinverse,
    get_circular_neighbors,
    rotate_coordinates,
)


def wvf_single_pixel(image, X0, Y0, np_count=15, order=4, theta=0.0):
    """Apply the Wide View Filter at a single pixel at orientation theta.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    X0, Y0 : int
        Global pixel coordinates of target pixel.
    np_count : int
        Number of neighbor pixels.
    order : int
        Taylor expansion order.
    theta : float
        Orientation angle in radians.

    Returns
    -------
    f_x : float
        Normal derivative.
    f_y : float
        Tangential derivative.
    cond : float
        Condition number of the system.
    """
    H, W = image.shape
    global_coords = get_circular_neighbors(np_count)

    rotated_local = rotate_coordinates(global_coords, theta)

    global_pixel_coords = global_coords.astype(int) + np.array([X0, Y0])

    valid = ((global_pixel_coords[:, 0] >= 0) &
             (global_pixel_coords[:, 0] < W) &
             (global_pixel_coords[:, 1] >= 0) &
             (global_pixel_coords[:, 1] < H))

    if np.sum(valid) < order * (order + 1) // 2 + 3:
        return 0.0, 0.0, np.inf

    local_coords = rotated_local[valid]
    pixel_values = image[global_pixel_coords[valid, 1],
                         global_pixel_coords[valid, 0]]

    A_pinv, cond = compute_wvf_pseudoinverse(local_coords, order=order)

    z = A_pinv @ pixel_values

    f_x = z[1] if len(z) > 1 else 0.0
    f_y = z[2] if len(z) > 2 else 0.0

    return f_x, f_y, cond


def wvf_image(image, np_count=15, order=4, n_orientations=36):
    """Apply WVF to entire image at multiple orientations (CPU).

    For each pixel, runs WVF at ``n_orientations`` angles and keeps the
    orientation with the maximum normal derivative magnitude.

    Note: sweeps angles over [0, 2*pi). For most edge detection purposes
    [0, pi) suffices since |f_x| is symmetric, but the full circle is
    retained for backward compatibility.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale image.
    np_count : int
        Number of neighbor pixels per WVF application.
    order : int
        Taylor expansion order.
    n_orientations : int
        Number of orientations to sample.

    Returns
    -------
    gradient_mag : ndarray, shape (H, W)
    gradient_angle : ndarray, shape (H, W)
    condition_numbers : ndarray, shape (H, W)
    """
    H, W = image.shape
    img = image.astype(np.float64)

    gradient_mag = np.zeros((H, W))
    gradient_angle = np.zeros((H, W))
    condition_numbers = np.zeros((H, W))

    angles = np.linspace(0, 2 * np.pi, n_orientations, endpoint=False)

    global_neighbors = get_circular_neighbors(np_count)

    border = int(np.max(np.abs(global_neighbors))) + 1

    for y in range(border, H - border):
        for x in range(border, W - border):
            max_mag = 0.0
            best_angle = 0.0
            best_cond = np.inf

            pixel_values_cache = {}
            for ni in range(len(global_neighbors)):
                gx = int(global_neighbors[ni, 0]) + x
                gy = int(global_neighbors[ni, 1]) + y
                pixel_values_cache[ni] = img[gy, gx]

            pixel_vals = np.array([pixel_values_cache[i]
                                   for i in range(len(global_neighbors))])

            for theta in angles:
                local_coords = rotate_coordinates(global_neighbors, theta)
                A = build_taylor_matrix(local_coords, order=order)
                try:
                    z = np.linalg.lstsq(A, pixel_vals, rcond=None)[0]
                    f_x = z[1]
                    mag = abs(f_x)
                    if mag > max_mag:
                        max_mag = mag
                        best_angle = theta
                        cond = np.linalg.cond(A.T @ A)
                        best_cond = cond
                except np.linalg.LinAlgError:
                    continue

            gradient_mag[y, x] = max_mag
            gradient_angle[y, x] = best_angle
            condition_numbers[y, x] = best_cond

    return gradient_mag, gradient_angle, condition_numbers
