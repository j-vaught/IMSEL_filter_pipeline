"""Taylor expansion design matrix and neighborhood utilities.

These are the shared mathematical building blocks used by both
WVF and LF implementations on both CPU and GPU backends.
"""

import numpy as np


def build_taylor_matrix(coords, order=4):
    """Build the design matrix A for the 2D Taylor expansion.

    Given neighbor pixel positions (x_i, y_i) in the local coordinate system,
    construct the Vandermonde-like matrix whose columns correspond to each
    monomial term up to the specified order.

    For order=4, the expansion has 15 derivative coefficients:
      f, f_x, f_y, f_xx/2, f_yy/2, f_xy, f_xxx/6, f_yyy/6, f_xxy/2,
      f_xyy/2, f_xxxx/24, f_yyyy/24, f_xxxy/6, f_xxyy/4, f_xyyy/6

    Parameters
    ----------
    coords : ndarray, shape (Np, 2)
        Local coordinates (x_i, y_i) of neighbor pixels.
    order : int
        Maximum order of Taylor expansion (default 4).

    Returns
    -------
    A : ndarray, shape (Np, num_coefficients)
        Design matrix for the least-squares system.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    n = len(x)

    columns = [np.ones(n)]  # f^0 (constant term)

    if order >= 1:
        columns.append(x)                          # f_x * x
        columns.append(y)                          # f_y * y

    if order >= 2:
        columns.append(x**2 / 2)                   # f_xx * x^2/2
        columns.append(y**2 / 2)                   # f_yy * y^2/2
        columns.append(x * y)                      # f_xy * xy

    if order >= 3:
        columns.append(x**3 / 6)                   # f_xxx * x^3/6
        columns.append(y**3 / 6)                   # f_yyy * y^3/6
        columns.append(x**2 * y / 2)               # f_xxy * x^2*y/2
        columns.append(x * y**2 / 2)               # f_xyy * x*y^2/2

    if order >= 4:
        columns.append(x**4 / 24)                  # f_xxxx * x^4/24
        columns.append(y**4 / 24)                  # f_yyyy * y^4/24
        columns.append(x**3 * y / 6)               # f_xxxy * x^3*y/6
        columns.append(x**2 * y**2 / 4)            # f_xxyy * x^2*y^2/4
        columns.append(x * y**3 / 6)               # f_xyyy * x*y^3/6

    if order >= 5:
        columns.append(x**5 / 120)
        columns.append(y**5 / 120)
        columns.append(x**4 * y / 24)
        columns.append(x**3 * y**2 / 12)
        columns.append(x**2 * y**3 / 12)
        columns.append(x * y**4 / 24)

    A = np.column_stack(columns)
    return A


def get_circular_neighbors(np_count, radius=None):
    """Get pixel positions within a circular neighborhood.

    Selects the ``np_count`` closest integer-coordinate pixels to the
    origin within a circle, excluding the origin itself.

    Parameters
    ----------
    np_count : int
        Number of neighbor pixels (Np).
    radius : float, optional
        If given, use this radius. Otherwise, auto-compute.

    Returns
    -------
    coords : ndarray, shape (Np, 2)
        Integer (x, y) coordinates relative to origin.
    """
    if radius is None:
        radius = int(np.ceil(np.sqrt(np_count / np.pi))) + 1

    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                candidates.append((dx, dy, dist))

    candidates.sort(key=lambda c: c[2])
    selected = candidates[:np_count]
    coords = np.array([(c[0], c[1]) for c in selected], dtype=np.float64)
    return coords


def rotate_coordinates(coords, theta):
    """Rotate coordinates into the local coordinate system at angle theta.

    The WVF operates in a local (x, y) system rotated by theta from the
    global (X, Y) system. The normal direction is along x, tangential along y.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Coordinates in global frame.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    rotated : ndarray, shape (N, 2)
        Coordinates in local frame.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, sin_t],
                  [-sin_t, cos_t]])
    return coords @ R.T


def get_square_neighbors(np_count, radius=None):
    """Get pixel positions within a square neighborhood.

    Selects the ``np_count`` closest integer-coordinate pixels to the
    origin within a square grid, excluding the origin itself.

    Parameters
    ----------
    np_count : int
        Number of neighbor pixels (Np).
    radius : int, optional
        If given, use this half-width. Otherwise, auto-compute from np_count.

    Returns
    -------
    coords : ndarray, shape (Np, 2)
        Integer (x, y) coordinates relative to origin.
    """
    if radius is None:
        # (2r+1)^2 - 1 >= np_count  =>  r >= (sqrt(np_count+1) - 1) / 2
        radius = int(np.ceil((np.sqrt(np_count + 1) - 1) / 2))

    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.sqrt(dx**2 + dy**2)
            candidates.append((dx, dy, dist))

    candidates.sort(key=lambda c: c[2])
    selected = candidates[:np_count]
    coords = np.array([(c[0], c[1]) for c in selected], dtype=np.float64)
    return coords


def compute_wvf_pseudoinverse(coords, order=4):
    """Compute the pseudo-inverse A* = (A^T A)^{-1} A^T.

    Parameters
    ----------
    coords : ndarray, shape (Np, 2)
        Local coordinates of neighbor pixels.
    order : int
        Taylor expansion order.

    Returns
    -------
    A_pinv : ndarray
        Pseudo-inverse matrix.
    cond_number : float
        Condition number of A^T A.
    """
    A = build_taylor_matrix(coords, order=order)
    ATA = A.T @ A
    cond = np.linalg.cond(ATA)
    A_pinv = np.linalg.pinv(A)
    return A_pinv, cond
