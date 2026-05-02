"""Taylor expansion design matrix and neighborhood utilities."""

import math

import numpy as np


def default_pinv_rcond(shape, dtype=np.float64):
    """Return the scaled-epsilon cutoff used for WVF pseudoinverses."""
    rows, cols = int(shape[-2]), int(shape[-1])
    return max(rows, cols) * np.finfo(dtype).eps


def compute_pseudoinverse(matrix, rcond=None):
    """Compute a WVF pseudoinverse with an explicit singular-value cutoff."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    cutoff = default_pinv_rcond(mat.shape, dtype=mat.dtype) if rcond is None else float(rcond)
    return np.linalg.pinv(mat, rcond=cutoff)


def _taylor_exponents(order):
    """Return WVF Taylor exponents through ``order`` with stable ordering."""
    if order < 0:
        raise ValueError("order must be nonnegative")

    exponents = [(0, 0)]
    if order >= 1:
        exponents.extend([(1, 0), (0, 1)])

    for degree in range(2, order + 1):
        exponents.append((degree, 0))
        exponents.append((0, degree))
        for px in range(degree - 1, 0, -1):
            exponents.append((px, degree - px))

    return exponents


def build_taylor_matrix(coords, order=4, normalize_radius=None):
    """Build the design matrix A for the 2D Taylor expansion.

    Given neighbor pixel positions (x_i, y_i) in the local coordinate system,
    construct the Vandermonde-like matrix whose columns correspond to each
    monomial term up to the specified order.

    For order=4, the expansion has 15 derivative coefficients:
      f, f_x, f_y, f_xx/2, f_yy/2, f_xy, f_xxx/6, f_yyy/6, f_xxy/2,
      f_xyy/2, f_xxxx/24, f_yyyy/24, f_xxxy/6, f_xxyy/4, f_xyyy/6

    Higher orders follow the same convention. Each monomial is scaled by
    ``1 / (p! q!)`` so the unknown coefficient corresponds to the derivative
    ``d^(p+q) f / dx^p dy^q``.

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
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")

    d = int(order)
    x = coords[:, 0]
    y = coords[:, 1]
    if normalize_radius is not None:
        radius = float(normalize_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("normalize_radius must be a positive finite value")
        x = x / radius
        y = y / radius

    columns = []
    for px, py in _taylor_exponents(d):
        scale = math.factorial(px) * math.factorial(py)
        columns.append((x**px) * (y**py) / scale)

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


def compute_wvf_pseudoinverse(coords, order=4, normalize_radius=None, rcond=None):
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
    A = build_taylor_matrix(coords, order=order, normalize_radius=normalize_radius)
    ATA = A.T @ A
    cond = np.linalg.cond(ATA)
    A_pinv = compute_pseudoinverse(A, rcond=rcond)
    return A_pinv, cond
