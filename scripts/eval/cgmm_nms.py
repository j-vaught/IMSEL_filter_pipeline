"""Enhanced non-maximum suppression for the c-GMM fused output.

Consumes the per-pixel five-tuple
    (theta_primary, M_primary, theta_sec, M_sec, v_fused)
and emits a thinned magnitude map.  The "enhanced" part is the corner
OR rule: at pixels where the c-GMM preserved a secondary slot
(M_sec > 0), we accept the pixel if EITHER the primary-direction
local-max check OR the secondary-direction local-max check passes.

Two axes of variation:
    neighborhood : N1 (radius=1), N2 (radius=2), N3 (radius=3)
    angular_fid  : 'A8' (8 bins), 'A16' (16 bins), 'Acont' (no binning)

Both nms_check calls use M_primary as the magnitude image being
thinned -- M_sec > 0 is just the corner indicator.

No hysteresis, no Gaussian smoothing, no v_fused=0 pixels considered.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------
# Bilinear sampling
# ---------------------------------------------------------------------

def bilinear_sample(M, x_coords, y_coords, default=0.0):
    """Vectorised bilinear sampling of array M at non-integer (x, y).

    Out-of-bounds samples get `default` (0.0).  The convention is that
    M[y, x] uses (row=y, col=x) indexing; x_coords and y_coords are
    in pixel-coordinate space (continuous), broadcasting compatible.
    """
    M = np.asarray(M, dtype=np.float64)
    H, W = M.shape
    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0

    in_bounds = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    v00 = M[y0c, x0c]
    v01 = M[y0c, x1c]
    v10 = M[y1c, x0c]
    v11 = M[y1c, x1c]

    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx * fy

    val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return np.where(in_bounds, val, default)


# ---------------------------------------------------------------------
# Per-orientation offset
# ---------------------------------------------------------------------

def gradient_offset(theta, angular_fidelity):
    """Return (offset_x, offset_y) unit-vector arrays along the
    gradient direction (perpendicular to edge orientation theta).

    angular_fidelity in {'A8', 'A16', 'Acont'}.  theta in radians on
    [0, pi).
    """
    grad = theta + math.pi / 2
    if angular_fidelity == "A8":
        bin_idx = np.round(grad / (math.pi / 8)).astype(np.int64) % 8
        ang = bin_idx * (math.pi / 8)
    elif angular_fidelity == "A16":
        bin_idx = np.round(grad / (math.pi / 16)).astype(np.int64) % 16
        ang = bin_idx * (math.pi / 16)
    elif angular_fidelity == "Acont":
        ang = grad
    else:
        raise ValueError(f"unknown angular_fidelity: {angular_fidelity!r}")
    return np.cos(ang), np.sin(ang)


# ---------------------------------------------------------------------
# Vectorised NMS check
# ---------------------------------------------------------------------

def nms_check_vec(M_image, x, y, theta_at_xy,
                  neighborhood, angular_fidelity):
    """For each (x[i], y[i]) with edge orientation theta_at_xy[i],
    return True if M_image[y, x] >= bilinear-sampled neighbour values
    at +- `neighborhood` units along the gradient direction.

    M_image  : (H, W) float — magnitude map being thinned
    x, y     : (P,) int     — center pixel coords
    theta_at_xy : (P,) float in [0, pi)
    neighborhood : int   — radial extent in pixels
    angular_fidelity : str
    """
    ox, oy = gradient_offset(theta_at_xy, angular_fidelity)
    n = float(neighborhood)
    M_plus  = bilinear_sample(M_image, x + n * ox, y + n * oy)
    M_minus = bilinear_sample(M_image, x - n * ox, y - n * oy)
    M_center = M_image[y, x]
    return (M_center >= M_plus) & (M_center >= M_minus)


# ---------------------------------------------------------------------
# Enhanced NMS (corner OR rule)
# ---------------------------------------------------------------------

def enhanced_nms(theta_primary, M_primary, theta_sec, M_sec, v_fused,
                 neighborhood=1, angular_fidelity="A8",
                 corner_method="or"):
    """Run two-orientation NMS on the c-GMM fused output.

    Pixels with v_fused = 0 are skipped (output zero).

    corner_method: how to handle pixels where M_sec > 0.
      'or'       - DEFAULT (per spec): keep if primary OR secondary
                   check passes.  Both checks compare against M_primary
                   neighbours.
      'bypass'   - at every M_sec > 0 pixel, force-keep (skip the NMS
                   check entirely).  Simplest corner recovery.
      'sec_mag'  - the secondary check uses M_sec as the magnitude
                   image being compared, instead of M_primary.  More
                   permissive on corners because M_sec is zero off the
                   corner-junction-mask and the secondary check
                   trivially passes.

    Output is M_primary where keep else 0.
    """
    if corner_method not in ("or", "bypass", "sec_mag"):
        raise ValueError(f"unknown corner_method: {corner_method!r}")
    M_primary = np.asarray(M_primary, dtype=np.float64)
    M_sec_arr = np.asarray(M_sec,     dtype=np.float64)
    out = np.zeros_like(M_primary)

    valid = (v_fused == 1) & np.isfinite(theta_primary) & (M_primary > 0)
    if not valid.any():
        return out
    ys, xs = np.where(valid)

    keep_primary = nms_check_vec(
        M_primary, xs, ys, theta_primary[ys, xs],
        neighborhood, angular_fidelity)

    sec_present = (
        (M_sec_arr[ys, xs] > 0)
        & np.isfinite(theta_sec[ys, xs])
    )
    keep = keep_primary.copy()

    if corner_method == "bypass":
        # Force-keep every flagged corner pixel regardless of any check.
        keep |= sec_present
    elif sec_present.any():
        idx = np.where(sec_present)[0]
        if corner_method == "or":
            keep_sec = nms_check_vec(
                M_primary, xs[idx], ys[idx],
                theta_sec[ys[idx], xs[idx]],
                neighborhood, angular_fidelity)
        else:  # 'sec_mag'
            keep_sec = nms_check_vec(
                M_sec_arr, xs[idx], ys[idx],
                theta_sec[ys[idx], xs[idx]],
                neighborhood, angular_fidelity)
        keep[idx] = keep[idx] | keep_sec

    out[ys[keep], xs[keep]] = M_primary[ys[keep], xs[keep]]
    return out


# ---------------------------------------------------------------------
# Hysteresis post-processing
# ---------------------------------------------------------------------

def hysteresis(nms_magnitudes, high_thresh, low_thresh):
    """Canny-style hysteresis: keep pixels above high_thresh as 'strong';
    keep pixels above low_thresh ('weak') only if they are connected
    (8-connectivity) to a strong pixel.

    nms_magnitudes : (H, W) float -- NMS output (magnitudes where kept).
    high_thresh, low_thresh : floats with low <= high.

    Returns a binary mask (H, W) of surviving edges.
    """
    from scipy import ndimage
    strong = nms_magnitudes >= high_thresh
    weak   = nms_magnitudes >= low_thresh
    # Label connected components in the weak mask; keep components that
    # contain at least one strong pixel.
    labels, _ = ndimage.label(weak, structure=np.ones((3, 3)))
    if labels.max() == 0:
        return strong.copy()
    keep_lbls = np.unique(labels[strong])
    keep_lbls = keep_lbls[keep_lbls > 0]
    return np.isin(labels, keep_lbls)


# ---------------------------------------------------------------------
# Standard NMS (no corner OR rule) -- diagnostic; used for validation 3
# ---------------------------------------------------------------------

def standard_nms(theta_primary, M_primary, v_fused,
                 neighborhood=1, angular_fidelity="A8"):
    M_primary = np.asarray(M_primary, dtype=np.float64)
    out = np.zeros_like(M_primary)
    valid = (v_fused == 1) & np.isfinite(theta_primary) & (M_primary > 0)
    if not valid.any():
        return out
    ys, xs = np.where(valid)
    keep = nms_check_vec(
        M_primary, xs, ys, theta_primary[ys, xs],
        neighborhood, angular_fidelity)
    out[ys[keep], xs[keep]] = M_primary[ys[keep], xs[keep]]
    return out
