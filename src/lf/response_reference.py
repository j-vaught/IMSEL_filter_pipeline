"""Numpy reference implementation of the LF response operator.

This is the EXACT operator the Metal implementation must match (within
1e-4 absolute tolerance).  It is the per-pixel-batch numpy code
currently used in `pipeline.synthetic_eval`, copied here
as the source of truth.

NOT performance-critical -- the goal of the Metal port is to be ~10-50x
faster while producing identical outputs.
"""

from __future__ import annotations

import numpy as np


def lf_response_at_pixels(g_x, g_y, px, py, theta, m):
    """Vectorised LF response over a batch of pixels.

    For each pixel (px[i], py[i]) and a single (theta, m), compute the
    Gaussian-weighted directional integral of the perpendicular gradient
    g_perp = -sin(theta) * g_x + cos(theta) * g_y along the line at
    orientation theta of half-length m.

    Inputs
    ------
    g_x, g_y : (H, W) float32 or float64 arrays
        Gradient field components produced by WVF.
    px, py : (P,) int arrays
        Center pixel coordinates (integer; px is column, py is row).
    theta : float
        Edge orientation in radians on [0, pi).  The line is along
        (cos theta, sin theta); the gradient direction is perpendicular.
    m : int
        Line half-width.  Total samples per pixel = 2*m + 1.

    Returns
    -------
    response : (P,) float64
        |sum_j w_j * g_perp_at(px + j*step*cos, py + j*step*sin)
         / sum_j w_j|
        Out-of-bounds samples contribute zero (numerator) and zero weight
        (denominator).  Pixels with all samples out of bounds yield 0.
    """
    H, W = g_x.shape
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    if m <= 0:
        gp = -sin_t * g_x[py, px] + cos_t * g_y[py, px]
        return np.abs(gp)
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0
    sigma = m / 2.0
    j_offsets = np.arange(-m, m + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)
    ix = np.round(j_offsets * step * cos_t).astype(np.int32)
    iy = np.round(j_offsets * step * sin_t).astype(np.int32)
    yy = py[:, None] + iy[None, :]
    xx = px[:, None] + ix[None, :]
    valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy_safe = np.clip(yy, 0, H - 1)
    xx_safe = np.clip(xx, 0, W - 1)
    g = -sin_t * g_x[yy_safe, xx_safe] + cos_t * g_y[yy_safe, xx_safe]
    g = g * valid
    w_eff = (weights[None, :] * valid)
    num = (g * weights[None, :]).sum(axis=1)
    den = w_eff.sum(axis=1)
    return np.abs(num / np.maximum(den, 1e-12))
