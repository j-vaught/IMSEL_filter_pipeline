"""Source-of-truth Python implementation of orientation recovery.

DO NOT MODIFY. This is the algorithm the Metal port must reproduce.

Operator: find_two_peaks
    input  : angles_rad   (K,) float64 monotonically increasing in [0, pi)
    input  : response_2d  (N, K) float64 LF responses at K equally
                          spaced orientations on [0, pi), one row per pixel
    output : theta_p   (N,) float64    primary peak location, rad in [0, pi)
             M_p       (N,) float64    primary peak response value
             theta_s   (N,) float64    secondary peak location, rad or NaN
             M_s       (N,) float64    secondary peak response, 0 if absent
             v         (N,) uint8      per-pixel validity flag (range rule)

Per-pixel quintuple (theta, M, theta_sec, M_sec, v) matches the
section-6 specification of the paper.

Algorithm
---------
1. Periodic cubic spline through the K samples of each row, wrap point
   y[K] := y[0] appended.
2. Evaluate the spline on a dense grid of dense_n equally spaced angles
   in [0, pi).
3. Mark dense-grid local maxima (>= both periodic neighbours). Sample-knot
   local maxima are included by this dense-grid test.
4. Primary = argmax over cubic-spline local maxima.
5. Fit an order-4 doubled-angle trig model to the K samples and evaluate
   it on the same dense grid.
6. Secondary candidate = argmax over trig-fit local maxima at periodic
   dense index distance > min_sep_frac * dense_n from the cubic primary.
7. Suppress secondary (M_s = 0, theta_s = NaN) if there is no such
   candidate, or if M_sec_candidate / M_p < tau_sec_floor.
8. Clamp emitted primary and secondary magnitudes to max_k y_k after
   the suppression decision.
9. Validity flag v (per-pixel range rule, eq. validity-range):
       R       = max_k y_k - min_k y_k
       R_ref   = max over rows of R     (image-wide reference)
       v       = 1 if R > tau * R_ref else 0
   tau in [0, 1] controls aggressiveness; default tau = 0.10.

The cubic dense-grid primary is the Python reference path. The Metal
primary may use closed-form per-segment quadratic roots as long as the
acceptance test passes. The secondary slot is trig-derived in both
reference and Metal implementations.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline


def _trig_fit_dense(angles_rad, response_2d, dense_a, order=4):
    cols = [np.ones_like(angles_rad)]
    for n in range(1, order + 1):
        cols.append(np.cos(2.0 * n * angles_rad))
        cols.append(np.sin(2.0 * n * angles_rad))
    design = np.column_stack(cols)
    coef = response_2d @ np.linalg.pinv(design).T

    dense_cols = [np.ones_like(dense_a)]
    for n in range(1, order + 1):
        dense_cols.append(np.cos(2.0 * n * dense_a))
        dense_cols.append(np.sin(2.0 * n * dense_a))
    return coef @ np.column_stack(dense_cols).T


def find_two_peaks(angles_rad, response_2d,
                   tau_sec_floor=0.40,
                   tau_validity=0.10,
                   dense_n=500,
                   min_sep_frac=0.125):
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    response_2d = np.asarray(response_2d, dtype=np.float64)
    N, K = response_2d.shape

    # ---- spline densification ----
    x = np.concatenate([angles_rad, [math.pi]])
    y = np.concatenate([response_2d, response_2d[:, :1]], axis=1)
    cs = CubicSpline(x, y, axis=1, bc_type="periodic")
    dense_a = np.linspace(0.0, math.pi, dense_n, endpoint=False)
    dy = cs(dense_a)                       # (N, dense_n)

    # ---- local maxima on the dense grid ----
    left  = np.roll(dy, 1, axis=1)
    right = np.roll(dy, -1, axis=1)
    is_peak = (dy >= left) & (dy >= right)
    masked = np.where(is_peak, dy, -np.inf)

    # ---- primary peak ----
    primary_idx = np.argmax(masked, axis=1)
    th_hat = dense_a[primary_idx]
    M_hat  = dy[np.arange(N), primary_idx]

    # ---- secondary peak from order-4 trig fit ----
    sep = max(1, int(min_sep_frac * dense_n))
    grid = np.arange(dense_n)
    d = np.abs(grid[None, :] - primary_idx[:, None])
    d = np.minimum(d, dense_n - d)

    trig_y = _trig_fit_dense(angles_rad, response_2d, dense_a, order=4)
    trig_left = np.roll(trig_y, 1, axis=1)
    trig_right = np.roll(trig_y, -1, axis=1)
    trig_is_peak = (trig_y >= trig_left) & (trig_y >= trig_right)
    masked2 = np.where((d > sep) & trig_is_peak, trig_y, -np.inf)
    sec_idx = np.argmax(masked2, axis=1)
    sec_max_val = masked2[np.arange(N), sec_idx]
    has_local_max = sec_max_val > -np.inf
    M_sec_raw = trig_y[np.arange(N), sec_idx]

    ratio = M_sec_raw / np.maximum(M_hat, 1e-30)
    weak  = ratio < tau_sec_floor

    suppress = ~has_local_max | weak
    th_sec = np.where(suppress, np.nan, dense_a[sec_idx])
    M_sec  = np.where(suppress, 0.0,    M_sec_raw)

    # Clamp emitted magnitudes so cubic overshoot cannot inflate downstream
    # fusion weights. The suppression ratio above uses the unclamped values.
    y_max = response_2d.max(axis=1)
    M_hat = np.minimum(M_hat, y_max)
    M_sec = np.minimum(M_sec, y_max)

    # ---- per-pixel validity flag (range rule) ----
    R     = response_2d.max(axis=1) - response_2d.min(axis=1)
    R_ref = float(R.max()) if R.size else 0.0
    v = (R > tau_validity * R_ref).astype(np.uint8)
    return th_hat, M_hat, th_sec, M_sec, v
