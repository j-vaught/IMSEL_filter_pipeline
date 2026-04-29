"""Spline orientation recovery (§6.3) for the LF response curves.

Given a per-pixel response array of shape (N, K) where each row is the
LF response sampled at K orientations on [0, pi), this module returns
the per-row primary and secondary peaks via periodic-cubic-spline
interpolation with a two-stage sentinel that suppresses weak / absent
secondary peaks.

This stage sits UPSTREAM of fusion (cgmm_vmm.py).  The threshold
tau_sec_floor lives here, not in the fusion stage, because the
secondary stream the fusion consumes is whatever this stage emits.

Sentinel logic:

  primary_idx = argmax over local maxima  (always exists, M_hat = peak value)

  secondary_idx = argmax over local maxima at distance > min_sep_frac * dense_n
                  from primary_idx. Dense-grid local maxima include sample-knot
                  local maxima.
  if no such local max exists:
      M_sec = 0;  theta_sec = NaN
  else:
      candidate = (theta_sec_idx, dy[theta_sec_idx])
      if dy[theta_sec_idx] / dy[primary_idx] < tau_sec_floor:
          M_sec = 0;  theta_sec = NaN
      else:
          theta_sec = candidate angle
          M_sec     = candidate response value

  M_hat and M_sec are clamped to max_k y_k after the suppression decision.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline


def find_two_peaks(angles_rad, response_2d,
                   tau_sec_floor=0.40,
                   dense_n=2000,
                   min_sep_frac=0.125):
    """Vectorised spline peak finder over rows of response_2d.

    Returns
    -------
    theta_hat : (N,) float64        primary peak location, rad in [0, pi)
    M_hat     : (N,) float64        primary peak response value
    theta_sec : (N,) float64        secondary peak location, rad or NaN
    M_sec     : (N,) float64        secondary peak response, 0 if absent

    The secondary slot is suppressed (M_sec=0, theta_sec=NaN) when EITHER:
      (a) no non-adjacent local maximum exists on the dense spline grid
      (b) M_sec_candidate / M_hat < tau_sec_floor
    """
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    response_2d = np.asarray(response_2d, dtype=np.float64)
    N, K = response_2d.shape

    x = np.concatenate([angles_rad, [math.pi]])
    y = np.concatenate([response_2d, response_2d[:, :1]], axis=1)
    cs = CubicSpline(x, y, axis=1, bc_type="periodic")
    dense_a = np.linspace(0.0, math.pi, dense_n, endpoint=False)
    dy = cs(dense_a)                       # (N, dense_n)

    # Local maxima (periodic comparison via np.roll).
    left  = np.roll(dy, 1, axis=1)
    right = np.roll(dy, -1, axis=1)
    is_peak = (dy >= left) & (dy >= right)
    masked = np.where(is_peak, dy, -np.inf)

    primary_idx = np.argmax(masked, axis=1)
    th_hat = dense_a[primary_idx]
    M_hat  = dy[np.arange(N), primary_idx]

    # Secondary candidate: largest local maximum at periodic distance
    # > min_sep_frac * dense_n from primary. Dense-grid local maxima already
    # include sample-knot maxima for this reference path.
    sep = max(1, int(min_sep_frac * dense_n))
    grid = np.arange(dense_n)
    d = np.abs(grid[None, :] - primary_idx[:, None])
    d = np.minimum(d, dense_n - d)

    masked2 = np.where((d > sep) & is_peak, dy, -np.inf)
    sec_idx = np.argmax(masked2, axis=1)
    sec_max_val = masked2[np.arange(N), sec_idx]    # -inf if no local max
    has_local_max = sec_max_val > -np.inf
    M_sec_raw = dy[np.arange(N), sec_idx]

    # Magnitude floor (tau_sec_floor).
    ratio = M_sec_raw / np.maximum(M_hat, 1e-30)
    weak  = ratio < tau_sec_floor

    suppress = ~has_local_max | weak
    th_sec = np.where(suppress, np.nan, dense_a[sec_idx])
    M_sec  = np.where(suppress, 0.0,    M_sec_raw)

    # Clamp emitted magnitudes while leaving the suppression ratio above on
    # the raw spline values.
    y_max = response_2d.max(axis=1)
    M_hat = np.minimum(M_hat, y_max)
    M_sec = np.minimum(M_sec, y_max)
    return th_hat, M_hat, th_sec, M_sec
