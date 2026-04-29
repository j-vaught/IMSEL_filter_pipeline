"""Source-of-truth Python implementation of c-GMM K=3 fusion.

DO NOT MODIFY. The Metal port must reproduce these outputs.

Operator: cgmm_fuse_two_pass
    inputs:
        phi_p, w_p   (P, N) float64    primary doubled-angle measurements + weights
        phi_s, w_s   (P, N) float64    secondary doubled-angle measurements + weights
        K            int               mixture component count (production: 3)
        n_iters      int               fixed EM iterations (production: 30)
        init_kappa   float             initial concentration scalar (production: 4.0)
        hard_em      bool              hard E-step (production: True)
        tau_M_rel    float              relative mass floor (production: 0.05)
        theta_min_deg float              geometric separation floor (production: 10.0)

    outputs (length P):
        theta_primary   float64 [0, pi) or NaN
        M_primary       float64
        theta_sec       float64 [0, pi) or NaN
        M_sec           float64 (0 if suppressed)
        v_fused         uint8 {0, 1}
        primary_pi      (P, K) float64
        primary_mu      (P, K) float64
        primary_kappa   (P, K) float64
        secondary_pi    (P, K) float64
        secondary_mu    (P, K) float64
        secondary_kappa (P, K) float64
        keep_secondary_mask  (P,) uint8

Inputs come from the upstream orientation-recovery stage.  Each pixel
has N independent (theta, M, v) measurements (one per
(channel, radius, degree, lf_half_length) configuration).  The
pre-processing function `theta_M_to_phi_w` converts a (theta_deg, M, v)
triple into the (phi, w) form the EM consumes:

    phi = (2 * theta_rad) mod (2*pi)
    w   = v * M

The "double the angle" trick maps the pi-periodic orientation to a
2*pi-periodic angle, which makes the per-pixel mixture fit into a
plain weighted circular EM.

Algorithm (production: K=3, hard_em=True, n_iters=30)
-----------------------------------------------------

PER-PIXEL VALIDITY (degenerate guard).
    primary_valid   := (sum(w_p) > 1e-12) AND (count(w_p > 1e-12) >= K)
    secondary_valid := primary_valid AND (sum(w_s) > 1e-12)
                       AND (count(w_s > 1e-12) >= K)
    v_fused         := primary_valid

INITIALIZATION (deterministic K-init on the circle).
    mu[0] = phi[argmax_n w[n]]
    mu[k] = phi[argmax_n  w[n] * min_{j<k} circ_dist(phi[n], mu[j])]
    pi[k] = 1/K   for all k
    kappa[k] = init_kappa

EM (n_iters fixed, hard E-step).
    E-step (hard):
        d[k, n] = circ_dist(phi[n], mu[k])
        gamma[k, n] = 1 if k == argmin_k d[k, n] else 0

    M-step:
        W[k]  = sum_n w[n] * gamma[k, n]
        C[k]  = sum_n w[n] * gamma[k, n] * cos(phi[n])
        S[k]  = sum_n w[n] * gamma[k, n] * sin(phi[n])
        mu[k] = atan2(S[k], C[k]) mod 2*pi
        pi[k] = W[k] / sum_k W[k]
        R[k]  = sqrt(C[k]^2 + S[k]^2) / max(W[k], eps)
        kappa[k] = clip(inv_A1_banerjee(R[k]), 0, 700)

    inv_A1_banerjee(R) = R * (2 - R^2) / (1 - R^2)   for R in [0, 1-1e-6]

    circ_dist(a, b) = abs(((a - b + pi) mod 2*pi) - pi)

    eps = 1e-12

PRIMARY/SECONDARY SELECTION.
    Run independent EM on (phi_p, w_p) -> primary fit.
    Run independent EM on (phi_s, w_s) -> secondary fit.
    Both passes use K=3 hard EM, fresh init, no shared state.

    primary slot   : k_p = argmax_k pi_p[k]
        theta_primary = (mu_p[k_p] mod 2*pi) / 2
        M_primary     = W_p[k_p]

    secondary slot : k_s = argmax_k pi_s[k]
        theta_sec_candidate = (mu_s[k_s] mod 2*pi) / 2
        M_sec_candidate     = W_s[k_s]

SUPPRESSION RULE (BOTH must hold else suppress).
    mass_ok  : M_sec_candidate / max(M_primary, 1e-30) > tau_M_rel
    sep_ok   : circ_dist(mu_p[k_p], mu_s[k_s]) / 2  >  theta_min_deg (in deg)
                  i.e., the geometric distance between primary and
                  secondary orientations exceeds theta_min_deg.

    If suppressed:
        theta_sec = NaN
        M_sec     = 0

INVALID PIXELS.
    primary_valid == False  ->  theta_primary = NaN, M_primary = 0,
                                theta_sec = NaN, M_sec = 0, v_fused = 0
    secondary_valid == False ->  theta_sec = NaN, M_sec = 0
                                  (primary still emitted if primary_valid)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ive


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

def circular_distance(a, b):
    """Smallest absolute distance on the circle.
    Returns values in [0, pi]."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.where(d > np.float32(np.pi), np.float32(2.0 * np.pi) - d, d)


def inv_A1_banerjee(R):
    """Banerjee 2005 closed-form inverse of A(kappa) = I_1 / I_0 in d=2.
    R in [0, 1-1e-6); R=0 -> kappa=0; R->1 -> kappa->inf."""
    R = np.clip(np.asarray(R, dtype=np.float32), 0.0, 1.0 - 1e-6)
    R2 = R * R
    return (R * (2.0 - R2)) / (1.0 - R2)


def log_I0_safe(kappa):
    """log I_0(kappa) without overflow.
    Used only on the soft-EM path; production uses hard EM."""
    kappa = np.maximum(np.asarray(kappa, dtype=np.float32), 0.0)
    return kappa + np.log(np.maximum(ive(0, kappa), 1e-300))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_centers(phi, w, K):
    """Deterministic K-init on the circle.
        mu[0] = phi[argmax_n w[n]]
        mu[k] = phi[argmax_n w[n] * min_{j<k} circ_dist(phi[n], mu[j])]
    """
    phi = np.asarray(phi, dtype=np.float32)
    w   = np.asarray(w,   dtype=np.float32)
    P, N = phi.shape
    mu = np.zeros((P, K), dtype=np.float32)
    idx0 = np.argmax(w, axis=1)
    mu[:, 0] = phi[np.arange(P), idx0]
    for k in range(1, K):
        d = circular_distance(phi[:, :, None], mu[:, None, :k])  # (P, N, k)
        d_min = d.min(axis=2)
        score = w * d_min
        idxk = np.argmax(score, axis=1)
        mu[:, k] = phi[np.arange(P), idxk]
    return mu


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

@dataclass
class CGMMResult:
    mu:    np.ndarray  # (P, K) means in [0, 2*pi)
    kappa: np.ndarray  # (P, K) concentrations
    pi:    np.ndarray  # (P, K) mixing weights, sum_k pi = 1
    W:     np.ndarray  # (P, K) total weight on each component
    gamma: np.ndarray  # (P, K, N) final responsibilities


def cgmm_em(phi, w, K,
            n_iters=30, init_kappa=4.0, kappa_max=700.0, eps=1e-12,
            hard_seed=False, hard_em=False):
    """Vectorized weighted circular mixture EM, fixed iterations.

    Production setting: hard_em=True, n_iters=30.
    """
    phi = np.asarray(phi, dtype=np.float32)
    w   = np.asarray(w,   dtype=np.float32)
    P, N = phi.shape
    mu    = init_centers(phi, w, K)
    kappa = np.full((P, K), init_kappa, dtype=np.float32)
    pi    = np.full((P, K), 1.0 / K,    dtype=np.float32)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    log_2pi = np.log(2.0 * np.pi)

    for it in range(n_iters):
        # E-step
        if hard_em or (hard_seed and it == 0):
            d = circular_distance(phi[:, None, :], mu[:, :, None])  # (P, K, N)
            hard_idx = np.argmin(d, axis=1)                           # (P, N)
            gamma = np.zeros((P, K, N), dtype=np.float32)
            pp = np.arange(P)[:, None]
            nn = np.arange(N)[None, :]
            gamma[pp, hard_idx, nn] = 1.0
        else:
            cos_diff = np.cos(phi[:, None, :] - mu[:, :, None])
            log_norm = log_I0_safe(kappa) + log_2pi
            log_pi   = np.log(np.maximum(pi, eps))
            log_unnorm = (log_pi[:, :, None]
                          + kappa[:, :, None] * cos_diff
                          - log_norm[:, :, None])
            m_max = np.max(log_unnorm, axis=1, keepdims=True)
            e_un  = np.exp(log_unnorm - m_max)
            gamma = e_un / np.maximum(e_un.sum(axis=1, keepdims=True), eps)

        # M-step
        w_gamma = w[:, None, :] * gamma
        W       = w_gamma.sum(axis=2)
        W_total = W.sum(axis=1, keepdims=True)
        pi      = W / np.maximum(W_total, eps)
        C       = (w_gamma * cos_phi[:, None, :]).sum(axis=2)
        S       = (w_gamma * sin_phi[:, None, :]).sum(axis=2)
        mu      = np.arctan2(S, C) % (2.0 * np.pi)
        R_bar   = np.sqrt(C * C + S * S) / np.maximum(W, eps)
        kappa   = np.minimum(inv_A1_banerjee(R_bar), kappa_max)

    return CGMMResult(mu=mu, kappa=kappa, pi=pi, W=W, gamma=gamma)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def theta_M_to_phi_w(theta_deg, M, v=None):
    """(theta in [0, 180), M >= 0, v in {0, 1}) -> (phi in [0, 2*pi),
    w = v * M).  NaN-safe.  If v is None it is derived from finiteness
    and M > 0."""
    theta_deg = np.asarray(theta_deg, dtype=np.float32)
    M         = np.asarray(M,         dtype=np.float32)
    finite_th = np.isfinite(theta_deg)
    finite_M  = np.isfinite(M)
    if v is None:
        v_arr = (finite_th & finite_M & (M > 0.0)).astype(np.float32)
    else:
        v_arr = np.asarray(v, dtype=np.float32)
    theta_safe = np.where(finite_th, theta_deg, 0.0)
    phi = (2.0 * np.deg2rad(theta_safe)) % (2.0 * np.pi)
    M_safe = np.where(finite_M, np.maximum(M, 0.0), 0.0)
    w = v_arr * M_safe
    return phi, w, v_arr


# ---------------------------------------------------------------------------
# Two-pass c-GMM fusion (production)
# ---------------------------------------------------------------------------

def cgmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, n_iters=30,
                       init_kappa=4.0, hard_seed=False, hard_em=True,
                       tau_M_rel=0.05, theta_min_deg=10.0):
    """Two-pass c-GMM K=3 fusion per pixel.

    Independent K=3 hard-EM fits on the primary stream (phi_p, w_p) and
    on the secondary stream (phi_s, w_s).  Primary slot = highest pi
    component of the primary fit.  Secondary slot = highest pi component
    of the secondary fit.  Suppression rule applied at the end.
    """
    phi_p = np.asarray(phi_p, dtype=np.float32)
    w_p   = np.asarray(w_p,   dtype=np.float32)
    phi_s = np.asarray(phi_s, dtype=np.float32)
    w_s   = np.asarray(w_s,   dtype=np.float32)
    P, N = phi_p.shape
    assert phi_s.shape == (P, N) and w_p.shape == (P, N) and w_s.shape == (P, N)

    # ---- per-pass validity ----
    W_total_p  = w_p.sum(axis=1)
    n_active_p = (w_p > 1e-12).sum(axis=1)
    primary_valid = (W_total_p > 1e-12) & (n_active_p >= K)

    W_total_s  = w_s.sum(axis=1)
    n_active_s = (w_s > 1e-12).sum(axis=1)
    secondary_valid = (
        primary_valid
        & (W_total_s > 1e-12)
        & (n_active_s >= K)
    )

    # ---- output buffers ----
    theta_primary = np.full(P, np.nan, dtype=np.float32)
    M_primary     = np.zeros(P,        dtype=np.float32)
    theta_sec     = np.full(P, np.nan, dtype=np.float32)
    M_sec         = np.zeros(P,        dtype=np.float32)
    v_fused       = primary_valid.astype(np.uint8)

    primary_pi    = np.full((P, K), np.nan, dtype=np.float32)
    primary_mu    = np.full((P, K), np.nan, dtype=np.float32)
    primary_kappa = np.full((P, K), np.nan, dtype=np.float32)
    secondary_pi    = np.full((P, K), np.nan, dtype=np.float32)
    secondary_mu    = np.full((P, K), np.nan, dtype=np.float32)
    secondary_kappa = np.full((P, K), np.nan, dtype=np.float32)
    keep_secondary_mask = np.zeros(P, dtype=np.uint8)

    mu_kp_phi = np.full(P, np.nan, dtype=np.float32)

    # ---- primary pass ----
    if primary_valid.any():
        res_p = cgmm_em(phi_p[primary_valid], w_p[primary_valid], K=K,
                        n_iters=n_iters, init_kappa=init_kappa,
                        hard_seed=hard_seed, hard_em=hard_em)
        Pv  = int(primary_valid.sum())
        rng = np.arange(Pv)
        k_p     = np.argmax(res_p.pi, axis=1)
        mu_kp_v = res_p.mu[rng, k_p]
        W_kp    = res_p.W[rng,  k_p]

        theta_primary[primary_valid] = (mu_kp_v % (2.0 * np.pi)) / 2.0
        M_primary[primary_valid]     = W_kp
        mu_kp_phi[primary_valid]     = mu_kp_v
        primary_pi[primary_valid]    = res_p.pi
        primary_mu[primary_valid]    = res_p.mu
        primary_kappa[primary_valid] = res_p.kappa

    # ---- secondary pass ----
    if secondary_valid.any():
        res_s = cgmm_em(phi_s[secondary_valid], w_s[secondary_valid], K=K,
                        n_iters=n_iters, init_kappa=init_kappa,
                        hard_seed=hard_seed, hard_em=hard_em)
        Ps  = int(secondary_valid.sum())
        rng = np.arange(Ps)
        k_s     = np.argmax(res_s.pi, axis=1)
        mu_ks_v = res_s.mu[rng, k_s]
        W_ks    = res_s.W[rng,  k_s]

        secondary_pi[secondary_valid]    = res_s.pi
        secondary_mu[secondary_valid]    = res_s.mu
        secondary_kappa[secondary_valid] = res_s.kappa

        M_p_at_sec    = M_primary[secondary_valid]
        mu_kp_at_sec  = mu_kp_phi[secondary_valid]

        mass_ratio = W_ks / np.maximum(M_p_at_sec, 1e-30)
        mass_ok    = mass_ratio > tau_M_rel

        sep_phi       = circular_distance(mu_kp_at_sec, mu_ks_v)
        sep_theta_deg = np.degrees(sep_phi) / 2.0
        sep_ok        = sep_theta_deg > theta_min_deg

        keep = mass_ok & sep_ok
        theta_s_v = (mu_ks_v % (2.0 * np.pi)) / 2.0
        theta_sec[secondary_valid] = np.where(keep, theta_s_v, np.nan)
        M_sec[secondary_valid]     = np.where(keep, W_ks,      0.0)
        keep_secondary_mask[secondary_valid] = keep.astype(np.uint8)

    return dict(
        theta_primary       = theta_primary,
        M_primary           = M_primary,
        theta_sec           = theta_sec,
        M_sec               = M_sec,
        v_fused             = v_fused,
        primary_pi          = primary_pi,
        primary_mu          = primary_mu,
        primary_kappa       = primary_kappa,
        secondary_pi        = secondary_pi,
        secondary_mu        = secondary_mu,
        secondary_kappa     = secondary_kappa,
        keep_secondary_mask = keep_secondary_mask,
    )
