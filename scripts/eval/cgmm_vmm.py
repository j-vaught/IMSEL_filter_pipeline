"""Weighted von Mises mixture EM, vectorised across pixels.

Replaces the sklearn.GaussianMixture-on-plane fusion with a model
that lives natively on the circle, has 2 parameters per component,
and emits a principled concentration scalar kappa_k that downstream
rules (e.g., secondary suppression) can threshold against.

Per-pixel inputs (length-N) for one fusion pass:
    theta[n] in [0, pi)        orientation from configuration n
    M[n]     in [0, inf)       magnitude
    v[n]     in {0, 1}         validity flag

Pre-processing:
    phi[n] = (2 * theta[n]) mod (2*pi)          # double the angle
    w[n]   = v[n] * M[n]                         # validity-gated weight
    W_total = sum_n w[n]

If W_total < eps the pixel is degenerate (all configs invalid or zero
magnitude); skip EM and emit v_fused = 0.

EM: 30 fixed iterations, no early stopping (deterministic per-pixel
cost). Initialisation is circle-aware k-means++-flavoured:
    mu[1] = phi[argmax_n w[n]]
    mu[k] = phi[argmax_n w[n] * min_{j<k} circ_dist(phi[n], mu[j])]

Primary/secondary selection by mixing weight pi[k]; the third
component (and beyond) is residual and discarded. A secondary peak
is reported only if both
    M_fused_sec > tau_M_rel * M_fused                 (mass-ratio test)
    |theta_signal - theta_sec| > theta_min_deg        (geometric test)
hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ive


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

def circular_distance(a, b):
    """Smallest absolute distance on the circle. Broadcasts; returns
    values in [0, pi]."""
    d = (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(d)


def inv_A1_banerjee(R):
    """Banerjee 2005 closed-form inverse of A(kappa) = I_1/I_0 in d=2.
    Accuracy ~1e-3 across [0, 1).  R=0 -> kappa=0.  R->1 -> kappa->inf."""
    R = np.clip(np.asarray(R, dtype=np.float64), 0.0, 1.0 - 1e-6)
    R2 = R * R
    return (R * (2.0 - R2)) / (1.0 - R2)


def log_I0_safe(kappa):
    """log I_0(kappa) without overflow, via log I_0 = kappa + log(ive(0, kappa))
    where ive is the exponentially-scaled modified Bessel.  ive(0, 0) = 1."""
    kappa = np.maximum(np.asarray(kappa, dtype=np.float64), 0.0)
    return kappa + np.log(np.maximum(ive(0, kappa), 1e-300))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_centers(phi, w, K):
    """Circle-aware deterministic K-init.
        mu[0] = phi[argmax_n w[n]]
        mu[k] = phi[argmax_n w[n] * min_{j<k} circ_dist(phi[n], mu[j])]
    """
    phi = np.asarray(phi, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    P, N = phi.shape
    mu = np.zeros((P, K), dtype=np.float64)
    idx0 = np.argmax(w, axis=1)
    mu[:, 0] = phi[np.arange(P), idx0]
    for k in range(1, K):
        d = circular_distance(phi[:, :, None], mu[:, None, :k])  # (P, N, k)
        d_min = d.min(axis=2)                                     # (P, N)
        score = w * d_min
        # Degenerate fallback: all-zero score (e.g., single-config pixel)
        # produces argmax = 0, dropping mu[k] back onto mu[0]. EM then
        # gives that empty component near-zero pi; harmless.
        idxk = np.argmax(score, axis=1)
        mu[:, k] = phi[np.arange(P), idxk]
    return mu


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

@dataclass
class VMMResult:
    mu:    np.ndarray  # (P, K) means in [0, 2*pi)
    kappa: np.ndarray  # (P, K) concentrations
    pi:    np.ndarray  # (P, K) mixing weights, sum_k pi = 1
    W:     np.ndarray  # (P, K) total weight on each component
    gamma: np.ndarray  # (P, K, N) final responsibilities


def vmm_em(phi, w, K,
           n_iters=30, init_kappa=4.0, kappa_max=700.0, eps=1e-12,
           hard_seed=False, hard_em=False, record_log_lik=False):
    """Vectorised weighted vM mixture EM, fixed iterations.

    hard_seed: if True, at iter 0 use one-hot responsibilities based on
        nearest-mu (circular distance) rather than soft vM densities.
        Breaks the symmetry when init_kappa is too low to discriminate
        closely-spaced clusters via soft assignment. From iter 1 onward,
        normal soft EM continues.
    hard_em: if True, every iteration uses one-hot responsibilities
        (i.e., this becomes a weighted k-means on the circle, equivalent
        to vM mixture EM in the kappa->infinity limit).
    """
    phi = np.asarray(phi, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    P, N = phi.shape
    mu    = init_centers(phi, w, K)
    kappa = np.full((P, K), init_kappa, dtype=np.float64)
    pi    = np.full((P, K), 1.0 / K,    dtype=np.float64)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    log_2pi = np.log(2.0 * np.pi)

    log_lik_trace = []

    for it in range(n_iters):
        # E-step
        if hard_em or (hard_seed and it == 0):
            d = circular_distance(phi[:, None, :], mu[:, :, None])  # (P, K, N)
            hard_idx = np.argmin(d, axis=1)                          # (P, N)
            gamma = np.zeros((P, K, N), dtype=np.float64)
            pp = np.arange(P)[:, None]
            nn = np.arange(N)[None, :]
            gamma[pp, hard_idx, nn] = 1.0
        else:
            cos_diff = np.cos(phi[:, None, :] - mu[:, :, None])  # (P, K, N)
            log_norm = log_I0_safe(kappa) + log_2pi               # (P, K)
            log_pi   = np.log(np.maximum(pi, eps))                # (P, K)
            log_unnorm = (log_pi[:, :, None]
                          + kappa[:, :, None] * cos_diff
                          - log_norm[:, :, None])                  # (P, K, N)
            m_max = np.max(log_unnorm, axis=1, keepdims=True)
            e_un  = np.exp(log_unnorm - m_max)
            gamma = e_un / np.maximum(e_un.sum(axis=1, keepdims=True), eps)

        if record_log_lik:
            if hard_em or (hard_seed and it == 0):
                # Hard E-step: no soft log-marginals available; push NaN.
                log_lik_trace.append(np.full(P, np.nan))
            else:
                log_marg = m_max[:, 0, :] + np.log(
                    np.maximum(e_un.sum(axis=1), eps))
                ll = (w * log_marg).sum(axis=1)
                log_lik_trace.append(ll.copy())

        # M-step
        w_gamma = w[:, None, :] * gamma                           # (P, K, N)
        W       = w_gamma.sum(axis=2)                             # (P, K)
        W_total = W.sum(axis=1, keepdims=True)                    # (P, 1)
        pi      = W / np.maximum(W_total, eps)
        C       = (w_gamma * cos_phi[:, None, :]).sum(axis=2)
        S       = (w_gamma * sin_phi[:, None, :]).sum(axis=2)
        mu      = np.arctan2(S, C) % (2.0 * np.pi)
        R_bar   = np.sqrt(C * C + S * S) / np.maximum(W, eps)
        kappa   = np.minimum(inv_A1_banerjee(R_bar), kappa_max)

    out = VMMResult(mu=mu, kappa=kappa, pi=pi, W=W, gamma=gamma)
    if record_log_lik:
        return out, np.stack(log_lik_trace, axis=0)  # (n_iters, P)
    return out


def vmm_em_with_trace(phi, w, K, n_iters=30,
                      init_kappa=4.0, hard_seed=False, hard_em=False):
    """Same EM but records the full per-iteration trajectory.  Use only
    on a small set of pixels (memory cost is (n_iters+1) * P * K * N)."""
    phi = np.asarray(phi, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    P, N = phi.shape
    mu    = init_centers(phi, w, K)
    kappa = np.full((P, K), init_kappa)
    pi    = np.full((P, K), 1.0 / K)
    cos_phi = np.cos(phi); sin_phi = np.sin(phi)
    log_2pi = np.log(2.0 * np.pi); eps = 1e-12

    mu_tr    = [mu.copy()]
    kappa_tr = [kappa.copy()]
    pi_tr    = [pi.copy()]
    ll_tr    = []
    gamma_tr = []
    W_tr     = []

    for it in range(n_iters):
        if hard_em or (hard_seed and it == 0):
            d = circular_distance(phi[:, None, :], mu[:, :, None])
            hard_idx = np.argmin(d, axis=1)
            gamma = np.zeros((P, K, N), dtype=np.float64)
            pp = np.arange(P)[:, None]
            nn = np.arange(N)[None, :]
            gamma[pp, hard_idx, nn] = 1.0
            ll_tr.append(np.full(P, np.nan))
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
            log_marg = m_max[:, 0, :] + np.log(np.maximum(e_un.sum(axis=1), eps))
            ll = (w * log_marg).sum(axis=1)
            ll_tr.append(ll.copy())
        gamma_tr.append(gamma.copy())

        w_gamma = w[:, None, :] * gamma
        W       = w_gamma.sum(axis=2)
        W_tr.append(W.copy())
        W_total = W.sum(axis=1, keepdims=True)
        pi      = W / np.maximum(W_total, eps)
        C       = (w_gamma * cos_phi[:, None, :]).sum(axis=2)
        S       = (w_gamma * sin_phi[:, None, :]).sum(axis=2)
        mu      = np.arctan2(S, C) % (2.0 * np.pi)
        R_bar   = np.sqrt(C * C + S * S) / np.maximum(W, eps)
        kappa   = np.minimum(inv_A1_banerjee(R_bar), 700.0)
        mu_tr.append(mu.copy())
        kappa_tr.append(kappa.copy())
        pi_tr.append(pi.copy())

    return dict(
        mu_trace      = np.stack(mu_tr,    axis=0),  # (n_iters+1, P, K)
        kappa_trace   = np.stack(kappa_tr, axis=0),
        pi_trace      = np.stack(pi_tr,    axis=0),
        gamma_trace   = np.stack(gamma_tr, axis=0),  # (n_iters, P, K, N)
        W_trace       = np.stack(W_tr,     axis=0),
        log_lik_trace = np.stack(ll_tr,    axis=0),  # (n_iters, P)
    )


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def theta_M_to_phi_w(theta_deg, M, v=None):
    """(theta in [0, 180), M >= 0, v in {0, 1}) -> (phi in [0, 2pi),
    w = v * M).  NaN-safe.  If v is None it is derived from finiteness
    and M > 0."""
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    M         = np.asarray(M,         dtype=np.float64)
    finite_th = np.isfinite(theta_deg)
    finite_M  = np.isfinite(M)
    if v is None:
        v_arr = (finite_th & finite_M & (M > 0.0)).astype(np.float64)
    else:
        v_arr = np.asarray(v, dtype=np.float64)
    theta_safe = np.where(finite_th, theta_deg, 0.0)
    phi = (2.0 * np.deg2rad(theta_safe)) % (2.0 * np.pi)
    M_safe = np.where(finite_M, np.maximum(M, 0.0), 0.0)
    w = v_arr * M_safe
    return phi, w, v_arr


# ---------------------------------------------------------------------------
# Two-pass fusion (production)
# ---------------------------------------------------------------------------

def vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, n_iters=30,
                      init_kappa=4.0, hard_seed=False, hard_em=True,
                      tau_M_rel=0.05, theta_min_deg=10.0):
    """Two-pass weighted vM mixture fusion per pixel.

    Runs K=3 hard-EM vMM independently on the primary measurement set
    (phi_p, w_p) and on the secondary measurement set (phi_s, w_s).
    The two fits do NOT share state, initialization, or components.

    Primary slot  = argmax-pi component of the PRIMARY fit.
    Secondary slot = argmax-pi component of the SECONDARY fit.

    Suppression rule (BOTH must hold, else theta_sec=NaN, M_sec=0):
      M_sec / M_primary > tau_M_rel    (absolute weights from each fit)
      |theta_primary - theta_sec| > theta_min_deg     (geometric)

    Per-pass validity (degenerate-pixel guard):
      primary_valid   := (sum(w_p) > 1e-12) AND (n_active(w_p) >= K)
      secondary_valid := primary_valid AND (sum(w_s) > 1e-12) AND (n_active(w_s) >= K)
      v_fused         := primary_valid

    Inputs:
      phi_p, phi_s : (P, N) doubled angles in [0, 2*pi)
      w_p,   w_s   : (P, N) non-negative weights w_n = v_n * M_n

    Output dict (length-P unless noted):
      theta_primary    float64 [0, pi) or NaN
      M_primary        float64                primary fit's W[k_primary]
      theta_sec        float64 [0, pi) or NaN  (NaN if suppressed/invalid)
      M_sec            float64                 (0 if suppressed/invalid)
      v_fused          uint8 {0, 1}
      primary_pi, primary_mu, primary_kappa        (P, K) diagnostics
      secondary_pi, secondary_mu, secondary_kappa  (P, K) diagnostics
      keep_secondary_mask    uint8 {0, 1}  pre-NaN-substitution suppression flag
    """
    phi_p = np.asarray(phi_p, dtype=np.float64)
    w_p   = np.asarray(w_p,   dtype=np.float64)
    phi_s = np.asarray(phi_s, dtype=np.float64)
    w_s   = np.asarray(w_s,   dtype=np.float64)
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
    theta_primary = np.full(P, np.nan, dtype=np.float64)
    M_primary     = np.zeros(P,        dtype=np.float64)
    theta_sec     = np.full(P, np.nan, dtype=np.float64)
    M_sec         = np.zeros(P,        dtype=np.float64)
    v_fused       = primary_valid.astype(np.uint8)

    primary_pi    = np.full((P, K), np.nan)
    primary_mu    = np.full((P, K), np.nan)
    primary_kappa = np.full((P, K), np.nan)
    secondary_pi    = np.full((P, K), np.nan)
    secondary_mu    = np.full((P, K), np.nan)
    secondary_kappa = np.full((P, K), np.nan)
    keep_secondary_mask = np.zeros(P, dtype=np.uint8)

    # phi-space mu of the primary signal (kept for the geometric test).
    mu_kp_phi = np.full(P, np.nan, dtype=np.float64)

    # ---- primary pass ----
    if primary_valid.any():
        res_p = vmm_em(phi_p[primary_valid], w_p[primary_valid], K=K,
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

    # ---- secondary pass (only on pixels where BOTH passes are valid) ----
    if secondary_valid.any():
        res_s = vmm_em(phi_s[secondary_valid], w_s[secondary_valid], K=K,
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

        # ---- suppression rule: ABSOLUTE mass ratio + geometric separation
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
