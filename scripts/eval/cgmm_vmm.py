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
    M_fused_sec > tau_M_rel * M_fused      (default tau_M_rel = 0.10)
    pi[k_sec] / pi[k_signal] > rho         (default rho = 0.40)
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
# Fusion
# ---------------------------------------------------------------------------

def vmm_fuse(phi, w, K=3, n_iters=30, init_kappa=4.0,
             hard_seed=False, hard_em=False,
             tau_M_rel=0.10, rho=0.40, select="pi"):
    """Fit weighted vM mixture per pixel and select primary/secondary.

    select : str
        "pi"        - signal = argmax_k pi_k (per spec).
        "pi_kappa"  - signal = argmax_k (pi_k * kappa_k); robust against
                       split-cluster instability at K>=3 by preferring
                       components that are both massive AND tight.

    Defaults:
        tau_M_rel = 0.10  (secondary M_fused must be >= 10% of primary's,
                           image-scale-invariant absolute floor)
        rho       = 0.40  (secondary mixing weight must be >= 40% of
                           primary's, relative mass floor)

    Output keys (length-P unless noted):
        theta_fused        float64 in [0, pi) or NaN  primary orientation
        M_fused            float64                    primary signal weight = W[k_signal]
        theta_fused_sec    float64 in [0, pi) or NaN  secondary (NaN if suppressed)
        M_fused_sec        float64                    secondary weight (0 if suppressed)
        v_fused            uint8   in {0, 1}          aggregate validity

    Diagnostic keys (full (P, K)):
        mu, kappa, pi, W
    """
    phi = np.asarray(phi, dtype=np.float64)
    w   = np.asarray(w,   dtype=np.float64)
    P, N = phi.shape

    # Degenerate-pixel guard: a K-component fit needs at least K
    # measurements with v_n = 1 AND nonzero magnitude.  If fewer than K
    # valid configs exist, the fit is statistically meaningless; skip
    # EM and emit v_fused = 0 with NaN angles.
    W_total  = w.sum(axis=1)
    n_active = (w > 1e-12).sum(axis=1)
    valid    = (W_total > 1e-12) & (n_active >= K)

    theta_fused     = np.full(P, np.nan, dtype=np.float64)
    M_fused         = np.zeros(P,         dtype=np.float64)
    theta_fused_sec = np.full(P, np.nan, dtype=np.float64)
    M_fused_sec     = np.zeros(P,         dtype=np.float64)
    v_fused         = valid.astype(np.uint8)
    mu_full         = np.full((P, K), np.nan)
    kappa_full      = np.full((P, K), np.nan)
    pi_full         = np.full((P, K), np.nan)
    W_full          = np.zeros((P, K))

    if not valid.any():
        return dict(theta_fused=theta_fused, M_fused=M_fused,
                    theta_fused_sec=theta_fused_sec,
                    M_fused_sec=M_fused_sec, v_fused=v_fused,
                    mu=mu_full, kappa=kappa_full, pi=pi_full, W=W_full)

    res = vmm_em(phi[valid], w[valid], K=K,
                 n_iters=n_iters, init_kappa=init_kappa,
                 hard_seed=hard_seed, hard_em=hard_em)
    Pv  = int(valid.sum())
    pi  = res.pi
    mu  = res.mu
    W   = res.W

    rng = np.arange(Pv)
    if select == "pi_kappa":
        score = pi * res.kappa
    else:
        score = pi
    k_signal = np.argmax(score, axis=1)
    if K > 1:
        score_msk = score.copy()
        score_msk[rng, k_signal] = -np.inf
        k_sec  = np.argmax(score_msk, axis=1)
    else:
        k_sec  = k_signal  # no secondary possible

    mu_signal = mu[rng, k_signal]
    mu_sec    = mu[rng, k_sec]
    pi_signal = pi[rng, k_signal]
    pi_sec    = pi[rng, k_sec]
    M_signal  = W[rng,  k_signal]
    M_sec_w   = W[rng,  k_sec]

    keep_sec = (
        (K > 1)
        & (M_sec_w > tau_M_rel * np.maximum(M_signal, 1e-30))
        & (pi_sec  > rho       * np.maximum(pi_signal, 1e-30))
    )

    theta_signal_half = (mu_signal % (2.0 * np.pi)) / 2.0
    theta_sec_half    = (mu_sec    % (2.0 * np.pi)) / 2.0

    theta_fused[valid]     = theta_signal_half
    M_fused[valid]         = M_signal
    theta_fused_sec[valid] = np.where(keep_sec, theta_sec_half, np.nan)
    M_fused_sec[valid]     = np.where(keep_sec, M_sec_w,        0.0)
    mu_full[valid]         = mu
    kappa_full[valid]      = res.kappa
    pi_full[valid]         = pi
    W_full[valid]          = W

    return dict(theta_fused=theta_fused, M_fused=M_fused,
                theta_fused_sec=theta_fused_sec,
                M_fused_sec=M_fused_sec, v_fused=v_fused,
                mu=mu_full, kappa=kappa_full, pi=pi_full, W=W_full)
