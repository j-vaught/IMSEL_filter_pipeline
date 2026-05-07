
inline float cgmm_nan() {
    return as_type<float>(0x7fc00000u);
}

inline float cgmm_wrap_2pi(float value) {
    float out = fmod(value, CGMM_TWO_PI);
    if (out < 0.0f) {
        out += CGMM_TWO_PI;
    }
    return out;
}

inline float cgmm_circ_dist(float a, float b) {
    float d = fabs(a - b);
    if (d > CGMM_PI) {
        d = CGMM_TWO_PI - d;
    }
    return d;
}

inline float cgmm_inv_a1_banerjee(float r) {
    const float rc = clamp(r, 0.0f, 1.0f - 1.0e-6f);
    const float r2 = rc * rc;
    return rc * (2.0f - r2) / (1.0f - r2);
}

inline uint cgmm_argmax3(thread const float* values) {
    uint best_idx = 0;
    float best_value = values[0];
    if (values[1] > best_value) {
        best_value = values[1];
        best_idx = 1;
    }
    if (values[2] > best_value) {
        best_idx = 2;
    }
    return best_idx;
}

inline void cgmm_init_centers_k3(
    device const float* w,
    ulong row_offset,
    uint N,
    thread const float* phi_values,
    thread float* mu
) {
    uint best_idx = 0;
    float best_weight = w[row_offset];
    for (uint n = 1; n < N; ++n) {
        const float value = w[row_offset + ulong(n)];
        if (value > best_weight) {
            best_weight = value;
            best_idx = n;
        }
    }
    mu[0] = phi_values[best_idx];

    for (uint k = 1; k < CGMM_K; ++k) {
        best_idx = 0;
        float best_score = w[row_offset] * cgmm_circ_dist(phi_values[0], mu[0]);
        if (k == 2) {
            best_score = w[row_offset] *
                min(cgmm_circ_dist(phi_values[0], mu[0]), cgmm_circ_dist(phi_values[0], mu[1]));
        }
        for (uint n = 1; n < N; ++n) {
            float d_min = cgmm_circ_dist(phi_values[n], mu[0]);
            if (k == 2) {
                d_min = min(d_min, cgmm_circ_dist(phi_values[n], mu[1]));
            }
            const float score = w[row_offset + ulong(n)] * d_min;
            if (score > best_score) {
                best_score = score;
                best_idx = n;
            }
        }
        mu[k] = phi_values[best_idx];
    }
}

inline void cgmm_fit_k3(
    device const float* phi,
    device const float* w,
    ulong row_offset,
    uint N,
    uint n_iters,
    float init_kappa,
    thread float* out_pi,
    thread float* out_mu,
    thread float* out_kappa,
    thread float* out_W
) {
    float phi_values[CGMM_N_MAX];
    float cos_values[CGMM_N_MAX];
    float sin_values[CGMM_N_MAX];

    for (uint n = 0; n < N; ++n) {
        const float value = phi[row_offset + ulong(n)];
        phi_values[n] = value;
        cos_values[n] = cos(value);
        sin_values[n] = sin(value);
    }

    cgmm_init_centers_k3(w, row_offset, N, phi_values, out_mu);
    for (uint k = 0; k < CGMM_K; ++k) {
        out_pi[k] = 1.0f / float(CGMM_K);
        out_kappa[k] = init_kappa;
        out_W[k] = 0.0f;
    }

    for (uint iter = 0; iter < n_iters; ++iter) {
        float W[CGMM_K];
        float C[CGMM_K];
        float S[CGMM_K];
        for (uint k = 0; k < CGMM_K; ++k) {
            W[k] = 0.0f;
            C[k] = 0.0f;
            S[k] = 0.0f;
        }

        for (uint n = 0; n < N; ++n) {
            uint component = 0;
            float best_dist = cgmm_circ_dist(phi_values[n], out_mu[0]);
            const float d1 = cgmm_circ_dist(phi_values[n], out_mu[1]);
            if (d1 < best_dist) {
                best_dist = d1;
                component = 1;
            }
            const float d2 = cgmm_circ_dist(phi_values[n], out_mu[2]);
            if (d2 < best_dist) {
                component = 2;
            }

            const float weight = w[row_offset + ulong(n)];
            W[component] += weight;
            C[component] += weight * cos_values[n];
            S[component] += weight * sin_values[n];
        }

        const float W_total = W[0] + W[1] + W[2];
        const float pi_den = max(W_total, CGMM_EPS);
        for (uint k = 0; k < CGMM_K; ++k) {
            out_W[k] = W[k];
            out_pi[k] = W[k] / pi_den;
            out_mu[k] = (C[k] == 0.0f && S[k] == 0.0f)
                ? 0.0f
                : cgmm_wrap_2pi(atan2(S[k], C[k]));
            const float r_bar = sqrt(C[k] * C[k] + S[k] * S[k]) / max(W[k], CGMM_EPS);
            out_kappa[k] = min(cgmm_inv_a1_banerjee(r_bar), CGMM_KAPPA_MAX);
        }
    }
}

kernel void cgmm_fuse_two_pass_k3(
    device const float* phi_p [[buffer(0)]],
    device const float* w_p [[buffer(1)]],
    device const float* phi_s [[buffer(2)]],
    device const float* w_s [[buffer(3)]],
    device float* theta_primary [[buffer(4)]],
    device float* M_primary [[buffer(5)]],
    device float* theta_sec [[buffer(6)]],
    device float* M_sec [[buffer(7)]],
    device uchar* v_fused [[buffer(8)]],
    device float* primary_pi [[buffer(9)]],
    device float* primary_mu [[buffer(10)]],
    device float* primary_kappa [[buffer(11)]],
    device float* secondary_pi [[buffer(12)]],
    device float* secondary_mu [[buffer(13)]],
    device float* secondary_kappa [[buffer(14)]],
    device uchar* keep_secondary_mask [[buffer(15)]],
    constant CgmmParams& params [[buffer(16)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.P || params.N == 0 || params.N > CGMM_N_MAX) {
        return;
    }

    const float nan_value = cgmm_nan();
    const ulong row_offset = ulong(row) * ulong(params.N);
    const ulong diag_offset = ulong(row) * ulong(CGMM_K);

    theta_primary[row] = nan_value;
    M_primary[row] = 0.0f;
    theta_sec[row] = nan_value;
    M_sec[row] = 0.0f;
    v_fused[row] = uchar(0);
    keep_secondary_mask[row] = uchar(0);
    for (uint k = 0; k < CGMM_K; ++k) {
        const ulong idx = diag_offset + ulong(k);
        primary_pi[idx] = nan_value;
        primary_mu[idx] = nan_value;
        primary_kappa[idx] = nan_value;
        secondary_pi[idx] = nan_value;
        secondary_mu[idx] = nan_value;
        secondary_kappa[idx] = nan_value;
    }

    float W_total_p = 0.0f;
    uint n_active_p = 0;
    for (uint n = 0; n < params.N; ++n) {
        const float weight = w_p[row_offset + ulong(n)];
        W_total_p += weight;
        if (weight > CGMM_EPS) {
            n_active_p += 1;
        }
    }
    const bool primary_valid = W_total_p > CGMM_EPS && n_active_p >= CGMM_K;
    if (!primary_valid) {
        return;
    }

    float p_pi[CGMM_K];
    float p_mu[CGMM_K];
    float p_kappa[CGMM_K];
    float p_W[CGMM_K];
    cgmm_fit_k3(
        phi_p, w_p, row_offset, params.N, params.n_iters, params.init_kappa,
        p_pi, p_mu, p_kappa, p_W);

    for (uint k = 0; k < CGMM_K; ++k) {
        const ulong idx = diag_offset + ulong(k);
        primary_pi[idx] = p_pi[k];
        primary_mu[idx] = p_mu[k];
        primary_kappa[idx] = p_kappa[k];
    }

    const uint k_p = cgmm_argmax3(p_pi);
    const float mu_kp = p_mu[k_p];
    const float M_kp = p_W[k_p];
    theta_primary[row] = cgmm_wrap_2pi(mu_kp) * 0.5f;
    M_primary[row] = M_kp;
    v_fused[row] = uchar(1);

    float W_total_s = 0.0f;
    uint n_active_s = 0;
    for (uint n = 0; n < params.N; ++n) {
        const float weight = w_s[row_offset + ulong(n)];
        W_total_s += weight;
        if (weight > CGMM_EPS) {
            n_active_s += 1;
        }
    }
    const bool secondary_valid = W_total_s > CGMM_EPS && n_active_s >= CGMM_K;
    if (!secondary_valid) {
        return;
    }

    float s_pi[CGMM_K];
    float s_mu[CGMM_K];
    float s_kappa[CGMM_K];
    float s_W[CGMM_K];
    cgmm_fit_k3(
        phi_s, w_s, row_offset, params.N, params.n_iters, params.init_kappa,
        s_pi, s_mu, s_kappa, s_W);

    for (uint k = 0; k < CGMM_K; ++k) {
        const ulong idx = diag_offset + ulong(k);
        secondary_pi[idx] = s_pi[k];
        secondary_mu[idx] = s_mu[k];
        secondary_kappa[idx] = s_kappa[k];
    }

    const uint k_s = cgmm_argmax3(s_pi);
    const float M_ks = s_W[k_s];
    const float mu_ks = s_mu[k_s];
    const bool mass_ok = M_ks / max(M_kp, 1.0e-30f) > params.tau_M_rel;
    const bool sep_ok = cgmm_circ_dist(mu_kp, mu_ks) > params.theta_min_phi;
    if (mass_ok && sep_ok) {
        theta_sec[row] = cgmm_wrap_2pi(mu_ks) * 0.5f;
        M_sec[row] = M_ks;
        keep_secondary_mask[row] = uchar(1);
    }
}
