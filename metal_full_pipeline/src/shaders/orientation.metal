
inline float recovery_eval_segment(
    threadgroup const float* y,
    threadgroup const float* m,
    constant RecoveryParams& params,
    uint seg,
    float u
) {
    const uint k = params.k;
    const uint next = (seg + 1 == k) ? 0 : seg + 1;
    const float omt = 1.0f - u;
    const float omt2 = omt * omt;
    const float u2 = u * u;
    return y[seg] * omt + y[next] * u +
        params.h2_over6 *
            (m[seg] * (omt2 * omt - omt) + m[next] * (u2 * u - u));
}

inline float recovery_eval_spline(
    threadgroup const float* y,
    threadgroup const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const ulong scaled = ulong(dense_idx) * ulong(params.k);
    const uint seg = uint(scaled / ulong(params.dense_n));
    const uint rem = uint(scaled - ulong(seg) * ulong(params.dense_n));
    return recovery_eval_segment(y, m, params, seg, float(rem) / float(params.dense_n));
}

struct RecoveryPeakCandidate {
    bool is_peak;
    uint dense_idx;
    float value;
};

inline uint recovery_dense_floor_idx(
    constant RecoveryParams& params,
    uint seg,
    float u
) {
    const float dense_pos =
        (float(seg) + u) * float(params.dense_n) / float(params.k);
    uint idx = uint(floor(dense_pos));
    if (idx >= params.dense_n) {
        idx -= params.dense_n;
    }
    return idx;
}

inline RecoveryPeakCandidate recovery_dense_peak_candidate(
    threadgroup const float* y,
    threadgroup const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint left_idx = dense_idx == 0 ? params.dense_n - 1 : dense_idx - 1;
    const uint right_idx = dense_idx + 1 == params.dense_n ? 0 : dense_idx + 1;
    const float left_value = recovery_eval_spline(y, m, params, left_idx);
    const float center_value = recovery_eval_spline(y, m, params, dense_idx);
    const float right_value = recovery_eval_spline(y, m, params, right_idx);
    RecoveryPeakCandidate candidate;
    candidate.is_peak = center_value >= left_value && center_value >= right_value;
    candidate.dense_idx = dense_idx;
    candidate.value = center_value;
    return candidate;
}

inline float recovery_eval_trig_dense(
    thread const float* coeff,
    device const float* trig_dense,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint base = dense_idx * RECOVERY_TRIG_COEFFS;
    return coeff[0] * trig_dense[base]
        + coeff[1] * trig_dense[base + 1]
        + coeff[2] * trig_dense[base + 2]
        + coeff[3] * trig_dense[base + 3]
        + coeff[4] * trig_dense[base + 4]
        + coeff[5] * trig_dense[base + 5]
        + coeff[6] * trig_dense[base + 6]
        + coeff[7] * trig_dense[base + 7]
        + coeff[8] * trig_dense[base + 8];
}

inline float recovery_eval_trig_dense_threadgroup(
    threadgroup const float* coeff,
    device const float* trig_dense,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint base = dense_idx * RECOVERY_TRIG_COEFFS;
    return coeff[0] * trig_dense[base]
        + coeff[1] * trig_dense[base + 1]
        + coeff[2] * trig_dense[base + 2]
        + coeff[3] * trig_dense[base + 3]
        + coeff[4] * trig_dense[base + 4]
        + coeff[5] * trig_dense[base + 5]
        + coeff[6] * trig_dense[base + 6]
        + coeff[7] * trig_dense[base + 7]
        + coeff[8] * trig_dense[base + 8];
}

kernel void recovery_peaks(
    device const float* response [[buffer(0)]],
    device const float* solver_inv [[buffer(1)]],
    device float* theta_p [[buffer(4)]],
    device float* m_p [[buffer(5)]],
    device float* theta_s [[buffer(6)]],
    device float* m_s [[buffer(7)]],
    device float* row_range [[buffer(8)]],
    constant RecoveryParams& params [[buffer(9)]],
    device const float* trig_solver [[buffer(10)]],
    device const float* trig_dense [[buffer(11)]],
    threadgroup float* y_scratch [[threadgroup(0)]],
    threadgroup float* rhs_scratch [[threadgroup(1)]],
    threadgroup float* m_scratch [[threadgroup(2)]],
    threadgroup float* candidate_value [[threadgroup(3)]],
    threadgroup uint* candidate_idx [[threadgroup(4)]],
    threadgroup float* primary_value_scratch [[threadgroup(5)]],
    threadgroup uint* primary_idx_scratch [[threadgroup(6)]],
    threadgroup float* trig_scratch [[threadgroup(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint k = params.k;
    const uint lane = tid.x;
    const uint row_slot = tid.y;
    const uint row = gid.y;
    const bool active = row < params.n_rows && lane < k;
    const uint scratch_offset = row_slot * k;
    threadgroup float* y = y_scratch + scratch_offset;
    threadgroup float* rhs_values = rhs_scratch + scratch_offset;
    threadgroup float* m = m_scratch + scratch_offset;
    threadgroup float* cand_value = candidate_value + scratch_offset;
    threadgroup uint* cand_idx = candidate_idx + scratch_offset;
    threadgroup float* trig_coeff = trig_scratch + row_slot * RECOVERY_TRIG_COEFFS;
    const ulong row_offset = ulong(row) * ulong(k);

    if (active) {
        y[lane] = response[row_offset + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        const uint prev = (lane == 0) ? k - 1 : lane - 1;
        const uint next = (lane + 1 == k) ? 0 : lane + 1;
        rhs_values[lane] = params.rhs_scale * (y[next] - 2.0f * y[lane] + y[prev]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        float sum = 0.0f;
        const uint solver_row = lane * k;
        for (uint j = 0; j < k; ++j) {
            sum += solver_inv[solver_row + j] * rhs_values[j];
        }
        m[lane] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active && lane == 0) {
        float ymin = y[0];
        float ymax = y[0];
        for (uint i = 1; i < k; ++i) {
            ymin = min(ymin, y[i]);
            ymax = max(ymax, y[i]);
        }
        row_range[row] = ymax - ymin;
    }

    if (active) {
        const uint i = lane;
        float best_value = -INFINITY;
        uint best_idx = 0;

        uint base_idx = recovery_dense_floor_idx(params, i, 0.0f);
        for (uint offset = 0; offset < 2; ++offset) {
            const uint dense_idx =
                (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
            const float value = recovery_eval_spline(y, m, params, dense_idx);
            if (value > best_value) {
                best_value = value;
                best_idx = dense_idx;
            }
        }

        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float a = 3.0f * params.h2_over6 * (m[next] - m[i]);
        const float b = 6.0f * params.h2_over6 * m[i];
        const float c = y[next] - y[i] - params.h2_over6 * (2.0f * m[i] + m[next]);

        float roots[2];
        uint n_roots = 0;
        if (fabs(a) <= 1.0e-20f) {
            if (fabs(b) > 1.0e-20f) {
                roots[0] = -c / b;
                n_roots = 1;
            }
        } else {
            const float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                const float sqrt_disc = sqrt(disc);
                const float denom = 2.0f * a;
                roots[0] = (-b - sqrt_disc) / denom;
                roots[1] = (-b + sqrt_disc) / denom;
                n_roots = 2;
            }
        }

        for (uint root_idx = 0; root_idx < n_roots; ++root_idx) {
            const float u = roots[root_idx];
            if (u < -1.0e-6f || u > 1.0f + 1.0e-6f) {
                continue;
            }
            const float u_eval = clamp(u, 0.0f, 1.0f);
            const float second_derivative = m[i] * (1.0f - u_eval) + m[next] * u_eval;
            if (second_derivative >= 0.0f) {
                continue;
            }
            base_idx = recovery_dense_floor_idx(params, i, u_eval);
            for (uint offset = 1; offset < 2; ++offset) {
                const uint dense_idx =
                    (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
                const float value = recovery_eval_spline(y, m, params, dense_idx);
                if (value > best_value) {
                    best_value = value;
                    best_idx = dense_idx;
                }
            }
        }

        cand_value[lane] = best_value;
        cand_idx[lane] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active && lane == 0) {
        float primary_value = cand_value[0];
        uint primary_idx = cand_idx[0];
        for (uint i = 1; i < k; ++i) {
            if (cand_value[i] > primary_value) {
                primary_value = cand_value[i];
                primary_idx = cand_idx[i];
            }
        }
        primary_value_scratch[row_slot] = primary_value;
        primary_idx_scratch[row_slot] = primary_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        for (uint c_idx = lane; c_idx < params.trig_coeffs; c_idx += k) {
            float sum = 0.0f;
            const uint solver_row = c_idx * k;
            for (uint j = 0; j < k; ++j) {
                sum += trig_solver[solver_row + j] * y[j];
            }
            trig_coeff[c_idx] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        const uint primary_idx = primary_idx_scratch[row_slot];
        float best_value = -INFINITY;
        uint best_idx = 0;

        for (uint dense_idx = lane; dense_idx < params.dense_n; dense_idx += k) {
            const uint left_idx = dense_idx == 0 ? params.dense_n - 1 : dense_idx - 1;
            const uint right_idx = dense_idx + 1 == params.dense_n ? 0 : dense_idx + 1;
            const float left_value =
                recovery_eval_trig_dense_threadgroup(trig_coeff, trig_dense, params, left_idx);
            const float center_value =
                recovery_eval_trig_dense_threadgroup(trig_coeff, trig_dense, params, dense_idx);
            const float right_value =
                recovery_eval_trig_dense_threadgroup(trig_coeff, trig_dense, params, right_idx);
            uint dist = dense_idx > primary_idx ? dense_idx - primary_idx : primary_idx - dense_idx;
            dist = min(dist, params.dense_n - dist);
            if (center_value >= left_value
                    && center_value >= right_value
                    && dist > params.sep
                    && center_value > best_value) {
                best_value = center_value;
                best_idx = dense_idx;
            }
        }

        cand_value[lane] = best_value;
        cand_idx[lane] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active && lane == 0) {
        const float primary_value = primary_value_scratch[row_slot];
        const uint primary_idx = primary_idx_scratch[row_slot];
        float secondary_value = -INFINITY;
        uint secondary_idx = 0;
        bool has_secondary = false;
        for (uint i = 0; i < k; ++i) {
            if (cand_value[i] > secondary_value) {
                secondary_value = cand_value[i];
                secondary_idx = cand_idx[i];
                has_secondary = true;
            }
        }

        float row_max = y[0];
        for (uint i = 1; i < k; ++i) {
            row_max = max(row_max, y[i]);
        }

        theta_p[row] = float(primary_idx) * params.pi_over_dense;
        m_p[row] = min(primary_value, row_max);

        const float ratio_den = max(primary_value, 1.0e-30f);
        const bool suppress = !has_secondary || (secondary_value / ratio_den) < params.tau_sec_floor;
        if (suppress) {
            theta_s[row] = as_type<float>(0x7fc00000u);
            m_s[row] = 0.0f;
        } else {
            theta_s[row] = float(secondary_idx) * params.pi_over_dense;
            m_s[row] = min(secondary_value, row_max);
        }
    }
}

inline float recovery_eval_segment_private(
    device const float* response,
    ulong row_offset,
    thread const float* m,
    constant RecoveryParams& params,
    uint seg,
    float u
) {
    const uint k = params.k;
    const uint next = (seg + 1 == k) ? 0 : seg + 1;
    const float y0 = response[row_offset + seg];
    const float y1 = response[row_offset + next];
    const float omt = 1.0f - u;
    const float omt2 = omt * omt;
    const float u2 = u * u;
    return y0 * omt + y1 * u +
        params.h2_over6 *
            (m[seg] * (omt2 * omt - omt) + m[next] * (u2 * u - u));
}

inline float recovery_eval_spline_private(
    device const float* response,
    ulong row_offset,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const ulong scaled = ulong(dense_idx) * ulong(params.k);
    const uint seg = uint(scaled / ulong(params.dense_n));
    const uint rem = uint(scaled - ulong(seg) * ulong(params.dense_n));
    return recovery_eval_segment_private(
        response, row_offset, m, params, seg, float(rem) / float(params.dense_n));
}

inline RecoveryPeakCandidate recovery_dense_peak_candidate_private(
    device const float* response,
    ulong row_offset,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint left_idx = dense_idx == 0 ? params.dense_n - 1 : dense_idx - 1;
    const uint right_idx = dense_idx + 1 == params.dense_n ? 0 : dense_idx + 1;
    const float left_value = recovery_eval_spline_private(response, row_offset, m, params, left_idx);
    const float center_value = recovery_eval_spline_private(response, row_offset, m, params, dense_idx);
    const float right_value = recovery_eval_spline_private(response, row_offset, m, params, right_idx);
    RecoveryPeakCandidate candidate;
    candidate.is_peak = center_value >= left_value && center_value >= right_value;
    candidate.dense_idx = dense_idx;
    candidate.value = center_value;
    return candidate;
}

inline float recovery_eval_near_segment_private(
    device const float* response,
    ulong row_offset,
    thread const float* m,
    constant RecoveryParams& params,
    uint seg_hint,
    uint dense_idx
) {
    int seg = int(seg_hint);
    int rel = int(dense_idx) * int(params.k) - int(seg_hint) * int(params.dense_n);
    while (rel < 0) {
        rel += int(params.dense_n);
        seg = (seg == 0) ? int(params.k) - 1 : seg - 1;
    }
    while (rel >= int(params.dense_n)) {
        rel -= int(params.dense_n);
        seg = (seg + 1 == int(params.k)) ? 0 : seg + 1;
    }
    return recovery_eval_segment_private(
        response, row_offset, m, params, uint(seg), float(rel) / float(params.dense_n));
}

kernel void recovery_peaks_private(
    device const float* response [[buffer(0)]],
    device const float* cprime [[buffer(1)]],
    device const float* inv_denom [[buffer(2)]],
    device const float* z_solve [[buffer(3)]],
    device float* theta_p [[buffer(4)]],
    device float* m_p [[buffer(5)]],
    device float* theta_s [[buffer(6)]],
    device float* m_s [[buffer(7)]],
    device float* row_range [[buffer(8)]],
    constant RecoveryParams& params [[buffer(9)]],
    device const float* trig_solver [[buffer(10)]],
    device const float* trig_dense [[buffer(11)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.n_rows || params.k > 64) {
        return;
    }

    const uint k = params.k;
    const ulong row_offset = ulong(row) * ulong(k);
    float m[64];

    float ymin = INFINITY;
    float ymax = -INFINITY;
    for (uint i = 0; i < k; ++i) {
        const float value = response[row_offset + i];
        ymin = min(ymin, value);
        ymax = max(ymax, value);
    }
    row_range[row] = ymax - ymin;

    for (uint i = 0; i < k; ++i) {
        const uint prev = (i == 0) ? k - 1 : i - 1;
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float y_prev = response[row_offset + prev];
        const float y_curr = response[row_offset + i];
        const float y_next = response[row_offset + next];
        const float rhs = params.rhs_scale * (y_next - 2.0f * y_curr + y_prev);
        if (i == 0) {
            m[i] = rhs * inv_denom[i];
        } else {
            m[i] = (rhs - m[i - 1]) * inv_denom[i];
        }
    }
    for (int i = int(k) - 2; i >= 0; --i) {
        m[uint(i)] = m[uint(i)] - cprime[uint(i)] * m[uint(i) + 1];
    }
    const float correction =
        (m[0] + params.gamma_inv * m[k - 1]) * params.cyclic_denom_inv;
    for (uint i = 0; i < k; ++i) {
        m[i] = m[i] - correction * z_solve[i];
    }

    float top_values[7];
    uint top_indices[7];
    for (uint i = 0; i < 7; ++i) {
        top_values[i] = -INFINITY;
        top_indices[i] = 0;
    }

    for (uint i = 0; i < k; ++i) {
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float a = 3.0f * params.h2_over6 * (m[next] - m[i]);
        const float b = 6.0f * params.h2_over6 * m[i];
        const float y_i = response[row_offset + i];
        const float y_next = response[row_offset + next];
        const float c = y_next - y_i - params.h2_over6 * (2.0f * m[i] + m[next]);

        float roots[2];
        uint n_roots = 0;
        if (fabs(a) <= 1.0e-20f) {
            if (fabs(b) > 1.0e-20f) {
                roots[0] = -c / b;
                n_roots = 1;
            }
        } else {
            const float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                const float sqrt_disc = sqrt(disc);
                const float denom = 2.0f * a;
                roots[0] = (-b - sqrt_disc) / denom;
                roots[1] = (-b + sqrt_disc) / denom;
                n_roots = 2;
            }
        }

        for (uint root_idx = 0; root_idx < n_roots; ++root_idx) {
            const float u = roots[root_idx];
            if (u < -1.0e-6f || u > 1.0f + 1.0e-6f) {
                continue;
            }
            const float u_eval = clamp(u, 0.0f, 1.0f);
            const float second_derivative = m[i] * (1.0f - u_eval) + m[next] * u_eval;
            if (second_derivative >= 0.0f) {
                continue;
            }
            const uint base_idx = recovery_dense_floor_idx(params, i, u_eval);
            uint best_idx = base_idx;
            float best_value =
                recovery_eval_near_segment_private(response, row_offset, m, params, i, base_idx);
            for (uint offset = 0; offset < 2; ++offset) {
                const uint dense_idx =
                    (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
                const float value =
                    recovery_eval_near_segment_private(response, row_offset, m, params, i, dense_idx);
                if (value > best_value) {
                    best_value = value;
                    best_idx = dense_idx;
                }
            }
            for (uint pos = 0; pos < 7; ++pos) {
                if (best_value > top_values[pos]) {
                    for (uint shift = 6; shift > pos; --shift) {
                        top_values[shift] = top_values[shift - 1];
                        top_indices[shift] = top_indices[shift - 1];
                    }
                    top_values[pos] = best_value;
                    top_indices[pos] = best_idx;
                    break;
                }
            }
        }
    }

    float primary_value = top_values[0];
    uint primary_idx = top_indices[0];
    if (primary_value == -INFINITY) {
        primary_value = response[row_offset];
        primary_idx = 0;
    }

    float trig_coeff[RECOVERY_TRIG_COEFFS];
    for (uint c_idx = 0; c_idx < params.trig_coeffs; ++c_idx) {
        float sum = 0.0f;
        const uint solver_row = c_idx * k;
        for (uint j = 0; j < k; ++j) {
            sum += trig_solver[solver_row + j] * response[row_offset + j];
        }
        trig_coeff[c_idx] = sum;
    }

    float secondary_value = -INFINITY;
    uint secondary_idx = 0;
    bool has_secondary = false;
    float left_value =
        recovery_eval_trig_dense(trig_coeff, trig_dense, params, params.dense_n - 1);
    float center_value =
        recovery_eval_trig_dense(trig_coeff, trig_dense, params, 0);
    float right_value =
        recovery_eval_trig_dense(
            trig_coeff, trig_dense, params, params.dense_n == 1 ? 0 : 1);
    for (uint dense_idx = 0; dense_idx < params.dense_n; ++dense_idx) {
        uint dist = dense_idx > primary_idx
            ? dense_idx - primary_idx
            : primary_idx - dense_idx;
        dist = min(dist, params.dense_n - dist);
        if (center_value >= left_value
                && center_value >= right_value
                && dist > params.sep
                && center_value > secondary_value) {
            secondary_value = center_value;
            secondary_idx = dense_idx;
            has_secondary = true;
        }
        left_value = center_value;
        center_value = right_value;
        uint next_idx = dense_idx + 2;
        if (next_idx >= params.dense_n) {
            next_idx -= params.dense_n;
        }
        right_value = recovery_eval_trig_dense(trig_coeff, trig_dense, params, next_idx);
    }

    theta_p[row] = float(primary_idx) * params.pi_over_dense;
    m_p[row] = min(primary_value, ymax);

    const float ratio_den = max(primary_value, 1.0e-30f);
    const bool suppress = !has_secondary || (secondary_value / ratio_den) < params.tau_sec_floor;
    if (suppress) {
        theta_s[row] = as_type<float>(0x7fc00000u);
        m_s[row] = 0.0f;
    } else {
        theta_s[row] = float(secondary_idx) * params.pi_over_dense;
        m_s[row] = min(secondary_value, ymax);
    }
}

inline float recovery_eval_segment_stack(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint seg,
    float u
) {
    const uint k = params.k;
    const uint next = (seg + 1 == k) ? 0 : seg + 1;
    const float y0 = response[ulong(seg) * ulong(params.plane_size) + ulong(row)];
    const float y1 = response[ulong(next) * ulong(params.plane_size) + ulong(row)];
    const float omt = 1.0f - u;
    const float omt2 = omt * omt;
    const float u2 = u * u;
    return y0 * omt + y1 * u +
        params.h2_over6 *
            (m[seg] * (omt2 * omt - omt) + m[next] * (u2 * u - u));
}

inline float recovery_eval_spline_stack(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const ulong scaled = ulong(dense_idx) * ulong(params.k);
    const uint seg = uint(scaled / ulong(params.dense_n));
    const uint rem = uint(scaled - ulong(seg) * ulong(params.dense_n));
    return recovery_eval_segment_stack(
        response, row, m, params, seg, float(rem) / float(params.dense_n));
}

inline RecoveryPeakCandidate recovery_dense_peak_candidate_stack(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint left_idx = dense_idx == 0 ? params.dense_n - 1 : dense_idx - 1;
    const uint right_idx = dense_idx + 1 == params.dense_n ? 0 : dense_idx + 1;
    const float left_value = recovery_eval_spline_stack(response, row, m, params, left_idx);
    const float center_value = recovery_eval_spline_stack(response, row, m, params, dense_idx);
    const float right_value = recovery_eval_spline_stack(response, row, m, params, right_idx);
    RecoveryPeakCandidate candidate;
    candidate.is_peak = center_value >= left_value && center_value >= right_value;
    candidate.dense_idx = dense_idx;
    candidate.value = center_value;
    return candidate;
}

inline float recovery_eval_near_segment_stack(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint seg_hint,
    uint dense_idx
) {
    int seg = int(seg_hint);
    int rel = int(dense_idx) * int(params.k) - int(seg_hint) * int(params.dense_n);
    while (rel < 0) {
        rel += int(params.dense_n);
        seg = (seg == 0) ? int(params.k) - 1 : seg - 1;
    }
    while (rel >= int(params.dense_n)) {
        rel -= int(params.dense_n);
        seg = (seg + 1 == int(params.k)) ? 0 : seg + 1;
    }
    return recovery_eval_segment_stack(
        response, row, m, params, uint(seg), float(rel) / float(params.dense_n));
}

kernel void recovery_peaks_private_stack(
    device const float* response [[buffer(0)]],
    device const float* cprime [[buffer(1)]],
    device const float* inv_denom [[buffer(2)]],
    device const float* z_solve [[buffer(3)]],
    device float* theta_p [[buffer(4)]],
    device float* m_p [[buffer(5)]],
    device float* theta_s [[buffer(6)]],
    device float* m_s [[buffer(7)]],
    device float* row_range [[buffer(8)]],
    constant RecoveryParams& params [[buffer(9)]],
    device const float* trig_solver [[buffer(10)]],
    device const float* trig_dense [[buffer(11)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.n_rows || params.k > 64) {
        return;
    }

    const uint k = params.k;
    float m[64];

    float ymin = INFINITY;
    float ymax = -INFINITY;
    for (uint i = 0; i < k; ++i) {
        const float value = response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        ymin = min(ymin, value);
        ymax = max(ymax, value);
    }
    row_range[row] = ymax - ymin;

    for (uint i = 0; i < k; ++i) {
        const uint prev = (i == 0) ? k - 1 : i - 1;
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float y_prev = response[ulong(prev) * ulong(params.plane_size) + ulong(row)];
        const float y_curr = response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        const float y_next = response[ulong(next) * ulong(params.plane_size) + ulong(row)];
        const float rhs = params.rhs_scale * (y_next - 2.0f * y_curr + y_prev);
        if (i == 0) {
            m[i] = rhs * inv_denom[i];
        } else {
            m[i] = (rhs - m[i - 1]) * inv_denom[i];
        }
    }
    for (int i = int(k) - 2; i >= 0; --i) {
        m[uint(i)] = m[uint(i)] - cprime[uint(i)] * m[uint(i) + 1];
    }
    const float correction =
        (m[0] + params.gamma_inv * m[k - 1]) * params.cyclic_denom_inv;
    for (uint i = 0; i < k; ++i) {
        m[i] = m[i] - correction * z_solve[i];
    }

    float top_values[7];
    uint top_indices[7];
    for (uint i = 0; i < 7; ++i) {
        top_values[i] = -INFINITY;
        top_indices[i] = 0;
    }

    for (uint i = 0; i < k; ++i) {
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float a = 3.0f * params.h2_over6 * (m[next] - m[i]);
        const float b = 6.0f * params.h2_over6 * m[i];
        const float y_i = response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        const float y_next = response[ulong(next) * ulong(params.plane_size) + ulong(row)];
        const float c = y_next - y_i - params.h2_over6 * (2.0f * m[i] + m[next]);

        float roots[2];
        uint n_roots = 0;
        if (fabs(a) <= 1.0e-20f) {
            if (fabs(b) > 1.0e-20f) {
                roots[0] = -c / b;
                n_roots = 1;
            }
        } else {
            const float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                const float sqrt_disc = sqrt(disc);
                const float denom = 2.0f * a;
                roots[0] = (-b - sqrt_disc) / denom;
                roots[1] = (-b + sqrt_disc) / denom;
                n_roots = 2;
            }
        }

        for (uint root_idx = 0; root_idx < n_roots; ++root_idx) {
            const float u = roots[root_idx];
            if (u < -1.0e-6f || u > 1.0f + 1.0e-6f) {
                continue;
            }
            const float u_eval = clamp(u, 0.0f, 1.0f);
            const float second_derivative = m[i] * (1.0f - u_eval) + m[next] * u_eval;
            if (second_derivative >= 0.0f) {
                continue;
            }
            const uint base_idx = recovery_dense_floor_idx(params, i, u_eval);
            uint best_idx = base_idx;
            float best_value =
                recovery_eval_near_segment_stack(response, row, m, params, i, base_idx);
            for (uint offset = 0; offset < 2; ++offset) {
                const uint dense_idx =
                    (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
                const float value =
                    recovery_eval_near_segment_stack(response, row, m, params, i, dense_idx);
                if (value > best_value) {
                    best_value = value;
                    best_idx = dense_idx;
                }
            }
            for (uint pos = 0; pos < 7; ++pos) {
                if (best_value > top_values[pos]) {
                    for (uint shift = 6; shift > pos; --shift) {
                        top_values[shift] = top_values[shift - 1];
                        top_indices[shift] = top_indices[shift - 1];
                    }
                    top_values[pos] = best_value;
                    top_indices[pos] = best_idx;
                    break;
                }
            }
        }
    }

    float primary_value = top_values[0];
    uint primary_idx = top_indices[0];
    if (primary_value == -INFINITY) {
        primary_value = response[ulong(row)];
        primary_idx = 0;
    }

    float trig_coeff[RECOVERY_TRIG_COEFFS];
    for (uint c_idx = 0; c_idx < params.trig_coeffs; ++c_idx) {
        float sum = 0.0f;
        const uint solver_row = c_idx * k;
        for (uint j = 0; j < k; ++j) {
            sum += trig_solver[solver_row + j]
                * response[ulong(j) * ulong(params.plane_size) + ulong(row)];
        }
        trig_coeff[c_idx] = sum;
    }

    float secondary_value = -INFINITY;
    uint secondary_idx = 0;
    bool has_secondary = false;
    float left_value =
        recovery_eval_trig_dense(trig_coeff, trig_dense, params, params.dense_n - 1);
    float center_value =
        recovery_eval_trig_dense(trig_coeff, trig_dense, params, 0);
    float right_value =
        recovery_eval_trig_dense(
            trig_coeff, trig_dense, params, params.dense_n == 1 ? 0 : 1);
    for (uint dense_idx = 0; dense_idx < params.dense_n; ++dense_idx) {
        uint dist = dense_idx > primary_idx
            ? dense_idx - primary_idx
            : primary_idx - dense_idx;
        dist = min(dist, params.dense_n - dist);
        if (center_value >= left_value
                && center_value >= right_value
                && dist > params.sep
                && center_value > secondary_value) {
            secondary_value = center_value;
            secondary_idx = dense_idx;
            has_secondary = true;
        }
        left_value = center_value;
        center_value = right_value;
        uint next_idx = dense_idx + 2;
        if (next_idx >= params.dense_n) {
            next_idx -= params.dense_n;
        }
        right_value = recovery_eval_trig_dense(trig_coeff, trig_dense, params, next_idx);
    }

    theta_p[row] = float(primary_idx) * params.pi_over_dense;
    m_p[row] = min(primary_value, ymax);

    const float ratio_den = max(primary_value, 1.0e-30f);
    const bool suppress = !has_secondary || (secondary_value / ratio_den) < params.tau_sec_floor;
    if (suppress) {
        theta_s[row] = as_type<float>(0x7fc00000u);
        m_s[row] = 0.0f;
    } else {
        theta_s[row] = float(secondary_idx) * params.pi_over_dense;
        m_s[row] = min(secondary_value, ymax);
    }
}

kernel void recovery_range_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RecoveryReduceParams& params [[buffer(2)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    const uint base = group_id * params.group_size * 2 + tid;
    float best = 0.0f;
    if (base < params.count) {
        best = input[base];
    }
    const uint second = base + params.group_size;
    if (second < params.count) {
        best = max(best, input[second]);
    }
    scratch[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = params.group_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = scratch[0];
    }
}

kernel void recovery_validity_flags(
    device const float* row_range [[buffer(0)]],
    device const float* ref_range [[buffer(1)]],
    device uchar* v [[buffer(2)]],
    constant uint& n_rows [[buffer(3)]],
    constant float& tau_validity [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= n_rows) {
        return;
    }
    v[row] = row_range[row] > tau_validity * ref_range[0] ? uchar(1) : uchar(0);
}
