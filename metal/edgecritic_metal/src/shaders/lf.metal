
kernel void lf_response(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device const int* px [[buffer(2)]],
    device const int* py [[buffer(3)]],
    device const int* dx [[buffer(4)]],
    device const int* dy [[buffer(5)]],
    device const float* weights [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant LfParams& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_pixels) {
        return;
    }

    const int x0 = px[gid];
    const int y0 = py[gid];
    float num = 0.0f;
    float den = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int x = x0 + dx[k];
        const int y = y0 + dy[k];
        if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
            continue;
        }
        const uint index = uint(y) * params.width + uint(x);
        const float sample = -params.sin_t * g_x[index] + params.cos_t * g_y[index];
        const float w = weights[k];
        num += sample * w;
        den += w;
    }

    out[gid] = den > 0.0f ? fabs(num / den) : 0.0f;
}

kernel void lf_response_batch(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device const int* px [[buffer(2)]],
    device const int* py [[buffer(3)]],
    device const float* cos_values [[buffer(4)]],
    device const float* sin_values [[buffer(5)]],
    device const int* dx [[buffer(6)]],
    device const int* dy [[buffer(7)]],
    device const float* weights [[buffer(8)]],
    device float* out [[buffer(9)]],
    constant LfBatchParams& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_threads = params.n_pixels * params.n_thetas;
    if (gid >= total_threads) {
        return;
    }

    const uint pixel_idx = gid % params.n_pixels;
    const uint theta_idx = gid / params.n_pixels;
    const int x0 = px[pixel_idx];
    const int y0 = py[pixel_idx];
    const float cos_t = cos_values[theta_idx];
    const float sin_t = sin_values[theta_idx];
    const uint theta_offset = theta_idx * params.n_samples;

    float nums[MAX_BATCH_MS];
    float dens[MAX_BATCH_MS];
    for (uint mi = 0; mi < params.n_ms; ++mi) {
        nums[mi] = 0.0f;
        dens[mi] = 0.0f;
    }

    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint offset_idx = theta_offset + sample_idx;
        const int x = x0 + dx[offset_idx];
        const int y = y0 + dy[offset_idx];
        if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
            continue;
        }

        const uint image_idx = uint(y) * params.width + uint(x);
        const float sample = -sin_t * g_x[image_idx] + cos_t * g_y[image_idx];
        for (uint mi = 0; mi < params.n_ms; ++mi) {
            const float w = weights[mi * params.n_samples + sample_idx];
            nums[mi] += sample * w;
            dens[mi] += w;
        }
    }

    for (uint mi = 0; mi < params.n_ms; ++mi) {
        const uint out_idx = (theta_idx * params.n_ms + mi) * params.n_pixels + pixel_idx;
        out[out_idx] = dens[mi] > 0.0f ? fabs(nums[mi] / dens[mi]) : 0.0f;
    }
}

kernel void lf_orientation_stack_interior(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device const float* cos_values [[buffer(2)]],
    device const float* sin_values [[buffer(3)]],
    device const int* dx [[buffer(4)]],
    device const int* dy [[buffer(5)]],
    device const float* weights [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant LfStackParams& params [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.n_orientations) {
        return;
    }
    if (params.border > 0) {
        if (params.width <= 2 * params.border || params.height <= 2 * params.border) {
            return;
        }
        if (gid.x < params.border || gid.x >= params.width - params.border ||
            gid.y < params.border || gid.y >= params.height - params.border) {
            return;
        }
    }

    const uint theta_idx = gid.z;
    const int x0 = int(gid.x);
    const int y0 = int(gid.y);
    const float cos_t = cos_values[theta_idx];
    const float sin_t = sin_values[theta_idx];
    const uint theta_offset = theta_idx * params.n_samples;
    float num = 0.0f;

    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint offset_idx = theta_offset + sample_idx;
        const int x = x0 + dx[offset_idx];
        const int y = y0 + dy[offset_idx];
        const uint image_idx = uint(y) * params.width + uint(x);
        const float sample = -sin_t * g_x[image_idx] + cos_t * g_y[image_idx];
        const float w = weights[sample_idx];
        num += sample * w;
    }

    const uint plane_size = params.width * params.height;
    const uint out_idx = theta_idx * plane_size + gid.y * params.width + gid.x;
    out[out_idx] = params.weight_sum > 0.0f ? fabs(num / params.weight_sum) : 0.0f;
}

kernel void lf_orientation_stack_boundary(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device const float* cos_values [[buffer(2)]],
    device const float* sin_values [[buffer(3)]],
    device const int* dx [[buffer(4)]],
    device const int* dy [[buffer(5)]],
    device const float* weights [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant LfStackParams& params [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.n_orientations) {
        return;
    }
    if (params.border == 0) {
        return;
    }
    if (params.width > 2 * params.border && params.height > 2 * params.border &&
        gid.x >= params.border && gid.x < params.width - params.border &&
        gid.y >= params.border && gid.y < params.height - params.border) {
        return;
    }

    const uint theta_idx = gid.z;
    const int x0 = int(gid.x);
    const int y0 = int(gid.y);
    const float cos_t = cos_values[theta_idx];
    const float sin_t = sin_values[theta_idx];
    const uint theta_offset = theta_idx * params.n_samples;
    float num = 0.0f;
    float den = 0.0f;

    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint offset_idx = theta_offset + sample_idx;
        const int x = x0 + dx[offset_idx];
        const int y = y0 + dy[offset_idx];
        if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
            continue;
        }

        const uint image_idx = uint(y) * params.width + uint(x);
        const float sample = -sin_t * g_x[image_idx] + cos_t * g_y[image_idx];
        const float w = weights[sample_idx];
        num += sample * w;
        den += w;
    }

    const uint plane_size = params.width * params.height;
    const uint out_idx = theta_idx * plane_size + gid.y * params.width + gid.x;
    out[out_idx] = den > 0.0f ? fabs(num / den) : 0.0f;
}

kernel void lf_project_perp(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device float* g_perp [[buffer(2)]],
    constant LfProjectParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    g_perp[idx] = -params.sin_t * g_x[idx] + params.cos_t * g_y[idx];
}

kernel void lf_orientation_stack_projected_interior(
    device const float* g_perp [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant LfPlaneParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    if (params.border > 0) {
        if (params.width <= 2 * params.border || params.height <= 2 * params.border) {
            return;
        }
        if (gid.x < params.border || gid.x >= params.width - params.border ||
            gid.y < params.border || gid.y >= params.height - params.border) {
            return;
        }
    }

    const int x0 = int(gid.x);
    const int y0 = int(gid.y);
    const uint theta_offset = params.theta_idx * params.n_samples;
    float num = 0.0f;

    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint offset_idx = theta_offset + sample_idx;
        const int x = x0 + dx[offset_idx];
        const int y = y0 + dy[offset_idx];
        const uint image_idx = uint(y) * params.width + uint(x);
        num += g_perp[image_idx] * weights[sample_idx];
    }

    const uint plane_size = params.width * params.height;
    const uint out_idx = params.theta_idx * plane_size + gid.y * params.width + gid.x;
    out[out_idx] = params.weight_sum > 0.0f ? fabs(num / params.weight_sum) : 0.0f;
}

kernel void lf_orientation_stack_projected_boundary(
    device const float* g_perp [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant LfPlaneParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    if (params.border == 0) {
        return;
    }
    if (params.width > 2 * params.border && params.height > 2 * params.border &&
        gid.x >= params.border && gid.x < params.width - params.border &&
        gid.y >= params.border && gid.y < params.height - params.border) {
        return;
    }

    const int x0 = int(gid.x);
    const int y0 = int(gid.y);
    const uint theta_offset = params.theta_idx * params.n_samples;
    float num = 0.0f;
    float den = 0.0f;

    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint offset_idx = theta_offset + sample_idx;
        const int x = x0 + dx[offset_idx];
        const int y = y0 + dy[offset_idx];
        if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
            continue;
        }

        const uint image_idx = uint(y) * params.width + uint(x);
        const float w = weights[sample_idx];
        num += g_perp[image_idx] * w;
        den += w;
    }

    const uint plane_size = params.width * params.height;
    const uint out_idx = params.theta_idx * plane_size + gid.y * params.width + gid.x;
    out[out_idx] = den > 0.0f ? fabs(num / den) : 0.0f;
}

kernel void lf_box_seed(
    device const float* g_x [[buffer(0)]],
    device const float* g_y [[buffer(1)]],
    device float* num [[buffer(2)]],
    device float* den [[buffer(3)]],
    constant LfBoxSeedParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    num[idx] = -params.sin_t * g_x[idx] + params.cos_t * g_y[idx];
    den[idx] = 1.0f;
}

kernel void lf_box_filter_x_major(
    device const float* in_num [[buffer(0)]],
    device const float* in_den [[buffer(1)]],
    device float* out_num [[buffer(2)]],
    device float* out_den [[buffer(3)]],
    device const int* line_offsets [[buffer(4)]],
    constant LfBoxPassParams& params [[buffer(5)]],
    uint line_id [[thread_position_in_grid]]
) {
    if (line_id >= params.line_count) {
        return;
    }

    const int key = int(line_id) + params.key_min;
    const int radius = int(params.radius);
    const float scale = 1.0f / float(2 * radius + 1);
    float sum_num = 0.0f;
    float sum_den = 0.0f;

    const uint initial_end = min(params.radius, params.width - 1);
    for (uint x = 0; x <= initial_end; ++x) {
        const int y = key + line_offsets[x];
        if (y >= 0 && y < int(params.height)) {
            const uint idx = uint(y) * params.width + x;
            sum_num += in_num[idx];
            sum_den += in_den[idx];
        }
    }

    for (uint x = 0; x < params.width; ++x) {
        const int y = key + line_offsets[x];
        if (y >= 0 && y < int(params.height)) {
            const uint idx = uint(y) * params.width + x;
            out_num[idx] = sum_num * scale;
            out_den[idx] = sum_den * scale;
        }

        const int remove_x = int(x) - radius;
        if (remove_x >= 0) {
            const int remove_y = key + line_offsets[uint(remove_x)];
            if (remove_y >= 0 && remove_y < int(params.height)) {
                const uint remove_idx = uint(remove_y) * params.width + uint(remove_x);
                sum_num -= in_num[remove_idx];
                sum_den -= in_den[remove_idx];
            }
        }

        const uint add_x = x + params.radius + 1;
        if (add_x < params.width) {
            const int add_y = key + line_offsets[add_x];
            if (add_y >= 0 && add_y < int(params.height)) {
                const uint add_idx = uint(add_y) * params.width + add_x;
                sum_num += in_num[add_idx];
                sum_den += in_den[add_idx];
            }
        }
    }
}

kernel void lf_box_filter_y_major(
    device const float* in_num [[buffer(0)]],
    device const float* in_den [[buffer(1)]],
    device float* out_num [[buffer(2)]],
    device float* out_den [[buffer(3)]],
    device const int* line_offsets [[buffer(4)]],
    constant LfBoxPassParams& params [[buffer(5)]],
    uint line_id [[thread_position_in_grid]]
) {
    if (line_id >= params.line_count) {
        return;
    }

    const int key = int(line_id) + params.key_min;
    const int radius = int(params.radius);
    const float scale = 1.0f / float(2 * radius + 1);
    float sum_num = 0.0f;
    float sum_den = 0.0f;

    const uint initial_end = min(params.radius, params.height - 1);
    for (uint y = 0; y <= initial_end; ++y) {
        const int x = key + line_offsets[y];
        if (x >= 0 && x < int(params.width)) {
            const uint idx = y * params.width + uint(x);
            sum_num += in_num[idx];
            sum_den += in_den[idx];
        }
    }

    for (uint y = 0; y < params.height; ++y) {
        const int x = key + line_offsets[y];
        if (x >= 0 && x < int(params.width)) {
            const uint idx = y * params.width + uint(x);
            out_num[idx] = sum_num * scale;
            out_den[idx] = sum_den * scale;
        }

        const int remove_y = int(y) - radius;
        if (remove_y >= 0) {
            const int remove_x = key + line_offsets[uint(remove_y)];
            if (remove_x >= 0 && remove_x < int(params.width)) {
                const uint remove_idx = uint(remove_y) * params.width + uint(remove_x);
                sum_num -= in_num[remove_idx];
                sum_den -= in_den[remove_idx];
            }
        }

        const uint add_y = y + params.radius + 1;
        if (add_y < params.height) {
            const int add_x = key + line_offsets[add_y];
            if (add_x >= 0 && add_x < int(params.width)) {
                const uint add_idx = add_y * params.width + uint(add_x);
                sum_num += in_num[add_idx];
                sum_den += in_den[add_idx];
            }
        }
    }
}

kernel void lf_box_finalize(
    device const float* num [[buffer(0)]],
    device const float* den [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant LfBoxFinalizeParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint plane_size = params.width * params.height;
    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = params.theta_idx * plane_size + idx;
    const float d = den[idx];
    out[out_idx] = d > 0.0f ? fabs(num[idx] / d) : 0.0f;
}

kernel void lf_box_multi_x_major(
    device const float* g_perp [[buffer(0)]],
    device const uint* radii [[buffer(1)]],
    device const int* line_offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant LfBoxMultiParams& params [[buffer(4)]],
    uint line_id [[thread_position_in_grid]]
) {
    if (line_id >= params.line_count || params.n_ms > MAX_BATCH_MS) {
        return;
    }

    const int key = int(line_id) + params.key_min;
    const uint plane_size = params.width * params.height;
    float sum_num[MAX_BATCH_MS];
    float sum_den[MAX_BATCH_MS];

    for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
        sum_num[m_idx] = 0.0f;
        sum_den[m_idx] = 0.0f;
        const uint initial_end = min(radii[m_idx], params.width - 1);
        for (uint x = 0; x <= initial_end; ++x) {
            const int y = key + line_offsets[x];
            if (y >= 0 && y < int(params.height)) {
                const uint idx = uint(y) * params.width + x;
                sum_num[m_idx] += g_perp[idx];
                sum_den[m_idx] += 1.0f;
            }
        }
    }

    for (uint x = 0; x < params.width; ++x) {
        const int y = key + line_offsets[x];
        if (y >= 0 && y < int(params.height)) {
            const uint idx = uint(y) * params.width + x;
            for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
                const float value =
                    sum_den[m_idx] > 0.0f ? fabs(sum_num[m_idx] / sum_den[m_idx]) : 0.0f;
                if (params.output_layout == 0) {
                    const ulong out_plane = ulong(params.theta_idx) * ulong(params.n_ms) + ulong(m_idx);
                    out[out_plane * plane_size + ulong(idx)] = value;
                } else {
                    out[(ulong(params.theta_idx) * plane_size + ulong(idx)) * ulong(params.n_ms) + ulong(m_idx)] = value;
                }
            }
        }

        for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
            const uint radius = radii[m_idx];
            const int remove_x = int(x) - int(radius);
            if (remove_x >= 0) {
                const int remove_y = key + line_offsets[uint(remove_x)];
                if (remove_y >= 0 && remove_y < int(params.height)) {
                    const uint remove_idx = uint(remove_y) * params.width + uint(remove_x);
                    sum_num[m_idx] -= g_perp[remove_idx];
                    sum_den[m_idx] -= 1.0f;
                }
            }

            const uint add_x = x + radius + 1;
            if (add_x < params.width) {
                const int add_y = key + line_offsets[add_x];
                if (add_y >= 0 && add_y < int(params.height)) {
                    const uint add_idx = uint(add_y) * params.width + add_x;
                    sum_num[m_idx] += g_perp[add_idx];
                    sum_den[m_idx] += 1.0f;
                }
            }
        }
    }
}

kernel void lf_box_multi_y_major(
    device const float* g_perp [[buffer(0)]],
    device const uint* radii [[buffer(1)]],
    device const int* line_offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant LfBoxMultiParams& params [[buffer(4)]],
    uint line_id [[thread_position_in_grid]]
) {
    if (line_id >= params.line_count || params.n_ms > MAX_BATCH_MS) {
        return;
    }

    const int key = int(line_id) + params.key_min;
    const ulong plane_size = ulong(params.width) * ulong(params.height);
    float sum_num[MAX_BATCH_MS];
    float sum_den[MAX_BATCH_MS];

    for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
        sum_num[m_idx] = 0.0f;
        sum_den[m_idx] = 0.0f;
        const uint initial_end = min(radii[m_idx], params.height - 1);
        for (uint y = 0; y <= initial_end; ++y) {
            const int x = key + line_offsets[y];
            if (x >= 0 && x < int(params.width)) {
                const uint idx = y * params.width + uint(x);
                sum_num[m_idx] += g_perp[idx];
                sum_den[m_idx] += 1.0f;
            }
        }
    }

    for (uint y = 0; y < params.height; ++y) {
        const int x = key + line_offsets[y];
        if (x >= 0 && x < int(params.width)) {
            const uint idx = y * params.width + uint(x);
            for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
                const float value =
                    sum_den[m_idx] > 0.0f ? fabs(sum_num[m_idx] / sum_den[m_idx]) : 0.0f;
                if (params.output_layout == 0) {
                    const ulong out_plane = ulong(params.theta_idx) * ulong(params.n_ms) + ulong(m_idx);
                    out[out_plane * plane_size + ulong(idx)] = value;
                } else {
                    out[(ulong(params.theta_idx) * plane_size + ulong(idx)) * ulong(params.n_ms) + ulong(m_idx)] = value;
                }
            }
        }

        for (uint m_idx = 0; m_idx < params.n_ms; ++m_idx) {
            const uint radius = radii[m_idx];
            const int remove_y = int(y) - int(radius);
            if (remove_y >= 0) {
                const int remove_x = key + line_offsets[uint(remove_y)];
                if (remove_x >= 0 && remove_x < int(params.width)) {
                    const uint remove_idx = uint(remove_y) * params.width + uint(remove_x);
                    sum_num[m_idx] -= g_perp[remove_idx];
                    sum_den[m_idx] -= 1.0f;
                }
            }

            const uint add_y = y + radius + 1;
            if (add_y < params.height) {
                const int add_x = key + line_offsets[add_y];
                if (add_x >= 0 && add_x < int(params.width)) {
                    const uint add_idx = add_y * params.width + uint(add_x);
                    sum_num[m_idx] += g_perp[add_idx];
                    sum_den[m_idx] += 1.0f;
                }
            }
        }
    }
}

kernel void lf_gaussian_scanline_x_major(
    device const float* g_perp [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* line_offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant LfScanlineParams& params [[buffer(4)]],
    threadgroup float* tile [[threadgroup(0)]],
    threadgroup float* valid [[threadgroup(1)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    const uint line_id = group_id.y;
    if (line_id >= params.line_count) {
        return;
    }

    const uint chunk_start = group_id.x * params.chunk_len;
    const int tile_start = int(chunk_start) - int(params.radius);
    const uint tile_len = params.chunk_len + 2 * params.radius;
    const int key = int(line_id) + params.key_min;
    const uint tid = tid2.x;

    for (uint tile_idx = tid; tile_idx < tile_len; tile_idx += params.chunk_len) {
        const int x = tile_start + int(tile_idx);
        float value = 0.0f;
        float is_valid = 0.0f;
        if (x >= 0 && x < int(params.width)) {
            const int y = key + line_offsets[uint(x)];
            if (y >= 0 && y < int(params.height)) {
                value = g_perp[uint(y) * params.width + uint(x)];
                is_valid = 1.0f;
            }
        }
        tile[tile_idx] = value;
        valid[tile_idx] = is_valid;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint x = chunk_start + tid;
    if (tid >= params.chunk_len || x >= params.width) {
        return;
    }

    const int y = key + line_offsets[x];
    if (y < 0 || y >= int(params.height)) {
        return;
    }

    float num = 0.0f;
    float den = 0.0f;
    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint tile_idx = tid + sample_idx;
        const float w = weights[sample_idx];
        num += tile[tile_idx] * w;
        den += valid[tile_idx] * w;
    }

    const uint plane_size = params.width * params.height;
    const uint idx = uint(y) * params.width + x;
    out[params.theta_idx * plane_size + idx] = den > 0.0f ? fabs(num / den) : 0.0f;
}

kernel void lf_gaussian_scanline_y_major(
    device const float* g_perp [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const int* line_offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant LfScanlineParams& params [[buffer(4)]],
    threadgroup float* tile [[threadgroup(0)]],
    threadgroup float* valid [[threadgroup(1)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    const uint line_id = group_id.y;
    if (line_id >= params.line_count) {
        return;
    }

    const uint chunk_start = group_id.x * params.chunk_len;
    const int tile_start = int(chunk_start) - int(params.radius);
    const uint tile_len = params.chunk_len + 2 * params.radius;
    const int key = int(line_id) + params.key_min;
    const uint tid = tid2.x;

    for (uint tile_idx = tid; tile_idx < tile_len; tile_idx += params.chunk_len) {
        const int y = tile_start + int(tile_idx);
        float value = 0.0f;
        float is_valid = 0.0f;
        if (y >= 0 && y < int(params.height)) {
            const int x = key + line_offsets[uint(y)];
            if (x >= 0 && x < int(params.width)) {
                value = g_perp[uint(y) * params.width + uint(x)];
                is_valid = 1.0f;
            }
        }
        tile[tile_idx] = value;
        valid[tile_idx] = is_valid;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint y = chunk_start + tid;
    if (tid >= params.chunk_len || y >= params.height) {
        return;
    }

    const int x = key + line_offsets[y];
    if (x < 0 || x >= int(params.width)) {
        return;
    }

    float num = 0.0f;
    float den = 0.0f;
    for (uint sample_idx = 0; sample_idx < params.n_samples; ++sample_idx) {
        const uint tile_idx = tid + sample_idx;
        const float w = weights[sample_idx];
        num += tile[tile_idx] * w;
        den += valid[tile_idx] * w;
    }

    const uint plane_size = params.width * params.height;
    const uint idx = y * params.width + uint(x);
    out[params.theta_idx * plane_size + idx] = den > 0.0f ? fabs(num / den) : 0.0f;
}
