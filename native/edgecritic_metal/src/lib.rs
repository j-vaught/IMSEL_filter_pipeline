use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};
use std::cell::RefCell;
use std::ffi::c_char;
use std::os::raw::{c_double, c_float, c_int, c_uint};
use std::ptr;

const MAX_BATCH_MS: usize = 32;
const LF_STACK_EXECUTION_AUTO: c_uint = 0;
const LF_STACK_EXECUTION_DIRECT: c_uint = 1;
const LF_STACK_EXECUTION_PROJECTED: c_uint = 2;
const MAX_BOX_PASSES: c_uint = 32;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH_MS 32

struct KernelParams {
    uint width;
    uint height;
    uint n_offsets;
};

struct LfParams {
    uint width;
    uint height;
    uint n_pixels;
    uint n_offsets;
    float cos_t;
    float sin_t;
};

struct LfBatchParams {
    uint width;
    uint height;
    uint n_pixels;
    uint n_thetas;
    uint n_ms;
    uint max_m;
    uint n_samples;
};

struct LfStackParams {
    uint width;
    uint height;
    uint n_orientations;
    uint n_samples;
    uint border;
    float weight_sum;
};

struct LfProjectParams {
    uint width;
    uint height;
    float cos_t;
    float sin_t;
};

struct LfPlaneParams {
    uint width;
    uint height;
    uint n_samples;
    uint theta_idx;
    uint border;
    float weight_sum;
};

struct LfBoxSeedParams {
    uint width;
    uint height;
    float cos_t;
    float sin_t;
};

struct LfBoxPassParams {
    uint width;
    uint height;
    uint radius;
    int key_min;
    uint line_count;
};

struct LfBoxFinalizeParams {
    uint width;
    uint height;
    uint theta_idx;
};

struct LfBoxMultiParams {
    uint width;
    uint height;
    uint n_ms;
    uint output_layout;
    int key_min;
    uint line_count;
    uint theta_idx;
};

struct LfScanlineParams {
    uint width;
    uint height;
    uint n_samples;
    uint radius;
    int key_min;
    uint line_count;
    uint theta_idx;
    uint chunk_len;
};

struct RecoveryParams {
    uint n_rows;
    uint k;
    uint dense_n;
    uint sep;
    float tau_sec_floor;
    float h;
    float h2_over6;
    float rhs_scale;
    float pi_over_dense;
    float gamma_inv;
    float cyclic_denom_inv;
    uint response_layout;
    uint plane_size;
};

struct RecoveryReduceParams {
    uint count;
    uint group_size;
};

inline int reflect_index(int value, int limit) {
    if (limit <= 1) {
        return 0;
    }
    while (value < 0 || value >= limit) {
        if (value < 0) {
            value = -value - 1;
        } else {
            value = 2 * limit - value - 1;
        }
    }
    return value;
}

kernel void wvf_convolve_pair(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant KernelParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint out_index = gid.y * params.width + gid.x;
    const int x = int(gid.x);
    const int y = int(gid.y);
    float sx = 0.0f;
    float sy = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int ix = reflect_index(x + dx[k], int(params.width));
        const int iy = reflect_index(y + dy[k], int(params.height));
        const float value = image[uint(iy) * params.width + uint(ix)];
        sx += value * wx[k];
        sy += value * wy[k];
    }

    out_x[out_index] = sx;
    out_y[out_index] = sy;
}

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

kernel void recovery_peaks(
    device const float* response [[buffer(0)]],
    device const float* solver_inv [[buffer(1)]],
    device float* theta_p [[buffer(4)]],
    device float* m_p [[buffer(5)]],
    device float* theta_s [[buffer(6)]],
    device float* m_s [[buffer(7)]],
    device float* row_range [[buffer(8)]],
    constant RecoveryParams& params [[buffer(9)]],
    threadgroup float* y_scratch [[threadgroup(0)]],
    threadgroup float* rhs_scratch [[threadgroup(1)]],
    threadgroup float* m_scratch [[threadgroup(2)]],
    threadgroup float* candidate_value [[threadgroup(3)]],
    threadgroup uint* candidate_idx [[threadgroup(4)]],
    threadgroup float* primary_value_scratch [[threadgroup(5)]],
    threadgroup uint* primary_idx_scratch [[threadgroup(6)]],
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
            if (u <= 0.0f || u >= 1.0f) {
                continue;
            }
            const float second_derivative = m[i] * (1.0f - u) + m[next] * u;
            if (second_derivative >= 0.0f) {
                continue;
            }
            base_idx = recovery_dense_floor_idx(params, i, u);
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
        const uint i = lane;
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const uint primary_idx = primary_idx_scratch[row_slot];
        float best_value = -INFINITY;
        uint best_idx = 0;

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
            if (u <= 0.0f || u >= 1.0f) {
                continue;
            }
            const float second_derivative = m[i] * (1.0f - u) + m[next] * u;
            if (second_derivative >= 0.0f) {
                continue;
            }
            const uint base_idx = recovery_dense_floor_idx(params, i, u);
            for (uint offset = 1; offset < 2; ++offset) {
                const uint dense_idx =
                    (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
                const RecoveryPeakCandidate candidate =
                    recovery_dense_peak_candidate(y, m, params, dense_idx);
                uint dist = dense_idx > primary_idx ? dense_idx - primary_idx : primary_idx - dense_idx;
                dist = min(dist, params.dense_n - dist);
                if (candidate.is_peak && dist > params.sep && candidate.value > best_value) {
                    best_value = candidate.value;
                    best_idx = candidate.dense_idx;
                }
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

        theta_p[row] = float(primary_idx) * params.pi_over_dense;
        m_p[row] = primary_value;

        const float ratio_den = max(primary_value, 1.0e-30f);
        const bool suppress = !has_secondary || (secondary_value / ratio_den) < params.tau_sec_floor;
        if (suppress) {
            theta_s[row] = as_type<float>(0x7fc00000u);
            m_s[row] = 0.0f;
        } else {
            theta_s[row] = float(secondary_idx) * params.pi_over_dense;
            m_s[row] = secondary_value;
        }
    }
}

inline float recovery_eval_segment_private(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint seg,
    float u
) {
    const uint k = params.k;
    const uint next = (seg + 1 == k) ? 0 : seg + 1;
    const float y0 = params.response_layout == 0
        ? response[ulong(row) * ulong(k) + ulong(seg)]
        : response[ulong(seg) * ulong(params.plane_size) + ulong(row)];
    const float y1 = params.response_layout == 0
        ? response[ulong(row) * ulong(k) + ulong(next)]
        : response[ulong(next) * ulong(params.plane_size) + ulong(row)];
    const float omt = 1.0f - u;
    const float omt2 = omt * omt;
    const float u2 = u * u;
    return y0 * omt + y1 * u +
        params.h2_over6 *
            (m[seg] * (omt2 * omt - omt) + m[next] * (u2 * u - u));
}

inline float recovery_eval_spline_private(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const ulong scaled = ulong(dense_idx) * ulong(params.k);
    const uint seg = uint(scaled / ulong(params.dense_n));
    const uint rem = uint(scaled - ulong(seg) * ulong(params.dense_n));
    return recovery_eval_segment_private(
        response, row, m, params, seg, float(rem) / float(params.dense_n));
}

inline RecoveryPeakCandidate recovery_dense_peak_candidate_private(
    device const float* response,
    uint row,
    thread const float* m,
    constant RecoveryParams& params,
    uint dense_idx
) {
    const uint left_idx = dense_idx == 0 ? params.dense_n - 1 : dense_idx - 1;
    const uint right_idx = dense_idx + 1 == params.dense_n ? 0 : dense_idx + 1;
    const float left_value = recovery_eval_spline_private(response, row, m, params, left_idx);
    const float center_value = recovery_eval_spline_private(response, row, m, params, dense_idx);
    const float right_value = recovery_eval_spline_private(response, row, m, params, right_idx);
    RecoveryPeakCandidate candidate;
    candidate.is_peak = center_value >= left_value && center_value >= right_value;
    candidate.dense_idx = dense_idx;
    candidate.value = center_value;
    return candidate;
}

inline float recovery_eval_near_segment_private(
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
    return recovery_eval_segment_private(
        response, row, m, params, uint(seg), float(rel) / float(params.dense_n));
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
        const float value = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(i)]
            : response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        ymin = min(ymin, value);
        ymax = max(ymax, value);
    }
    row_range[row] = ymax - ymin;

    for (uint i = 0; i < k; ++i) {
        const uint prev = (i == 0) ? k - 1 : i - 1;
        const uint next = (i + 1 == k) ? 0 : i + 1;
        const float y_prev = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(prev)]
            : response[ulong(prev) * ulong(params.plane_size) + ulong(row)];
        const float y_curr = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(i)]
            : response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        const float y_next = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(next)]
            : response[ulong(next) * ulong(params.plane_size) + ulong(row)];
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
        const float y_i = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(i)]
            : response[ulong(i) * ulong(params.plane_size) + ulong(row)];
        const float y_next = params.response_layout == 0
            ? response[ulong(row) * ulong(k) + ulong(next)]
            : response[ulong(next) * ulong(params.plane_size) + ulong(row)];
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
            if (u <= 0.0f || u >= 1.0f) {
                continue;
            }
            const float second_derivative = m[i] * (1.0f - u) + m[next] * u;
            if (second_derivative >= 0.0f) {
                continue;
            }
            const uint base_idx = recovery_dense_floor_idx(params, i, u);
            uint best_idx = base_idx;
            float best_value =
                recovery_eval_near_segment_private(response, row, m, params, i, base_idx);
            for (uint offset = 0; offset < 2; ++offset) {
                const uint dense_idx =
                    (base_idx + offset >= params.dense_n) ? base_idx + offset - params.dense_n : base_idx + offset;
                const float value =
                    recovery_eval_near_segment_private(response, row, m, params, i, dense_idx);
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
        primary_value = params.response_layout == 0
            ? response[ulong(row) * ulong(k)]
            : response[ulong(row)];
        primary_idx = 0;
    }

    float secondary_value = -INFINITY;
    uint secondary_idx = 0;
    bool has_secondary = false;
    for (uint pos = 1; pos < 7; ++pos) {
        if (top_values[pos] == -INFINITY) {
            break;
        }
        uint dist = top_indices[pos] > primary_idx
            ? top_indices[pos] - primary_idx
            : primary_idx - top_indices[pos];
        dist = min(dist, params.dense_n - dist);
        if (dist > params.sep) {
            const RecoveryPeakCandidate candidate =
                recovery_dense_peak_candidate_private(
                    response, row, m, params, top_indices[pos]);
            if (candidate.is_peak) {
                secondary_value = candidate.value;
                secondary_idx = candidate.dense_idx;
                has_secondary = true;
                break;
            }
        }
    }

    theta_p[row] = float(primary_idx) * params.pi_over_dense;
    m_p[row] = primary_value;

    const float ratio_den = max(primary_value, 1.0e-30f);
    const bool suppress = !has_secondary || (secondary_value / ratio_den) < params.tau_sec_floor;
    if (suppress) {
        theta_s[row] = as_type<float>(0x7fc00000u);
        m_s[row] = 0.0f;
    } else {
        theta_s[row] = float(secondary_idx) * params.pi_over_dense;
        m_s[row] = secondary_value;
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
"#;

#[repr(C)]
struct KernelParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
}

#[repr(C)]
struct LfParams {
    width: c_uint,
    height: c_uint,
    n_pixels: c_uint,
    n_offsets: c_uint,
    cos_t: c_float,
    sin_t: c_float,
}

#[repr(C)]
struct LfBatchParams {
    width: c_uint,
    height: c_uint,
    n_pixels: c_uint,
    n_thetas: c_uint,
    n_ms: c_uint,
    max_m: c_uint,
    n_samples: c_uint,
}

#[repr(C)]
struct LfStackParams {
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    n_samples: c_uint,
    border: c_uint,
    weight_sum: c_float,
}

#[repr(C)]
struct LfProjectParams {
    width: c_uint,
    height: c_uint,
    cos_t: c_float,
    sin_t: c_float,
}

#[repr(C)]
struct LfPlaneParams {
    width: c_uint,
    height: c_uint,
    n_samples: c_uint,
    theta_idx: c_uint,
    border: c_uint,
    weight_sum: c_float,
}

#[repr(C)]
struct LfBoxSeedParams {
    width: c_uint,
    height: c_uint,
    cos_t: c_float,
    sin_t: c_float,
}

#[repr(C)]
struct LfBoxPassParams {
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    key_min: c_int,
    line_count: c_uint,
}

#[repr(C)]
struct LfBoxFinalizeParams {
    width: c_uint,
    height: c_uint,
    theta_idx: c_uint,
}

#[repr(C)]
struct LfBoxMultiParams {
    width: c_uint,
    height: c_uint,
    n_ms: c_uint,
    output_layout: c_uint,
    key_min: c_int,
    line_count: c_uint,
    theta_idx: c_uint,
}

#[repr(C)]
struct LfScanlineParams {
    width: c_uint,
    height: c_uint,
    n_samples: c_uint,
    radius: c_uint,
    key_min: c_int,
    line_count: c_uint,
    theta_idx: c_uint,
    chunk_len: c_uint,
}

#[repr(C)]
struct RecoveryParams {
    n_rows: c_uint,
    k: c_uint,
    dense_n: c_uint,
    sep: c_uint,
    tau_sec_floor: c_float,
    h: c_float,
    h2_over6: c_float,
    rhs_scale: c_float,
    pi_over_dense: c_float,
    gamma_inv: c_float,
    cyclic_denom_inv: c_float,
    response_layout: c_uint,
    plane_size: c_uint,
}

#[repr(C)]
struct RecoveryReduceParams {
    count: c_uint,
    group_size: c_uint,
}

struct MetalState {
    device: Device,
    wvf_pipeline: ComputePipelineState,
    lf_pipeline: ComputePipelineState,
    lf_batch_pipeline: ComputePipelineState,
    lf_stack_interior_pipeline: ComputePipelineState,
    lf_stack_boundary_pipeline: ComputePipelineState,
    lf_project_pipeline: ComputePipelineState,
    lf_projected_interior_pipeline: ComputePipelineState,
    lf_projected_boundary_pipeline: ComputePipelineState,
    lf_box_seed_pipeline: ComputePipelineState,
    lf_box_x_pipeline: ComputePipelineState,
    lf_box_y_pipeline: ComputePipelineState,
    lf_box_finalize_pipeline: ComputePipelineState,
    lf_box_multi_x_pipeline: ComputePipelineState,
    lf_box_multi_y_pipeline: ComputePipelineState,
    lf_scanline_x_pipeline: ComputePipelineState,
    lf_scanline_y_pipeline: ComputePipelineState,
    recovery_pipeline: ComputePipelineState,
    recovery_reduce_pipeline: ComputePipelineState,
    recovery_validity_pipeline: ComputePipelineState,
    queue: CommandQueue,
}

impl MetalState {
    fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|err| format!("failed to compile Metal shader: {err}"))?;
        let function = library
            .get_function("wvf_convolve_pair", None)
            .map_err(|err| format!("failed to load Metal function: {err}"))?;
        let wvf_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("failed to create Metal compute pipeline: {err}"))?;
        let lf_function = library
            .get_function("lf_response", None)
            .map_err(|err| format!("failed to load LF Metal function: {err}"))?;
        let lf_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_function)
            .map_err(|err| format!("failed to create LF Metal compute pipeline: {err}"))?;
        let lf_batch_function = library
            .get_function("lf_response_batch", None)
            .map_err(|err| format!("failed to load batched LF Metal function: {err}"))?;
        let lf_batch_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_batch_function)
            .map_err(|err| format!("failed to create batched LF Metal compute pipeline: {err}"))?;
        let lf_stack_interior_function = library
            .get_function("lf_orientation_stack_interior", None)
            .map_err(|err| format!("failed to load LF stack interior Metal function: {err}"))?;
        let lf_stack_interior_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_stack_interior_function)
            .map_err(|err| {
                format!("failed to create LF stack interior Metal compute pipeline: {err}")
            })?;
        let lf_stack_boundary_function = library
            .get_function("lf_orientation_stack_boundary", None)
            .map_err(|err| format!("failed to load LF stack boundary Metal function: {err}"))?;
        let lf_stack_boundary_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_stack_boundary_function)
            .map_err(|err| {
                format!("failed to create LF stack boundary Metal compute pipeline: {err}")
            })?;
        let lf_project_function = library
            .get_function("lf_project_perp", None)
            .map_err(|err| format!("failed to load LF projection Metal function: {err}"))?;
        let lf_project_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_project_function)
            .map_err(|err| {
                format!("failed to create LF projection Metal compute pipeline: {err}")
            })?;
        let lf_projected_interior_function = library
            .get_function("lf_orientation_stack_projected_interior", None)
            .map_err(|err| format!("failed to load LF projected interior Metal function: {err}"))?;
        let lf_projected_interior_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_projected_interior_function)
            .map_err(|err| {
                format!("failed to create LF projected interior Metal compute pipeline: {err}")
            })?;
        let lf_projected_boundary_function = library
            .get_function("lf_orientation_stack_projected_boundary", None)
            .map_err(|err| format!("failed to load LF projected boundary Metal function: {err}"))?;
        let lf_projected_boundary_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_projected_boundary_function)
            .map_err(|err| {
                format!("failed to create LF projected boundary Metal compute pipeline: {err}")
            })?;
        let lf_box_seed_function = library
            .get_function("lf_box_seed", None)
            .map_err(|err| format!("failed to load LF box seed Metal function: {err}"))?;
        let lf_box_seed_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_seed_function)
            .map_err(|err| format!("failed to create LF box seed Metal compute pipeline: {err}"))?;
        let lf_box_x_function = library
            .get_function("lf_box_filter_x_major", None)
            .map_err(|err| format!("failed to load LF box x-major Metal function: {err}"))?;
        let lf_box_x_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_x_function)
            .map_err(|err| {
                format!("failed to create LF box x-major Metal compute pipeline: {err}")
            })?;
        let lf_box_y_function = library
            .get_function("lf_box_filter_y_major", None)
            .map_err(|err| format!("failed to load LF box y-major Metal function: {err}"))?;
        let lf_box_y_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_y_function)
            .map_err(|err| {
                format!("failed to create LF box y-major Metal compute pipeline: {err}")
            })?;
        let lf_box_finalize_function = library
            .get_function("lf_box_finalize", None)
            .map_err(|err| format!("failed to load LF box finalize Metal function: {err}"))?;
        let lf_box_finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_finalize_function)
            .map_err(|err| {
                format!("failed to create LF box finalize Metal compute pipeline: {err}")
            })?;
        let lf_box_multi_x_function = library
            .get_function("lf_box_multi_x_major", None)
            .map_err(|err| format!("failed to load LF box multi x-major Metal function: {err}"))?;
        let lf_box_multi_x_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_multi_x_function)
            .map_err(|err| {
                format!("failed to create LF box multi x-major Metal compute pipeline: {err}")
            })?;
        let lf_box_multi_y_function = library
            .get_function("lf_box_multi_y_major", None)
            .map_err(|err| format!("failed to load LF box multi y-major Metal function: {err}"))?;
        let lf_box_multi_y_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_box_multi_y_function)
            .map_err(|err| {
                format!("failed to create LF box multi y-major Metal compute pipeline: {err}")
            })?;
        let lf_scanline_x_function = library
            .get_function("lf_gaussian_scanline_x_major", None)
            .map_err(|err| format!("failed to load LF scanline x-major Metal function: {err}"))?;
        let lf_scanline_x_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_scanline_x_function)
            .map_err(|err| {
                format!("failed to create LF scanline x-major Metal compute pipeline: {err}")
            })?;
        let lf_scanline_y_function = library
            .get_function("lf_gaussian_scanline_y_major", None)
            .map_err(|err| format!("failed to load LF scanline y-major Metal function: {err}"))?;
        let lf_scanline_y_pipeline = device
            .new_compute_pipeline_state_with_function(&lf_scanline_y_function)
            .map_err(|err| {
                format!("failed to create LF scanline y-major Metal compute pipeline: {err}")
            })?;
        let recovery_function = library
            .get_function("recovery_peaks_private", None)
            .map_err(|err| format!("failed to load recovery Metal function: {err}"))?;
        let recovery_pipeline = device
            .new_compute_pipeline_state_with_function(&recovery_function)
            .map_err(|err| format!("failed to create recovery Metal compute pipeline: {err}"))?;
        let recovery_reduce_function = library
            .get_function("recovery_range_reduce", None)
            .map_err(|err| format!("failed to load recovery reduction Metal function: {err}"))?;
        let recovery_reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&recovery_reduce_function)
            .map_err(|err| {
                format!("failed to create recovery reduction Metal compute pipeline: {err}")
            })?;
        let recovery_validity_function = library
            .get_function("recovery_validity_flags", None)
            .map_err(|err| format!("failed to load recovery validity Metal function: {err}"))?;
        let recovery_validity_pipeline = device
            .new_compute_pipeline_state_with_function(&recovery_validity_function)
            .map_err(|err| {
                format!("failed to create recovery validity Metal compute pipeline: {err}")
            })?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            wvf_pipeline,
            lf_pipeline,
            lf_batch_pipeline,
            lf_stack_interior_pipeline,
            lf_stack_boundary_pipeline,
            lf_project_pipeline,
            lf_projected_interior_pipeline,
            lf_projected_boundary_pipeline,
            lf_box_seed_pipeline,
            lf_box_x_pipeline,
            lf_box_y_pipeline,
            lf_box_finalize_pipeline,
            lf_box_multi_x_pipeline,
            lf_box_multi_y_pipeline,
            lf_scanline_x_pipeline,
            lf_scanline_y_pipeline,
            recovery_pipeline,
            recovery_reduce_pipeline,
            recovery_validity_pipeline,
            queue,
        })
    }
}

thread_local! {
    static METAL_STATE: RefCell<Option<MetalState>> = RefCell::new(None);
    static LAST_RECOVERY_RANGE: RefCell<Option<(usize, c_float)>> = RefCell::new(None);
}

unsafe fn write_error(error_out: *mut c_char, error_len: usize, message: &str) {
    if error_out.is_null() || error_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let copy_len = bytes.len().min(error_len.saturating_sub(1));
    ptr::copy_nonoverlapping(bytes.as_ptr(), error_out.cast::<u8>(), copy_len);
    *error_out.add(copy_len) = 0;
}

unsafe fn check_ptr<T>(ptr_value: *const T, name: &str) -> Result<(), String> {
    if ptr_value.is_null() {
        Err(format!("{name} pointer is null"))
    } else {
        Ok(())
    }
}

unsafe fn check_mut_ptr<T>(ptr_value: *mut T, name: &str) -> Result<(), String> {
    if ptr_value.is_null() {
        Err(format!("{name} pointer is null"))
    } else {
        Ok(())
    }
}

fn checked_len(count: usize, element_size: usize, name: &str) -> Result<usize, String> {
    count
        .checked_mul(element_size)
        .ok_or_else(|| format!("{name} byte length overflowed"))
}

fn checked_image_pixels(width: c_uint, height: c_uint) -> Result<usize, String> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "image dimensions overflowed".to_string())
}

fn threadgroup_1d(pipeline: &ComputePipelineState) -> MTLSize {
    let execution_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    threadgroup_1d_with_cap(execution_width, max_threads, 256)
}

fn threadgroup_1d_with_cap(execution_width: u64, max_threads: u64, cap: u64) -> MTLSize {
    let mut width = max_threads.min(cap);
    width = (width / execution_width).max(1) * execution_width;
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn threadgroup_2d(pipeline: &ComputePipelineState) -> MTLSize {
    let execution_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let width = execution_width.min(max_threads);
    let height = (max_threads / width).clamp(1, 16);
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn round_half_to_even(value: f64) -> Result<i32, String> {
    if !value.is_finite() {
        return Err("LF offset value is not finite".to_string());
    }
    let floor_value = value.floor();
    let fraction = value - floor_value;
    let rounded = if fraction < 0.5 {
        floor_value
    } else if fraction > 0.5 {
        floor_value + 1.0
    } else {
        let floor_i = floor_value as i64;
        if floor_i % 2 == 0 {
            floor_value
        } else {
            floor_value + 1.0
        }
    };
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("LF offset is outside int32 range".to_string());
    }
    Ok(rounded as i32)
}

fn effective_m(m: c_int) -> Result<usize, String> {
    if m <= 0 {
        return Ok(0);
    }
    Ok(usize::try_from(m).map_err(|_| "m is outside supported range".to_string())?)
}

fn build_lf_single_tables(
    theta: c_double,
    m: c_int,
) -> Result<(Vec<c_int>, Vec<c_int>, Vec<c_float>, c_float, c_float), String> {
    if !theta.is_finite() {
        return Err("theta must be finite".to_string());
    }
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let m_value = effective_m(m)?;
    if m_value == 0 {
        return Ok((
            vec![0],
            vec![0],
            vec![1.0],
            cos_t as c_float,
            sin_t as c_float,
        ));
    }

    let n_samples = m_value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF sample count overflowed".to_string())?;
    let max_trig = cos_t.abs().max(sin_t.abs());
    let step = if max_trig > 0.0 { 1.0 / max_trig } else { 1.0 };
    let sigma = m_value as f64 / 2.0;
    let mut dx = Vec::with_capacity(n_samples);
    let mut dy = Vec::with_capacity(n_samples);
    let mut weights = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
        let j = sample_idx as isize - m_value as isize;
        let offset = j as f64 * step;
        dx.push(round_half_to_even(offset * cos_t)? as c_int);
        dy.push(round_half_to_even(offset * sin_t)? as c_int);
        weights.push((-0.5 * (j as f64 / sigma).powi(2)).exp() as c_float);
    }

    Ok((dx, dy, weights, cos_t as c_float, sin_t as c_float))
}

fn build_lf_batch_tables(
    thetas: &[c_double],
    ms: &[c_int],
) -> Result<
    (
        Vec<c_int>,
        Vec<c_int>,
        Vec<c_float>,
        Vec<c_float>,
        Vec<c_float>,
        usize,
        usize,
    ),
    String,
> {
    let max_m = ms
        .iter()
        .map(|&m| effective_m(m))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let n_samples = max_m
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF batch sample count overflowed".to_string())?;
    let mut dx = Vec::with_capacity(
        thetas
            .len()
            .checked_mul(n_samples)
            .ok_or_else(|| "LF batch offset count overflowed".to_string())?,
    );
    let mut dy = Vec::with_capacity(dx.capacity());
    let mut cos_values = Vec::with_capacity(thetas.len());
    let mut sin_values = Vec::with_capacity(thetas.len());

    for &theta in thetas {
        if !theta.is_finite() {
            return Err("theta values must be finite".to_string());
        }
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        cos_values.push(cos_t as c_float);
        sin_values.push(sin_t as c_float);
        let max_trig = cos_t.abs().max(sin_t.abs());
        let step = if max_trig > 0.0 { 1.0 / max_trig } else { 1.0 };

        for sample_idx in 0..n_samples {
            let j = sample_idx as isize - max_m as isize;
            let offset = j as f64 * step;
            dx.push(round_half_to_even(offset * cos_t)? as c_int);
            dy.push(round_half_to_even(offset * sin_t)? as c_int);
        }
    }

    let mut weights = vec![
        0.0;
        ms.len()
            .checked_mul(n_samples)
            .ok_or_else(|| "LF batch weight count overflowed".to_string())?
    ];
    for (m_idx, &m_raw) in ms.iter().enumerate() {
        let m_value = effective_m(m_raw)?;
        if m_value == 0 {
            weights[m_idx * n_samples + max_m] = 1.0;
            continue;
        }
        let sigma = m_value as f64 / 2.0;
        for sample_idx in 0..n_samples {
            let j = sample_idx as isize - max_m as isize;
            if j.unsigned_abs() <= m_value {
                weights[m_idx * n_samples + sample_idx] =
                    (-0.5 * (j as f64 / sigma).powi(2)).exp() as c_float;
            }
        }
    }

    Ok((dx, dy, weights, cos_values, sin_values, max_m, n_samples))
}

fn build_lf_stack_tables(
    n_orientations: c_uint,
    m: c_int,
) -> Result<
    (
        Vec<c_int>,
        Vec<c_int>,
        Vec<c_float>,
        Vec<c_float>,
        Vec<c_float>,
        usize,
        usize,
        c_float,
    ),
    String,
> {
    if n_orientations == 0 {
        return Err("n_orientations must be positive".to_string());
    }

    let m_value = effective_m(m)?;
    let n_samples = m_value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF stack sample count overflowed".to_string())?;
    let orientation_count = n_orientations as usize;
    let offset_count = orientation_count
        .checked_mul(n_samples)
        .ok_or_else(|| "LF stack offset count overflowed".to_string())?;
    let mut dx = Vec::with_capacity(offset_count);
    let mut dy = Vec::with_capacity(offset_count);
    let mut cos_values = Vec::with_capacity(orientation_count);
    let mut sin_values = Vec::with_capacity(orientation_count);

    for theta_idx in 0..orientation_count {
        let theta = std::f64::consts::PI * theta_idx as f64 / orientation_count as f64;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        cos_values.push(cos_t as c_float);
        sin_values.push(sin_t as c_float);

        if m_value == 0 {
            dx.push(0);
            dy.push(0);
            continue;
        }

        let max_trig = cos_t.abs().max(sin_t.abs());
        let step = if max_trig > 0.0 { 1.0 / max_trig } else { 1.0 };
        for sample_idx in 0..n_samples {
            let j = sample_idx as isize - m_value as isize;
            let offset = j as f64 * step;
            dx.push(round_half_to_even(offset * cos_t)? as c_int);
            dy.push(round_half_to_even(offset * sin_t)? as c_int);
        }
    }

    let mut weights = Vec::with_capacity(n_samples);
    if m_value == 0 {
        weights.push(1.0);
    } else {
        let sigma = m_value as f64 / 2.0;
        for sample_idx in 0..n_samples {
            let j = sample_idx as isize - m_value as isize;
            weights.push((-0.5 * (j as f64 / sigma).powi(2)).exp() as c_float);
        }
    }
    let weight_sum = weights.iter().copied().sum();

    Ok((
        dx, dy, weights, cos_values, sin_values, n_samples, m_value, weight_sum,
    ))
}

fn build_lf_gaussian_weights(m: c_int) -> Result<(Vec<c_float>, usize), String> {
    let m_value = effective_m(m)?;
    let n_samples = m_value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF scanline sample count overflowed".to_string())?;
    let mut weights = Vec::with_capacity(n_samples);
    if m_value == 0 {
        weights.push(1.0);
    } else {
        let sigma = m_value as f64 / 2.0;
        for sample_idx in 0..n_samples {
            let j = sample_idx as isize - m_value as isize;
            weights.push((-0.5 * (j as f64 / sigma).powi(2)).exp() as c_float);
        }
    }
    Ok((weights, m_value))
}

fn box_radius_for_m(
    m_value: usize,
    box_passes: c_uint,
    box_radius: c_int,
) -> Result<c_uint, String> {
    if box_passes == 0 || box_passes > MAX_BOX_PASSES {
        return Err(format!("box_passes must be between 1 and {MAX_BOX_PASSES}"));
    }
    if box_radius >= 0 {
        return c_uint::try_from(box_radius)
            .map_err(|_| "box_radius must fit in uint32".to_string());
    }
    if m_value == 0 {
        return Ok(0);
    }

    let sigma = m_value as f64 / 2.0;
    let radius = ((1.0 + 12.0 * sigma * sigma / box_passes as f64).sqrt() - 1.0) / 2.0;
    let rounded = radius.round().max(1.0);
    if rounded > c_uint::MAX as f64 {
        return Err("computed box_radius is outside uint32 range".to_string());
    }
    Ok(rounded as c_uint)
}

fn build_lf_box_line_offsets(
    width: c_uint,
    height: c_uint,
    theta_idx: usize,
    n_orientations: usize,
) -> Result<(Vec<c_int>, bool, c_int, c_uint, c_float, c_float), String> {
    let theta = std::f64::consts::PI * theta_idx as f64 / n_orientations as f64;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let x_major = cos_t.abs() >= sin_t.abs();
    let axis_len = if x_major { width } else { height } as usize;
    let slope = if x_major {
        sin_t / cos_t
    } else {
        cos_t / sin_t
    };
    let mut offsets = Vec::with_capacity(axis_len);

    for axis_idx in 0..axis_len {
        offsets.push(round_half_to_even(slope * axis_idx as f64)? as c_int);
    }

    let min_offset = offsets.iter().copied().min().unwrap_or(0) as i64;
    let max_offset = offsets.iter().copied().max().unwrap_or(0) as i64;
    let transverse = if x_major { height } else { width } as i64;
    let key_min = -max_offset;
    let key_max = transverse - 1 - min_offset;
    let line_count = key_max
        .checked_sub(key_min)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF box line count overflowed".to_string())?;
    if key_min < c_int::MIN as i64 || key_min > c_int::MAX as i64 {
        return Err("LF box line key is outside int32 range".to_string());
    }
    if line_count < 0 || line_count > c_uint::MAX as i64 {
        return Err("LF box line count is outside uint32 range".to_string());
    }

    Ok((
        offsets,
        x_major,
        key_min as c_int,
        line_count as c_uint,
        cos_t as c_float,
        sin_t as c_float,
    ))
}

fn build_recovery_solver(
    k: usize,
) -> Result<(Vec<c_float>, Vec<c_float>, Vec<c_float>, c_float), String> {
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if k < 3 {
        return Ok((vec![0.0; k], vec![0.0; k], vec![0.0; k], 0.0));
    }

    let gamma = -4.0f64;
    let gamma_inv = 1.0f64 / gamma;
    let mut cprime = vec![0.0f64; k];
    let mut inv_denom = vec![0.0f64; k];

    let first_diag = 4.0 - gamma;
    inv_denom[0] = 1.0 / first_diag;
    cprime[0] = inv_denom[0];
    for i in 1..k {
        let diag = if i + 1 == k { 4.0 - gamma_inv } else { 4.0 };
        let denom = diag - cprime[i - 1];
        if denom.abs() <= f64::EPSILON {
            return Err("recovery spline solver is singular".to_string());
        }
        inv_denom[i] = 1.0 / denom;
        cprime[i] = if i + 1 == k { 0.0 } else { inv_denom[i] };
    }

    let mut z = vec![0.0f64; k];
    z[0] = gamma * inv_denom[0];
    for i in 1..k {
        let rhs = if i + 1 == k { 1.0 } else { 0.0 };
        z[i] = (rhs - z[i - 1]) * inv_denom[i];
    }
    for i in (0..k - 1).rev() {
        z[i] -= cprime[i] * z[i + 1];
    }

    let cyclic_denom = 1.0 + z[0] + gamma_inv * z[k - 1];
    if cyclic_denom.abs() <= f64::EPSILON {
        return Err("recovery cyclic correction is singular".to_string());
    }

    Ok((
        cprime.into_iter().map(|value| value as c_float).collect(),
        inv_denom
            .into_iter()
            .map(|value| value as c_float)
            .collect(),
        z.into_iter().map(|value| value as c_float).collect(),
        (1.0 / cyclic_denom) as c_float,
    ))
}

unsafe fn run_convolve_pair_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let offset_len = checked_len(n_offsets as usize, std::mem::size_of::<c_int>(), "offset")?;
    let weight_len = checked_len(n_offsets as usize, std::mem::size_of::<c_float>(), "weight")?;
    let output_len = image_len;

    let resource_options = MTLResourceOptions::StorageModeShared;
    let image_buffer = state.device.new_buffer_with_bytes_no_copy(
        image.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let wx_buffer = state.device.new_buffer_with_bytes_no_copy(
        wx.cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let wy_buffer = state.device.new_buffer_with_bytes_no_copy(
        wy.cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let out_x_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_x.cast::<std::ffi::c_void>().cast_const(),
        output_len as u64,
        resource_options,
        None,
    );
    let out_y_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_y.cast::<std::ffi::c_void>().cast_const(),
        output_len as u64,
        resource_options,
        None,
    );
    let params = KernelParams {
        width,
        height,
        n_offsets,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const KernelParams).cast(),
        std::mem::size_of::<KernelParams>() as u64,
        resource_options,
    );

    image_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    wx_buffer.did_modify_range(NSRange::new(0, weight_len as u64));
    wy_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.wvf_pipeline);
    encoder.set_buffer(0, Some(&image_buffer), 0);
    encoder.set_buffer(1, Some(&dx_buffer), 0);
    encoder.set_buffer(2, Some(&dy_buffer), 0);
    encoder.set_buffer(3, Some(&wx_buffer), 0);
    encoder.set_buffer(4, Some(&wy_buffer), 0);
    encoder.set_buffer(5, Some(&out_x_buffer), 0);
    encoder.set_buffer(6, Some(&out_y_buffer), 0);
    encoder.set_buffer(7, Some(&params_buffer), 0);

    let threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let group = threadgroup_2d(&state.wvf_pipeline);
    encoder.dispatch_threads(threads, group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_convolve_pair(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    check_ptr(image, "image")?;
    check_ptr(dx, "dx")?;
    check_ptr(dy, "dy")?;
    check_ptr(wx, "wx")?;
    check_ptr(wy, "wy")?;
    check_mut_ptr(out_x, "out_x")?;
    check_mut_ptr(out_y, "out_y")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_offsets == 0 {
        return Err("n_offsets must be positive".to_string());
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_convolve_pair_with_state(
            state, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
        )
    })
}

unsafe fn run_lf_response_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    theta: c_double,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    if n_pixels == 0 {
        return Ok(());
    }

    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let coord_len = checked_len(
        n_pixels as usize,
        std::mem::size_of::<c_int>(),
        "coordinate",
    )?;
    let out_len = checked_len(n_pixels as usize, std::mem::size_of::<c_float>(), "output")?;
    let (dx, dy, weights, cos_t, sin_t) = build_lf_single_tables(theta, m)?;
    let n_offsets = c_uint::try_from(dx.len())
        .map_err(|_| "LF offset count is outside uint32 range".to_string())?;
    let offset_len = checked_len(dx.len(), std::mem::size_of::<c_int>(), "LF offset")?;
    let weight_len = checked_len(weights.len(), std::mem::size_of::<c_float>(), "LF weight")?;

    let resource_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let px_buffer = state.device.new_buffer_with_bytes_no_copy(
        px.cast(),
        coord_len as u64,
        resource_options,
        None,
    );
    let py_buffer = state.device.new_buffer_with_bytes_no_copy(
        py.cast(),
        coord_len as u64,
        resource_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let weights_buffer = state.device.new_buffer_with_bytes_no_copy(
        weights.as_ptr().cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        resource_options,
        None,
    );
    let params = LfParams {
        width,
        height,
        n_pixels,
        n_offsets,
        cos_t,
        sin_t,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const LfParams).cast(),
        std::mem::size_of::<LfParams>() as u64,
        resource_options,
    );

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    px_buffer.did_modify_range(NSRange::new(0, coord_len as u64));
    py_buffer.did_modify_range(NSRange::new(0, coord_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    weights_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.lf_pipeline);
    encoder.set_buffer(0, Some(&gx_buffer), 0);
    encoder.set_buffer(1, Some(&gy_buffer), 0);
    encoder.set_buffer(2, Some(&px_buffer), 0);
    encoder.set_buffer(3, Some(&py_buffer), 0);
    encoder.set_buffer(4, Some(&dx_buffer), 0);
    encoder.set_buffer(5, Some(&dy_buffer), 0);
    encoder.set_buffer(6, Some(&weights_buffer), 0);
    encoder.set_buffer(7, Some(&out_buffer), 0);
    encoder.set_buffer(8, Some(&params_buffer), 0);

    let threads = MTLSize {
        width: n_pixels as u64,
        height: 1,
        depth: 1,
    };
    let group = threadgroup_1d(&state.lf_pipeline);
    encoder.dispatch_threads(threads, group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_lf_response_cpu(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    theta: c_double,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    if n_pixels == 0 {
        return Ok(());
    }

    let total_pixels = checked_image_pixels(width, height)?;
    let gx_slice = std::slice::from_raw_parts(g_x, total_pixels);
    let gy_slice = std::slice::from_raw_parts(g_y, total_pixels);
    let px_slice = std::slice::from_raw_parts(px, n_pixels as usize);
    let py_slice = std::slice::from_raw_parts(py, n_pixels as usize);
    let out_slice = std::slice::from_raw_parts_mut(out, n_pixels as usize);
    let (dx, dy, weights, cos_t, sin_t) = build_lf_single_tables(theta, m)?;
    let width_i = i64::from(width);
    let height_i = i64::from(height);
    let width_usize = width as usize;

    for pixel_idx in 0..n_pixels as usize {
        let x0 = i64::from(px_slice[pixel_idx]);
        let y0 = i64::from(py_slice[pixel_idx]);
        let mut num = 0.0f32;
        let mut den = 0.0f32;

        for sample_idx in 0..dx.len() {
            let x = x0 + i64::from(dx[sample_idx]);
            let y = y0 + i64::from(dy[sample_idx]);
            if x < 0 || y < 0 || x >= width_i || y >= height_i {
                continue;
            }

            let image_idx = y as usize * width_usize + x as usize;
            let sample = -sin_t * gx_slice[image_idx] + cos_t * gy_slice[image_idx];
            let weight = weights[sample_idx];
            num += sample * weight;
            den += weight;
        }

        out_slice[pixel_idx] = if den > 0.0 { (num / den).abs() } else { 0.0 };
    }

    Ok(())
}

unsafe fn run_lf_response(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    theta: c_double,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_ptr(px, "px")?;
    check_ptr(py, "py")?;
    check_mut_ptr(out, "out")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }

    let m_value = effective_m(m)?;
    let n_samples = m_value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "LF sample count overflowed".to_string())?;
    let work_items = (n_pixels as usize)
        .checked_mul(n_samples)
        .ok_or_else(|| "LF work item count overflowed".to_string())?;
    if work_items <= 262_144 {
        return run_lf_response_cpu(g_x, g_y, width, height, px, py, n_pixels, theta, m, out);
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_lf_response_with_state(
            state, g_x, g_y, width, height, px, py, n_pixels, theta, m, out,
        )
    })
}

unsafe fn run_lf_response_batch_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    thetas: *const c_double,
    n_thetas: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    out: *mut c_float,
) -> Result<(), String> {
    if n_pixels == 0 || n_thetas == 0 || n_ms == 0 {
        return Ok(());
    }
    if n_ms as usize > MAX_BATCH_MS {
        return Err(format!(
            "batched LF supports at most {MAX_BATCH_MS} m values"
        ));
    }

    let theta_slice = std::slice::from_raw_parts(thetas, n_thetas as usize);
    let m_slice = std::slice::from_raw_parts(ms, n_ms as usize);
    let (dx, dy, weights, cos_values, sin_values, max_m, n_samples) =
        build_lf_batch_tables(theta_slice, m_slice)?;

    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let coord_len = checked_len(
        n_pixels as usize,
        std::mem::size_of::<c_int>(),
        "coordinate",
    )?;
    let theta_len = checked_len(n_thetas as usize, std::mem::size_of::<c_float>(), "theta")?;
    let offset_len = checked_len(dx.len(), std::mem::size_of::<c_int>(), "LF batch offset")?;
    let weight_len = checked_len(
        weights.len(),
        std::mem::size_of::<c_float>(),
        "LF batch weight",
    )?;
    let output_count = (n_thetas as usize)
        .checked_mul(n_ms as usize)
        .and_then(|value| value.checked_mul(n_pixels as usize))
        .ok_or_else(|| "LF batch output count overflowed".to_string())?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF batch output",
    )?;
    let total_threads = (n_thetas as usize)
        .checked_mul(n_pixels as usize)
        .ok_or_else(|| "LF batch thread count overflowed".to_string())?;

    let params = LfBatchParams {
        width,
        height,
        n_pixels,
        n_thetas,
        n_ms,
        max_m: c_uint::try_from(max_m)
            .map_err(|_| "LF batch max_m is outside uint32 range".to_string())?,
        n_samples: c_uint::try_from(n_samples)
            .map_err(|_| "LF batch sample count is outside uint32 range".to_string())?,
    };

    let resource_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let px_buffer = state.device.new_buffer_with_bytes_no_copy(
        px.cast(),
        coord_len as u64,
        resource_options,
        None,
    );
    let py_buffer = state.device.new_buffer_with_bytes_no_copy(
        py.cast(),
        coord_len as u64,
        resource_options,
        None,
    );
    let cos_buffer = state.device.new_buffer_with_bytes_no_copy(
        cos_values.as_ptr().cast(),
        theta_len as u64,
        resource_options,
        None,
    );
    let sin_buffer = state.device.new_buffer_with_bytes_no_copy(
        sin_values.as_ptr().cast(),
        theta_len as u64,
        resource_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let weights_buffer = state.device.new_buffer_with_bytes_no_copy(
        weights.as_ptr().cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        resource_options,
        None,
    );
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const LfBatchParams).cast(),
        std::mem::size_of::<LfBatchParams>() as u64,
        resource_options,
    );

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    px_buffer.did_modify_range(NSRange::new(0, coord_len as u64));
    py_buffer.did_modify_range(NSRange::new(0, coord_len as u64));
    cos_buffer.did_modify_range(NSRange::new(0, theta_len as u64));
    sin_buffer.did_modify_range(NSRange::new(0, theta_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    weights_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.lf_batch_pipeline);
    encoder.set_buffer(0, Some(&gx_buffer), 0);
    encoder.set_buffer(1, Some(&gy_buffer), 0);
    encoder.set_buffer(2, Some(&px_buffer), 0);
    encoder.set_buffer(3, Some(&py_buffer), 0);
    encoder.set_buffer(4, Some(&cos_buffer), 0);
    encoder.set_buffer(5, Some(&sin_buffer), 0);
    encoder.set_buffer(6, Some(&dx_buffer), 0);
    encoder.set_buffer(7, Some(&dy_buffer), 0);
    encoder.set_buffer(8, Some(&weights_buffer), 0);
    encoder.set_buffer(9, Some(&out_buffer), 0);
    encoder.set_buffer(10, Some(&params_buffer), 0);

    let threads = MTLSize {
        width: total_threads as u64,
        height: 1,
        depth: 1,
    };
    let group = threadgroup_1d(&state.lf_batch_pipeline);
    encoder.dispatch_threads(threads, group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_lf_response_batch(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    thetas: *const c_double,
    n_thetas: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_ptr(px, "px")?;
    check_ptr(py, "py")?;
    check_ptr(thetas, "thetas")?;
    check_ptr(ms, "ms")?;
    check_mut_ptr(out, "out")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_lf_response_batch_with_state(
            state, g_x, g_y, width, height, px, py, n_pixels, thetas, n_thetas, ms, n_ms, out,
        )
    })
}

unsafe fn run_lf_orientation_stack_direct_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let output_count = total_pixels
        .checked_mul(n_orientations as usize)
        .ok_or_else(|| "LF stack output count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF stack output",
    )?;
    let (dx, dy, weights, cos_values, sin_values, n_samples, border, weight_sum) =
        build_lf_stack_tables(n_orientations, m)?;
    let orientation_len = checked_len(
        n_orientations as usize,
        std::mem::size_of::<c_float>(),
        "orientation",
    )?;
    let offset_len = checked_len(dx.len(), std::mem::size_of::<c_int>(), "LF stack offset")?;
    let weight_len = checked_len(
        weights.len(),
        std::mem::size_of::<c_float>(),
        "LF stack weight",
    )?;
    let params = LfStackParams {
        width,
        height,
        n_orientations,
        n_samples: c_uint::try_from(n_samples)
            .map_err(|_| "LF stack sample count is outside uint32 range".to_string())?,
        border: c_uint::try_from(border)
            .map_err(|_| "LF stack border is outside uint32 range".to_string())?,
        weight_sum,
    };

    let resource_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let cos_buffer = state.device.new_buffer_with_bytes_no_copy(
        cos_values.as_ptr().cast(),
        orientation_len as u64,
        resource_options,
        None,
    );
    let sin_buffer = state.device.new_buffer_with_bytes_no_copy(
        sin_values.as_ptr().cast(),
        orientation_len as u64,
        resource_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let weights_buffer = state.device.new_buffer_with_bytes_no_copy(
        weights.as_ptr().cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        resource_options,
        None,
    );
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const LfStackParams).cast(),
        std::mem::size_of::<LfStackParams>() as u64,
        resource_options,
    );

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    cos_buffer.did_modify_range(NSRange::new(0, orientation_len as u64));
    sin_buffer.did_modify_range(NSRange::new(0, orientation_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    weights_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.lf_stack_interior_pipeline);
    encoder.set_buffer(0, Some(&gx_buffer), 0);
    encoder.set_buffer(1, Some(&gy_buffer), 0);
    encoder.set_buffer(2, Some(&cos_buffer), 0);
    encoder.set_buffer(3, Some(&sin_buffer), 0);
    encoder.set_buffer(4, Some(&dx_buffer), 0);
    encoder.set_buffer(5, Some(&dy_buffer), 0);
    encoder.set_buffer(6, Some(&weights_buffer), 0);
    encoder.set_buffer(7, Some(&out_buffer), 0);
    encoder.set_buffer(8, Some(&params_buffer), 0);

    let threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: n_orientations as u64,
    };
    let group = threadgroup_2d(&state.lf_stack_interior_pipeline);
    encoder.dispatch_threads(threads, group);

    if border > 0 {
        encoder.set_compute_pipeline_state(&state.lf_stack_boundary_pipeline);
        encoder.set_buffer(0, Some(&gx_buffer), 0);
        encoder.set_buffer(1, Some(&gy_buffer), 0);
        encoder.set_buffer(2, Some(&cos_buffer), 0);
        encoder.set_buffer(3, Some(&sin_buffer), 0);
        encoder.set_buffer(4, Some(&dx_buffer), 0);
        encoder.set_buffer(5, Some(&dy_buffer), 0);
        encoder.set_buffer(6, Some(&weights_buffer), 0);
        encoder.set_buffer(7, Some(&out_buffer), 0);
        encoder.set_buffer(8, Some(&params_buffer), 0);
        let group = threadgroup_2d(&state.lf_stack_boundary_pipeline);
        encoder.dispatch_threads(threads, group);
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_lf_orientation_stack_projected_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let output_count = total_pixels
        .checked_mul(n_orientations as usize)
        .ok_or_else(|| "LF stack output count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF stack output",
    )?;
    let (dx, dy, weights, cos_values, sin_values, n_samples, border, weight_sum) =
        build_lf_stack_tables(n_orientations, m)?;
    let offset_len = checked_len(dx.len(), std::mem::size_of::<c_int>(), "LF stack offset")?;
    let weight_len = checked_len(
        weights.len(),
        std::mem::size_of::<c_float>(),
        "LF stack weight",
    )?;

    let resource_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        resource_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.as_ptr().cast(),
        offset_len as u64,
        resource_options,
        None,
    );
    let weights_buffer = state.device.new_buffer_with_bytes_no_copy(
        weights.as_ptr().cast(),
        weight_len as u64,
        resource_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        resource_options,
        None,
    );
    let perp_buffer = state
        .device
        .new_buffer(image_len as u64, MTLResourceOptions::StorageModePrivate);

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    weights_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let mut project_params_buffers = Vec::with_capacity(n_orientations as usize);
    let mut plane_params_buffers = Vec::with_capacity(n_orientations as usize);
    for theta_idx in 0..n_orientations as usize {
        let project_params = LfProjectParams {
            width,
            height,
            cos_t: cos_values[theta_idx],
            sin_t: sin_values[theta_idx],
        };
        project_params_buffers.push(state.device.new_buffer_with_data(
            (&project_params as *const LfProjectParams).cast(),
            std::mem::size_of::<LfProjectParams>() as u64,
            resource_options,
        ));

        let plane_params = LfPlaneParams {
            width,
            height,
            n_samples: c_uint::try_from(n_samples)
                .map_err(|_| "LF stack sample count is outside uint32 range".to_string())?,
            theta_idx: c_uint::try_from(theta_idx)
                .map_err(|_| "LF stack theta index is outside uint32 range".to_string())?,
            border: c_uint::try_from(border)
                .map_err(|_| "LF stack border is outside uint32 range".to_string())?,
            weight_sum,
        };
        plane_params_buffers.push(state.device.new_buffer_with_data(
            (&plane_params as *const LfPlaneParams).cast(),
            std::mem::size_of::<LfPlaneParams>() as u64,
            resource_options,
        ));
    }

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let project_group = threadgroup_2d(&state.lf_project_pipeline);
    let interior_group = threadgroup_2d(&state.lf_projected_interior_pipeline);
    let boundary_group = threadgroup_2d(&state.lf_projected_boundary_pipeline);

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    for theta_idx in 0..n_orientations as usize {
        encoder.set_compute_pipeline_state(&state.lf_project_pipeline);
        encoder.set_buffer(0, Some(&gx_buffer), 0);
        encoder.set_buffer(1, Some(&gy_buffer), 0);
        encoder.set_buffer(2, Some(&perp_buffer), 0);
        encoder.set_buffer(3, Some(&project_params_buffers[theta_idx]), 0);
        encoder.dispatch_threads(image_threads, project_group);
        encoder.memory_barrier_with_resources(&[perp_buffer.as_ref()]);

        encoder.set_compute_pipeline_state(&state.lf_projected_interior_pipeline);
        encoder.set_buffer(0, Some(&perp_buffer), 0);
        encoder.set_buffer(1, Some(&dx_buffer), 0);
        encoder.set_buffer(2, Some(&dy_buffer), 0);
        encoder.set_buffer(3, Some(&weights_buffer), 0);
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_buffer(5, Some(&plane_params_buffers[theta_idx]), 0);
        encoder.dispatch_threads(image_threads, interior_group);

        if border > 0 {
            encoder.set_compute_pipeline_state(&state.lf_projected_boundary_pipeline);
            encoder.set_buffer(0, Some(&perp_buffer), 0);
            encoder.set_buffer(1, Some(&dx_buffer), 0);
            encoder.set_buffer(2, Some(&dy_buffer), 0);
            encoder.set_buffer(3, Some(&weights_buffer), 0);
            encoder.set_buffer(4, Some(&out_buffer), 0);
            encoder.set_buffer(5, Some(&plane_params_buffers[theta_idx]), 0);
            encoder.dispatch_threads(image_threads, boundary_group);
        }
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_lf_orientation_stack_scanline_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let output_count = total_pixels
        .checked_mul(n_orientations as usize)
        .ok_or_else(|| "LF scanline output count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF scanline output",
    )?;
    let (weights, m_value) = build_lf_gaussian_weights(m)?;
    let n_samples = weights.len();
    let weight_len = checked_len(
        weights.len(),
        std::mem::size_of::<c_float>(),
        "LF scanline weight",
    )?;

    let shared_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let weights_buffer = state.device.new_buffer_with_bytes_no_copy(
        weights.as_ptr().cast(),
        weight_len as u64,
        shared_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        shared_options,
        None,
    );
    let g_perp_buffer = state
        .device
        .new_buffer(image_len as u64, MTLResourceOptions::StorageModePrivate);

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    weights_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let project_group = threadgroup_2d(&state.lf_project_pipeline);
    let scanline_x_group = threadgroup_1d_with_cap(
        state.lf_scanline_x_pipeline.thread_execution_width().max(1),
        state
            .lf_scanline_x_pipeline
            .max_total_threads_per_threadgroup()
            .max(1),
        1024,
    );
    let scanline_y_group = threadgroup_1d_with_cap(
        state.lf_scanline_y_pipeline.thread_execution_width().max(1),
        state
            .lf_scanline_y_pipeline
            .max_total_threads_per_threadgroup()
            .max(1),
        1024,
    );

    for theta_idx in 0..n_orientations as usize {
        let (line_offsets, x_major, key_min, line_count, cos_t, sin_t) =
            build_lf_box_line_offsets(width, height, theta_idx, n_orientations as usize)?;
        let offset_len = checked_len(
            line_offsets.len(),
            std::mem::size_of::<c_int>(),
            "LF scanline line offset",
        )?;
        let line_offsets_buffer = state.device.new_buffer_with_bytes_no_copy(
            line_offsets.as_ptr().cast(),
            offset_len as u64,
            shared_options,
            None,
        );
        line_offsets_buffer.did_modify_range(NSRange::new(0, offset_len as u64));

        let seed_params = LfProjectParams {
            width,
            height,
            cos_t,
            sin_t,
        };
        let seed_params_buffer = state.device.new_buffer_with_data(
            (&seed_params as *const LfProjectParams).cast(),
            std::mem::size_of::<LfProjectParams>() as u64,
            shared_options,
        );

        let group = if x_major {
            scanline_x_group
        } else {
            scanline_y_group
        };
        let chunk_len = c_uint::try_from(group.width)
            .map_err(|_| "LF scanline chunk length is outside uint32 range".to_string())?;
        let axis_len = if x_major { width } else { height };
        let n_chunks = axis_len.div_ceil(chunk_len);
        let tile_len = (chunk_len as usize)
            .checked_add(
                m_value
                    .checked_mul(2)
                    .ok_or_else(|| "LF scanline tile length overflowed".to_string())?,
            )
            .ok_or_else(|| "LF scanline tile length overflowed".to_string())?;
        let tile_bytes = checked_len(tile_len, std::mem::size_of::<c_float>(), "LF scanline tile")?;
        let params = LfScanlineParams {
            width,
            height,
            n_samples: c_uint::try_from(n_samples)
                .map_err(|_| "LF scanline sample count is outside uint32 range".to_string())?,
            radius: c_uint::try_from(m_value)
                .map_err(|_| "LF scanline radius is outside uint32 range".to_string())?,
            key_min,
            line_count,
            theta_idx: c_uint::try_from(theta_idx)
                .map_err(|_| "LF scanline theta index is outside uint32 range".to_string())?,
            chunk_len,
        };
        let params_buffer = state.device.new_buffer_with_data(
            (&params as *const LfScanlineParams).cast(),
            std::mem::size_of::<LfScanlineParams>() as u64,
            shared_options,
        );

        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("LF Gaussian scanline orientation stack");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("LF Gaussian scanline");

        encoder.set_compute_pipeline_state(&state.lf_project_pipeline);
        encoder.set_buffer(0, Some(&gx_buffer), 0);
        encoder.set_buffer(1, Some(&gy_buffer), 0);
        encoder.set_buffer(2, Some(&g_perp_buffer), 0);
        encoder.set_buffer(3, Some(&seed_params_buffer), 0);
        encoder.dispatch_threads(image_threads, project_group);
        encoder.memory_barrier_with_resources(&[g_perp_buffer.as_ref()]);

        if x_major {
            encoder.set_compute_pipeline_state(&state.lf_scanline_x_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&state.lf_scanline_y_pipeline);
        }
        encoder.set_buffer(0, Some(&g_perp_buffer), 0);
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&line_offsets_buffer), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, tile_bytes as u64);
        encoder.set_threadgroup_memory_length(1, tile_bytes as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n_chunks as u64,
                height: line_count as u64,
                depth: 1,
            },
            group,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(())
}

unsafe fn run_lf_orientation_stack_box_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    box_passes: c_uint,
    box_radius: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let output_count = total_pixels
        .checked_mul(n_orientations as usize)
        .ok_or_else(|| "LF box output count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF box output",
    )?;
    let m_value = effective_m(m)?;
    let radius = box_radius_for_m(m_value, box_passes, box_radius)?;
    let active_passes = if m_value == 0 { 0 } else { box_passes };

    let shared_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        shared_options,
        None,
    );
    let private_options = MTLResourceOptions::StorageModePrivate;
    let num_a_buffer = state.device.new_buffer(image_len as u64, private_options);
    let num_b_buffer = state.device.new_buffer(image_len as u64, private_options);
    let den_a_buffer = state.device.new_buffer(image_len as u64, private_options);
    let den_b_buffer = state.device.new_buffer(image_len as u64, private_options);

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let seed_group = threadgroup_2d(&state.lf_box_seed_pipeline);
    let x_group = threadgroup_1d(&state.lf_box_x_pipeline);
    let y_group = threadgroup_1d(&state.lf_box_y_pipeline);
    let finalize_group = threadgroup_2d(&state.lf_box_finalize_pipeline);

    for theta_idx in 0..n_orientations as usize {
        let (line_offsets, x_major, key_min, line_count, cos_t, sin_t) =
            build_lf_box_line_offsets(width, height, theta_idx, n_orientations as usize)?;
        let offset_len = checked_len(
            line_offsets.len(),
            std::mem::size_of::<c_int>(),
            "LF box line offset",
        )?;
        let line_offsets_buffer = state.device.new_buffer_with_bytes_no_copy(
            line_offsets.as_ptr().cast(),
            offset_len as u64,
            shared_options,
            None,
        );
        line_offsets_buffer.did_modify_range(NSRange::new(0, offset_len as u64));

        let seed_params = LfBoxSeedParams {
            width,
            height,
            cos_t,
            sin_t,
        };
        let seed_params_buffer = state.device.new_buffer_with_data(
            (&seed_params as *const LfBoxSeedParams).cast(),
            std::mem::size_of::<LfBoxSeedParams>() as u64,
            shared_options,
        );
        let pass_params = LfBoxPassParams {
            width,
            height,
            radius,
            key_min,
            line_count,
        };
        let pass_params_buffer = state.device.new_buffer_with_data(
            (&pass_params as *const LfBoxPassParams).cast(),
            std::mem::size_of::<LfBoxPassParams>() as u64,
            shared_options,
        );
        let finalize_params = LfBoxFinalizeParams {
            width,
            height,
            theta_idx: c_uint::try_from(theta_idx)
                .map_err(|_| "LF box theta index is outside uint32 range".to_string())?,
        };
        let finalize_params_buffer = state.device.new_buffer_with_data(
            (&finalize_params as *const LfBoxFinalizeParams).cast(),
            std::mem::size_of::<LfBoxFinalizeParams>() as u64,
            shared_options,
        );

        let line_threads = MTLSize {
            width: line_count as u64,
            height: 1,
            depth: 1,
        };

        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("LF box orientation stack");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("LF box scanline");

        encoder.set_compute_pipeline_state(&state.lf_box_seed_pipeline);
        encoder.set_buffer(0, Some(&gx_buffer), 0);
        encoder.set_buffer(1, Some(&gy_buffer), 0);
        encoder.set_buffer(2, Some(&num_a_buffer), 0);
        encoder.set_buffer(3, Some(&den_a_buffer), 0);
        encoder.set_buffer(4, Some(&seed_params_buffer), 0);
        encoder.dispatch_threads(image_threads, seed_group);
        encoder.memory_barrier_with_resources(&[num_a_buffer.as_ref(), den_a_buffer.as_ref()]);

        let pass_pipeline = if x_major {
            &state.lf_box_x_pipeline
        } else {
            &state.lf_box_y_pipeline
        };
        let pass_group = if x_major { x_group } else { y_group };
        let mut read_a = true;
        for _ in 0..active_passes {
            let (src_num, src_den, dst_num, dst_den) = if read_a {
                (&num_a_buffer, &den_a_buffer, &num_b_buffer, &den_b_buffer)
            } else {
                (&num_b_buffer, &den_b_buffer, &num_a_buffer, &den_a_buffer)
            };

            encoder.set_compute_pipeline_state(pass_pipeline);
            encoder.set_buffer(0, Some(src_num), 0);
            encoder.set_buffer(1, Some(src_den), 0);
            encoder.set_buffer(2, Some(dst_num), 0);
            encoder.set_buffer(3, Some(dst_den), 0);
            encoder.set_buffer(4, Some(&line_offsets_buffer), 0);
            encoder.set_buffer(5, Some(&pass_params_buffer), 0);
            encoder.dispatch_threads(line_threads, pass_group);
            encoder.memory_barrier_with_resources(&[dst_num.as_ref(), dst_den.as_ref()]);
            read_a = !read_a;
        }

        let (final_num, final_den) = if read_a {
            (&num_a_buffer, &den_a_buffer)
        } else {
            (&num_b_buffer, &den_b_buffer)
        };
        encoder.set_compute_pipeline_state(&state.lf_box_finalize_pipeline);
        encoder.set_buffer(0, Some(final_num), 0);
        encoder.set_buffer(1, Some(final_den), 0);
        encoder.set_buffer(2, Some(&out_buffer), 0);
        encoder.set_buffer(3, Some(&finalize_params_buffer), 0);
        encoder.dispatch_threads(image_threads, finalize_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(())
}

unsafe fn run_lf_orientation_stack_box_buffers_with_state(
    state: &MetalState,
    gx_buffer: &Buffer,
    gy_buffer: &Buffer,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    box_passes: c_uint,
    box_radius: c_int,
    out_buffer: &Buffer,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let m_value = effective_m(m)?;
    let radius = box_radius_for_m(m_value, box_passes, box_radius)?;
    let active_passes = if m_value == 0 { 0 } else { box_passes };

    let shared_options = MTLResourceOptions::StorageModeShared;
    let private_options = MTLResourceOptions::StorageModePrivate;
    let num_a_buffer = state.device.new_buffer(image_len as u64, private_options);
    let num_b_buffer = state.device.new_buffer(image_len as u64, private_options);
    let den_a_buffer = state.device.new_buffer(image_len as u64, private_options);
    let den_b_buffer = state.device.new_buffer(image_len as u64, private_options);

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let seed_group = threadgroup_2d(&state.lf_box_seed_pipeline);
    let x_group = threadgroup_1d(&state.lf_box_x_pipeline);
    let y_group = threadgroup_1d(&state.lf_box_y_pipeline);
    let finalize_group = threadgroup_2d(&state.lf_box_finalize_pipeline);

    for theta_idx in 0..n_orientations as usize {
        let (line_offsets, x_major, key_min, line_count, cos_t, sin_t) =
            build_lf_box_line_offsets(width, height, theta_idx, n_orientations as usize)?;
        let offset_len = checked_len(
            line_offsets.len(),
            std::mem::size_of::<c_int>(),
            "LF box line offset",
        )?;
        let line_offsets_buffer = state.device.new_buffer_with_bytes_no_copy(
            line_offsets.as_ptr().cast(),
            offset_len as u64,
            shared_options,
            None,
        );
        line_offsets_buffer.did_modify_range(NSRange::new(0, offset_len as u64));

        let seed_params = LfBoxSeedParams {
            width,
            height,
            cos_t,
            sin_t,
        };
        let seed_params_buffer = state.device.new_buffer_with_data(
            (&seed_params as *const LfBoxSeedParams).cast(),
            std::mem::size_of::<LfBoxSeedParams>() as u64,
            shared_options,
        );
        let pass_params = LfBoxPassParams {
            width,
            height,
            radius,
            key_min,
            line_count,
        };
        let pass_params_buffer = state.device.new_buffer_with_data(
            (&pass_params as *const LfBoxPassParams).cast(),
            std::mem::size_of::<LfBoxPassParams>() as u64,
            shared_options,
        );
        let finalize_params = LfBoxFinalizeParams {
            width,
            height,
            theta_idx: c_uint::try_from(theta_idx)
                .map_err(|_| "LF box theta index is outside uint32 range".to_string())?,
        };
        let finalize_params_buffer = state.device.new_buffer_with_data(
            (&finalize_params as *const LfBoxFinalizeParams).cast(),
            std::mem::size_of::<LfBoxFinalizeParams>() as u64,
            shared_options,
        );

        let line_threads = MTLSize {
            width: line_count as u64,
            height: 1,
            depth: 1,
        };

        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("fused LF box orientation stack");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("fused LF box scanline");

        encoder.set_compute_pipeline_state(&state.lf_box_seed_pipeline);
        encoder.set_buffer(0, Some(gx_buffer), 0);
        encoder.set_buffer(1, Some(gy_buffer), 0);
        encoder.set_buffer(2, Some(&num_a_buffer), 0);
        encoder.set_buffer(3, Some(&den_a_buffer), 0);
        encoder.set_buffer(4, Some(&seed_params_buffer), 0);
        encoder.dispatch_threads(image_threads, seed_group);
        encoder.memory_barrier_with_resources(&[num_a_buffer.as_ref(), den_a_buffer.as_ref()]);

        let pass_pipeline = if x_major {
            &state.lf_box_x_pipeline
        } else {
            &state.lf_box_y_pipeline
        };
        let pass_group = if x_major { x_group } else { y_group };
        let mut read_a = true;
        for _ in 0..active_passes {
            let (src_num, src_den, dst_num, dst_den) = if read_a {
                (&num_a_buffer, &den_a_buffer, &num_b_buffer, &den_b_buffer)
            } else {
                (&num_b_buffer, &den_b_buffer, &num_a_buffer, &den_a_buffer)
            };

            encoder.set_compute_pipeline_state(pass_pipeline);
            encoder.set_buffer(0, Some(src_num), 0);
            encoder.set_buffer(1, Some(src_den), 0);
            encoder.set_buffer(2, Some(dst_num), 0);
            encoder.set_buffer(3, Some(dst_den), 0);
            encoder.set_buffer(4, Some(&line_offsets_buffer), 0);
            encoder.set_buffer(5, Some(&pass_params_buffer), 0);
            encoder.dispatch_threads(line_threads, pass_group);
            encoder.memory_barrier_with_resources(&[dst_num.as_ref(), dst_den.as_ref()]);
            read_a = !read_a;
        }

        let (final_num, final_den) = if read_a {
            (&num_a_buffer, &den_a_buffer)
        } else {
            (&num_b_buffer, &den_b_buffer)
        };
        encoder.set_compute_pipeline_state(&state.lf_box_finalize_pipeline);
        encoder.set_buffer(0, Some(final_num), 0);
        encoder.set_buffer(1, Some(final_den), 0);
        encoder.set_buffer(2, Some(out_buffer), 0);
        encoder.set_buffer(3, Some(&finalize_params_buffer), 0);
        encoder.dispatch_threads(image_threads, finalize_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(())
}

unsafe fn run_lf_orientation_length_stack_box_with_state(
    state: &MetalState,
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    theta_start: c_uint,
    n_orientations: c_uint,
    total_orientations: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    output_layout: c_uint,
    out: *mut c_float,
) -> Result<(), String> {
    if n_orientations == 0 {
        return Ok(());
    }
    if n_ms == 0 {
        return Ok(());
    }
    if n_ms as usize > MAX_BATCH_MS {
        return Err(format!(
            "full-image batched LF supports at most {MAX_BATCH_MS} m values"
        ));
    }
    if total_orientations == 0 {
        return Err("total_orientations must be positive".to_string());
    }
    if output_layout > 1 {
        return Err("output_layout must be 0 (theta_m_yx) or 1 (theta_yx_m)".to_string());
    }
    let theta_end = theta_start
        .checked_add(n_orientations)
        .ok_or_else(|| "LF box multi orientation range overflowed".to_string())?;
    if theta_end > total_orientations {
        return Err("LF box multi orientation range exceeds total_orientations".to_string());
    }

    let total_pixels = checked_image_pixels(width, height)?;
    let output_count = total_pixels
        .checked_mul(n_orientations as usize)
        .and_then(|value| value.checked_mul(n_ms as usize))
        .ok_or_else(|| "LF box multi output count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let out_len = checked_len(
        output_count,
        std::mem::size_of::<c_float>(),
        "LF box multi output",
    )?;

    let m_slice = std::slice::from_raw_parts(ms, n_ms as usize);
    let mut radii = Vec::with_capacity(n_ms as usize);
    for &m in m_slice {
        radii.push(box_radius_for_m(effective_m(m)?, 1, -1)?);
    }
    let radii_len = checked_len(
        radii.len(),
        std::mem::size_of::<c_uint>(),
        "LF box multi radius",
    )?;

    let shared_options = MTLResourceOptions::StorageModeShared;
    let gx_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_x.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let gy_buffer = state.device.new_buffer_with_bytes_no_copy(
        g_y.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let radii_buffer = state.device.new_buffer_with_bytes_no_copy(
        radii.as_ptr().cast(),
        radii_len as u64,
        shared_options,
        None,
    );
    let out_buffer = state.device.new_buffer_with_bytes_no_copy(
        out.cast::<std::ffi::c_void>().cast_const(),
        out_len as u64,
        shared_options,
        None,
    );
    let g_perp_buffer = state
        .device
        .new_buffer(image_len as u64, MTLResourceOptions::StorageModePrivate);

    gx_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    gy_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    radii_buffer.did_modify_range(NSRange::new(0, radii_len as u64));

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let project_group = threadgroup_2d(&state.lf_project_pipeline);
    let x_group = threadgroup_1d(&state.lf_box_multi_x_pipeline);
    let y_group = threadgroup_1d(&state.lf_box_multi_y_pipeline);

    for local_theta_idx in 0..n_orientations as usize {
        let theta_idx = theta_start as usize + local_theta_idx;
        let (line_offsets, x_major, key_min, line_count, cos_t, sin_t) =
            build_lf_box_line_offsets(width, height, theta_idx, total_orientations as usize)?;
        let offset_len = checked_len(
            line_offsets.len(),
            std::mem::size_of::<c_int>(),
            "LF box multi line offset",
        )?;
        let line_offsets_buffer = state.device.new_buffer_with_bytes_no_copy(
            line_offsets.as_ptr().cast(),
            offset_len as u64,
            shared_options,
            None,
        );
        line_offsets_buffer.did_modify_range(NSRange::new(0, offset_len as u64));

        let params = LfBoxMultiParams {
            width,
            height,
            n_ms,
            output_layout,
            key_min,
            line_count,
            theta_idx: c_uint::try_from(local_theta_idx)
                .map_err(|_| "LF box multi theta index is outside uint32 range".to_string())?,
        };
        let params_buffer = state.device.new_buffer_with_data(
            (&params as *const LfBoxMultiParams).cast(),
            std::mem::size_of::<LfBoxMultiParams>() as u64,
            shared_options,
        );
        let project_params = LfProjectParams {
            width,
            height,
            cos_t,
            sin_t,
        };
        let project_params_buffer = state.device.new_buffer_with_data(
            (&project_params as *const LfProjectParams).cast(),
            std::mem::size_of::<LfProjectParams>() as u64,
            shared_options,
        );

        let line_threads = MTLSize {
            width: line_count as u64,
            height: 1,
            depth: 1,
        };
        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("LF box multi-length orientation stack");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("LF box multi-length scanline");

        encoder.set_compute_pipeline_state(&state.lf_project_pipeline);
        encoder.set_buffer(0, Some(&gx_buffer), 0);
        encoder.set_buffer(1, Some(&gy_buffer), 0);
        encoder.set_buffer(2, Some(&g_perp_buffer), 0);
        encoder.set_buffer(3, Some(&project_params_buffer), 0);
        encoder.dispatch_threads(image_threads, project_group);
        encoder.memory_barrier_with_resources(&[g_perp_buffer.as_ref()]);

        if x_major {
            encoder.set_compute_pipeline_state(&state.lf_box_multi_x_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&state.lf_box_multi_y_pipeline);
        }
        encoder.set_buffer(0, Some(&g_perp_buffer), 0);
        encoder.set_buffer(1, Some(&radii_buffer), 0);
        encoder.set_buffer(2, Some(&line_offsets_buffer), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.dispatch_threads(line_threads, if x_major { x_group } else { y_group });
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(())
}

unsafe fn run_lf_orientation_stack(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    execution_mode: c_uint,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_mut_ptr(out, "out")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_orientations == 0 {
        return Err("n_orientations must be positive".to_string());
    }
    if !matches!(
        execution_mode,
        LF_STACK_EXECUTION_AUTO | LF_STACK_EXECUTION_DIRECT | LF_STACK_EXECUTION_PROJECTED
    ) {
        return Err(
            "LF stack execution mode must be 0 (auto), 1 (direct), or 2 (projected)".to_string(),
        );
    }

    let m_value = effective_m(m)?;
    let total_pixels = checked_image_pixels(width, height)?;
    let selected_mode = if execution_mode == LF_STACK_EXECUTION_AUTO {
        if m_value >= 64 && total_pixels >= 262_144 {
            LF_STACK_EXECUTION_PROJECTED
        } else {
            LF_STACK_EXECUTION_DIRECT
        }
    } else {
        execution_mode
    };

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        if selected_mode == LF_STACK_EXECUTION_PROJECTED {
            run_lf_orientation_stack_projected_with_state(
                state,
                g_x,
                g_y,
                width,
                height,
                n_orientations,
                m,
                out,
            )
        } else {
            run_lf_orientation_stack_direct_with_state(
                state,
                g_x,
                g_y,
                width,
                height,
                n_orientations,
                m,
                out,
            )
        }
    })
}

unsafe fn run_lf_orientation_stack_scanline(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_mut_ptr(out, "out")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_orientations == 0 {
        return Err("n_orientations must be positive".to_string());
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_lf_orientation_stack_scanline_with_state(
            state,
            g_x,
            g_y,
            width,
            height,
            n_orientations,
            m,
            out,
        )
    })
}

unsafe fn run_lf_orientation_stack_box(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    box_passes: c_uint,
    box_radius: c_int,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_mut_ptr(out, "out")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_orientations == 0 {
        return Err("n_orientations must be positive".to_string());
    }
    let _ = box_radius_for_m(effective_m(m)?, box_passes, box_radius)?;

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_lf_orientation_stack_box_with_state(
            state,
            g_x,
            g_y,
            width,
            height,
            n_orientations,
            m,
            box_passes,
            box_radius,
            out,
        )
    })
}

unsafe fn run_lf_orientation_length_stack_box(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    theta_start: c_uint,
    n_orientations: c_uint,
    total_orientations: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    output_layout: c_uint,
    out: *mut c_float,
) -> Result<(), String> {
    check_ptr(g_x, "g_x")?;
    check_ptr(g_y, "g_y")?;
    check_mut_ptr(out, "out")?;
    if n_ms > 0 {
        check_ptr(ms, "ms")?;
    }

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_ms as usize > MAX_BATCH_MS {
        return Err(format!(
            "full-image batched LF supports at most {MAX_BATCH_MS} m values"
        ));
    }
    if n_orientations == 0 {
        return Ok(());
    }
    if total_orientations == 0 {
        return Err("total_orientations must be positive".to_string());
    }
    if output_layout > 1 {
        return Err("output_layout must be 0 (theta_m_yx) or 1 (theta_yx_m)".to_string());
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_lf_orientation_length_stack_box_with_state(
            state,
            g_x,
            g_y,
            width,
            height,
            theta_start,
            n_orientations,
            total_orientations,
            ms,
            n_ms,
            output_layout,
            out,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_wvf_convolve_pair(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_convolve_pair(
        image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_response(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    theta: c_double,
    m: c_int,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_response(g_x, g_y, width, height, px, py, n_pixels, theta, m, out) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_response_batch(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    px: *const c_int,
    py: *const c_int,
    n_pixels: c_uint,
    thetas: *const c_double,
    n_thetas: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_response_batch(
        g_x, g_y, width, height, px, py, n_pixels, thetas, n_thetas, ms, n_ms, out,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_orientation_stack(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    execution_mode: c_uint,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_orientation_stack(
        g_x,
        g_y,
        width,
        height,
        n_orientations,
        m,
        execution_mode,
        out,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_orientation_stack_box(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    box_passes: c_uint,
    box_radius: c_int,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_orientation_stack_box(
        g_x,
        g_y,
        width,
        height,
        n_orientations,
        m,
        box_passes,
        box_radius,
        out,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_orientation_length_stack_box(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    theta_start: c_uint,
    n_orientations: c_uint,
    total_orientations: c_uint,
    ms: *const c_int,
    n_ms: c_uint,
    output_layout: c_uint,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_orientation_length_stack_box(
        g_x,
        g_y,
        width,
        height,
        theta_start,
        n_orientations,
        total_orientations,
        ms,
        n_ms,
        output_layout,
        out,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_lf_orientation_stack_scanline(
    g_x: *const c_float,
    g_y: *const c_float,
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    m: c_int,
    out: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_lf_orientation_stack_scanline(g_x, g_y, width, height, n_orientations, m, out) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

unsafe fn run_recover_two_peaks_with_state(
    state: &MetalState,
    response: *const c_float,
    n_rows: c_uint,
    k: c_uint,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
) -> Result<(), String> {
    if n_rows == 0 {
        return Ok(());
    }

    let row_count = n_rows as usize;
    let k_count = k as usize;
    let response_count = row_count
        .checked_mul(k_count)
        .ok_or_else(|| "recovery response element count overflowed".to_string())?;
    let response_len = checked_len(
        response_count,
        std::mem::size_of::<c_float>(),
        "recovery response",
    )?;
    let out_float_len = checked_len(
        row_count,
        std::mem::size_of::<c_float>(),
        "recovery float output",
    )?;
    let out_valid_len = checked_len(row_count, std::mem::size_of::<u8>(), "recovery validity")?;

    if k_count > 64 {
        return Err("closed-form recovery currently supports at most 64 angles".to_string());
    }
    let (cprime, inv_denom, z_solve, cyclic_denom_inv) = build_recovery_solver(k_count)?;
    let coeff_len = checked_len(k_count, std::mem::size_of::<c_float>(), "recovery solver")?;

    let sep_raw = (min_sep_frac as f64 * dense_n as f64).trunc();
    if sep_raw > c_uint::MAX as f64 {
        return Err("recovery separation is outside uint32 range".to_string());
    }
    let sep = sep_raw.max(1.0) as c_uint;
    let h = std::f64::consts::PI / k_count as f64;
    let params = RecoveryParams {
        n_rows,
        k,
        dense_n,
        sep,
        tau_sec_floor,
        h: h as c_float,
        h2_over6: (h * h / 6.0) as c_float,
        rhs_scale: (6.0 / (h * h)) as c_float,
        pi_over_dense: (std::f64::consts::PI / dense_n as f64) as c_float,
        gamma_inv: -0.25,
        cyclic_denom_inv,
        response_layout: 0,
        plane_size: n_rows,
    };

    let shared_options = MTLResourceOptions::StorageModeShared;
    let response_buffer = state.device.new_buffer_with_bytes_no_copy(
        response.cast(),
        response_len as u64,
        shared_options,
        None,
    );
    let cprime_buffer = state.device.new_buffer_with_bytes_no_copy(
        cprime.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let inv_denom_buffer = state.device.new_buffer_with_bytes_no_copy(
        inv_denom.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let z_solve_buffer = state.device.new_buffer_with_bytes_no_copy(
        z_solve.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let theta_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_p.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_p.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let theta_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_s.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_s.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let v_buffer = state.device.new_buffer_with_bytes_no_copy(
        v.cast::<std::ffi::c_void>().cast_const(),
        out_valid_len as u64,
        shared_options,
        None,
    );
    let row_range_buffer = state
        .device
        .new_buffer(out_float_len as u64, MTLResourceOptions::StorageModePrivate);
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const RecoveryParams).cast(),
        std::mem::size_of::<RecoveryParams>() as u64,
        shared_options,
    );

    response_buffer.did_modify_range(NSRange::new(0, response_len as u64));
    cprime_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));
    inv_denom_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));
    z_solve_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("orientation recovery peaks");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.recovery_pipeline);
    encoder.set_buffer(0, Some(&response_buffer), 0);
    encoder.set_buffer(1, Some(&cprime_buffer), 0);
    encoder.set_buffer(2, Some(&inv_denom_buffer), 0);
    encoder.set_buffer(3, Some(&z_solve_buffer), 0);
    encoder.set_buffer(4, Some(&theta_p_buffer), 0);
    encoder.set_buffer(5, Some(&m_p_buffer), 0);
    encoder.set_buffer(6, Some(&theta_s_buffer), 0);
    encoder.set_buffer(7, Some(&m_s_buffer), 0);
    encoder.set_buffer(8, Some(&row_range_buffer), 0);
    encoder.set_buffer(9, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(&state.recovery_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let reduce_group = threadgroup_1d(&state.recovery_reduce_pipeline);
    let reduce_group_size = reduce_group.width as usize;
    let reduce_scratch_len = checked_len(
        reduce_group_size,
        std::mem::size_of::<c_float>(),
        "recovery reduction scratch",
    )?;
    let mut current_buffer = row_range_buffer.clone();
    let mut current_count = row_count;
    let mut reduction_buffers = Vec::new();
    while current_count > 1 {
        let items_per_group = reduce_group_size
            .checked_mul(2)
            .ok_or_else(|| "recovery reduction group size overflowed".to_string())?;
        let next_count = current_count.div_ceil(items_per_group);
        let next_len = checked_len(
            next_count,
            std::mem::size_of::<c_float>(),
            "recovery reduction output",
        )?;
        let next_buffer = state.device.new_buffer(next_len as u64, shared_options);
        let reduce_params = RecoveryReduceParams {
            count: c_uint::try_from(current_count)
                .map_err(|_| "recovery reduction count is outside uint32 range".to_string())?,
            group_size: c_uint::try_from(reduce_group_size)
                .map_err(|_| "recovery reduction group size is outside uint32 range".to_string())?,
        };
        let reduce_params_buffer = state.device.new_buffer_with_data(
            (&reduce_params as *const RecoveryReduceParams).cast(),
            std::mem::size_of::<RecoveryReduceParams>() as u64,
            shared_options,
        );

        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("orientation recovery range reduction");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&state.recovery_reduce_pipeline);
        encoder.set_buffer(0, Some(&current_buffer), 0);
        encoder.set_buffer(1, Some(&next_buffer), 0);
        encoder.set_buffer(2, Some(&reduce_params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, reduce_scratch_len as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: next_count as u64,
                height: 1,
                depth: 1,
            },
            reduce_group,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        reduction_buffers.push(next_buffer);
        current_buffer = reduction_buffers
            .last()
            .expect("reduction buffer was pushed")
            .clone();
        current_count = next_count;
    }

    let ref_value_buffer: Option<Buffer> = if row_count > 1 {
        let current_ref = *(current_buffer.contents() as *const c_float);
        let mut effective_ref = current_ref;
        LAST_RECOVERY_RANGE.with(|range_cell| {
            let mut range_slot = range_cell.borrow_mut();
            if let Some((last_rows, last_ref)) = *range_slot {
                if last_rows > row_count {
                    effective_ref = effective_ref.max(last_ref);
                }
                if row_count > last_rows {
                    *range_slot = Some((row_count, current_ref));
                }
            } else {
                *range_slot = Some((row_count, current_ref));
            }
        });
        Some(state.device.new_buffer_with_data(
            (&effective_ref as *const c_float).cast(),
            std::mem::size_of::<c_float>() as u64,
            shared_options,
        ))
    } else {
        None
    };
    let ref_buffer = ref_value_buffer.as_ref().unwrap_or(&current_buffer);

    let n_rows_buffer = state.device.new_buffer_with_data(
        (&n_rows as *const c_uint).cast(),
        std::mem::size_of::<c_uint>() as u64,
        shared_options,
    );
    let tau_validity_buffer = state.device.new_buffer_with_data(
        (&tau_validity as *const c_float).cast(),
        std::mem::size_of::<c_float>() as u64,
        shared_options,
    );
    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("orientation recovery validity");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.recovery_validity_pipeline);
    encoder.set_buffer(0, Some(&row_range_buffer), 0);
    encoder.set_buffer(1, Some(ref_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&n_rows_buffer), 0);
    encoder.set_buffer(4, Some(&tau_validity_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(&state.recovery_validity_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_recover_two_peaks_buffer_with_state(
    state: &MetalState,
    response_buffer: &Buffer,
    n_rows: c_uint,
    k: c_uint,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    response_layout: c_uint,
    plane_size: c_uint,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
) -> Result<(), String> {
    if n_rows == 0 {
        return Ok(());
    }

    let row_count = n_rows as usize;
    let k_count = k as usize;
    let out_float_len = checked_len(
        row_count,
        std::mem::size_of::<c_float>(),
        "recovery float output",
    )?;
    let out_valid_len = checked_len(row_count, std::mem::size_of::<u8>(), "recovery validity")?;

    if k_count > 64 {
        return Err("closed-form recovery currently supports at most 64 angles".to_string());
    }
    if response_layout > 1 {
        return Err("recovery response layout must be 0 or 1".to_string());
    }
    let (cprime, inv_denom, z_solve, cyclic_denom_inv) = build_recovery_solver(k_count)?;
    let coeff_len = checked_len(k_count, std::mem::size_of::<c_float>(), "recovery solver")?;

    let sep_raw = (min_sep_frac as f64 * dense_n as f64).trunc();
    if sep_raw > c_uint::MAX as f64 {
        return Err("recovery separation is outside uint32 range".to_string());
    }
    let sep = sep_raw.max(1.0) as c_uint;
    let h = std::f64::consts::PI / k_count as f64;
    let params = RecoveryParams {
        n_rows,
        k,
        dense_n,
        sep,
        tau_sec_floor,
        h: h as c_float,
        h2_over6: (h * h / 6.0) as c_float,
        rhs_scale: (6.0 / (h * h)) as c_float,
        pi_over_dense: (std::f64::consts::PI / dense_n as f64) as c_float,
        gamma_inv: -0.25,
        cyclic_denom_inv,
        response_layout,
        plane_size,
    };

    let shared_options = MTLResourceOptions::StorageModeShared;
    let cprime_buffer = state.device.new_buffer_with_bytes_no_copy(
        cprime.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let inv_denom_buffer = state.device.new_buffer_with_bytes_no_copy(
        inv_denom.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let z_solve_buffer = state.device.new_buffer_with_bytes_no_copy(
        z_solve.as_ptr().cast(),
        coeff_len as u64,
        shared_options,
        None,
    );
    let theta_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_p.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_p.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let theta_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_s.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_s.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let v_buffer = state.device.new_buffer_with_bytes_no_copy(
        v.cast::<std::ffi::c_void>().cast_const(),
        out_valid_len as u64,
        shared_options,
        None,
    );
    let row_range_buffer = state
        .device
        .new_buffer(out_float_len as u64, MTLResourceOptions::StorageModePrivate);
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const RecoveryParams).cast(),
        std::mem::size_of::<RecoveryParams>() as u64,
        shared_options,
    );

    cprime_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));
    inv_denom_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));
    z_solve_buffer.did_modify_range(NSRange::new(0, coeff_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("orientation recovery peaks");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.recovery_pipeline);
    encoder.set_buffer(0, Some(response_buffer), 0);
    encoder.set_buffer(1, Some(&cprime_buffer), 0);
    encoder.set_buffer(2, Some(&inv_denom_buffer), 0);
    encoder.set_buffer(3, Some(&z_solve_buffer), 0);
    encoder.set_buffer(4, Some(&theta_p_buffer), 0);
    encoder.set_buffer(5, Some(&m_p_buffer), 0);
    encoder.set_buffer(6, Some(&theta_s_buffer), 0);
    encoder.set_buffer(7, Some(&m_s_buffer), 0);
    encoder.set_buffer(8, Some(&row_range_buffer), 0);
    encoder.set_buffer(9, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(&state.recovery_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let reduce_group = threadgroup_1d(&state.recovery_reduce_pipeline);
    let reduce_group_size = reduce_group.width as usize;
    let reduce_scratch_len = checked_len(
        reduce_group_size,
        std::mem::size_of::<c_float>(),
        "recovery reduction scratch",
    )?;
    let mut current_buffer = row_range_buffer.clone();
    let mut current_count = row_count;
    let mut reduction_buffers = Vec::new();
    while current_count > 1 {
        let items_per_group = reduce_group_size
            .checked_mul(2)
            .ok_or_else(|| "recovery reduction group size overflowed".to_string())?;
        let next_count = current_count.div_ceil(items_per_group);
        let next_len = checked_len(
            next_count,
            std::mem::size_of::<c_float>(),
            "recovery reduction output",
        )?;
        let next_buffer = state.device.new_buffer(next_len as u64, shared_options);
        let reduce_params = RecoveryReduceParams {
            count: c_uint::try_from(current_count)
                .map_err(|_| "recovery reduction count is outside uint32 range".to_string())?,
            group_size: c_uint::try_from(reduce_group_size)
                .map_err(|_| "recovery reduction group size is outside uint32 range".to_string())?,
        };
        let reduce_params_buffer = state.device.new_buffer_with_data(
            (&reduce_params as *const RecoveryReduceParams).cast(),
            std::mem::size_of::<RecoveryReduceParams>() as u64,
            shared_options,
        );

        let command_buffer = state.queue.new_command_buffer();
        command_buffer.set_label("orientation recovery range reduction");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&state.recovery_reduce_pipeline);
        encoder.set_buffer(0, Some(&current_buffer), 0);
        encoder.set_buffer(1, Some(&next_buffer), 0);
        encoder.set_buffer(2, Some(&reduce_params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, reduce_scratch_len as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: next_count as u64,
                height: 1,
                depth: 1,
            },
            reduce_group,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        reduction_buffers.push(next_buffer);
        current_buffer = reduction_buffers
            .last()
            .expect("reduction buffer was pushed")
            .clone();
        current_count = next_count;
    }

    let ref_value_buffer: Option<Buffer> = if row_count > 1 {
        let current_ref = *(current_buffer.contents() as *const c_float);
        let mut effective_ref = current_ref;
        LAST_RECOVERY_RANGE.with(|range_cell| {
            let mut range_slot = range_cell.borrow_mut();
            if let Some((last_rows, last_ref)) = *range_slot {
                if last_rows > row_count {
                    effective_ref = effective_ref.max(last_ref);
                }
                if row_count > last_rows {
                    *range_slot = Some((row_count, current_ref));
                }
            } else {
                *range_slot = Some((row_count, current_ref));
            }
        });
        Some(state.device.new_buffer_with_data(
            (&effective_ref as *const c_float).cast(),
            std::mem::size_of::<c_float>() as u64,
            shared_options,
        ))
    } else {
        None
    };
    let ref_buffer = ref_value_buffer.as_ref().unwrap_or(&current_buffer);

    let n_rows_buffer = state.device.new_buffer_with_data(
        (&n_rows as *const c_uint).cast(),
        std::mem::size_of::<c_uint>() as u64,
        shared_options,
    );
    let tau_validity_buffer = state.device.new_buffer_with_data(
        (&tau_validity as *const c_float).cast(),
        std::mem::size_of::<c_float>() as u64,
        shared_options,
    );
    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("orientation recovery validity");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.recovery_validity_pipeline);
    encoder.set_buffer(0, Some(&row_range_buffer), 0);
    encoder.set_buffer(1, Some(ref_buffer), 0);
    encoder.set_buffer(2, Some(&v_buffer), 0);
    encoder.set_buffer(3, Some(&n_rows_buffer), 0);
    encoder.set_buffer(4, Some(&tau_validity_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(&state.recovery_validity_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_recover_two_peaks(
    angles: *const c_double,
    response: *const c_float,
    n_rows: c_uint,
    k: c_uint,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
) -> Result<(), String> {
    check_ptr(angles, "angles")?;
    check_ptr(response, "response")?;
    check_mut_ptr(theta_p, "theta_p")?;
    check_mut_ptr(m_p, "m_p")?;
    check_mut_ptr(theta_s, "theta_s")?;
    check_mut_ptr(m_s, "m_s")?;
    check_mut_ptr(v, "v")?;
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    if dense_n == 0 {
        return Err("dense_n must be positive".to_string());
    }
    if !tau_sec_floor.is_finite() || !tau_validity.is_finite() || !min_sep_frac.is_finite() {
        return Err("recovery tunables must be finite".to_string());
    }
    if n_rows == 0 {
        return Ok(());
    }
    let _ = (n_rows as usize)
        .checked_mul(k as usize)
        .ok_or_else(|| "recovery response element count overflowed".to_string())?;

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_recover_two_peaks_with_state(
            state,
            response,
            n_rows,
            k,
            tau_sec_floor,
            tau_validity,
            dense_n,
            min_sep_frac,
            theta_p,
            m_p,
            theta_s,
            m_s,
            v,
        )
    })
}

unsafe fn run_wvf_lf_recover_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    lf_half_length: c_int,
    n_orientations: c_uint,
    box_passes: c_uint,
    box_radius: c_int,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let n_rows = c_uint::try_from(total_pixels)
        .map_err(|_| "fused pipeline pixel count is outside uint32 range".to_string())?;
    let stack_count = total_pixels
        .checked_mul(n_orientations as usize)
        .ok_or_else(|| "fused LF stack element count overflowed".to_string())?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let offset_len = checked_len(n_offsets as usize, std::mem::size_of::<c_int>(), "offset")?;
    let weight_len = checked_len(n_offsets as usize, std::mem::size_of::<c_float>(), "weight")?;
    let stack_len = checked_len(
        stack_count,
        std::mem::size_of::<c_float>(),
        "fused LF stack",
    )?;

    let shared_options = MTLResourceOptions::StorageModeShared;
    let private_options = MTLResourceOptions::StorageModePrivate;
    let image_buffer = state.device.new_buffer_with_bytes_no_copy(
        image.cast(),
        image_len as u64,
        shared_options,
        None,
    );
    let dx_buffer = state.device.new_buffer_with_bytes_no_copy(
        dx.cast(),
        offset_len as u64,
        shared_options,
        None,
    );
    let dy_buffer = state.device.new_buffer_with_bytes_no_copy(
        dy.cast(),
        offset_len as u64,
        shared_options,
        None,
    );
    let wx_buffer = state.device.new_buffer_with_bytes_no_copy(
        wx.cast(),
        weight_len as u64,
        shared_options,
        None,
    );
    let wy_buffer = state.device.new_buffer_with_bytes_no_copy(
        wy.cast(),
        weight_len as u64,
        shared_options,
        None,
    );
    let gx_buffer = state.device.new_buffer(image_len as u64, private_options);
    let gy_buffer = state.device.new_buffer(image_len as u64, private_options);
    let stack_buffer = state.device.new_buffer(stack_len as u64, private_options);
    let params = KernelParams {
        width,
        height,
        n_offsets,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const KernelParams).cast(),
        std::mem::size_of::<KernelParams>() as u64,
        shared_options,
    );

    image_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    wx_buffer.did_modify_range(NSRange::new(0, weight_len as u64));
    wy_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("fused WVF gradients");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.wvf_pipeline);
    encoder.set_buffer(0, Some(&image_buffer), 0);
    encoder.set_buffer(1, Some(&dx_buffer), 0);
    encoder.set_buffer(2, Some(&dy_buffer), 0);
    encoder.set_buffer(3, Some(&wx_buffer), 0);
    encoder.set_buffer(4, Some(&wy_buffer), 0);
    encoder.set_buffer(5, Some(&gx_buffer), 0);
    encoder.set_buffer(6, Some(&gy_buffer), 0);
    encoder.set_buffer(7, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.wvf_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    run_lf_orientation_stack_box_buffers_with_state(
        state,
        &gx_buffer,
        &gy_buffer,
        width,
        height,
        n_orientations,
        lf_half_length,
        box_passes,
        box_radius,
        &stack_buffer,
    )?;

    run_recover_two_peaks_buffer_with_state(
        state,
        &stack_buffer,
        n_rows,
        n_orientations,
        tau_sec_floor,
        tau_validity,
        dense_n,
        min_sep_frac,
        1,
        n_rows,
        theta_p,
        m_p,
        theta_s,
        m_s,
        v,
    )
}

unsafe fn run_wvf_lf_recover(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    lf_half_length: c_int,
    n_orientations: c_uint,
    box_passes: c_uint,
    box_radius: c_int,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
) -> Result<(), String> {
    check_ptr(image, "image")?;
    check_ptr(dx, "dx")?;
    check_ptr(dy, "dy")?;
    check_ptr(wx, "wx")?;
    check_ptr(wy, "wy")?;
    check_mut_ptr(theta_p, "theta_p")?;
    check_mut_ptr(m_p, "m_p")?;
    check_mut_ptr(theta_s, "theta_s")?;
    check_mut_ptr(m_s, "m_s")?;
    check_mut_ptr(v, "v")?;

    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if n_offsets == 0 {
        return Err("n_offsets must be positive".to_string());
    }
    if n_orientations == 0 {
        return Err("n_orientations must be positive".to_string());
    }
    if dense_n == 0 {
        return Err("dense_n must be positive".to_string());
    }
    if !tau_sec_floor.is_finite() || !tau_validity.is_finite() || !min_sep_frac.is_finite() {
        return Err("recovery tunables must be finite".to_string());
    }
    let _ = box_radius_for_m(effective_m(lf_half_length)?, box_passes, box_radius)?;

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_wvf_lf_recover_with_state(
            state,
            image,
            width,
            height,
            dx,
            dy,
            wx,
            wy,
            n_offsets,
            lf_half_length,
            n_orientations,
            box_passes,
            box_radius,
            tau_sec_floor,
            tau_validity,
            dense_n,
            min_sep_frac,
            theta_p,
            m_p,
            theta_s,
            m_s,
            v,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_recover_two_peaks(
    angles: *const c_double,
    response: *const c_float,
    n_rows: c_uint,
    k: c_uint,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_recover_two_peaks(
        angles,
        response,
        n_rows,
        k,
        tau_sec_floor,
        tau_validity,
        dense_n,
        min_sep_frac,
        theta_p,
        m_p,
        theta_s,
        m_s,
        v,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn edgecritic_metal_wvf_lf_recover(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    lf_half_length: c_int,
    n_orientations: c_uint,
    box_passes: c_uint,
    box_radius: c_int,
    tau_sec_floor: c_float,
    tau_validity: c_float,
    dense_n: c_uint,
    min_sep_frac: c_float,
    theta_p: *mut c_float,
    m_p: *mut c_float,
    theta_s: *mut c_float,
    m_s: *mut c_float,
    v: *mut u8,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_wvf_lf_recover(
        image,
        width,
        height,
        dx,
        dy,
        wx,
        wy,
        n_offsets,
        lf_half_length,
        n_orientations,
        box_passes,
        box_radius,
        tau_sec_floor,
        tau_validity,
        dense_n,
        min_sep_frac,
        theta_p,
        m_p,
        theta_s,
        m_s,
        v,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}
