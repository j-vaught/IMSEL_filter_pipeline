#include <metal_stdlib>
using namespace metal;

#define WVF_MAX_FFT_STAGES 16
#define WVF_TRANSPOSE_TILE 16

struct KernelParams {
    uint width;
    uint height;
    uint n_offsets;
};

struct SplitParams {
    uint width;
    uint height;
    uint n_offsets;
    uint radius;
};

struct WvfFftPadParams {
    uint image_width;
    uint image_height;
    uint padded_width;
    uint padded_height;
    uint fft_width;
    uint fft_height;
    uint radius;
};

struct WvfFftPostprocessParams {
    uint width;
    uint height;
    uint crop;
    uint fft_width;
    uint plane_stride;
    float scale;
};

struct WvfFftStageParams {
    uint row_len;
    uint row_count;
    uint stride;
    uint prev;
    uint radix;
};

struct WvfFftTransposeParams {
    uint width;
    uint height;
    uint batch_count;
};

struct WvfFftHermitianParams {
    uint fft_width;
    uint fft_height;
    uint complex_width;
    uint batch_count;
};

struct WvfFftRowPlanParams {
    uint row_len;
    uint row_count;
    uint stage_count;
    uint _reserved;
    uint radix[WVF_MAX_FFT_STAGES];
    uint stride[WVF_MAX_FFT_STAGES];
    uint prev[WVF_MAX_FFT_STAGES];
    uint weight_offset[WVF_MAX_FFT_STAGES];
};

struct WvfFftStridedParams {
    uint row_stride;
    uint rows_per_batch;
    uint plane_stride;
    uint _reserved;
};

struct WvfFftRealWidthParams {
    uint fft_width;
    uint half_width;
    uint complex_width;
    uint row_count;
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

inline float wvf_unsigned_angle(float y, float x) {
    float theta = atan2(y, x);
    if (theta < 0.0f) {
        theta += M_PI_F;
    }
    if (theta >= M_PI_F) {
        theta -= M_PI_F;
    }
    return theta;
}

inline float2 complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 complex_add(float2 a, float2 b) {
    return float2(a.x + b.x, a.y + b.y);
}

inline void write_wvf_outputs(
    uint out_index,
    float sx,
    float sy,
    device float* out_x,
    device float* out_y,
    device float* magnitude,
    device float* angle
) {
    out_x[out_index] = sx;
    out_y[out_index] = sy;
    magnitude[out_index] = sqrt(sx * sx + sy * sy);
    angle[out_index] = wvf_unsigned_angle(sy, sx);
}

kernel void wvf_fft_reflect_pad_real_dense(
    device const float* image [[buffer(0)]],
    device float* padded [[buffer(1)]],
    constant WvfFftPadParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.fft_width || gid.y >= params.fft_height) {
        return;
    }

    const uint dst_index = gid.y * params.fft_width + gid.x;
    if (gid.x >= params.padded_width || gid.y >= params.padded_height) {
        padded[dst_index] = 0.0f;
        return;
    }

    const int src_x =
        reflect_index(int(gid.x) - int(params.radius), int(params.image_width));
    const int src_y =
        reflect_index(int(gid.y) - int(params.radius), int(params.image_height));
    padded[dst_index] = image[uint(src_y) * params.image_width + uint(src_x)];
}

kernel void wvf_fft_reflect_pad_complex_dense(
    device const float* image [[buffer(0)]],
    device float2* padded [[buffer(1)]],
    constant WvfFftPadParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.fft_width || gid.y >= params.fft_height) {
        return;
    }

    const uint dst_index = gid.y * params.fft_width + gid.x;
    if (gid.x >= params.padded_width || gid.y >= params.padded_height) {
        padded[dst_index] = float2(0.0f, 0.0f);
        return;
    }

    const int src_x =
        reflect_index(int(gid.x) - int(params.radius), int(params.image_width));
    const int src_y =
        reflect_index(int(gid.y) - int(params.radius), int(params.image_height));
    padded[dst_index] = float2(
        image[uint(src_y) * params.image_width + uint(src_x)],
        0.0f
    );
}

kernel void wvf_fft_stage_c2c(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftStageParams& params [[buffer(2)]],
    device const float2* weights [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.row_len || gid.y >= params.row_count) {
        return;
    }

    const uint group = gid.x / params.stride;
    const uint q = gid.x - group * params.stride;
    const uint p = group % params.prev;
    const uint row_base = gid.y * params.row_len;
    const uint input_group_base = params.radix * p;
    const uint weight_base = group * params.radix;

    float2 sum = float2(0.0f, 0.0f);
    for (uint l = 0; l < 8; ++l) {
        if (l >= params.radix) {
            break;
        }
        const uint input_index =
            row_base + q + params.stride * (input_group_base + l);
        sum = complex_add(sum, complex_mul(src[input_index], weights[weight_base + l]));
    }

    dst[row_base + gid.x] = sum;
}

kernel void wvf_fft_row_c2c_fused(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftRowPlanParams& plan [[buffer(2)]],
    device const float2* weights [[buffer(3)]],
    threadgroup float2* shared_row [[threadgroup(0)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_position_in_threadgroup.x;
    const uint threads_per_group = threads_per_threadgroup.x;
    const uint row_index = threadgroup_position.y;
    if (row_index >= plan.row_count) {
        return;
    }

    const uint row_base = row_index * plan.row_len;
    for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
        shared_row[idx] = src[row_base + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool source_in_shared = true;
    for (uint stage = 0; stage < plan.stage_count; ++stage) {
        const uint radix = plan.radix[stage];
        const uint stride = plan.stride[stage];
        const uint prev = plan.prev[stage];
        const uint weight_offset = plan.weight_offset[stage];

        for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
            const uint group = idx / stride;
            const uint q = idx - group * stride;
            const uint p = group % prev;

            float2 sum = float2(0.0f, 0.0f);
            for (uint l = 0; l < WVF_MAX_FFT_STAGES; ++l) {
                if (l >= radix) {
                    break;
                }
                const uint input_index = q + stride * (radix * p + l);
                const float2 value =
                    source_in_shared ? shared_row[input_index] : dst[row_base + input_index];
                sum = complex_add(sum, complex_mul(value, weights[weight_offset + group * radix + l]));
            }

            if (source_in_shared) {
                dst[row_base + idx] = sum;
            } else {
                shared_row[idx] = sum;
            }
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        source_in_shared = !source_in_shared;
    }

    if (source_in_shared) {
        for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
            dst[row_base + idx] = shared_row[idx];
        }
    }
}

kernel void wvf_fft_row_c2c_strided_fused(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftRowPlanParams& plan [[buffer(2)]],
    constant WvfFftStridedParams& layout [[buffer(3)]],
    device const float2* weights [[buffer(4)]],
    threadgroup float2* shared_row [[threadgroup(0)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]]
) {
    const uint tid = thread_position_in_threadgroup.x;
    const uint threads_per_group = threads_per_threadgroup.x;
    const uint logical_row = threadgroup_position.y;
    if (logical_row >= plan.row_count) {
        return;
    }

    const uint batch = logical_row / layout.rows_per_batch;
    const uint row_in_batch = logical_row - batch * layout.rows_per_batch;
    const uint row_base = batch * layout.plane_stride + row_in_batch;
    const uint row_stride = layout.row_stride;

    for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
        shared_row[idx] = src[row_base + idx * row_stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool source_in_shared = true;
    for (uint stage = 0; stage < plan.stage_count; ++stage) {
        const uint radix = plan.radix[stage];
        const uint stride = plan.stride[stage];
        const uint prev = plan.prev[stage];
        const uint weight_offset = plan.weight_offset[stage];

        for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
            const uint group = idx / stride;
            const uint q = idx - group * stride;
            const uint p = group % prev;

            float2 sum = float2(0.0f, 0.0f);
            for (uint l = 0; l < WVF_MAX_FFT_STAGES; ++l) {
                if (l >= radix) {
                    break;
                }
                const uint input_index = q + stride * (radix * p + l);
                const float2 value = source_in_shared
                    ? shared_row[input_index]
                    : dst[row_base + input_index * row_stride];
                sum = complex_add(sum, complex_mul(value, weights[weight_offset + group * radix + l]));
            }

            if (source_in_shared) {
                dst[row_base + idx * row_stride] = sum;
            } else {
                shared_row[idx] = sum;
            }
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        source_in_shared = !source_in_shared;
    }

    if (source_in_shared) {
        for (uint idx = tid; idx < plan.row_len; idx += threads_per_group) {
            dst[row_base + idx * row_stride] = shared_row[idx];
        }
    }
}

kernel void wvf_fft_pack_real_pairs(
    device const float* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftRealWidthParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.half_width || gid.y >= params.row_count) {
        return;
    }

    const uint src_base = gid.y * params.fft_width + gid.x * 2u;
    const uint dst_index = gid.y * params.half_width + gid.x;
    dst[dst_index] = float2(src[src_base], src[src_base + 1u]);
}

kernel void wvf_fft_finalize_r2c(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftRealWidthParams& params [[buffer(2)]],
    device const float2* twiddles [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.complex_width || gid.y >= params.row_count) {
        return;
    }

    const uint src_row = gid.y * params.half_width;
    const uint dst_row = gid.y * params.complex_width;
    const uint k = gid.x;
    const float2 z0 = src[src_row];
    if (k == 0u) {
        dst[dst_row] = float2(z0.x + z0.y, 0.0f);
        return;
    }
    if (k == params.half_width) {
        dst[dst_row + k] = float2(z0.x - z0.y, 0.0f);
        return;
    }

    const float2 a = src[src_row + k];
    const float2 mirrored = src[src_row + (params.half_width - k)];
    const float2 b = float2(mirrored.x, -mirrored.y);
    const float2 diff = a - b;
    const float2 twiddled = complex_mul(twiddles[k], diff);
    dst[dst_row + k] = 0.5f * ((a + b) + float2(twiddled.y, -twiddled.x));
}

kernel void wvf_fft_prepare_c2r(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftRealWidthParams& params [[buffer(2)]],
    device const float2* twiddles [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.half_width || gid.y >= params.row_count) {
        return;
    }

    const uint src_row = gid.y * params.complex_width;
    const uint dst_row = gid.y * params.half_width;
    const uint k = gid.x;
    if (k == 0u) {
        const float x0 = src[src_row].x;
        const float xm = src[src_row + params.half_width].x;
        dst[dst_row] = float2(0.5f * (x0 + xm), 0.5f * (x0 - xm));
        return;
    }

    const float2 xk = src[src_row + k];
    const float2 mirrored = src[src_row + (params.half_width - k)];
    const float2 y = float2(mirrored.x, -mirrored.y);
    const float2 diff = xk - y;
    const float2 conj_twiddle = float2(twiddles[k].x, -twiddles[k].y);
    const float2 twiddled = complex_mul(conj_twiddle, diff);
    dst[dst_row + k] = 0.5f * ((xk + y) + float2(-twiddled.y, twiddled.x));
}

kernel void wvf_fft_unpack_real_pairs(
    device const float2* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant WvfFftRealWidthParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.half_width || gid.y >= params.row_count) {
        return;
    }

    const uint src_index = gid.y * params.half_width + gid.x;
    const uint dst_base = gid.y * params.fft_width + gid.x * 2u;
    const float2 value = src[src_index];
    dst[dst_base] = value.x;
    dst[dst_base + 1u] = value.y;
}

kernel void wvf_fft_transpose_c2c(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftTransposeParams& params [[buffer(2)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]]
) {
    threadgroup float2 tile[WVF_TRANSPOSE_TILE][WVF_TRANSPOSE_TILE + 1];

    const uint local_x = thread_position_in_threadgroup.x;
    const uint local_y = thread_position_in_threadgroup.y;
    const uint batch = threadgroup_position.z;
    if (batch >= params.batch_count) {
        return;
    }

    const uint x = threadgroup_position.x * WVF_TRANSPOSE_TILE + local_x;
    const uint y = threadgroup_position.y * WVF_TRANSPOSE_TILE + local_y;
    const uint plane_stride = params.width * params.height;
    const uint batch_offset = batch * plane_stride;

    if (x < params.width && y < params.height) {
        tile[local_y][local_x] = src[batch_offset + y * params.width + x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_x = threadgroup_position.y * WVF_TRANSPOSE_TILE + local_x;
    const uint out_y = threadgroup_position.x * WVF_TRANSPOSE_TILE + local_y;
    if (out_x < params.height && out_y < params.width) {
        dst[batch_offset + out_y * params.height + out_x] = tile[local_x][local_y];
    }
}

kernel void wvf_fft_pack_hermitian(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftHermitianParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.complex_width || gid.y >= params.fft_height || gid.z >= params.batch_count) {
        return;
    }

    const uint full_plane_stride = params.fft_width * params.fft_height;
    const uint reduced_plane_stride = params.complex_width * params.fft_height;
    const uint batch_full = gid.z * full_plane_stride;
    const uint batch_reduced = gid.z * reduced_plane_stride;
    dst[batch_reduced + gid.y * params.complex_width + gid.x] =
        src[batch_full + gid.y * params.fft_width + gid.x];
}

kernel void wvf_fft_unpack_hermitian(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant WvfFftHermitianParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.fft_width || gid.y >= params.fft_height || gid.z >= params.batch_count) {
        return;
    }

    const uint full_plane_stride = params.fft_width * params.fft_height;
    const uint reduced_plane_stride = params.complex_width * params.fft_height;
    const uint batch_full = gid.z * full_plane_stride;
    const uint batch_reduced = gid.z * reduced_plane_stride;
    const uint full_index = batch_full + gid.y * params.fft_width + gid.x;

    if (gid.x < params.complex_width) {
        dst[full_index] = src[batch_reduced + gid.y * params.complex_width + gid.x];
        return;
    }

    const uint mirrored = params.fft_width - gid.x;
    const float2 value = src[batch_reduced + gid.y * params.complex_width + mirrored];
    dst[full_index] = float2(value.x, -value.y);
}

kernel void wvf_fft_multiply_spectra(
    device const float2* input [[buffer(0)]],
    device const float2* kernels [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& n_complex [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = n_complex * 2u;
    if (gid >= total) {
        return;
    }

    const uint plane = gid / n_complex;
    const uint index = gid - plane * n_complex;
    const float2 a = input[index];
    const float2 b = kernels[gid];
    output[gid] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void wvf_fft_postprocess_dense(
    device const float* planes [[buffer(0)]],
    device float* out_x [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    device float* magnitude [[buffer(3)]],
    device float* angle [[buffer(4)]],
    constant WvfFftPostprocessParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint out_index = gid.y * params.width + gid.x;
    const uint src_index =
        (gid.y + params.crop) * params.fft_width + gid.x + params.crop;
    const float gx = planes[src_index] * params.scale;
    const float gy = planes[params.plane_stride + src_index] * params.scale;
    write_wvf_outputs(out_index, gx, gy, out_x, out_y, magnitude, angle);
}

kernel void wvf_fft_postprocess_complex_dense(
    device const float2* planes [[buffer(0)]],
    device float* out_x [[buffer(1)]],
    device float* out_y [[buffer(2)]],
    device float* magnitude [[buffer(3)]],
    device float* angle [[buffer(4)]],
    constant WvfFftPostprocessParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint out_index = gid.y * params.width + gid.x;
    const uint src_index =
        (gid.y + params.crop) * params.fft_width + gid.x + params.crop;
    const float gx = planes[src_index].x * params.scale;
    const float gy = planes[params.plane_stride + src_index].x * params.scale;
    write_wvf_outputs(out_index, gx, gy, out_x, out_y, magnitude, angle);
}

kernel void wvf_direct(
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

kernel void wvf_direct_magnitude_angle(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant KernelParams& params [[buffer(7)]],
    device float* magnitude [[buffer(8)]],
    device float* angle [[buffer(9)]],
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

    write_wvf_outputs(out_index, sx, sy, out_x, out_y, magnitude, angle);
}

kernel void wvf_antipodal(
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
        const int kx = dx[k];
        const int ky = dy[k];
        const int ix_pos = reflect_index(x + kx, int(params.width));
        const int iy_pos = reflect_index(y + ky, int(params.height));
        const int ix_neg = reflect_index(x - kx, int(params.width));
        const int iy_neg = reflect_index(y - ky, int(params.height));
        const float delta =
            image[uint(iy_pos) * params.width + uint(ix_pos)] -
            image[uint(iy_neg) * params.width + uint(ix_neg)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    out_x[out_index] = sx;
    out_y[out_index] = sy;
}

kernel void wvf_antipodal_magnitude_angle(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant KernelParams& params [[buffer(7)]],
    device float* magnitude [[buffer(8)]],
    device float* angle [[buffer(9)]],
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
        const int kx = dx[k];
        const int ky = dy[k];
        const int ix_pos = reflect_index(x + kx, int(params.width));
        const int iy_pos = reflect_index(y + ky, int(params.height));
        const int ix_neg = reflect_index(x - kx, int(params.width));
        const int iy_neg = reflect_index(y - ky, int(params.height));
        const float delta =
            image[uint(iy_pos) * params.width + uint(ix_pos)] -
            image[uint(iy_neg) * params.width + uint(ix_neg)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    write_wvf_outputs(out_index, sx, sy, out_x, out_y, magnitude, angle);
}

kernel void wvf_split_interior(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant SplitParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint x = gid.x + params.radius;
    const uint y = gid.y + params.radius;
    if (x + params.radius >= params.width || y + params.radius >= params.height) {
        return;
    }

    const uint out_index = y * params.width + x;
    const int xi = int(x);
    const int yi = int(y);
    float sx = 0.0f;
    float sy = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int kx = dx[k];
        const int ky = dy[k];
        const float delta =
            image[uint(yi + ky) * params.width + uint(xi + kx)] -
            image[uint(yi - ky) * params.width + uint(xi - kx)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    out_x[out_index] = sx;
    out_y[out_index] = sy;
}

kernel void wvf_split_interior_magnitude_angle(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant SplitParams& params [[buffer(7)]],
    device float* magnitude [[buffer(8)]],
    device float* angle [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint x = gid.x + params.radius;
    const uint y = gid.y + params.radius;
    if (x + params.radius >= params.width || y + params.radius >= params.height) {
        return;
    }

    const uint out_index = y * params.width + x;
    const int xi = int(x);
    const int yi = int(y);
    float sx = 0.0f;
    float sy = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int kx = dx[k];
        const int ky = dy[k];
        const float delta =
            image[uint(yi + ky) * params.width + uint(xi + kx)] -
            image[uint(yi - ky) * params.width + uint(xi - kx)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    write_wvf_outputs(out_index, sx, sy, out_x, out_y, magnitude, angle);
}

kernel void wvf_split_boundary(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant SplitParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    if (
        gid.x >= params.radius &&
        gid.y >= params.radius &&
        gid.x + params.radius < params.width &&
        gid.y + params.radius < params.height
    ) {
        return;
    }

    const uint out_index = gid.y * params.width + gid.x;
    const int x = int(gid.x);
    const int y = int(gid.y);
    float sx = 0.0f;
    float sy = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int kx = dx[k];
        const int ky = dy[k];
        const int ix_pos = reflect_index(x + kx, int(params.width));
        const int iy_pos = reflect_index(y + ky, int(params.height));
        const int ix_neg = reflect_index(x - kx, int(params.width));
        const int iy_neg = reflect_index(y - ky, int(params.height));
        const float delta =
            image[uint(iy_pos) * params.width + uint(ix_pos)] -
            image[uint(iy_neg) * params.width + uint(ix_neg)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    out_x[out_index] = sx;
    out_y[out_index] = sy;
}

kernel void wvf_split_boundary_magnitude_angle(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant SplitParams& params [[buffer(7)]],
    device float* magnitude [[buffer(8)]],
    device float* angle [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    if (
        gid.x >= params.radius &&
        gid.y >= params.radius &&
        gid.x + params.radius < params.width &&
        gid.y + params.radius < params.height
    ) {
        return;
    }

    const uint out_index = gid.y * params.width + gid.x;
    const int x = int(gid.x);
    const int y = int(gid.y);
    float sx = 0.0f;
    float sy = 0.0f;

    for (uint k = 0; k < params.n_offsets; ++k) {
        const int kx = dx[k];
        const int ky = dy[k];
        const int ix_pos = reflect_index(x + kx, int(params.width));
        const int iy_pos = reflect_index(y + ky, int(params.height));
        const int ix_neg = reflect_index(x - kx, int(params.width));
        const int iy_neg = reflect_index(y - ky, int(params.height));
        const float delta =
            image[uint(iy_pos) * params.width + uint(ix_pos)] -
            image[uint(iy_neg) * params.width + uint(ix_neg)];
        sx += delta * wx[k];
        sy += delta * wy[k];
    }

    write_wvf_outputs(out_index, sx, sy, out_x, out_y, magnitude, angle);
}
