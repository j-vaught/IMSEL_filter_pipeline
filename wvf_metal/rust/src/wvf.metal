#include <metal_stdlib>
using namespace metal;

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
    const float gx = planes[src_index];
    const float gy = planes[params.plane_stride + src_index];
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
