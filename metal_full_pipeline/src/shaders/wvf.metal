
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

kernel void wvf_convolve_antipodal(
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

kernel void wvf_convolve_antipodal_interior(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant WvfInteriorParams& params [[buffer(7)]],
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

kernel void wvf_convolve_antipodal_boundary(
    device const float* image [[buffer(0)]],
    device const int* dx [[buffer(1)]],
    device const int* dy [[buffer(2)]],
    device const float* wx [[buffer(3)]],
    device const float* wy [[buffer(4)]],
    device float* out_x [[buffer(5)]],
    device float* out_y [[buffer(6)]],
    constant WvfInteriorParams& params [[buffer(7)]],
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
