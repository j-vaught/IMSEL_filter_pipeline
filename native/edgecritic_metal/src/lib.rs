use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    NSRange,
};
use std::cell::RefCell;
use std::ffi::c_char;
use std::os::raw::{c_double, c_float, c_int, c_uint};
use std::ptr;

const MAX_BATCH_MS: usize = 32;
const LF_STACK_EXECUTION_AUTO: c_uint = 0;
const LF_STACK_EXECUTION_DIRECT: c_uint = 1;
const LF_STACK_EXECUTION_PROJECTED: c_uint = 2;

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
    queue: CommandQueue,
}

impl MetalState {
    fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let options = CompileOptions::new();
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
            queue,
        })
    }
}

thread_local! {
    static METAL_STATE: RefCell<Option<MetalState>> = RefCell::new(None);
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
    let mut width = max_threads.min(256);
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
