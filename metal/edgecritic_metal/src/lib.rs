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
const WVF_DIRECT: c_uint = 0;
const WVF_ANTIPODAL: c_uint = 1;
const WVF_ANTIPODAL_SPLIT: c_uint = 2;

mod shaders;
use shaders::SHADER_SOURCE;

#[repr(C)]
struct KernelParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
}

#[repr(C)]
struct WvfInteriorParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
    radius: c_uint,
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
    trig_order: c_uint,
    trig_coeffs: c_uint,
}

#[repr(C)]
struct RecoveryReduceParams {
    count: c_uint,
    group_size: c_uint,
}

#[repr(C)]
struct CgmmParams {
    p: c_uint,
    n: c_uint,
    n_iters: c_uint,
    init_kappa: c_float,
    tau_m_rel: c_float,
    theta_min_phi: c_float,
}

struct MetalState {
    device: Device,
    wvf_pipeline: ComputePipelineState,
    wvf_antipodal_pipeline: ComputePipelineState,
    wvf_antipodal_interior_pipeline: ComputePipelineState,
    wvf_antipodal_boundary_pipeline: ComputePipelineState,
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
    recovery_stack_pipeline: ComputePipelineState,
    recovery_reduce_pipeline: ComputePipelineState,
    recovery_validity_pipeline: ComputePipelineState,
    cgmm_pipeline: ComputePipelineState,
    queue: CommandQueue,
}

struct FusedScratchBuffers {
    width: c_uint,
    height: c_uint,
    n_orientations: c_uint,
    image_len: usize,
    stack_len: usize,
    gx_buffer: Buffer,
    gy_buffer: Buffer,
    stack_buffer: Buffer,
    num_a_buffer: Buffer,
    num_b_buffer: Buffer,
    den_a_buffer: Buffer,
    den_b_buffer: Buffer,
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
        let antipodal_function = library
            .get_function("wvf_convolve_antipodal", None)
            .map_err(|err| format!("failed to load antipodal WVF Metal function: {err}"))?;
        let wvf_antipodal_pipeline = device
            .new_compute_pipeline_state_with_function(&antipodal_function)
            .map_err(|err| {
                format!("failed to create antipodal WVF Metal compute pipeline: {err}")
            })?;
        let antipodal_interior_function = library
            .get_function("wvf_convolve_antipodal_interior", None)
            .map_err(|err| {
                format!("failed to load antipodal WVF interior Metal function: {err}")
            })?;
        let wvf_antipodal_interior_pipeline = device
            .new_compute_pipeline_state_with_function(&antipodal_interior_function)
            .map_err(|err| {
                format!("failed to create antipodal WVF interior Metal compute pipeline: {err}")
            })?;
        let antipodal_boundary_function = library
            .get_function("wvf_convolve_antipodal_boundary", None)
            .map_err(|err| {
                format!("failed to load antipodal WVF boundary Metal function: {err}")
            })?;
        let wvf_antipodal_boundary_pipeline = device
            .new_compute_pipeline_state_with_function(&antipodal_boundary_function)
            .map_err(|err| {
                format!("failed to create antipodal WVF boundary Metal compute pipeline: {err}")
            })?;
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
        let recovery_stack_function = library
            .get_function("recovery_peaks_private_stack", None)
            .map_err(|err| format!("failed to load stack recovery Metal function: {err}"))?;
        let recovery_stack_pipeline = device
            .new_compute_pipeline_state_with_function(&recovery_stack_function)
            .map_err(|err| {
                format!("failed to create stack recovery Metal compute pipeline: {err}")
            })?;
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
        let cgmm_function = library
            .get_function("cgmm_fuse_two_pass_k3", None)
            .map_err(|err| format!("failed to load c-GMM Metal function: {err}"))?;
        let cgmm_pipeline = device
            .new_compute_pipeline_state_with_function(&cgmm_function)
            .map_err(|err| format!("failed to create c-GMM Metal compute pipeline: {err}"))?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            wvf_pipeline,
            wvf_antipodal_pipeline,
            wvf_antipodal_interior_pipeline,
            wvf_antipodal_boundary_pipeline,
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
            recovery_stack_pipeline,
            recovery_reduce_pipeline,
            recovery_validity_pipeline,
            cgmm_pipeline,
            queue,
        })
    }
}

thread_local! {
    static METAL_STATE: RefCell<Option<MetalState>> = RefCell::new(None);
    static LAST_RECOVERY_RANGE: RefCell<Option<(usize, c_float)>> = RefCell::new(None);
    static FUSED_SCRATCH: RefCell<Option<FusedScratchBuffers>> = RefCell::new(None);
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

fn build_recovery_trig_tables(
    k: usize,
    dense_n: usize,
) -> Result<(Vec<c_float>, Vec<c_float>), String> {
    const ORDER: usize = 4;
    const COEFFS: usize = 2 * ORDER + 1;
    if k < 2 * ORDER {
        return Err("hybrid trig recovery requires at least 8 angles".to_string());
    }
    if dense_n == 0 {
        return Err("dense_n must be positive".to_string());
    }

    let mut solver = vec![0.0f32; COEFFS * k];
    let inv_k = 1.0f64 / k as f64;
    for j in 0..k {
        let theta = std::f64::consts::PI * j as f64 / k as f64;
        solver[j] = inv_k as c_float;
        for n in 1..=ORDER {
            let base = 2 * n - 1;
            let phase = 2.0 * n as f64 * theta;
            let nyquist = 2 * n == k;
            let scale = if nyquist { inv_k } else { 2.0 * inv_k };
            solver[base * k + j] = (scale * phase.cos()) as c_float;
            solver[(base + 1) * k + j] = if nyquist {
                0.0
            } else {
                (scale * phase.sin()) as c_float
            };
        }
    }

    let mut dense = vec![0.0f32; dense_n * COEFFS];
    for idx in 0..dense_n {
        let theta = std::f64::consts::PI * idx as f64 / dense_n as f64;
        let row = idx * COEFFS;
        dense[row] = 1.0;
        for n in 1..=ORDER {
            let base = row + 2 * n - 1;
            let phase = 2.0 * n as f64 * theta;
            dense[base] = phase.cos() as c_float;
            dense[base + 1] = phase.sin() as c_float;
        }
    }

    Ok((solver, dense))
}

unsafe fn run_convolve_pair_with_state(
    state: &MetalState,
    antipodal: bool,
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

    let pipeline = if antipodal {
        &state.wvf_antipodal_pipeline
    } else {
        &state.wvf_pipeline
    };
    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
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
    let group = threadgroup_2d(pipeline);
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
            state, false, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
        )
    })
}

unsafe fn run_convolve_antipodal(
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
            state, true, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
        )
    })
}

unsafe fn run_convolve_antipodal_split_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    radius: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let double_radius = radius.saturating_mul(2);
    let interior_width = width.saturating_sub(double_radius);
    let interior_height = height.saturating_sub(double_radius);
    if radius == 0 || interior_width == 0 || interior_height == 0 {
        return run_convolve_pair_with_state(
            state, true, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
        );
    }

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
    let params = WvfInteriorParams {
        width,
        height,
        n_offsets,
        radius,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const WvfInteriorParams).cast(),
        std::mem::size_of::<WvfInteriorParams>() as u64,
        resource_options,
    );

    image_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    wx_buffer.did_modify_range(NSRange::new(0, weight_len as u64));
    wy_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_buffer(0, Some(&image_buffer), 0);
    encoder.set_buffer(1, Some(&dx_buffer), 0);
    encoder.set_buffer(2, Some(&dy_buffer), 0);
    encoder.set_buffer(3, Some(&wx_buffer), 0);
    encoder.set_buffer(4, Some(&wy_buffer), 0);
    encoder.set_buffer(5, Some(&out_x_buffer), 0);
    encoder.set_buffer(6, Some(&out_y_buffer), 0);
    encoder.set_buffer(7, Some(&params_buffer), 0);

    encoder.set_compute_pipeline_state(&state.wvf_antipodal_interior_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: interior_width as u64,
            height: interior_height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.wvf_antipodal_interior_pipeline),
    );

    encoder.set_compute_pipeline_state(&state.wvf_antipodal_boundary_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.wvf_antipodal_boundary_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_convolve_antipodal_split(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    radius: c_uint,
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
        run_convolve_antipodal_split_with_state(
            state, image, width, height, dx, dy, wx, wy, n_offsets, radius, out_x, out_y,
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
    num_a_buffer: &Buffer,
    num_b_buffer: &Buffer,
    den_a_buffer: &Buffer,
    den_b_buffer: &Buffer,
) -> Result<(), String> {
    let m_value = effective_m(m)?;
    let radius = box_radius_for_m(m_value, box_passes, box_radius)?;
    let active_passes = if m_value == 0 { 0 } else { box_passes };

    let shared_options = MTLResourceOptions::StorageModeShared;

    let image_threads = MTLSize {
        width: width as u64,
        height: height as u64,
        depth: 1,
    };
    let seed_group = threadgroup_2d(&state.lf_box_seed_pipeline);
    let x_group = threadgroup_1d(&state.lf_box_x_pipeline);
    let y_group = threadgroup_1d(&state.lf_box_y_pipeline);
    let finalize_group = threadgroup_2d(&state.lf_box_finalize_pipeline);
    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("fused LF box orientation stack");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_label("fused LF box scanline");
    let mut retained_buffers: Vec<Buffer> = Vec::with_capacity(n_orientations as usize * 4);

    for theta_idx in 0..n_orientations as usize {
        let (line_offsets, x_major, key_min, line_count, cos_t, sin_t) =
            build_lf_box_line_offsets(width, height, theta_idx, n_orientations as usize)?;
        let offset_len = checked_len(
            line_offsets.len(),
            std::mem::size_of::<c_int>(),
            "LF box line offset",
        )?;
        let line_offsets_buffer = state.device.new_buffer_with_data(
            line_offsets.as_ptr().cast(),
            offset_len as u64,
            shared_options,
        );

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
        retained_buffers.push(line_offsets_buffer.clone());
        retained_buffers.push(seed_params_buffer.clone());
        retained_buffers.push(pass_params_buffer.clone());
        retained_buffers.push(finalize_params_buffer.clone());

        encoder.set_compute_pipeline_state(&state.lf_box_seed_pipeline);
        encoder.set_buffer(0, Some(gx_buffer), 0);
        encoder.set_buffer(1, Some(gy_buffer), 0);
        encoder.set_buffer(2, Some(num_a_buffer), 0);
        encoder.set_buffer(3, Some(den_a_buffer), 0);
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
                (num_a_buffer, den_a_buffer, num_b_buffer, den_b_buffer)
            } else {
                (num_b_buffer, den_b_buffer, num_a_buffer, den_a_buffer)
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
            (num_a_buffer, den_a_buffer)
        } else {
            (num_b_buffer, den_b_buffer)
        };
        encoder.set_compute_pipeline_state(&state.lf_box_finalize_pipeline);
        encoder.set_buffer(0, Some(final_num), 0);
        encoder.set_buffer(1, Some(final_den), 0);
        encoder.set_buffer(2, Some(out_buffer), 0);
        encoder.set_buffer(3, Some(&finalize_params_buffer), 0);
        encoder.dispatch_threads(image_threads, finalize_group);
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

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

unsafe fn run_cgmm_fuse_two_pass_with_state(
    state: &MetalState,
    phi_p: *const c_float,
    w_p: *const c_float,
    phi_s: *const c_float,
    w_s: *const c_float,
    p_count: c_uint,
    n_count: c_uint,
    n_iters: c_uint,
    init_kappa: c_float,
    tau_m_rel: c_float,
    theta_min_phi: c_float,
    theta_primary: *mut c_float,
    m_primary: *mut c_float,
    theta_sec: *mut c_float,
    m_sec: *mut c_float,
    v_fused: *mut u8,
    primary_pi: *mut c_float,
    primary_mu: *mut c_float,
    primary_kappa: *mut c_float,
    secondary_pi: *mut c_float,
    secondary_mu: *mut c_float,
    secondary_kappa: *mut c_float,
    keep_secondary_mask: *mut u8,
) -> Result<(), String> {
    if p_count == 0 {
        return Ok(());
    }

    let row_count = p_count as usize;
    let n = n_count as usize;
    let input_count = row_count
        .checked_mul(n)
        .ok_or_else(|| "c-GMM input element count overflowed".to_string())?;
    let diag_count = row_count
        .checked_mul(3)
        .ok_or_else(|| "c-GMM diagnostic element count overflowed".to_string())?;
    let input_len = checked_len(input_count, std::mem::size_of::<c_float>(), "c-GMM input")?;
    let out_float_len = checked_len(
        row_count,
        std::mem::size_of::<c_float>(),
        "c-GMM float output",
    )?;
    let out_valid_len = checked_len(row_count, std::mem::size_of::<u8>(), "c-GMM mask output")?;
    let diag_len = checked_len(
        diag_count,
        std::mem::size_of::<c_float>(),
        "c-GMM diagnostic output",
    )?;

    let shared_options = MTLResourceOptions::StorageModeShared;
    let phi_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        phi_p.cast(),
        input_len as u64,
        shared_options,
        None,
    );
    let w_p_buffer = state.device.new_buffer_with_bytes_no_copy(
        w_p.cast(),
        input_len as u64,
        shared_options,
        None,
    );
    let phi_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        phi_s.cast(),
        input_len as u64,
        shared_options,
        None,
    );
    let w_s_buffer = state.device.new_buffer_with_bytes_no_copy(
        w_s.cast(),
        input_len as u64,
        shared_options,
        None,
    );
    let theta_primary_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_primary.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_primary_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_primary.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let theta_sec_buffer = state.device.new_buffer_with_bytes_no_copy(
        theta_sec.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let m_sec_buffer = state.device.new_buffer_with_bytes_no_copy(
        m_sec.cast::<std::ffi::c_void>().cast_const(),
        out_float_len as u64,
        shared_options,
        None,
    );
    let v_fused_buffer = state.device.new_buffer_with_bytes_no_copy(
        v_fused.cast::<std::ffi::c_void>().cast_const(),
        out_valid_len as u64,
        shared_options,
        None,
    );
    let primary_pi_buffer = state.device.new_buffer_with_bytes_no_copy(
        primary_pi.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let primary_mu_buffer = state.device.new_buffer_with_bytes_no_copy(
        primary_mu.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let primary_kappa_buffer = state.device.new_buffer_with_bytes_no_copy(
        primary_kappa.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let secondary_pi_buffer = state.device.new_buffer_with_bytes_no_copy(
        secondary_pi.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let secondary_mu_buffer = state.device.new_buffer_with_bytes_no_copy(
        secondary_mu.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let secondary_kappa_buffer = state.device.new_buffer_with_bytes_no_copy(
        secondary_kappa.cast::<std::ffi::c_void>().cast_const(),
        diag_len as u64,
        shared_options,
        None,
    );
    let keep_secondary_buffer = state.device.new_buffer_with_bytes_no_copy(
        keep_secondary_mask.cast::<std::ffi::c_void>().cast_const(),
        out_valid_len as u64,
        shared_options,
        None,
    );
    let params = CgmmParams {
        p: p_count,
        n: n_count,
        n_iters,
        init_kappa,
        tau_m_rel,
        theta_min_phi,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const CgmmParams).cast(),
        std::mem::size_of::<CgmmParams>() as u64,
        shared_options,
    );

    phi_p_buffer.did_modify_range(NSRange::new(0, input_len as u64));
    w_p_buffer.did_modify_range(NSRange::new(0, input_len as u64));
    phi_s_buffer.did_modify_range(NSRange::new(0, input_len as u64));
    w_s_buffer.did_modify_range(NSRange::new(0, input_len as u64));

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&state.cgmm_pipeline);
    encoder.set_buffer(0, Some(&phi_p_buffer), 0);
    encoder.set_buffer(1, Some(&w_p_buffer), 0);
    encoder.set_buffer(2, Some(&phi_s_buffer), 0);
    encoder.set_buffer(3, Some(&w_s_buffer), 0);
    encoder.set_buffer(4, Some(&theta_primary_buffer), 0);
    encoder.set_buffer(5, Some(&m_primary_buffer), 0);
    encoder.set_buffer(6, Some(&theta_sec_buffer), 0);
    encoder.set_buffer(7, Some(&m_sec_buffer), 0);
    encoder.set_buffer(8, Some(&v_fused_buffer), 0);
    encoder.set_buffer(9, Some(&primary_pi_buffer), 0);
    encoder.set_buffer(10, Some(&primary_mu_buffer), 0);
    encoder.set_buffer(11, Some(&primary_kappa_buffer), 0);
    encoder.set_buffer(12, Some(&secondary_pi_buffer), 0);
    encoder.set_buffer(13, Some(&secondary_mu_buffer), 0);
    encoder.set_buffer(14, Some(&secondary_kappa_buffer), 0);
    encoder.set_buffer(15, Some(&keep_secondary_buffer), 0);
    encoder.set_buffer(16, Some(&params_buffer), 0);

    let threads = MTLSize {
        width: p_count as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(threads, threadgroup_1d(&state.cgmm_pipeline));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

unsafe fn run_cgmm_fuse_two_pass(
    phi_p: *const c_float,
    w_p: *const c_float,
    phi_s: *const c_float,
    w_s: *const c_float,
    p_count: c_uint,
    n_count: c_uint,
    n_iters: c_uint,
    init_kappa: c_float,
    tau_m_rel: c_float,
    theta_min_phi: c_float,
    theta_primary: *mut c_float,
    m_primary: *mut c_float,
    theta_sec: *mut c_float,
    m_sec: *mut c_float,
    v_fused: *mut u8,
    primary_pi: *mut c_float,
    primary_mu: *mut c_float,
    primary_kappa: *mut c_float,
    secondary_pi: *mut c_float,
    secondary_mu: *mut c_float,
    secondary_kappa: *mut c_float,
    keep_secondary_mask: *mut u8,
) -> Result<(), String> {
    check_ptr(phi_p, "phi_p")?;
    check_ptr(w_p, "w_p")?;
    check_ptr(phi_s, "phi_s")?;
    check_ptr(w_s, "w_s")?;
    check_mut_ptr(theta_primary, "theta_primary")?;
    check_mut_ptr(m_primary, "M_primary")?;
    check_mut_ptr(theta_sec, "theta_sec")?;
    check_mut_ptr(m_sec, "M_sec")?;
    check_mut_ptr(v_fused, "v_fused")?;
    check_mut_ptr(primary_pi, "primary_pi")?;
    check_mut_ptr(primary_mu, "primary_mu")?;
    check_mut_ptr(primary_kappa, "primary_kappa")?;
    check_mut_ptr(secondary_pi, "secondary_pi")?;
    check_mut_ptr(secondary_mu, "secondary_mu")?;
    check_mut_ptr(secondary_kappa, "secondary_kappa")?;
    check_mut_ptr(keep_secondary_mask, "keep_secondary_mask")?;

    if n_count == 0 || n_count > 64 {
        return Err("c-GMM N must satisfy 0 < N <= 64".to_string());
    }
    if n_iters != 30 {
        return Err("c-GMM Metal path supports n_iters=30 only".to_string());
    }
    if !init_kappa.is_finite() {
        return Err("init_kappa must be finite".to_string());
    }
    if !tau_m_rel.is_finite() {
        return Err("tau_M_rel must be finite".to_string());
    }
    if !theta_min_phi.is_finite() {
        return Err("theta_min_phi must be finite".to_string());
    }

    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        run_cgmm_fuse_two_pass_with_state(
            state,
            phi_p,
            w_p,
            phi_s,
            w_s,
            p_count,
            n_count,
            n_iters,
            init_kappa,
            tau_m_rel,
            theta_min_phi,
            theta_primary,
            m_primary,
            theta_sec,
            m_sec,
            v_fused,
            primary_pi,
            primary_mu,
            primary_kappa,
            secondary_pi,
            secondary_mu,
            secondary_kappa,
            keep_secondary_mask,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn metal_full_pipeline_wvf_convolve_pair(
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
pub unsafe extern "C" fn metal_full_pipeline_wvf_convolve_antipodal(
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
    match run_convolve_antipodal(
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
pub unsafe extern "C" fn metal_full_pipeline_wvf_convolve_antipodal_split(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    radius: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_convolve_antipodal_split(
        image, width, height, dx, dy, wx, wy, n_offsets, radius, out_x, out_y,
    ) {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn metal_full_pipeline_lf_response(
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
pub unsafe extern "C" fn metal_full_pipeline_lf_response_batch(
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
pub unsafe extern "C" fn metal_full_pipeline_lf_orientation_stack(
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
pub unsafe extern "C" fn metal_full_pipeline_lf_orientation_stack_box(
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
pub unsafe extern "C" fn metal_full_pipeline_lf_orientation_length_stack_box(
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
pub unsafe extern "C" fn metal_full_pipeline_lf_orientation_stack_scanline(
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

#[no_mangle]
pub unsafe extern "C" fn metal_full_pipeline_cgmm_fuse_two_pass(
    phi_p: *const c_float,
    w_p: *const c_float,
    phi_s: *const c_float,
    w_s: *const c_float,
    p_count: c_uint,
    n_count: c_uint,
    n_iters: c_uint,
    init_kappa: c_float,
    tau_m_rel: c_float,
    theta_min_phi: c_float,
    theta_primary: *mut c_float,
    m_primary: *mut c_float,
    theta_sec: *mut c_float,
    m_sec: *mut c_float,
    v_fused: *mut u8,
    primary_pi: *mut c_float,
    primary_mu: *mut c_float,
    primary_kappa: *mut c_float,
    secondary_pi: *mut c_float,
    secondary_mu: *mut c_float,
    secondary_kappa: *mut c_float,
    keep_secondary_mask: *mut u8,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    match run_cgmm_fuse_two_pass(
        phi_p,
        w_p,
        phi_s,
        w_s,
        p_count,
        n_count,
        n_iters,
        init_kappa,
        tau_m_rel,
        theta_min_phi,
        theta_primary,
        m_primary,
        theta_sec,
        m_sec,
        v_fused,
        primary_pi,
        primary_mu,
        primary_kappa,
        secondary_pi,
        secondary_mu,
        secondary_kappa,
        keep_secondary_mask,
    ) {
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
    let (trig_solver, trig_dense) = build_recovery_trig_tables(k_count, dense_n as usize)?;
    let coeff_len = checked_len(k_count, std::mem::size_of::<c_float>(), "recovery solver")?;
    let trig_solver_len = checked_len(
        trig_solver.len(),
        std::mem::size_of::<c_float>(),
        "recovery trig solver",
    )?;
    let trig_dense_len = checked_len(
        trig_dense.len(),
        std::mem::size_of::<c_float>(),
        "recovery trig dense table",
    )?;

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
        trig_order: 4,
        trig_coeffs: 9,
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
    let trig_solver_buffer = state.device.new_buffer_with_bytes_no_copy(
        trig_solver.as_ptr().cast(),
        trig_solver_len as u64,
        shared_options,
        None,
    );
    let trig_dense_buffer = state.device.new_buffer_with_bytes_no_copy(
        trig_dense.as_ptr().cast(),
        trig_dense_len as u64,
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
    trig_solver_buffer.did_modify_range(NSRange::new(0, trig_solver_len as u64));
    trig_dense_buffer.did_modify_range(NSRange::new(0, trig_dense_len as u64));

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
    encoder.set_buffer(10, Some(&trig_solver_buffer), 0);
    encoder.set_buffer(11, Some(&trig_dense_buffer), 0);
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
    let (trig_solver, trig_dense) = build_recovery_trig_tables(k_count, dense_n as usize)?;
    let coeff_len = checked_len(k_count, std::mem::size_of::<c_float>(), "recovery solver")?;
    let trig_solver_len = checked_len(
        trig_solver.len(),
        std::mem::size_of::<c_float>(),
        "recovery trig solver",
    )?;
    let trig_dense_len = checked_len(
        trig_dense.len(),
        std::mem::size_of::<c_float>(),
        "recovery trig dense table",
    )?;

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
        trig_order: 4,
        trig_coeffs: 9,
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
    let trig_solver_buffer = state.device.new_buffer_with_bytes_no_copy(
        trig_solver.as_ptr().cast(),
        trig_solver_len as u64,
        shared_options,
        None,
    );
    let trig_dense_buffer = state.device.new_buffer_with_bytes_no_copy(
        trig_dense.as_ptr().cast(),
        trig_dense_len as u64,
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
    trig_solver_buffer.did_modify_range(NSRange::new(0, trig_solver_len as u64));
    trig_dense_buffer.did_modify_range(NSRange::new(0, trig_dense_len as u64));

    let recovery_pipeline = if response_layout == 1 {
        &state.recovery_stack_pipeline
    } else {
        &state.recovery_pipeline
    };
    let command_buffer = state.queue.new_command_buffer();
    command_buffer.set_label("orientation recovery peaks");
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(recovery_pipeline);
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
    encoder.set_buffer(10, Some(&trig_solver_buffer), 0);
    encoder.set_buffer(11, Some(&trig_dense_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: n_rows as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(recovery_pipeline),
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
    wvf_mode: c_uint,
    radius: c_uint,
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
    let (
        gx_buffer,
        gy_buffer,
        stack_buffer,
        num_a_buffer,
        num_b_buffer,
        den_a_buffer,
        den_b_buffer,
    ) = FUSED_SCRATCH.with(|scratch_cell| {
        let mut scratch_slot = scratch_cell.borrow_mut();
        let needs_new = match scratch_slot.as_ref() {
            Some(scratch) => {
                scratch.width != width
                    || scratch.height != height
                    || scratch.n_orientations != n_orientations
                    || scratch.image_len != image_len
                    || scratch.stack_len != stack_len
            }
            None => true,
        };
        if needs_new {
            *scratch_slot = Some(FusedScratchBuffers {
                width,
                height,
                n_orientations,
                image_len,
                stack_len,
                gx_buffer: state.device.new_buffer(image_len as u64, private_options),
                gy_buffer: state.device.new_buffer(image_len as u64, private_options),
                stack_buffer: state.device.new_buffer(stack_len as u64, private_options),
                num_a_buffer: state.device.new_buffer(image_len as u64, private_options),
                num_b_buffer: state.device.new_buffer(image_len as u64, private_options),
                den_a_buffer: state.device.new_buffer(image_len as u64, private_options),
                den_b_buffer: state.device.new_buffer(image_len as u64, private_options),
            });
        }
        let scratch = scratch_slot
            .as_ref()
            .expect("fused scratch buffers were initialized");
        (
            scratch.gx_buffer.clone(),
            scratch.gy_buffer.clone(),
            scratch.stack_buffer.clone(),
            scratch.num_a_buffer.clone(),
            scratch.num_b_buffer.clone(),
            scratch.den_a_buffer.clone(),
            scratch.den_b_buffer.clone(),
        )
    });
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
    let split_params = WvfInteriorParams {
        width,
        height,
        n_offsets,
        radius,
    };
    let split_params_buffer = state.device.new_buffer_with_data(
        (&split_params as *const WvfInteriorParams).cast(),
        std::mem::size_of::<WvfInteriorParams>() as u64,
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
    encoder.set_buffer(0, Some(&image_buffer), 0);
    encoder.set_buffer(1, Some(&dx_buffer), 0);
    encoder.set_buffer(2, Some(&dy_buffer), 0);
    encoder.set_buffer(3, Some(&wx_buffer), 0);
    encoder.set_buffer(4, Some(&wy_buffer), 0);
    encoder.set_buffer(5, Some(&gx_buffer), 0);
    encoder.set_buffer(6, Some(&gy_buffer), 0);
    if wvf_mode == WVF_ANTIPODAL_SPLIT {
        let double_radius = radius.saturating_mul(2);
        let interior_width = width.saturating_sub(double_radius);
        let interior_height = height.saturating_sub(double_radius);
        if radius > 0 && interior_width > 0 && interior_height > 0 {
            encoder.set_buffer(7, Some(&split_params_buffer), 0);
            encoder.set_compute_pipeline_state(&state.wvf_antipodal_interior_pipeline);
            encoder.dispatch_threads(
                MTLSize {
                    width: interior_width as u64,
                    height: interior_height as u64,
                    depth: 1,
                },
                threadgroup_2d(&state.wvf_antipodal_interior_pipeline),
            );
            encoder.set_compute_pipeline_state(&state.wvf_antipodal_boundary_pipeline);
            encoder.dispatch_threads(
                MTLSize {
                    width: width as u64,
                    height: height as u64,
                    depth: 1,
                },
                threadgroup_2d(&state.wvf_antipodal_boundary_pipeline),
            );
        } else {
            encoder.set_buffer(7, Some(&params_buffer), 0);
            encoder.set_compute_pipeline_state(&state.wvf_antipodal_pipeline);
            encoder.dispatch_threads(
                MTLSize {
                    width: width as u64,
                    height: height as u64,
                    depth: 1,
                },
                threadgroup_2d(&state.wvf_antipodal_pipeline),
            );
        }
    } else {
        let wvf_pipeline = if wvf_mode == WVF_ANTIPODAL {
            &state.wvf_antipodal_pipeline
        } else {
            &state.wvf_pipeline
        };
        encoder.set_buffer(7, Some(&params_buffer), 0);
        encoder.set_compute_pipeline_state(wvf_pipeline);
        encoder.dispatch_threads(
            MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
            threadgroup_2d(wvf_pipeline),
        );
    }
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
        &num_a_buffer,
        &num_b_buffer,
        &den_a_buffer,
        &den_b_buffer,
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
    wvf_mode: c_uint,
    radius: c_uint,
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
            wvf_mode,
            radius,
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
pub unsafe extern "C" fn metal_full_pipeline_recover_two_peaks(
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
pub unsafe extern "C" fn metal_full_pipeline_wvf_lf_recover(
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
        WVF_DIRECT,
        0,
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

#[no_mangle]
pub unsafe extern "C" fn metal_full_pipeline_wvf_lf_recover_antipodal(
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
        WVF_ANTIPODAL,
        0,
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

#[no_mangle]
pub unsafe extern "C" fn metal_full_pipeline_wvf_lf_recover_antipodal_split(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    dx: *const c_int,
    dy: *const c_int,
    wx: *const c_float,
    wy: *const c_float,
    n_offsets: c_uint,
    radius: c_uint,
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
        WVF_ANTIPODAL_SPLIT,
        radius,
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
