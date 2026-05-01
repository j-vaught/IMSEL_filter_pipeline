use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::os::raw::{c_float, c_int, c_uint};
use std::ptr;

const SHADER_SOURCE: &str = include_str!("wvf.metal");
const WVF_VARIANT_DIRECT: c_uint = 0;
const WVF_VARIANT_ANTIPODAL: c_uint = 1;
const WVF_VARIANT_SPLIT: c_uint = 2;
const WVF_VARIANT_FFT: c_uint = 3;

extern "C" {
    fn wvf_vkfft_magnitude_angle(
        image: *const c_float,
        width: c_uint,
        height: c_uint,
        radius: c_uint,
        kernel_x: *const c_float,
        kernel_y: *const c_float,
        kernel_width: c_uint,
        out_x: *mut c_float,
        out_y: *mut c_float,
        magnitude: *mut c_float,
        angle: *mut c_float,
        error_out: *mut c_char,
        error_len: usize,
    ) -> c_int;
}

#[repr(C)]
struct KernelParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
}

#[repr(C)]
struct SplitParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
    radius: c_uint,
}

struct MetalState {
    device: Device,
    direct_pipeline: ComputePipelineState,
    direct_magnitude_pipeline: ComputePipelineState,
    antipodal_pipeline: ComputePipelineState,
    antipodal_magnitude_pipeline: ComputePipelineState,
    split_interior_pipeline: ComputePipelineState,
    split_boundary_pipeline: ComputePipelineState,
    split_interior_magnitude_pipeline: ComputePipelineState,
    split_boundary_magnitude_pipeline: ComputePipelineState,
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

        let direct_pipeline = pipeline(&device, &library, "wvf_direct")?;
        let direct_magnitude_pipeline = pipeline(&device, &library, "wvf_direct_magnitude_angle")?;
        let antipodal_pipeline = pipeline(&device, &library, "wvf_antipodal")?;
        let antipodal_magnitude_pipeline =
            pipeline(&device, &library, "wvf_antipodal_magnitude_angle")?;
        let split_interior_pipeline = pipeline(&device, &library, "wvf_split_interior")?;
        let split_boundary_pipeline = pipeline(&device, &library, "wvf_split_boundary")?;
        let split_interior_magnitude_pipeline =
            pipeline(&device, &library, "wvf_split_interior_magnitude_angle")?;
        let split_boundary_magnitude_pipeline =
            pipeline(&device, &library, "wvf_split_boundary_magnitude_angle")?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            direct_pipeline,
            direct_magnitude_pipeline,
            antipodal_pipeline,
            antipodal_magnitude_pipeline,
            split_interior_pipeline,
            split_boundary_pipeline,
            split_interior_magnitude_pipeline,
            split_boundary_magnitude_pipeline,
            queue,
        })
    }
}

fn pipeline(
    device: &Device,
    library: &metal::Library,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|err| format!("failed to load Metal function {name}: {err}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| format!("failed to create Metal pipeline {name}: {err}"))
}

thread_local! {
    static METAL_STATE: RefCell<Option<MetalState>> = const { RefCell::new(None) };
    static KERNEL_CACHE: RefCell<HashMap<(c_uint, c_uint, c_uint), GeneratedKernels>> = RefCell::new(HashMap::new());
    static PLAN_CACHE: RefCell<HashMap<(c_uint, c_uint, c_uint), KernelPlan>> = RefCell::new(HashMap::new());
    static DENSE_KERNEL_CACHE: RefCell<HashMap<(c_uint, c_uint), DenseConvolutionKernels>> = RefCell::new(HashMap::new());
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

fn checked_image_pixels(width: c_uint, height: c_uint) -> Result<usize, String> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "image dimensions overflowed".to_string())
}

fn checked_len(count: usize, elem_size: usize, name: &str) -> Result<usize, String> {
    count
        .checked_mul(elem_size)
        .ok_or_else(|| format!("{name} byte length overflowed"))
}

unsafe fn check_ptr<T>(ptr: *const T, name: &str) -> Result<(), String> {
    if ptr.is_null() {
        Err(format!("{name} pointer is null"))
    } else {
        Ok(())
    }
}

unsafe fn check_mut_ptr<T>(ptr: *mut T, name: &str) -> Result<(), String> {
    if ptr.is_null() {
        Err(format!("{name} pointer is null"))
    } else {
        Ok(())
    }
}

unsafe fn write_error(error_out: *mut c_char, error_len: usize, message: &str) {
    if error_out.is_null() || error_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let copy_len = bytes.len().min(error_len.saturating_sub(1));
    ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), error_out, copy_len);
    *error_out.add(copy_len) = 0;
}

#[derive(Clone)]
struct GeneratedKernels {
    radius: c_uint,
    dx: Vec<c_int>,
    dy: Vec<c_int>,
    wx: Vec<c_float>,
    wy: Vec<c_float>,
}

impl GeneratedKernels {
    fn n_offsets(&self) -> Result<c_uint, String> {
        c_uint::try_from(self.dx.len())
            .map_err(|_| "WVF support is too large for uint32".to_string())
    }
}

fn taylor_exponents(degree: c_uint) -> Vec<(u32, u32)> {
    let mut exponents = vec![(0, 0)];
    if degree >= 1 {
        exponents.push((1, 0));
        exponents.push((0, 1));
    }
    for total_degree in 2..=degree {
        exponents.push((total_degree, 0));
        exponents.push((0, total_degree));
        for px in (1..total_degree).rev() {
            exponents.push((px, total_degree - px));
        }
    }
    exponents
}

fn factorial(value: u32) -> f64 {
    (2..=value).fold(1.0, |acc, x| acc * f64::from(x))
}

fn disk_offsets(radius: c_uint) -> Result<Vec<(c_int, c_int)>, String> {
    let r = c_int::try_from(radius).map_err(|_| "radius is too large".to_string())?;
    if r < 1 {
        return Err("radius must be positive".to_string());
    }
    let radius2 = r
        .checked_mul(r)
        .ok_or_else(|| "radius is too large".to_string())?;

    let mut offsets = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            if dx
                .checked_mul(dx)
                .and_then(|x2| dy.checked_mul(dy).and_then(|y2| x2.checked_add(y2)))
                .is_some_and(|dist2| dist2 <= radius2)
            {
                offsets.push((dx, dy));
            }
        }
    }
    Ok(offsets)
}

fn build_design(offsets: &[(c_int, c_int)], degree: c_uint) -> Vec<Vec<f64>> {
    let exponents = taylor_exponents(degree);
    offsets
        .iter()
        .map(|&(dx, dy)| {
            let x = f64::from(dx);
            let y = f64::from(dy);
            exponents
                .iter()
                .map(|&(px, py)| {
                    let scale = factorial(px) * factorial(py);
                    x.powi(px as i32) * y.powi(py as i32) / scale
                })
                .collect()
        })
        .collect()
}

fn invert_square(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut aug = vec![0.0; n * 2 * n];
    for row in 0..n {
        for col in 0..n {
            aug[row * 2 * n + col] = matrix[row * n + col];
        }
        aug[row * 2 * n + n + row] = 1.0;
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = aug[pivot_row * 2 * n + col].abs();
        for row in (col + 1)..n {
            let candidate = aug[row * 2 * n + col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= 1.0e-14 {
            return Err("Taylor normal matrix is singular".to_string());
        }
        if pivot_row != col {
            for k in 0..(2 * n) {
                aug.swap(col * 2 * n + k, pivot_row * 2 * n + k);
            }
        }

        let pivot = aug[col * 2 * n + col];
        for k in 0..(2 * n) {
            aug[col * 2 * n + k] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..(2 * n) {
                aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
            }
        }
    }

    let mut inverse = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            inverse[row * n + col] = aug[row * 2 * n + n + col];
        }
    }
    Ok(inverse)
}

fn build_direct_kernels(radius: c_uint, degree: c_uint) -> Result<GeneratedKernels, String> {
    if degree < 1 {
        return Err("degree must be at least 1 for WVF gradients".to_string());
    }

    let offsets = disk_offsets(radius)?;
    let design = build_design(&offsets, degree);
    let n_samples = design.len();
    let n_coeffs = taylor_exponents(degree).len();
    if n_samples < n_coeffs {
        return Err(format!(
            "radius {radius} gives {n_samples} samples, fewer than {n_coeffs} Taylor coefficients for degree {degree}"
        ));
    }

    let mut ata = vec![0.0; n_coeffs * n_coeffs];
    for row in &design {
        for i in 0..n_coeffs {
            for j in 0..n_coeffs {
                ata[i * n_coeffs + j] += row[i] * row[j];
            }
        }
    }
    let inv = invert_square(&ata, n_coeffs)?;

    let mut dx = Vec::with_capacity(n_samples);
    let mut dy = Vec::with_capacity(n_samples);
    let mut wx = Vec::with_capacity(n_samples);
    let mut wy = Vec::with_capacity(n_samples);
    for (sample_idx, &(offset_x, offset_y)) in offsets.iter().enumerate() {
        dx.push(offset_x);
        dy.push(offset_y);
        let row = &design[sample_idx];
        let weight_x = (0..n_coeffs)
            .map(|coeff| inv[n_coeffs + coeff] * row[coeff])
            .sum::<f64>();
        let weight_y = (0..n_coeffs)
            .map(|coeff| inv[2 * n_coeffs + coeff] * row[coeff])
            .sum::<f64>();
        wx.push(weight_x as c_float);
        wy.push(weight_y as c_float);
    }

    Ok(GeneratedKernels {
        radius,
        dx,
        dy,
        wx,
        wy,
    })
}

fn build_antipodal_kernels(radius: c_uint, degree: c_uint) -> Result<GeneratedKernels, String> {
    let direct = build_direct_kernels(radius, degree)?;
    let index: HashMap<(c_int, c_int), usize> = direct
        .dx
        .iter()
        .zip(&direct.dy)
        .enumerate()
        .map(|(idx, (&dx, &dy))| ((dx, dy), idx))
        .collect();
    let mut used = vec![false; direct.dx.len()];
    let mut dx = Vec::with_capacity(direct.dx.len() / 2);
    let mut dy = Vec::with_capacity(direct.dy.len() / 2);
    let mut wx = Vec::with_capacity(direct.wx.len() / 2);
    let mut wy = Vec::with_capacity(direct.wy.len() / 2);

    for i in 0..direct.dx.len() {
        if used[i] {
            continue;
        }
        let offset_x = direct.dx[i];
        let offset_y = direct.dy[i];
        let opposite = (-offset_x, -offset_y);
        let j = *index
            .get(&opposite)
            .ok_or_else(|| format!("offset ({offset_x}, {offset_y}) has no antipodal partner"))?;
        used[i] = true;
        used[j] = true;

        let (pos, neg, pair_dx, pair_dy) = if offset_y > 0 || (offset_y == 0 && offset_x > 0) {
            (i, j, offset_x, offset_y)
        } else {
            (j, i, opposite.0, opposite.1)
        };
        dx.push(pair_dx);
        dy.push(pair_dy);
        wx.push(0.5 * (direct.wx[pos] - direct.wx[neg]));
        wy.push(0.5 * (direct.wy[pos] - direct.wy[neg]));
    }

    Ok(GeneratedKernels {
        radius,
        dx,
        dy,
        wx,
        wy,
    })
}

fn generated_kernels(
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
) -> Result<GeneratedKernels, String> {
    KERNEL_CACHE.with(|cache_cell| {
        let key = (radius, degree, variant);
        if let Some(kernels) = cache_cell.borrow().get(&key) {
            return Ok(kernels.clone());
        }

        let kernels = match variant {
            WVF_VARIANT_DIRECT => build_direct_kernels(radius, degree),
            WVF_VARIANT_ANTIPODAL | WVF_VARIANT_SPLIT => build_antipodal_kernels(radius, degree),
            _ => Err("variant must be 0=direct, 1=antipodal, or 2=split".to_string()),
        }?;
        cache_cell.borrow_mut().insert(key, kernels.clone());
        Ok(kernels)
    })
}

struct BoundBuffers {
    image: Buffer,
    dx: Buffer,
    dy: Buffer,
    wx: Buffer,
    wy: Buffer,
    out_x: Buffer,
    out_y: Buffer,
}

#[derive(Clone)]
struct KernelPlan {
    radius: c_uint,
    n_offsets: c_uint,
    dx: Buffer,
    dy: Buffer,
    wx: Buffer,
    wy: Buffer,
}

struct ImageOutputBuffers {
    image: Buffer,
    out_x: Buffer,
    out_y: Buffer,
}

struct MagnitudeBuffers {
    magnitude: Buffer,
    angle: Buffer,
}

struct DenseConvolutionKernels {
    kernel_width: c_uint,
    kernel_x: Vec<c_float>,
    kernel_y: Vec<c_float>,
}

fn build_vkfft_convolution_kernels(
    radius: c_uint,
    degree: c_uint,
) -> Result<DenseConvolutionKernels, String> {
    let direct = generated_kernels(radius, degree, WVF_VARIANT_DIRECT)?;
    let kernel_width = radius
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "radius is too large".to_string())?;
    let dense_len = checked_len(kernel_width as usize, kernel_width as usize, "dense kernel")?;
    let mut kernel_x = vec![0.0; dense_len];
    let mut kernel_y = vec![0.0; dense_len];
    let r = c_int::try_from(radius).map_err(|_| "radius is too large".to_string())?;
    let width_i = c_int::try_from(kernel_width).map_err(|_| "radius is too large".to_string())?;

    for idx in 0..direct.dx.len() {
        let x = r - direct.dx[idx];
        let y = r - direct.dy[idx];
        if x < 0 || y < 0 || x >= width_i || y >= width_i {
            return Err("WVF offset fell outside dense convolution kernel".to_string());
        }
        let dense_idx = y as usize * kernel_width as usize + x as usize;
        kernel_x[dense_idx] += direct.wx[idx];
        kernel_y[dense_idx] += direct.wy[idx];
    }

    Ok(DenseConvolutionKernels {
        kernel_width,
        kernel_x,
        kernel_y,
    })
}

fn with_vkfft_convolution_kernels<T>(
    radius: c_uint,
    degree: c_uint,
    f: impl FnOnce(&DenseConvolutionKernels) -> T,
) -> Result<T, String> {
    DENSE_KERNEL_CACHE.with(|cache_cell| {
        let key = (radius, degree);
        if !cache_cell.borrow().contains_key(&key) {
            let kernels = build_vkfft_convolution_kernels(radius, degree)?;
            cache_cell.borrow_mut().insert(key, kernels);
        }
        let cache = cache_cell.borrow();
        let kernels = cache
            .get(&key)
            .ok_or_else(|| "dense VkFFT kernel cache insertion failed".to_string())?;
        Ok(f(kernels))
    })
}

unsafe fn bind_buffers(
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
) -> Result<BoundBuffers, String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let offset_len = checked_len(n_offsets as usize, std::mem::size_of::<c_int>(), "offset")?;
    let weight_len = checked_len(n_offsets as usize, std::mem::size_of::<c_float>(), "weight")?;

    let options = MTLResourceOptions::StorageModeShared;
    let image_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(image.cast(), image_len as u64, options, None);
    let dx_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(dx.cast(), offset_len as u64, options, None);
    let dy_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(dy.cast(), offset_len as u64, options, None);
    let wx_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(wx.cast(), weight_len as u64, options, None);
    let wy_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(wy.cast(), weight_len as u64, options, None);
    let out_x_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_x.cast::<std::ffi::c_void>().cast_const(),
        image_len as u64,
        options,
        None,
    );
    let out_y_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_y.cast::<std::ffi::c_void>().cast_const(),
        image_len as u64,
        options,
        None,
    );

    image_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    dx_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    dy_buffer.did_modify_range(NSRange::new(0, offset_len as u64));
    wx_buffer.did_modify_range(NSRange::new(0, weight_len as u64));
    wy_buffer.did_modify_range(NSRange::new(0, weight_len as u64));

    Ok(BoundBuffers {
        image: image_buffer,
        dx: dx_buffer,
        dy: dy_buffer,
        wx: wx_buffer,
        wy: wy_buffer,
        out_x: out_x_buffer,
        out_y: out_y_buffer,
    })
}

unsafe fn bind_image_output_buffers(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<ImageOutputBuffers, String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let image_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "image")?;
    let options = MTLResourceOptions::StorageModeShared;
    let image_buffer =
        state
            .device
            .new_buffer_with_bytes_no_copy(image.cast(), image_len as u64, options, None);
    let out_x_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_x.cast::<std::ffi::c_void>().cast_const(),
        image_len as u64,
        options,
        None,
    );
    let out_y_buffer = state.device.new_buffer_with_bytes_no_copy(
        out_y.cast::<std::ffi::c_void>().cast_const(),
        image_len as u64,
        options,
        None,
    );
    image_buffer.did_modify_range(NSRange::new(0, image_len as u64));
    Ok(ImageOutputBuffers {
        image: image_buffer,
        out_x: out_x_buffer,
        out_y: out_y_buffer,
    })
}

unsafe fn bind_magnitude_buffers(
    state: &MetalState,
    width: c_uint,
    height: c_uint,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<MagnitudeBuffers, String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let output_len = checked_len(total_pixels, std::mem::size_of::<c_float>(), "output")?;
    let options = MTLResourceOptions::StorageModeShared;
    let magnitude_buffer = state.device.new_buffer_with_bytes_no_copy(
        magnitude.cast::<std::ffi::c_void>().cast_const(),
        output_len as u64,
        options,
        None,
    );
    let angle_buffer = state.device.new_buffer_with_bytes_no_copy(
        angle.cast::<std::ffi::c_void>().cast_const(),
        output_len as u64,
        options,
        None,
    );
    Ok(MagnitudeBuffers {
        magnitude: magnitude_buffer,
        angle: angle_buffer,
    })
}

fn set_common_buffers(encoder: &metal::ComputeCommandEncoderRef, buffers: &BoundBuffers) {
    encoder.set_buffer(0, Some(&buffers.image), 0);
    encoder.set_buffer(1, Some(&buffers.dx), 0);
    encoder.set_buffer(2, Some(&buffers.dy), 0);
    encoder.set_buffer(3, Some(&buffers.wx), 0);
    encoder.set_buffer(4, Some(&buffers.wy), 0);
    encoder.set_buffer(5, Some(&buffers.out_x), 0);
    encoder.set_buffer(6, Some(&buffers.out_y), 0);
}

fn set_magnitude_buffers(encoder: &metal::ComputeCommandEncoderRef, buffers: &MagnitudeBuffers) {
    encoder.set_buffer(8, Some(&buffers.magnitude), 0);
    encoder.set_buffer(9, Some(&buffers.angle), 0);
}

fn set_plan_buffers(
    encoder: &metal::ComputeCommandEncoderRef,
    buffers: &ImageOutputBuffers,
    plan: &KernelPlan,
) {
    encoder.set_buffer(0, Some(&buffers.image), 0);
    encoder.set_buffer(1, Some(&plan.dx), 0);
    encoder.set_buffer(2, Some(&plan.dy), 0);
    encoder.set_buffer(3, Some(&plan.wx), 0);
    encoder.set_buffer(4, Some(&plan.wy), 0);
    encoder.set_buffer(5, Some(&buffers.out_x), 0);
    encoder.set_buffer(6, Some(&buffers.out_y), 0);
}

fn build_kernel_plan(
    state: &MetalState,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
) -> Result<KernelPlan, String> {
    let kernels = generated_kernels(radius, degree, variant)?;
    let n_offsets = kernels.n_offsets()?;
    let offset_len = checked_len(kernels.dx.len(), std::mem::size_of::<c_int>(), "offset")?;
    let weight_len = checked_len(kernels.wx.len(), std::mem::size_of::<c_float>(), "weight")?;
    let options = MTLResourceOptions::StorageModeShared;
    let dx =
        state
            .device
            .new_buffer_with_data(kernels.dx.as_ptr().cast(), offset_len as u64, options);
    let dy =
        state
            .device
            .new_buffer_with_data(kernels.dy.as_ptr().cast(), offset_len as u64, options);
    let wx =
        state
            .device
            .new_buffer_with_data(kernels.wx.as_ptr().cast(), weight_len as u64, options);
    let wy =
        state
            .device
            .new_buffer_with_data(kernels.wy.as_ptr().cast(), weight_len as u64, options);
    Ok(KernelPlan {
        radius: kernels.radius,
        n_offsets,
        dx,
        dy,
        wx,
        wy,
    })
}

fn generated_kernel_plan(
    state: &MetalState,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
) -> Result<KernelPlan, String> {
    PLAN_CACHE.with(|cache_cell| {
        let key = (radius, degree, variant);
        if let Some(plan) = cache_cell.borrow().get(&key) {
            return Ok(plan.clone());
        }
        let plan = build_kernel_plan(state, radius, degree, variant)?;
        cache_cell.borrow_mut().insert(key, plan.clone());
        Ok(plan)
    })
}

unsafe fn run_convolve_with_state(
    state: &MetalState,
    pipeline: &ComputePipelineState,
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
    let buffers = bind_buffers(
        state, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )?;
    let params = KernelParams {
        width,
        height,
        n_offsets,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const KernelParams).cast(),
        std::mem::size_of::<KernelParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    set_common_buffers(encoder, &buffers);
    encoder.set_buffer(7, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_plan_convolve_with_state(
    state: &MetalState,
    pipeline: &ComputePipelineState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    plan: &KernelPlan,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let buffers = bind_image_output_buffers(state, image, width, height, out_x, out_y)?;
    let params = KernelParams {
        width,
        height,
        n_offsets: plan.n_offsets,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const KernelParams).cast(),
        std::mem::size_of::<KernelParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    set_plan_buffers(encoder, &buffers, plan);
    encoder.set_buffer(7, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_plan_convolve_magnitude_angle_with_state(
    state: &MetalState,
    pipeline: &ComputePipelineState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    plan: &KernelPlan,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    let buffers = bind_image_output_buffers(state, image, width, height, out_x, out_y)?;
    let magnitude_buffers = bind_magnitude_buffers(state, width, height, magnitude, angle)?;
    let params = KernelParams {
        width,
        height,
        n_offsets: plan.n_offsets,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const KernelParams).cast(),
        std::mem::size_of::<KernelParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    set_plan_buffers(encoder, &buffers, plan);
    set_magnitude_buffers(encoder, &magnitude_buffers);
    encoder.set_buffer(7, Some(&params_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_split_with_state(
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
        return run_convolve_with_state(
            state,
            &state.antipodal_pipeline,
            image,
            width,
            height,
            dx,
            dy,
            wx,
            wy,
            n_offsets,
            out_x,
            out_y,
        );
    }

    let buffers = bind_buffers(
        state, image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )?;
    let params = SplitParams {
        width,
        height,
        n_offsets,
        radius,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const SplitParams).cast(),
        std::mem::size_of::<SplitParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    set_common_buffers(encoder, &buffers);
    encoder.set_buffer(7, Some(&params_buffer), 0);

    encoder.set_compute_pipeline_state(&state.split_interior_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: interior_width as u64,
            height: interior_height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_interior_pipeline),
    );

    encoder.set_compute_pipeline_state(&state.split_boundary_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_boundary_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_plan_split_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    plan: &KernelPlan,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let double_radius = plan.radius.saturating_mul(2);
    let interior_width = width.saturating_sub(double_radius);
    let interior_height = height.saturating_sub(double_radius);
    if plan.radius == 0 || interior_width == 0 || interior_height == 0 {
        return run_plan_convolve_with_state(
            state,
            &state.antipodal_pipeline,
            image,
            width,
            height,
            plan,
            out_x,
            out_y,
        );
    }

    let buffers = bind_image_output_buffers(state, image, width, height, out_x, out_y)?;
    let params = SplitParams {
        width,
        height,
        n_offsets: plan.n_offsets,
        radius: plan.radius,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const SplitParams).cast(),
        std::mem::size_of::<SplitParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    set_plan_buffers(encoder, &buffers, plan);
    encoder.set_buffer(7, Some(&params_buffer), 0);

    encoder.set_compute_pipeline_state(&state.split_interior_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: interior_width as u64,
            height: interior_height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_interior_pipeline),
    );

    encoder.set_compute_pipeline_state(&state.split_boundary_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_boundary_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_plan_split_magnitude_angle_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    plan: &KernelPlan,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    let double_radius = plan.radius.saturating_mul(2);
    let interior_width = width.saturating_sub(double_radius);
    let interior_height = height.saturating_sub(double_radius);
    if plan.radius == 0 || interior_width == 0 || interior_height == 0 {
        return run_plan_convolve_magnitude_angle_with_state(
            state,
            &state.antipodal_magnitude_pipeline,
            image,
            width,
            height,
            plan,
            out_x,
            out_y,
            magnitude,
            angle,
        );
    }

    let buffers = bind_image_output_buffers(state, image, width, height, out_x, out_y)?;
    let magnitude_buffers = bind_magnitude_buffers(state, width, height, magnitude, angle)?;
    let params = SplitParams {
        width,
        height,
        n_offsets: plan.n_offsets,
        radius: plan.radius,
    };
    let params_buffer = state.device.new_buffer_with_data(
        (&params as *const SplitParams).cast(),
        std::mem::size_of::<SplitParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    set_plan_buffers(encoder, &buffers, plan);
    set_magnitude_buffers(encoder, &magnitude_buffers);
    encoder.set_buffer(7, Some(&params_buffer), 0);

    encoder.set_compute_pipeline_state(&state.split_interior_magnitude_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: interior_width as u64,
            height: interior_height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_interior_magnitude_pipeline),
    );

    encoder.set_compute_pipeline_state(&state.split_boundary_magnitude_pipeline);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(&state.split_boundary_magnitude_pipeline),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

unsafe fn run_checked<F>(runner: F) -> Result<(), String>
where
    F: FnOnce(&MetalState) -> Result<(), String>,
{
    METAL_STATE.with(|state_cell| {
        let mut state_slot = state_cell.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(MetalState::new()?);
        }
        let state = state_slot.as_ref().expect("Metal state was initialized");
        runner(state)
    })
}

unsafe fn run_vkfft_magnitude_angle(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    with_vkfft_convolution_kernels(radius, degree, |kernels| {
        let mut error = vec![0 as c_char; 4096];
        let status = wvf_vkfft_magnitude_angle(
            image,
            width,
            height,
            radius,
            kernels.kernel_x.as_ptr(),
            kernels.kernel_y.as_ptr(),
            kernels.kernel_width,
            out_x,
            out_y,
            magnitude,
            angle,
            error.as_mut_ptr(),
            error.len(),
        );
        if status == 0 {
            return Ok(());
        }

        let message = CStr::from_ptr(error.as_ptr())
            .to_string_lossy()
            .trim()
            .to_string();
        if message.is_empty() {
            Err(format!("VkFFT WVF backend failed with status {status}"))
        } else {
            Err(message)
        }
    })?
}

unsafe fn run_vkfft_gradients(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let total_pixels = checked_image_pixels(width, height)?;
    let mut magnitude = vec![0.0; total_pixels];
    let mut angle = vec![0.0; total_pixels];
    run_vkfft_magnitude_angle(
        image,
        width,
        height,
        radius,
        degree,
        out_x,
        out_y,
        magnitude.as_mut_ptr(),
        angle.as_mut_ptr(),
    )
}

unsafe fn validate_common(
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
    Ok(())
}

unsafe fn validate_generated(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    check_ptr(image, "image")?;
    check_mut_ptr(out_x, "out_x")?;
    check_mut_ptr(out_y, "out_y")?;
    if width == 0 || height == 0 {
        return Err("image width and height must be positive".to_string());
    }
    if radius == 0 {
        return Err("radius must be positive".to_string());
    }
    if degree < 1 {
        return Err("degree must be at least 1".to_string());
    }
    if !matches!(
        variant,
        WVF_VARIANT_DIRECT | WVF_VARIANT_ANTIPODAL | WVF_VARIANT_SPLIT | WVF_VARIANT_FFT
    ) {
        return Err("variant must be 0=direct, 1=antipodal, 2=split, or 3=fft/vkfft".to_string());
    }
    Ok(())
}

unsafe fn run_generated_gradients_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    if variant == WVF_VARIANT_FFT {
        return run_vkfft_gradients(image, width, height, radius, degree, out_x, out_y);
    }

    let plan = generated_kernel_plan(state, radius, degree, variant)?;
    match variant {
        WVF_VARIANT_DIRECT => run_plan_convolve_with_state(
            state,
            &state.direct_pipeline,
            image,
            width,
            height,
            &plan,
            out_x,
            out_y,
        ),
        WVF_VARIANT_ANTIPODAL => run_plan_convolve_with_state(
            state,
            &state.antipodal_pipeline,
            image,
            width,
            height,
            &plan,
            out_x,
            out_y,
        ),
        WVF_VARIANT_SPLIT => {
            run_plan_split_with_state(state, image, width, height, &plan, out_x, out_y)
        }
        _ => Err("variant must be 0=direct, 1=antipodal, 2=split, or 3=fft/vkfft".to_string()),
    }
}

unsafe fn run_generated_magnitude_angle_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    if variant == WVF_VARIANT_FFT {
        return run_vkfft_magnitude_angle(
            image, width, height, radius, degree, out_x, out_y, magnitude, angle,
        );
    }

    let plan = generated_kernel_plan(state, radius, degree, variant)?;
    match variant {
        WVF_VARIANT_DIRECT => run_plan_convolve_magnitude_angle_with_state(
            state,
            &state.direct_magnitude_pipeline,
            image,
            width,
            height,
            &plan,
            out_x,
            out_y,
            magnitude,
            angle,
        ),
        WVF_VARIANT_ANTIPODAL => run_plan_convolve_magnitude_angle_with_state(
            state,
            &state.antipodal_magnitude_pipeline,
            image,
            width,
            height,
            &plan,
            out_x,
            out_y,
            magnitude,
            angle,
        ),
        WVF_VARIANT_SPLIT => run_plan_split_magnitude_angle_with_state(
            state, image, width, height, &plan, out_x, out_y, magnitude, angle,
        ),
        _ => Err("variant must be 0=direct, 1=antipodal, 2=split, or 3=fft/vkfft".to_string()),
    }
}

#[no_mangle]
pub unsafe extern "C" fn wvf_metal_gradients(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    let result = validate_generated(image, width, height, radius, degree, variant, out_x, out_y)
        .and_then(|()| {
            run_checked(|state| {
                run_generated_gradients_with_state(
                    state, image, width, height, radius, degree, variant, out_x, out_y,
                )
            })
        });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wvf_metal_magnitude_angle(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    let result = validate_generated(image, width, height, radius, degree, variant, out_x, out_y)
        .and_then(|()| check_mut_ptr(magnitude, "magnitude"))
        .and_then(|()| check_mut_ptr(angle, "angle"))
        .and_then(|()| {
            run_checked(|state| {
                run_generated_magnitude_angle_with_state(
                    state, image, width, height, radius, degree, variant, out_x, out_y, magnitude,
                    angle,
                )
            })
        });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wvf_metal_convolve_direct(
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
    let result = validate_common(
        image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )
    .and_then(|()| {
        run_checked(|state| {
            run_convolve_with_state(
                state,
                &state.direct_pipeline,
                image,
                width,
                height,
                dx,
                dy,
                wx,
                wy,
                n_offsets,
                out_x,
                out_y,
            )
        })
    });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wvf_metal_convolve_antipodal(
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
    let result = validate_common(
        image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )
    .and_then(|()| {
        run_checked(|state| {
            run_convolve_with_state(
                state,
                &state.antipodal_pipeline,
                image,
                width,
                height,
                dx,
                dy,
                wx,
                wy,
                n_offsets,
                out_x,
                out_y,
            )
        })
    });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wvf_metal_convolve_split(
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
    let result = validate_common(
        image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )
    .and_then(|()| {
        run_checked(|state| {
            run_split_with_state(
                state, image, width, height, dx, dy, wx, wy, n_offsets, radius, out_x, out_y,
            )
        })
    });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}
