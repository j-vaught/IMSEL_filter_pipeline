use super::*;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};
use std::cell::RefCell;
use std::collections::HashMap;

const SHADER_SOURCE: &str = include_str!("wvf.metal");

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

pub(crate) struct MetalState {
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
    fn new(device: Device) -> Result<Self, String> {
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
    static METAL_STATE: RefCell<HashMap<i32, MetalState>> = RefCell::new(HashMap::new());
    static PLAN_CACHE: RefCell<HashMap<(c_uint, c_uint, c_uint, bool, u8), KernelPlan>> = RefCell::new(HashMap::new());
}

fn selected_metal_device() -> Result<(i32, Device), String> {
    let requested = selected_gpu_device_index()?;
    let device_key = requested
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| "GPU device index exceeded supported range".to_string())
        })
        .transpose()?
        .unwrap_or(-1);

    if device_key < 0 {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        return Ok((device_key, device));
    }

    let devices = Device::all();
    let index = device_key as usize;
    if index >= devices.len() {
        return Err(format!(
            "GPU device index {index} is out of range for {} Metal device(s)",
            devices.len()
        ));
    }
    let device = devices
        .into_iter()
        .nth(index)
        .ok_or_else(|| "failed to select requested Metal device".to_string())?;
    Ok((device_key, device))
}

pub(crate) fn metal_device_available() -> Result<bool, String> {
    match selected_metal_device() {
        Ok(_) => Ok(true),
        Err(message) if message == "no Metal device is available" => Ok(false),
        Err(message) => Err(message),
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
    normalize_coords: bool,
    variant: c_uint,
) -> Result<KernelPlan, String> {
    let kernels = generated_kernels(radius, degree, variant, normalize_coords)?;
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
    normalize_coords: bool,
    variant: c_uint,
) -> Result<KernelPlan, String> {
    PLAN_CACHE.with(|cache_cell| {
        let key = (
            radius,
            degree,
            variant,
            normalize_coords,
            KERNEL_WEIGHT_PRECISION_F32,
        );
        if let Some(plan) = cache_cell.borrow().get(&key) {
            return Ok(plan.clone());
        }
        let plan = build_kernel_plan(state, radius, degree, normalize_coords, variant)?;
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

pub(crate) unsafe fn run_checked<F>(runner: F) -> Result<(), String>
where
    F: FnOnce(&MetalState) -> Result<(), String>,
{
    let (device_key, device) = selected_metal_device()?;
    METAL_STATE.with(|state_cell| {
        let mut states = state_cell.borrow_mut();
        if !states.contains_key(&device_key) {
            states.insert(device_key, MetalState::new(device)?);
        }
        let state = states
            .get(&device_key)
            .ok_or_else(|| "Metal state initialization failed".to_string())?;
        runner(state)
    })
}

pub(crate) unsafe fn run_generated_gradients_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    normalize_coords: bool,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let plan = generated_kernel_plan(state, radius, degree, normalize_coords, variant)?;
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
        _ => Err("variant must be 0=direct, 1=antipodal, 2=split".to_string()),
    }
}

pub(crate) unsafe fn run_generated_magnitude_angle_with_state(
    state: &MetalState,
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    normalize_coords: bool,
    variant: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    let plan = generated_kernel_plan(state, radius, degree, normalize_coords, variant)?;
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
        _ => Err("variant must be 0=direct, 1=antipodal, 2=split".to_string()),
    }
}

pub(crate) unsafe fn run_convolve_direct(
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
}

pub(crate) unsafe fn run_convolve_antipodal(
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
}

pub(crate) unsafe fn run_convolve_split(
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
    run_checked(|state| {
        run_split_with_state(
            state, image, width, height, dx, dy, wx, wy, n_offsets, radius, out_x, out_y,
        )
    })
}
