use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};
use std::cell::RefCell;
use std::ffi::c_char;
use std::os::raw::{c_float, c_int, c_uint};
use std::ptr;

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

struct MetalState {
    device: Device,
    direct_pipeline: ComputePipelineState,
    antipodal_pipeline: ComputePipelineState,
    split_interior_pipeline: ComputePipelineState,
    split_boundary_pipeline: ComputePipelineState,
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
        let antipodal_pipeline = pipeline(&device, &library, "wvf_antipodal")?;
        let split_interior_pipeline = pipeline(&device, &library, "wvf_split_interior")?;
        let split_boundary_pipeline = pipeline(&device, &library, "wvf_split_boundary")?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            direct_pipeline,
            antipodal_pipeline,
            split_interior_pipeline,
            split_boundary_pipeline,
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

struct BoundBuffers {
    image: Buffer,
    dx: Buffer,
    dy: Buffer,
    wx: Buffer,
    wy: Buffer,
    out_x: Buffer,
    out_y: Buffer,
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

fn set_common_buffers(encoder: &metal::ComputeCommandEncoderRef, buffers: &BoundBuffers) {
    encoder.set_buffer(0, Some(&buffers.image), 0);
    encoder.set_buffer(1, Some(&buffers.dx), 0);
    encoder.set_buffer(2, Some(&buffers.dy), 0);
    encoder.set_buffer(3, Some(&buffers.wx), 0);
    encoder.set_buffer(4, Some(&buffers.wy), 0);
    encoder.set_buffer(5, Some(&buffers.out_x), 0);
    encoder.set_buffer(6, Some(&buffers.out_y), 0);
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
