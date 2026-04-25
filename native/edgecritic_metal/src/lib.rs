use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    NSRange,
};
use std::cell::RefCell;
use std::ffi::c_char;
use std::os::raw::{c_float, c_int, c_uint};
use std::ptr;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    uint width;
    uint height;
    uint n_offsets;
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
"#;

#[repr(C)]
struct KernelParams {
    width: c_uint,
    height: c_uint,
    n_offsets: c_uint,
}

struct MetalState {
    device: Device,
    pipeline: ComputePipelineState,
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
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("failed to create Metal compute pipeline: {err}"))?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            pipeline,
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
    let total_pixels = width as usize * height as usize;
    let image_len = total_pixels * std::mem::size_of::<c_float>();
    let offset_len = n_offsets as usize * std::mem::size_of::<c_int>();
    let weight_len = n_offsets as usize * std::mem::size_of::<c_float>();
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
    encoder.set_compute_pipeline_state(&state.pipeline);
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
    let group = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
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
