use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::{c_float, c_int, c_uint};

mod interop;
mod mps_graph;

thread_local! {
    static DENSE_KERNEL_CACHE: RefCell<HashMap<(c_uint, c_uint), DenseConvolutionKernels>> =
        RefCell::new(HashMap::new());
    static FFT_BACKEND: RefCell<Option<mps_graph::MpsGraphFftBackend>> = const { RefCell::new(None) };
}

#[derive(Clone)]
pub(super) struct DenseConvolutionKernels {
    pub(super) kernel_width: c_uint,
    pub(super) kernel_x: Vec<c_float>,
    pub(super) kernel_y: Vec<c_float>,
}

fn build_dense_convolution_kernels(
    radius: c_uint,
    degree: c_uint,
) -> Result<DenseConvolutionKernels, String> {
    let direct = crate::generated_kernels(radius, degree, crate::WVF_VARIANT_DIRECT)?;
    let kernel_width = radius
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "radius is too large".to_string())?;
    let dense_len = crate::checked_len(
        kernel_width as usize,
        kernel_width as usize,
        "dense kernel",
    )?;
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

fn with_dense_convolution_kernels<T>(
    radius: c_uint,
    degree: c_uint,
    f: impl FnOnce(&DenseConvolutionKernels) -> T,
) -> Result<T, String> {
    DENSE_KERNEL_CACHE.with(|cache_cell| {
        let key = (radius, degree);
        if !cache_cell.borrow().contains_key(&key) {
            let kernels = build_dense_convolution_kernels(radius, degree)?;
            cache_cell.borrow_mut().insert(key, kernels);
        }
        let cache = cache_cell.borrow();
        let kernels = cache
            .get(&key)
            .ok_or_else(|| "dense FFT kernel cache insertion failed".to_string())?;
        Ok(f(kernels))
    })
}

fn with_backend<T>(
    f: impl FnOnce(&mut mps_graph::MpsGraphFftBackend) -> Result<T, String>,
) -> Result<T, String> {
    FFT_BACKEND.with(|backend_cell| {
        let mut backend_slot = backend_cell.borrow_mut();
        if backend_slot.is_none() {
            *backend_slot = Some(mps_graph::MpsGraphFftBackend::new()?);
        }
        let backend = backend_slot
            .as_mut()
            .ok_or_else(|| "FFT backend initialization failed".to_string())?;
        f(backend)
    })
}

pub(crate) unsafe fn run_fft_magnitude_angle(
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
    with_dense_convolution_kernels(radius, degree, |kernels| {
        with_backend(|backend| {
            backend.run_magnitude_angle(
                image, width, height, radius, kernels, out_x, out_y, magnitude, angle,
            )
        })
    })??;
    Ok(())
}

pub(crate) unsafe fn run_fft_gradients(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
) -> Result<(), String> {
    let total_pixels = crate::checked_image_pixels(width, height)?;
    let mut magnitude = vec![0.0; total_pixels];
    let mut angle = vec![0.0; total_pixels];
    run_fft_magnitude_angle(
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
