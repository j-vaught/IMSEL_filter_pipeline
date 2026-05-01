use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::os::raw::{c_float, c_int, c_uint};
use std::ptr;

mod fft_backend;
#[cfg(target_os = "macos")]
mod platform_metal;

const WVF_VARIANT_DIRECT: c_uint = 0;
const WVF_VARIANT_ANTIPODAL: c_uint = 1;
const WVF_VARIANT_SPLIT: c_uint = 2;
const WVF_VARIANT_FFT: c_uint = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FftBackendMode {
    Auto,
    Cpu,
    Vkfft,
}

#[cfg(wvf_has_vkfft)]
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

#[cfg(wvf_has_vkfft)]
extern "C" {
    fn wvf_vkfft_gradients(
        image: *const c_float,
        width: c_uint,
        height: c_uint,
        radius: c_uint,
        kernel_x: *const c_float,
        kernel_y: *const c_float,
        kernel_width: c_uint,
        out_x: *mut c_float,
        out_y: *mut c_float,
        error_out: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn wvf_vkfft_magnitude(
        image: *const c_float,
        width: c_uint,
        height: c_uint,
        radius: c_uint,
        kernel_x: *const c_float,
        kernel_y: *const c_float,
        kernel_width: c_uint,
        magnitude: *mut c_float,
        error_out: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn wvf_vkfft_magnitude_orientation(
        image: *const c_float,
        width: c_uint,
        height: c_uint,
        radius: c_uint,
        kernel_x: *const c_float,
        kernel_y: *const c_float,
        kernel_width: c_uint,
        magnitude: *mut c_float,
        angle: *mut c_float,
        error_out: *mut c_char,
        error_len: usize,
    ) -> c_int;
}

thread_local! {
    static KERNEL_CACHE: RefCell<HashMap<(c_uint, c_uint, c_uint), GeneratedKernels>> = RefCell::new(HashMap::new());
    static DENSE_KERNEL_CACHE: RefCell<HashMap<(c_uint, c_uint), DenseConvolutionKernels>> = RefCell::new(HashMap::new());
}

fn selected_gpu_device_index() -> Result<Option<usize>, String> {
    let raw_index = std::env::var("WVF_GPU_DEVICE_INDEX")
        .ok()
        .or_else(|| std::env::var("WVF_METAL_DEVICE_INDEX").ok());
    match raw_index.as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|_| "GPU device index must be a non-negative integer".to_string()),
    }
}

fn selected_fft_backend() -> Result<FftBackendMode, String> {
    match std::env::var("WVF_METAL_FFT_BACKEND")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("") | Some("auto") => Ok(FftBackendMode::Auto),
        Some("cpu") => Ok(FftBackendMode::Cpu),
        Some("vkfft") | Some("metal") | Some("gpu") => Ok(FftBackendMode::Vkfft),
        Some(other) => Err(format!(
            "WVF_METAL_FFT_BACKEND must be auto, cpu, or vkfft, got {other}"
        )),
    }
}

#[cfg(target_os = "macos")]
fn vkfft_backend_available_for_auto() -> Result<bool, String> {
    platform_metal::metal_device_available()
}

#[cfg(not(target_os = "macos"))]
fn vkfft_backend_available_for_auto() -> Result<bool, String> {
    Ok(cfg!(wvf_has_vkfft))
}

fn should_use_cpu_fft_backend() -> Result<bool, String> {
    match selected_fft_backend()? {
        FftBackendMode::Cpu => Ok(true),
        FftBackendMode::Vkfft => Ok(false),
        FftBackendMode::Auto => Ok(!vkfft_backend_available_for_auto()?),
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

pub(crate) struct DenseConvolutionKernels {
    pub(crate) kernel_width: c_uint,
    pub(crate) kernel_x: Vec<c_float>,
    pub(crate) kernel_y: Vec<c_float>,
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

#[cfg(wvf_has_vkfft)]
unsafe fn vkfft_status_to_result(status: c_int, error: &[c_char]) -> Result<(), String> {
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
}

#[cfg(wvf_has_vkfft)]
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
        vkfft_status_to_result(status, &error)
    })?
}

#[cfg(not(wvf_has_vkfft))]
unsafe fn run_vkfft_magnitude_angle(
    _image: *const c_float,
    _width: c_uint,
    _height: c_uint,
    _radius: c_uint,
    _degree: c_uint,
    _out_x: *mut c_float,
    _out_y: *mut c_float,
    _magnitude: *mut c_float,
    _angle: *mut c_float,
) -> Result<(), String> {
    Err("VkFFT GPU backend is not available in this build".to_string())
}

unsafe fn run_vkfft_magnitude_orientation(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    magnitude: *mut c_float,
    angle: *mut c_float,
) -> Result<(), String> {
    #[cfg(wvf_has_vkfft)]
    {
        return with_vkfft_convolution_kernels(radius, degree, |kernels| {
            let mut error = vec![0 as c_char; 4096];
            let status = wvf_vkfft_magnitude_orientation(
                image,
                width,
                height,
                radius,
                kernels.kernel_x.as_ptr(),
                kernels.kernel_y.as_ptr(),
                kernels.kernel_width,
                magnitude,
                angle,
                error.as_mut_ptr(),
                error.len(),
            );
            vkfft_status_to_result(status, &error)
        })?;
    }

    #[cfg(not(wvf_has_vkfft))]
    {
        let _ = (image, width, height, radius, degree, magnitude, angle);
        Err("VkFFT GPU backend is not available in this build".to_string())
    }
}

unsafe fn run_vkfft_magnitude(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    magnitude: *mut c_float,
) -> Result<(), String> {
    #[cfg(wvf_has_vkfft)]
    {
        return with_vkfft_convolution_kernels(radius, degree, |kernels| {
            let mut error = vec![0 as c_char; 4096];
            let status = wvf_vkfft_magnitude(
                image,
                width,
                height,
                radius,
                kernels.kernel_x.as_ptr(),
                kernels.kernel_y.as_ptr(),
                kernels.kernel_width,
                magnitude,
                error.as_mut_ptr(),
                error.len(),
            );
            vkfft_status_to_result(status, &error)
        })?;
    }

    #[cfg(not(wvf_has_vkfft))]
    {
        let _ = (image, width, height, radius, degree, magnitude);
        Err("VkFFT GPU backend is not available in this build".to_string())
    }
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
    #[cfg(wvf_has_vkfft)]
    {
        return with_vkfft_convolution_kernels(radius, degree, |kernels| {
            let mut error = vec![0 as c_char; 4096];
            let status = wvf_vkfft_gradients(
                image,
                width,
                height,
                radius,
                kernels.kernel_x.as_ptr(),
                kernels.kernel_y.as_ptr(),
                kernels.kernel_width,
                out_x,
                out_y,
                error.as_mut_ptr(),
                error.len(),
            );
            vkfft_status_to_result(status, &error)
        })?;
    }

    #[cfg(not(wvf_has_vkfft))]
    {
        let _ = (image, width, height, radius, degree, out_x, out_y);
        Err("VkFFT GPU backend is not available in this build".to_string())
    }
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

unsafe fn validate_generated_common(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
) -> Result<(), String> {
    check_ptr(image, "image")?;
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
    validate_generated_common(image, width, height, radius, degree, variant)?;
    check_mut_ptr(out_x, "out_x")?;
    check_mut_ptr(out_y, "out_y")?;
    Ok(())
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
            if variant == WVF_VARIANT_FFT && should_use_cpu_fft_backend()? {
                fft_backend::run_fft_gradients_cpu(
                    image, width, height, radius, degree, out_x, out_y,
                )
            } else if variant == WVF_VARIANT_FFT {
                run_vkfft_gradients(image, width, height, radius, degree, out_x, out_y)
            } else {
                #[cfg(target_os = "macos")]
                {
                    platform_metal::run_checked(|state| {
                        platform_metal::run_generated_gradients_with_state(
                            state, image, width, height, radius, degree, variant, out_x, out_y,
                        )
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("direct, antipodal, and split variants require macOS Metal".to_string())
                }
            }
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
            if variant == WVF_VARIANT_FFT && should_use_cpu_fft_backend()? {
                fft_backend::run_fft_magnitude_angle_cpu(
                    image, width, height, radius, degree, out_x, out_y, magnitude, angle,
                )
            } else if variant == WVF_VARIANT_FFT {
                run_vkfft_magnitude_angle(
                    image, width, height, radius, degree, out_x, out_y, magnitude, angle,
                )
            } else {
                #[cfg(target_os = "macos")]
                {
                    platform_metal::run_checked(|state| {
                        platform_metal::run_generated_magnitude_angle_with_state(
                            state, image, width, height, radius, degree, variant, out_x, out_y,
                            magnitude, angle,
                        )
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("direct, antipodal, and split variants require macOS Metal".to_string())
                }
            }
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
pub unsafe extern "C" fn wvf_metal_magnitude(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    magnitude: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    let result = validate_generated_common(image, width, height, radius, degree, variant)
        .and_then(|()| check_mut_ptr(magnitude, "magnitude"))
        .and_then(|()| {
            if variant == WVF_VARIANT_FFT && should_use_cpu_fft_backend()? {
                fft_backend::run_fft_magnitude_cpu(image, width, height, radius, degree, magnitude)
            } else if variant == WVF_VARIANT_FFT {
                run_vkfft_magnitude(image, width, height, radius, degree, magnitude)
            } else {
                #[cfg(target_os = "macos")]
                {
                    let total_pixels = checked_image_pixels(width, height)?;
                    let mut out_x = vec![0.0; total_pixels];
                    let mut out_y = vec![0.0; total_pixels];
                    let mut angle = vec![0.0; total_pixels];
                    platform_metal::run_checked(|state| {
                        platform_metal::run_generated_magnitude_angle_with_state(
                            state,
                            image,
                            width,
                            height,
                            radius,
                            degree,
                            variant,
                            out_x.as_mut_ptr(),
                            out_y.as_mut_ptr(),
                            magnitude,
                            angle.as_mut_ptr(),
                        )
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("direct, antipodal, and split variants require macOS Metal".to_string())
                }
            }
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
pub unsafe extern "C" fn wvf_metal_magnitude_orientation(
    image: *const c_float,
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    degree: c_uint,
    variant: c_uint,
    magnitude: *mut c_float,
    angle: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    let result = validate_generated_common(image, width, height, radius, degree, variant)
        .and_then(|()| check_mut_ptr(magnitude, "magnitude"))
        .and_then(|()| check_mut_ptr(angle, "angle"))
        .and_then(|()| {
            if variant == WVF_VARIANT_FFT && should_use_cpu_fft_backend()? {
                fft_backend::run_fft_magnitude_orientation_cpu(
                    image, width, height, radius, degree, magnitude, angle,
                )
            } else if variant == WVF_VARIANT_FFT {
                run_vkfft_magnitude_orientation(
                    image, width, height, radius, degree, magnitude, angle,
                )
            } else {
                #[cfg(target_os = "macos")]
                {
                    let total_pixels = checked_image_pixels(width, height)?;
                    let mut out_x = vec![0.0; total_pixels];
                    let mut out_y = vec![0.0; total_pixels];
                    platform_metal::run_checked(|state| {
                        platform_metal::run_generated_magnitude_angle_with_state(
                            state,
                            image,
                            width,
                            height,
                            radius,
                            degree,
                            variant,
                            out_x.as_mut_ptr(),
                            out_y.as_mut_ptr(),
                            magnitude,
                            angle,
                        )
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err("direct, antipodal, and split variants require macOS Metal".to_string())
                }
            }
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
        #[cfg(target_os = "macos")]
        {
            platform_metal::run_convolve_direct(
                image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("direct convolution requires macOS Metal".to_string())
        }
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
        #[cfg(target_os = "macos")]
        {
            platform_metal::run_convolve_antipodal(
                image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("antipodal convolution requires macOS Metal".to_string())
        }
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
    _radius: c_uint,
    out_x: *mut c_float,
    out_y: *mut c_float,
    error_out: *mut c_char,
    error_len: usize,
) -> c_int {
    let result = validate_common(
        image, width, height, dx, dy, wx, wy, n_offsets, out_x, out_y,
    )
    .and_then(|()| {
        #[cfg(target_os = "macos")]
        {
            platform_metal::run_convolve_split(
                image, width, height, dx, dy, wx, wy, n_offsets, _radius, out_x, out_y,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err("split convolution requires macOS Metal".to_string())
        }
    });
    match result {
        Ok(()) => 0,
        Err(message) => {
            write_error(error_out, error_len, &message);
            1
        }
    }
}
