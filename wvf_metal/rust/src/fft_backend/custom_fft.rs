use super::DenseConvolutionKernels;
use num_complex::Complex32;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::collections::HashMap;
use std::mem::size_of;
use std::os::raw::{c_float, c_uint};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlanKey {
    fft_w: c_uint,
    fft_h: c_uint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct KernelKey {
    fft_w: c_uint,
    fft_h: c_uint,
    radius: c_uint,
    kernel_width: c_uint,
    kernel_hash: u64,
}

struct FftPlan {
    fft_w: usize,
    fft_h: usize,
    plane_len: usize,
    forward_width: Arc<dyn Fft<f32>>,
    inverse_width: Arc<dyn Fft<f32>>,
    forward_height: Arc<dyn Fft<f32>>,
    inverse_height: Arc<dyn Fft<f32>>,
}

struct KernelSpectra {
    values: Arc<[Complex32]>,
}

pub(super) struct CpuFftBackend {
    planner: FftPlanner<f32>,
    plan_cache: HashMap<PlanKey, FftPlan>,
    kernel_cache: HashMap<KernelKey, KernelSpectra>,
}

impl CpuFftBackend {
    pub(super) fn new() -> Result<Self, String> {
        Ok(Self {
            planner: FftPlanner::new(),
            plan_cache: HashMap::new(),
            kernel_cache: HashMap::new(),
        })
    }

    pub(super) unsafe fn run_magnitude_angle(
        &mut self,
        image: *const c_float,
        width: c_uint,
        height: c_uint,
        radius: c_uint,
        kernels: &DenseConvolutionKernels,
        out_x: *mut c_float,
        out_y: *mut c_float,
        magnitude: *mut c_float,
        angle: *mut c_float,
    ) -> Result<(), String> {
        if image.is_null() || out_x.is_null() || out_y.is_null() || magnitude.is_null() || angle.is_null() {
            return Err("null pointer passed to FFT backend".to_string());
        }

        let fft_w = c_uint::try_from(next_smooth_fft_size(u64::from(width) + 4 * u64::from(radius)))
            .map_err(|_| "FFT width exceeded uint32".to_string())?;
        let fft_h = c_uint::try_from(next_smooth_fft_size(u64::from(height) + 4 * u64::from(radius)))
            .map_err(|_| "FFT height exceeded uint32".to_string())?;
        let plan_key = PlanKey { fft_w, fft_h };
        let spectra = self.kernel_spectra(plan_key, radius, kernels)?;
        let plan = self.plan(plan_key)?;

        let width_usize = width as usize;
        let height_usize = height as usize;
        let radius_usize = radius as usize;
        let padded_w = width_usize
            .checked_add(radius_usize * 2)
            .ok_or_else(|| "padded width overflowed".to_string())?;
        let padded_h = height_usize
            .checked_add(radius_usize * 2)
            .ok_or_else(|| "padded height overflowed".to_string())?;
        let image_len = crate::checked_image_pixels(width, height)?;
        let image_slice = std::slice::from_raw_parts(image, image_len);

        let mut image_plane = vec![Complex32::new(0.0, 0.0); plan.plane_len];
        let mut work_plane = vec![Complex32::new(0.0, 0.0); plan.plane_len];
        let mut output_planes = vec![Complex32::new(0.0, 0.0); plan.plane_len * 2];

        reflect_pad_image(
            image_slice,
            width_usize,
            height_usize,
            radius_usize,
            padded_w,
            padded_h,
            plan.fft_w,
            plan.fft_h,
            &mut image_plane,
        );
        fft2_in_place(plan, &mut image_plane, &mut work_plane, false);

        let spectra_slice: &[Complex32] = &spectra;
        let (out_x_plane, out_y_plane) = output_planes.split_at_mut(plan.plane_len);
        for idx in 0..plan.plane_len {
            let image_value = image_plane[idx];
            out_x_plane[idx] = image_value * spectra_slice[idx];
            out_y_plane[idx] = image_value * spectra_slice[plan.plane_len + idx];
        }

        fft2_in_place(plan, out_x_plane, &mut work_plane, true);
        fft2_in_place(plan, out_y_plane, &mut work_plane, true);

        let out_x_slice = std::slice::from_raw_parts_mut(out_x, image_len);
        let out_y_slice = std::slice::from_raw_parts_mut(out_y, image_len);
        let magnitude_slice = std::slice::from_raw_parts_mut(magnitude, image_len);
        let angle_slice = std::slice::from_raw_parts_mut(angle, image_len);
        let crop = radius_usize * 2;
        for y in 0..height_usize {
            for x in 0..width_usize {
                let out_index = y * width_usize + x;
                let src_index = (y + crop) * plan.fft_w + (x + crop);
                let gx = out_x_plane[src_index].re;
                let gy = out_y_plane[src_index].re;
                out_x_slice[out_index] = gx;
                out_y_slice[out_index] = gy;
                magnitude_slice[out_index] = (gx * gx + gy * gy).sqrt();
                angle_slice[out_index] = wvf_unsigned_angle(gy, gx);
            }
        }

        Ok(())
    }

    fn plan(&mut self, key: PlanKey) -> Result<&FftPlan, String> {
        if !self.plan_cache.contains_key(&key) {
            let fft_w = key.fft_w as usize;
            let fft_h = key.fft_h as usize;
            let plane_len = crate::checked_len(fft_w, fft_h, "FFT plane")?;
            let plan = FftPlan {
                fft_w,
                fft_h,
                plane_len,
                forward_width: self.planner.plan_fft_forward(fft_w),
                inverse_width: self.planner.plan_fft_inverse(fft_w),
                forward_height: self.planner.plan_fft_forward(fft_h),
                inverse_height: self.planner.plan_fft_inverse(fft_h),
            };
            self.plan_cache.insert(key, plan);
        }
        self.plan_cache
            .get(&key)
            .ok_or_else(|| "FFT plan cache insertion failed".to_string())
    }

    fn kernel_spectra(
        &mut self,
        plan_key: PlanKey,
        radius: c_uint,
        kernels: &DenseConvolutionKernels,
    ) -> Result<Arc<[Complex32]>, String> {
        let kernel_hash = hash_f32_bytes(&kernels.kernel_x, 14695981039346656037);
        let kernel_hash = hash_f32_bytes(&kernels.kernel_y, kernel_hash);
        let key = KernelKey {
            fft_w: plan_key.fft_w,
            fft_h: plan_key.fft_h,
            radius,
            kernel_width: kernels.kernel_width,
            kernel_hash,
        };
        if !self.kernel_cache.contains_key(&key) {
            let values = {
                let plan = self.plan(plan_key)?;
                build_kernel_spectra(plan, kernels)?
            };
            self.kernel_cache.insert(
                key,
                KernelSpectra {
                    values: Arc::from(values.into_boxed_slice()),
                },
            );
        }
        self.kernel_cache
            .get(&key)
            .map(|spectra| spectra.values.clone())
            .ok_or_else(|| "FFT kernel cache insertion failed".to_string())
    }
}

fn build_kernel_spectra(
    plan: &FftPlan,
    kernels: &DenseConvolutionKernels,
) -> Result<Vec<Complex32>, String> {
    let mut values = vec![Complex32::new(0.0, 0.0); plan.plane_len * 2];
    let mut scratch = vec![Complex32::new(0.0, 0.0); plan.plane_len];
    let kernel_width = kernels.kernel_width as usize;
    for y in 0..kernel_width {
        for x in 0..kernel_width {
            let src = y * kernel_width + x;
            let dst = y * plan.fft_w + x;
            values[dst] = Complex32::new(kernels.kernel_x[src], 0.0);
            values[plan.plane_len + dst] = Complex32::new(kernels.kernel_y[src], 0.0);
        }
    }
    let (kernel_x, kernel_y) = values.split_at_mut(plan.plane_len);
    fft2_in_place(plan, kernel_x, &mut scratch, false);
    fft2_in_place(plan, kernel_y, &mut scratch, false);
    Ok(values)
}

fn fft2_in_place(
    plan: &FftPlan,
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    inverse: bool,
) {
    debug_assert_eq!(data.len(), plan.plane_len);
    debug_assert_eq!(scratch.len(), plan.plane_len);

    let row_fft = if inverse {
        &plan.inverse_width
    } else {
        &plan.forward_width
    };
    data.par_chunks_mut(plan.fft_w)
        .for_each(|row| row_fft.process(row));

    transpose(data, scratch, plan.fft_w, plan.fft_h);

    let col_fft = if inverse {
        &plan.inverse_height
    } else {
        &plan.forward_height
    };
    scratch
        .par_chunks_mut(plan.fft_h)
        .for_each(|row| col_fft.process(row));

    transpose(scratch, data, plan.fft_h, plan.fft_w);

    if inverse {
        let scale = 1.0 / plan.plane_len as f32;
        for value in data.iter_mut() {
            *value *= scale;
        }
    }
}

fn transpose(
    src: &[Complex32],
    dst: &mut [Complex32],
    width: usize,
    height: usize,
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    dst.par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, column)| {
            for y in 0..height {
                column[y] = src[y * width + x];
            }
        });
}

fn reflect_pad_image(
    image: &[f32],
    image_width: usize,
    image_height: usize,
    radius: usize,
    padded_width: usize,
    padded_height: usize,
    fft_width: usize,
    fft_height: usize,
    padded: &mut [Complex32],
) {
    for y in 0..fft_height {
        let row_offset = y * fft_width;
        for x in 0..fft_width {
            let dst_index = row_offset + x;
            if x >= padded_width || y >= padded_height {
                padded[dst_index] = Complex32::new(0.0, 0.0);
                continue;
            }
            let src_x = reflect_index(x as i64 - radius as i64, image_width as i64) as usize;
            let src_y = reflect_index(y as i64 - radius as i64, image_height as i64) as usize;
            padded[dst_index] = Complex32::new(image[src_y * image_width + src_x], 0.0);
        }
    }
}

fn reflect_index(mut value: i64, limit: i64) -> i64 {
    if limit <= 1 {
        return 0;
    }
    let period = 2 * limit;
    value %= period;
    if value < 0 {
        value += period;
    }
    if value >= limit {
        value = period - value - 1;
    }
    value
}

fn wvf_unsigned_angle(y: f32, x: f32) -> f32 {
    let mut theta = y.atan2(x);
    if theta < 0.0 {
        theta += std::f32::consts::PI;
    }
    if theta >= std::f32::consts::PI {
        theta -= std::f32::consts::PI;
    }
    theta
}

fn is_smooth_fft_size(mut value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for factor in [2_u64, 3, 5, 7] {
        while value % factor == 0 {
            value /= factor;
        }
    }
    value == 1
}

fn next_smooth_fft_size(mut value: u64) -> u64 {
    value = value.max(2);
    while !is_smooth_fft_size(value) {
        value += 1;
    }
    value
}

fn hash_f32_bytes(values: &[f32], seed: u64) -> u64 {
    let byte_len = values.len() * size_of::<f32>();
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
    let mut hash = seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}
