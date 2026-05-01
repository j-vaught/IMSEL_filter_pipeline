use super::{custom_fft::CpuFftBackend, DenseConvolutionKernels};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
use num_complex::Complex32;
use std::collections::HashMap;
use std::mem::size_of;
use std::os::raw::{c_float, c_uint};

const FFT_SHADER_SOURCE: &str = include_str!("../wvf.metal");
const MAX_RADIX: usize = 8;
const MAX_STAGES: usize = 16;
const TRANSPOSE_TILE: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftPadParams {
    image_width: c_uint,
    image_height: c_uint,
    padded_width: c_uint,
    padded_height: c_uint,
    fft_width: c_uint,
    fft_height: c_uint,
    radius: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftPostprocessParams {
    width: c_uint,
    height: c_uint,
    crop: c_uint,
    fft_width: c_uint,
    plane_stride: c_uint,
    scale: c_float,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftTransposeParams {
    width: c_uint,
    height: c_uint,
    batch_count: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftRealWidthParams {
    fft_width: c_uint,
    half_width: c_uint,
    complex_width: c_uint,
    row_count: c_uint,
    rows_per_batch: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftRowPlanParams {
    row_len: c_uint,
    row_count: c_uint,
    stage_count: c_uint,
    _reserved: c_uint,
    radix: [c_uint; MAX_STAGES],
    stride: [c_uint; MAX_STAGES],
    prev: [c_uint; MAX_STAGES],
    weight_offset: [c_uint; MAX_STAGES],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftStridedParams {
    row_stride: c_uint,
    rows_per_batch: c_uint,
    plane_stride: c_uint,
    _reserved: c_uint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlanKey {
    width: c_uint,
    height: c_uint,
    radius: c_uint,
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
    transposed_layout: bool,
}

struct RowPlan {
    len: usize,
    row_count: usize,
    params_buffer: Buffer,
    weights_buffer: Buffer,
    threadgroup_width: u64,
    threadgroup_bytes: u64,
    double_shared: bool,
}

struct ImagePlan {
    fft_w: usize,
    fft_h: usize,
    full_plane_len: usize,
    complex_w: usize,
    reduced_plane_len: usize,
    single_padded: Buffer,
    single_half: Buffer,
    single_reduced_a: Buffer,
    single_reduced_b: Buffer,
    single_transpose: Buffer,
    double_half: Buffer,
    double_output: Buffer,
    double_reduced_a: Buffer,
    double_reduced_b: Buffer,
    double_transpose: Buffer,
    pad_params_buffer: Buffer,
    postprocess_params_buffer: Buffer,
    reduced_complex_count_buffer: Buffer,
    transpose_forward_single_buffer: Buffer,
    transpose_reverse_single_buffer: Buffer,
    transpose_forward_double_buffer: Buffer,
    transpose_reverse_double_buffer: Buffer,
    real_width_single_buffer: Buffer,
    real_width_double_buffer: Buffer,
    real_width_twiddles_buffer: Buffer,
    height_strided_single_buffer: Buffer,
    height_strided_double_buffer: Buffer,
    width_forward_single: RowPlan,
    width_forward_double: RowPlan,
    width_inverse_double: RowPlan,
    height_forward_single: RowPlan,
    height_forward_double: RowPlan,
    height_inverse_double: RowPlan,
}

#[derive(Clone)]
struct KernelSpectra {
    buffer: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeightFftMode {
    Transpose,
    Strided,
}

pub(super) struct GpuFftBackend {
    device: Device,
    queue: CommandQueue,
    pad_pipeline: ComputePipelineState,
    row_pipeline: ComputePipelineState,
    row_double_shared_pipeline: ComputePipelineState,
    row_strided_pipeline: ComputePipelineState,
    pack_real_pipeline: ComputePipelineState,
    finalize_r2c_pipeline: ComputePipelineState,
    finalize_r2c_transposed_pipeline: ComputePipelineState,
    prepare_c2r_pipeline: ComputePipelineState,
    prepare_c2r_transposed_pipeline: ComputePipelineState,
    unpack_real_pipeline: ComputePipelineState,
    transpose_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    postprocess_pipeline: ComputePipelineState,
    image_plans: HashMap<PlanKey, ImagePlan>,
    kernel_spectra: HashMap<KernelKey, KernelSpectra>,
    cpu_fallback: Option<CpuFftBackend>,
}

impl GpuFftBackend {
    pub(super) fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let queue = device.new_command_queue();
        let library = compile_library(&device, FFT_SHADER_SOURCE)?;
        let pad_pipeline = pipeline(&device, &library, "wvf_fft_reflect_pad_real_dense")?;
        let row_pipeline = pipeline(&device, &library, "wvf_fft_row_c2c_fused")?;
        let row_double_shared_pipeline =
            pipeline(&device, &library, "wvf_fft_row_c2c_double_shared_fused")?;
        let row_strided_pipeline = pipeline(&device, &library, "wvf_fft_row_c2c_strided_fused")?;
        let pack_real_pipeline = pipeline(&device, &library, "wvf_fft_pack_real_pairs")?;
        let finalize_r2c_pipeline = pipeline(&device, &library, "wvf_fft_finalize_r2c")?;
        let finalize_r2c_transposed_pipeline =
            pipeline(&device, &library, "wvf_fft_finalize_r2c_transposed")?;
        let prepare_c2r_pipeline = pipeline(&device, &library, "wvf_fft_prepare_c2r")?;
        let prepare_c2r_transposed_pipeline =
            pipeline(&device, &library, "wvf_fft_prepare_c2r_transposed")?;
        let unpack_real_pipeline = pipeline(&device, &library, "wvf_fft_unpack_real_pairs")?;
        let transpose_pipeline = pipeline(&device, &library, "wvf_fft_transpose_c2c")?;
        let multiply_pipeline = pipeline(&device, &library, "wvf_fft_multiply_spectra")?;
        let postprocess_pipeline = pipeline(&device, &library, "wvf_fft_postprocess_dense")?;
        Ok(Self {
            device,
            queue,
            pad_pipeline,
            row_pipeline,
            row_double_shared_pipeline,
            row_strided_pipeline,
            pack_real_pipeline,
            finalize_r2c_pipeline,
            finalize_r2c_transposed_pipeline,
            prepare_c2r_pipeline,
            prepare_c2r_transposed_pipeline,
            unpack_real_pipeline,
            transpose_pipeline,
            multiply_pipeline,
            postprocess_pipeline,
            image_plans: HashMap::new(),
            kernel_spectra: HashMap::new(),
            cpu_fallback: None,
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
            return Err("null pointer passed to GPU FFT backend".to_string());
        }

        let fft_w = c_uint::try_from(next_smooth_fft_size(u64::from(width) + 4 * u64::from(radius)))
            .map_err(|_| "FFT width exceeded uint32".to_string())?;
        let fft_h = c_uint::try_from(next_smooth_fft_size(u64::from(height) + 4 * u64::from(radius)))
            .map_err(|_| "FFT height exceeded uint32".to_string())?;
        if fft_w < 4 || fft_w % 2 != 0 {
            if self.cpu_fallback.is_none() {
                self.cpu_fallback = Some(CpuFftBackend::new()?);
            }
            let fallback = self
                .cpu_fallback
                .as_mut()
                .ok_or_else(|| "CPU FFT fallback initialization failed".to_string())?;
            return fallback.run_magnitude_angle(
                image, width, height, radius, kernels, out_x, out_y, magnitude, angle,
            );
        }
        let key = PlanKey {
            width,
            height,
            radius,
            fft_w,
            fft_h,
        };
        if !self.image_plans.contains_key(&key) {
            let plan = self.build_image_plan(key)?;
            self.image_plans.insert(key, plan);
        }
        let forward_height_mode = height_forward_mode();
        let inverse_height_mode = height_inverse_mode();
        let spectrum_buffer = self.kernel_spectra(key, kernels, forward_height_mode)?;
        let plan = self
            .image_plans
            .get(&key)
            .ok_or_else(|| "GPU FFT image plan cache insertion failed".to_string())?;
        let single_spectrum = &plan.single_reduced_b;

        let image_len = crate::checked_image_pixels(width, height)?;
        let image_bytes = crate::checked_len(image_len, size_of::<c_float>(), "image")?;
        let output_bytes = image_bytes;
        let shared = MTLResourceOptions::StorageModeShared;

        let image_buffer = self.device.new_buffer_with_bytes_no_copy(
            image.cast(),
            image_bytes as u64,
            shared,
            None,
        );
        image_buffer.did_modify_range(crate::NSRange::new(0, image_bytes as u64));

        let out_x_buffer = self.device.new_buffer_with_bytes_no_copy(
            out_x.cast::<std::ffi::c_void>().cast_const(),
            output_bytes as u64,
            shared,
            None,
        );
        let out_y_buffer = self.device.new_buffer_with_bytes_no_copy(
            out_y.cast::<std::ffi::c_void>().cast_const(),
            output_bytes as u64,
            shared,
            None,
        );
        let magnitude_buffer = self.device.new_buffer_with_bytes_no_copy(
            magnitude.cast::<std::ffi::c_void>().cast_const(),
            output_bytes as u64,
            shared,
            None,
        );
        let angle_buffer = self.device.new_buffer_with_bytes_no_copy(
            angle.cast::<std::ffi::c_void>().cast_const(),
            output_bytes as u64,
            shared,
            None,
        );

        let command_buffer = self.queue.new_command_buffer();
        encode_reflect_pad(
            &self.pad_pipeline,
            &command_buffer,
            &image_buffer,
            &plan.single_padded,
            &plan.pad_params_buffer,
            plan.fft_w,
            plan.fft_h,
        );
        encode_forward_2d_real(
            &self.row_pipeline,
            &self.row_double_shared_pipeline,
            &self.row_strided_pipeline,
            &self.pack_real_pipeline,
            &self.finalize_r2c_pipeline,
            &self.finalize_r2c_transposed_pipeline,
            &self.transpose_pipeline,
            &command_buffer,
            &plan.width_forward_single,
            &plan.height_forward_single,
            &plan.single_padded,
            &plan.single_half,
            &plan.single_transpose,
            &plan.single_reduced_a,
            &plan.single_reduced_b,
            &plan.single_transpose,
            single_spectrum,
            &plan.real_width_single_buffer,
            &plan.real_width_twiddles_buffer,
            &plan.transpose_forward_single_buffer,
            &plan.transpose_reverse_single_buffer,
            &plan.height_strided_single_buffer,
            forward_height_mode,
            plan.complex_w,
            1,
        );
        encode_multiply(
            &self.multiply_pipeline,
            &command_buffer,
            single_spectrum,
            &spectrum_buffer,
            &plan.double_reduced_a,
            &plan.reduced_complex_count_buffer,
            plan.reduced_plane_len,
        );
        encode_inverse_2d_real(
            &self.row_pipeline,
            &self.row_double_shared_pipeline,
            &self.row_strided_pipeline,
            &self.prepare_c2r_pipeline,
            &self.prepare_c2r_transposed_pipeline,
            &self.unpack_real_pipeline,
            &self.transpose_pipeline,
            &command_buffer,
            &plan.width_inverse_double,
            &plan.height_inverse_double,
            &plan.double_reduced_a,
            &plan.double_reduced_b,
            &plan.double_half,
            &plan.double_transpose,
            &plan.double_output,
            &plan.real_width_double_buffer,
            &plan.real_width_twiddles_buffer,
            &plan.transpose_forward_double_buffer,
            &plan.transpose_reverse_double_buffer,
            &plan.height_strided_double_buffer,
            forward_height_mode,
            inverse_height_mode,
            plan.complex_w,
            plan.fft_h,
            2,
        );
        encode_postprocess(
            &self.postprocess_pipeline,
            &command_buffer,
            &plan.double_output,
            &out_x_buffer,
            &out_y_buffer,
            &magnitude_buffer,
            &angle_buffer,
            &plan.postprocess_params_buffer,
            width as usize,
            height as usize,
        );

        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    fn kernel_spectra(
        &mut self,
        plan_key: PlanKey,
        kernels: &DenseConvolutionKernels,
        height_mode: HeightFftMode,
    ) -> Result<Buffer, String> {
        let kernel_hash = hash_f32_bytes(&kernels.kernel_x, 14695981039346656037);
        let kernel_hash = hash_f32_bytes(&kernels.kernel_y, kernel_hash);
        let key = KernelKey {
            fft_w: plan_key.fft_w,
            fft_h: plan_key.fft_h,
            radius: plan_key.radius,
            kernel_width: kernels.kernel_width,
            kernel_hash,
            transposed_layout: height_mode == HeightFftMode::Transpose,
        };
        if !self.kernel_spectra.contains_key(&key) {
            let plan = self
                .image_plans
                .get(&plan_key)
                .ok_or_else(|| "GPU FFT image plan was missing".to_string())?;
            let values = build_real_kernel_values(plan, kernels)?;
            let input_bytes = crate::checked_len(
                values.len(),
                size_of::<c_float>(),
                "real kernel upload",
            )?;
            let upload = self.device.new_buffer_with_data(
                values.as_ptr().cast(),
                input_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let spectrum_buffer = self.device.new_buffer(
                crate::checked_len(
                    plan.reduced_plane_len * 2,
                    size_of::<Complex32>(),
                    "cached reduced kernel spectra",
                )? as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let command_buffer = self.queue.new_command_buffer();
            encode_forward_2d_real(
                &self.row_pipeline,
                &self.row_double_shared_pipeline,
                &self.row_strided_pipeline,
                &self.pack_real_pipeline,
                &self.finalize_r2c_pipeline,
                &self.finalize_r2c_transposed_pipeline,
                &self.transpose_pipeline,
                &command_buffer,
                &plan.width_forward_double,
                &plan.height_forward_double,
                &upload,
                &plan.double_half,
                &plan.double_transpose,
                &plan.double_reduced_a,
                &plan.double_reduced_b,
                &plan.double_transpose,
                &spectrum_buffer,
                &plan.real_width_double_buffer,
                &plan.real_width_twiddles_buffer,
                &plan.transpose_forward_double_buffer,
                &plan.transpose_reverse_double_buffer,
                &plan.height_strided_double_buffer,
                height_mode,
                plan.complex_w,
                2,
            );
            command_buffer.commit();
            command_buffer.wait_until_completed();
            self.kernel_spectra
                .insert(key, KernelSpectra { buffer: spectrum_buffer });
        }
        self.kernel_spectra
            .get(&key)
            .map(|spectra| spectra.buffer.to_owned())
            .ok_or_else(|| "GPU FFT kernel cache insertion failed".to_string())
    }

    fn build_image_plan(&self, key: PlanKey) -> Result<ImagePlan, String> {
        let fft_w = key.fft_w as usize;
        let fft_h = key.fft_h as usize;
        let full_plane_len = crate::checked_len(fft_w, fft_h, "FFT plane")?;
        let half_w = fft_w / 2;
        let complex_w = fft_w / 2 + 1;
        let reduced_plane_len = crate::checked_len(complex_w, fft_h, "reduced FFT plane")?;
        let half_plane_len = crate::checked_len(half_w, fft_h, "half FFT plane")?;
        let padded_width = key
            .width
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded width exceeded uint32".to_string())?;
        let padded_height = key
            .height
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded height exceeded uint32".to_string())?;

        let real_plane_bytes = crate::checked_len(
            full_plane_len,
            size_of::<c_float>(),
            "real FFT plane",
        )?;
        let real_double_bytes = crate::checked_len(real_plane_bytes, 2, "double real plane")?;
        let half_complex_plane_bytes = crate::checked_len(
            half_plane_len,
            size_of::<Complex32>(),
            "half complex FFT plane",
        )?;
        let half_complex_double_bytes =
            crate::checked_len(half_complex_plane_bytes, 2, "double half complex plane")?;
        let reduced_complex_plane_bytes = crate::checked_len(
            reduced_plane_len,
            size_of::<Complex32>(),
            "reduced complex FFT plane",
        )?;
        let reduced_complex_double_bytes = crate::checked_len(
            reduced_complex_plane_bytes,
            2,
            "double reduced complex plane",
        )?;
        let reduced_complex_count =
            c_uint::try_from(reduced_plane_len).map_err(|_| "reduced FFT plane exceeded uint32".to_string())?;

        let width_radices = factor_fft_length(half_w)?;
        let height_radices = factor_fft_length(fft_h)?;
        let width_forward_single =
            build_row_plan(&self.device, &self.row_pipeline, half_w, fft_h, 1, &width_radices, false, true)?;
        let width_forward_double =
            build_row_plan(&self.device, &self.row_pipeline, half_w, fft_h, 2, &width_radices, false, true)?;
        let width_inverse_double =
            build_row_plan(&self.device, &self.row_pipeline, half_w, fft_h, 2, &width_radices, true, true)?;
        let height_forward_single =
            build_row_plan(&self.device, &self.row_strided_pipeline, fft_h, complex_w, 1, &height_radices, false, false)?;
        let height_forward_double =
            build_row_plan(&self.device, &self.row_strided_pipeline, fft_h, complex_w, 2, &height_radices, false, false)?;
        let height_inverse_double =
            build_row_plan(&self.device, &self.row_strided_pipeline, fft_h, complex_w, 2, &height_radices, true, false)?;
        let real_width_twiddles = build_real_width_twiddles(fft_w);

        let private = MTLResourceOptions::StorageModePrivate;
        let single_padded = self.device.new_buffer(real_plane_bytes as u64, private);
        let single_half = self.device.new_buffer(half_complex_plane_bytes as u64, private);
        let single_reduced_a = self.device.new_buffer(reduced_complex_plane_bytes as u64, private);
        let single_reduced_b = self.device.new_buffer(reduced_complex_plane_bytes as u64, private);
        let single_transpose = self.device.new_buffer(reduced_complex_plane_bytes as u64, private);
        let double_half = self.device.new_buffer(half_complex_double_bytes as u64, private);
        let double_output = self.device.new_buffer(real_double_bytes as u64, private);
        let double_reduced_a = self.device.new_buffer(reduced_complex_double_bytes as u64, private);
        let double_reduced_b = self.device.new_buffer(reduced_complex_double_bytes as u64, private);
        let double_transpose = self.device.new_buffer(reduced_complex_double_bytes as u64, private);

        let pad_params = WvfFftPadParams {
            image_width: key.width,
            image_height: key.height,
            padded_width,
            padded_height,
            fft_width: key.fft_w,
            fft_height: key.fft_h,
            radius: key.radius,
        };
        let postprocess_params = WvfFftPostprocessParams {
            width: key.width,
            height: key.height,
            crop: key.radius.saturating_mul(2),
            fft_width: key.fft_w,
            plane_stride: c_uint::try_from(full_plane_len).map_err(|_| "FFT plane exceeded uint32".to_string())?,
            scale: 2.0 / full_plane_len as f32,
        };
        let real_width_single = WvfFftRealWidthParams {
            fft_width: key.fft_w,
            half_width: half_w as c_uint,
            complex_width: complex_w as c_uint,
            row_count: key.fft_h,
            rows_per_batch: key.fft_h,
        };
        let real_width_double = WvfFftRealWidthParams {
            fft_width: key.fft_w,
            half_width: half_w as c_uint,
            complex_width: complex_w as c_uint,
            row_count: key
                .fft_h
                .checked_mul(2)
                .ok_or_else(|| "batched row count exceeded uint32".to_string())?,
            rows_per_batch: key.fft_h,
        };
        let transpose_forward_single = WvfFftTransposeParams {
            width: complex_w as c_uint,
            height: key.fft_h,
            batch_count: 1,
        };
        let transpose_reverse_single = WvfFftTransposeParams {
            width: key.fft_h,
            height: complex_w as c_uint,
            batch_count: 1,
        };
        let transpose_forward_double = WvfFftTransposeParams {
            width: complex_w as c_uint,
            height: key.fft_h,
            batch_count: 2,
        };
        let transpose_reverse_double = WvfFftTransposeParams {
            width: key.fft_h,
            height: complex_w as c_uint,
            batch_count: 2,
        };
        let height_strided_single = WvfFftStridedParams {
            row_stride: complex_w as c_uint,
            rows_per_batch: complex_w as c_uint,
            plane_stride: reduced_plane_len as c_uint,
            _reserved: 0,
        };
        let height_strided_double = WvfFftStridedParams {
            row_stride: complex_w as c_uint,
            rows_per_batch: complex_w as c_uint,
            plane_stride: reduced_plane_len as c_uint,
            _reserved: 0,
        };

        Ok(ImagePlan {
            fft_w,
            fft_h,
            full_plane_len,
            complex_w,
            reduced_plane_len,
            single_padded,
            single_half,
            single_reduced_a,
            single_reduced_b,
            single_transpose,
            double_half,
            double_output,
            double_reduced_a,
            double_reduced_b,
            double_transpose,
            pad_params_buffer: param_buffer(&self.device, &pad_params),
            postprocess_params_buffer: param_buffer(&self.device, &postprocess_params),
            reduced_complex_count_buffer: param_buffer(&self.device, &reduced_complex_count),
            transpose_forward_single_buffer: param_buffer(&self.device, &transpose_forward_single),
            transpose_reverse_single_buffer: param_buffer(&self.device, &transpose_reverse_single),
            transpose_forward_double_buffer: param_buffer(&self.device, &transpose_forward_double),
            transpose_reverse_double_buffer: param_buffer(&self.device, &transpose_reverse_double),
            real_width_single_buffer: param_buffer(&self.device, &real_width_single),
            real_width_double_buffer: param_buffer(&self.device, &real_width_double),
            real_width_twiddles_buffer: self.device.new_buffer_with_data(
                real_width_twiddles.as_ptr().cast(),
                (real_width_twiddles.len() * size_of::<Complex32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            height_strided_single_buffer: param_buffer(&self.device, &height_strided_single),
            height_strided_double_buffer: param_buffer(&self.device, &height_strided_double),
            width_forward_single,
            width_forward_double,
            width_inverse_double,
            height_forward_single,
            height_forward_double,
            height_inverse_double,
        })
    }
}

fn build_real_kernel_values(
    plan: &ImagePlan,
    kernels: &DenseConvolutionKernels,
) -> Result<Vec<c_float>, String> {
    let mut values = vec![0.0; plan.full_plane_len * 2];
    let kernel_width = kernels.kernel_width as usize;
    for y in 0..kernel_width {
        for x in 0..kernel_width {
            let src = y * kernel_width + x;
            let dst = y * plan.fft_w + x;
            values[dst] = kernels.kernel_x[src];
            values[plan.full_plane_len + dst] = kernels.kernel_y[src];
        }
    }
    Ok(values)
}

fn build_real_width_twiddles(fft_w: usize) -> Vec<Complex32> {
    let complex_w = fft_w / 2 + 1;
    let mut twiddles = Vec::with_capacity(complex_w);
    for k in 0..complex_w {
        let phase = -2.0_f64 * std::f64::consts::PI * k as f64 / fft_w as f64;
        twiddles.push(Complex32::new(phase.cos() as f32, phase.sin() as f32));
    }
    twiddles
}

fn build_row_plan(
    device: &Device,
    pipeline: &ComputePipelineState,
    len: usize,
    rows: usize,
    batch_count: usize,
    radices: &[u32],
    inverse: bool,
    allow_double_shared: bool,
) -> Result<RowPlan, String> {
    if radices.len() > MAX_STAGES {
        return Err("FFT stage count exceeded static GPU plan limit".to_string());
    }
    let mut params = WvfFftRowPlanParams {
        row_len: len as c_uint,
        row_count: (rows * batch_count) as c_uint,
        stage_count: radices.len() as c_uint,
        _reserved: 0,
        radix: [0; MAX_STAGES],
        stride: [0; MAX_STAGES],
        prev: [0; MAX_STAGES],
        weight_offset: [0; MAX_STAGES],
    };
    let mut packed_weights = Vec::<Complex32>::new();
    let mut prev = 1usize;
    for (stage_index, &radix) in radices.iter().enumerate() {
        let stride = len
            .checked_div(prev * radix as usize)
            .ok_or_else(|| "invalid FFT stage factorization".to_string())?;
        let weights = build_stage_weights(radix as usize, prev, inverse)?;
        params.radix[stage_index] = radix;
        params.stride[stage_index] = stride as c_uint;
        params.prev[stage_index] = prev as c_uint;
        params.weight_offset[stage_index] = packed_weights.len() as c_uint;
        packed_weights.extend_from_slice(&weights);
        prev *= radix as usize;
    }
    let weights_buffer = device.new_buffer_with_data(
        packed_weights.as_ptr().cast(),
        (packed_weights.len() * size_of::<Complex32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let threadgroup_width = choose_row_threadgroup_width(pipeline, len);
    let threadgroup_bytes = (len * size_of::<Complex32>()) as u64;
    let double_shared = allow_double_shared && threadgroup_bytes.saturating_mul(2) <= 32 * 1024;
    Ok(RowPlan {
        len,
        row_count: rows * batch_count,
        params_buffer: param_buffer(device, &params),
        weights_buffer,
        threadgroup_width,
        threadgroup_bytes,
        double_shared,
    })
}

fn build_stage_weights(
    radix: usize,
    prev: usize,
    inverse: bool,
) -> Result<Vec<Complex32>, String> {
    if radix == 0 || radix > MAX_RADIX {
        return Err("unsupported FFT radix".to_string());
    }
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let groups = prev
        .checked_mul(radix)
        .ok_or_else(|| "FFT stage group count overflowed".to_string())?;
    let total = groups
        .checked_mul(radix)
        .ok_or_else(|| "FFT stage weight count overflowed".to_string())?;
    let mut weights = vec![Complex32::new(0.0, 0.0); total];
    for m in 0..radix {
        for p in 0..prev {
            let group = p + m * prev;
            let denom = (radix * prev) as f64;
            for l in 0..radix {
                let twiddle_phase = sign * 2.0 * std::f64::consts::PI * (l * p) as f64 / denom;
                let butterfly_phase =
                    sign * 2.0 * std::f64::consts::PI * (l * m) as f64 / radix as f64;
                let phase = twiddle_phase + butterfly_phase;
                weights[group * radix + l] = Complex32::new(phase.cos() as f32, phase.sin() as f32);
            }
        }
    }
    Ok(weights)
}

fn factor_fft_length(mut len: usize) -> Result<Vec<u32>, String> {
    let mut radices = Vec::new();
    for radix in [8usize, 7, 5, 4, 3, 2] {
        while len % radix == 0 {
            radices.push(radix as u32);
            len /= radix;
        }
    }
    if len != 1 {
        return Err("FFT length is not 2/3/5/7-smooth".to_string());
    }
    if radices.is_empty() {
        return Err("FFT length must be at least 2".to_string());
    }
    Ok(radices)
}

fn compile_library(device: &Device, source: &str) -> Result<Library, String> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    device
        .new_library_with_source(source, &options)
        .map_err(|err| format!("failed to compile Metal FFT shaders: {err}"))
}

fn pipeline(device: &Device, library: &Library, name: &str) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|err| format!("failed to load Metal FFT function {name}: {err}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| format!("failed to create Metal FFT pipeline {name}: {err}"))
}

fn param_buffer<T>(device: &Device, value: &T) -> Buffer {
    device.new_buffer_with_data(
        (value as *const T).cast(),
        size_of::<T>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn encode_reflect_pad(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    image: &Buffer,
    padded: &Buffer,
    params: &Buffer,
    fft_w: usize,
    fft_h: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(image), 0);
    encoder.set_buffer(1, Some(padded), 0);
    encoder.set_buffer(2, Some(params), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: fft_w as u64,
            height: fft_h as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
}

fn encode_row_fft(
    row_pipeline: &ComputePipelineState,
    row_double_shared_pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    plan: &RowPlan,
    input: &Buffer,
    output: &Buffer,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    let pipeline = if plan.double_shared {
        row_double_shared_pipeline
    } else {
        row_pipeline
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_threadgroup_memory_length(0, plan.threadgroup_bytes);
    if plan.double_shared {
        encoder.set_threadgroup_memory_length(1, plan.threadgroup_bytes);
    }
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(&plan.params_buffer), 0);
    encoder.set_buffer(3, Some(&plan.weights_buffer), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: plan.row_count as u64,
            depth: 1,
        },
        MTLSize {
            width: plan.threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

fn encode_row_fft_strided(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    plan: &RowPlan,
    layout: &Buffer,
    input: &Buffer,
    output: &Buffer,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_threadgroup_memory_length(0, plan.threadgroup_bytes);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(&plan.params_buffer), 0);
    encoder.set_buffer(3, Some(layout), 0);
    encoder.set_buffer(4, Some(&plan.weights_buffer), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: plan.row_count as u64,
            depth: 1,
        },
        MTLSize {
            width: plan.threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

fn encode_transpose(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    width: usize,
    height: usize,
    batch_count: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(params), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: width.div_ceil(TRANSPOSE_TILE) as u64,
            height: height.div_ceil(TRANSPOSE_TILE) as u64,
            depth: batch_count as u64,
        },
        MTLSize {
            width: TRANSPOSE_TILE as u64,
            height: TRANSPOSE_TILE as u64,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

fn encode_real_width_stage(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    width: usize,
    row_count: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(params), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: row_count as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
}

fn encode_real_width_twiddled_stage(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    twiddles: &Buffer,
    width: usize,
    row_count: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(params), 0);
    encoder.set_buffer(3, Some(twiddles), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: row_count as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
}

fn encode_forward_2d_real(
    row_pipeline: &ComputePipelineState,
    row_double_shared_pipeline: &ComputePipelineState,
    row_strided_pipeline: &ComputePipelineState,
    pack_real_pipeline: &ComputePipelineState,
    finalize_r2c_pipeline: &ComputePipelineState,
    _finalize_r2c_transposed_pipeline: &ComputePipelineState,
    transpose_pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    width_plan: &RowPlan,
    height_plan: &RowPlan,
    input: &Buffer,
    half_input: &Buffer,
    half_work: &Buffer,
    reduced_a: &Buffer,
    _reduced_b: &Buffer,
    transpose: &Buffer,
    output: &Buffer,
    real_width_params: &Buffer,
    twiddles: &Buffer,
    transpose_forward_params: &Buffer,
    _transpose_reverse_params: &Buffer,
    height_strided_params: &Buffer,
    height_mode: HeightFftMode,
    complex_w: usize,
    batch_count: usize,
) {
    let row_count = width_plan.row_count;
    encode_real_width_stage(
        pack_real_pipeline,
        command_buffer,
        input,
        half_input,
        real_width_params,
        width_plan.len,
        row_count,
    );
    encode_row_fft(
        row_pipeline,
        row_double_shared_pipeline,
        command_buffer,
        width_plan,
        half_input,
        half_work,
    );
    encode_real_width_twiddled_stage(
        finalize_r2c_pipeline,
        command_buffer,
        half_work,
        reduced_a,
        real_width_params,
        twiddles,
        complex_w,
        row_count,
    );
    match height_mode {
        HeightFftMode::Transpose => {
            encode_transpose(
                transpose_pipeline,
                command_buffer,
                reduced_a,
                transpose,
                transpose_forward_params,
                complex_w,
                height_plan.len,
                batch_count,
            );
            encode_row_fft(
                row_pipeline,
                row_double_shared_pipeline,
                command_buffer,
                height_plan,
                transpose,
                output,
            );
        }
        HeightFftMode::Strided => {
            encode_row_fft_strided(
                row_strided_pipeline,
                command_buffer,
                height_plan,
                height_strided_params,
                reduced_a,
                output,
            );
        }
    }
}

fn encode_inverse_2d_real(
    row_pipeline: &ComputePipelineState,
    row_double_shared_pipeline: &ComputePipelineState,
    row_strided_pipeline: &ComputePipelineState,
    prepare_c2r_pipeline: &ComputePipelineState,
    prepare_c2r_transposed_pipeline: &ComputePipelineState,
    unpack_real_pipeline: &ComputePipelineState,
    transpose_pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    width_plan: &RowPlan,
    height_plan: &RowPlan,
    input: &Buffer,
    reduced_b: &Buffer,
    half_buffer: &Buffer,
    transpose: &Buffer,
    output: &Buffer,
    real_width_params: &Buffer,
    twiddles: &Buffer,
    transpose_forward_params: &Buffer,
    transpose_reverse_params: &Buffer,
    height_strided_params: &Buffer,
    input_layout: HeightFftMode,
    height_mode: HeightFftMode,
    complex_w: usize,
    fft_h: usize,
    batch_count: usize,
) {
    let width_inverse_input = match height_mode {
        HeightFftMode::Transpose => {
            let height_input = match input_layout {
                HeightFftMode::Transpose => input,
                HeightFftMode::Strided => {
                    encode_transpose(
                        transpose_pipeline,
                        command_buffer,
                        input,
                        transpose,
                        transpose_forward_params,
                        complex_w,
                        fft_h,
                        batch_count,
                    );
                    transpose
                }
            };
            encode_row_fft(
                row_pipeline,
                row_double_shared_pipeline,
                command_buffer,
                height_plan,
                height_input,
                reduced_b,
            );
            let prepare_input = match input_layout {
                HeightFftMode::Transpose => reduced_b,
                HeightFftMode::Strided => {
                    encode_transpose(
                        transpose_pipeline,
                        command_buffer,
                        reduced_b,
                        input,
                        transpose_reverse_params,
                        height_plan.len,
                        complex_w,
                        batch_count,
                    );
                    input
                }
            };
            encode_real_width_twiddled_stage(
                match input_layout {
                    HeightFftMode::Transpose => prepare_c2r_transposed_pipeline,
                    HeightFftMode::Strided => prepare_c2r_pipeline,
                },
                command_buffer,
                prepare_input,
                half_buffer,
                real_width_params,
                twiddles,
                width_plan.len,
                width_plan.row_count,
            );
            half_buffer
        }
        HeightFftMode::Strided => {
            let height_input = match input_layout {
                HeightFftMode::Transpose => {
                    encode_transpose(
                        transpose_pipeline,
                        command_buffer,
                        input,
                        transpose,
                        transpose_reverse_params,
                        fft_h,
                        complex_w,
                        batch_count,
                    );
                    transpose
                }
                HeightFftMode::Strided => input,
            };
            encode_row_fft_strided(
                row_strided_pipeline,
                command_buffer,
                height_plan,
                height_strided_params,
                height_input,
                reduced_b,
            );
            encode_real_width_twiddled_stage(
                prepare_c2r_pipeline,
                command_buffer,
                reduced_b,
                half_buffer,
                real_width_params,
                twiddles,
                width_plan.len,
                width_plan.row_count,
            );
            half_buffer
        }
    };
    encode_row_fft(
        row_pipeline,
        row_double_shared_pipeline,
        command_buffer,
        width_plan,
        width_inverse_input,
        transpose,
    );
    encode_real_width_stage(
        unpack_real_pipeline,
        command_buffer,
        transpose,
        output,
        real_width_params,
        width_plan.len,
        width_plan.row_count,
    );
}

fn encode_multiply(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    input: &Buffer,
    kernels: &Buffer,
    output: &Buffer,
    complex_count_buffer: &Buffer,
    complex_count: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(kernels), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_buffer(3, Some(complex_count_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: (complex_count * 2) as u64,
            height: 1,
            depth: 1,
        },
        threadgroup_1d(pipeline),
    );
    encoder.end_encoding();
}

fn encode_postprocess(
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    planes: &Buffer,
    out_x: &Buffer,
    out_y: &Buffer,
    magnitude: &Buffer,
    angle: &Buffer,
    params: &Buffer,
    width: usize,
    height: usize,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(planes), 0);
    encoder.set_buffer(1, Some(out_x), 0);
    encoder.set_buffer(2, Some(out_y), 0);
    encoder.set_buffer(3, Some(magnitude), 0);
    encoder.set_buffer(4, Some(angle), 0);
    encoder.set_buffer(5, Some(params), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
}

fn threadgroup_2d(pipeline: &ComputePipelineState) -> MTLSize {
    let execution_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let width = execution_width.min(max_threads).min(16);
    let height = (max_threads / width).clamp(1, 16).min(16);
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn threadgroup_1d(pipeline: &ComputePipelineState) -> MTLSize {
    let execution_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let width = (execution_width * 4).min(max_threads).max(execution_width);
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn choose_row_threadgroup_width(
    pipeline: &ComputePipelineState,
    row_len: usize,
) -> u64 {
    let execution_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let preferred = execution_width
        .saturating_mul(4)
        .min(max_threads)
        .max(execution_width);
    preferred.min(row_len as u64).max(1)
}

fn height_inverse_mode() -> HeightFftMode {
    match std::env::var("WVF_METAL_GPU_HEIGHT_INVERSE_MODE").ok().as_deref() {
        Some("strided") => HeightFftMode::Strided,
        Some("transpose") => HeightFftMode::Transpose,
        _ => HeightFftMode::Transpose,
    }
}

fn height_forward_mode() -> HeightFftMode {
    match std::env::var("WVF_METAL_GPU_HEIGHT_FORWARD_MODE").ok().as_deref() {
        Some("strided") => HeightFftMode::Strided,
        Some("transpose") => HeightFftMode::Transpose,
        _ => HeightFftMode::Transpose,
    }
}

fn next_smooth_fft_size(mut value: u64) -> u64 {
    value = value.max(2);
    while !is_smooth_fft_size(value) {
        value += 1;
    }
    value
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
