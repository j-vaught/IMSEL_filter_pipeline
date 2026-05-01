use super::DenseConvolutionKernels;
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
struct WvfFftStageParams {
    row_len: c_uint,
    row_count: c_uint,
    stride: c_uint,
    prev: c_uint,
    radix: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WvfFftTransposeParams {
    width: c_uint,
    height: c_uint,
    batch_count: c_uint,
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
}

struct StagePlan {
    params: WvfFftStageParams,
    params_buffer: Buffer,
    weights_buffer: Buffer,
}

struct RowPlan {
    len: usize,
    stages: Vec<StagePlan>,
}

struct ImagePlan {
    fft_w: usize,
    fft_h: usize,
    plane_len: usize,
    single_padded: Buffer,
    single_a: Buffer,
    single_b: Buffer,
    single_transpose: Buffer,
    single_output: Buffer,
    double_input: Buffer,
    double_a: Buffer,
    double_b: Buffer,
    double_transpose: Buffer,
    double_output: Buffer,
    pad_params_buffer: Buffer,
    postprocess_params_buffer: Buffer,
    complex_count_buffer: Buffer,
    transpose_forward_single_buffer: Buffer,
    transpose_reverse_single_buffer: Buffer,
    transpose_forward_double_buffer: Buffer,
    transpose_reverse_double_buffer: Buffer,
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

pub(super) struct GpuFftBackend {
    device: Device,
    queue: CommandQueue,
    pad_pipeline: ComputePipelineState,
    stage_pipeline: ComputePipelineState,
    transpose_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    postprocess_pipeline: ComputePipelineState,
    image_plans: HashMap<PlanKey, ImagePlan>,
    kernel_spectra: HashMap<KernelKey, KernelSpectra>,
}

impl GpuFftBackend {
    pub(super) fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let queue = device.new_command_queue();
        let library = compile_library(&device, FFT_SHADER_SOURCE)?;
        let pad_pipeline = pipeline(&device, &library, "wvf_fft_reflect_pad_complex_dense")?;
        let stage_pipeline = pipeline(&device, &library, "wvf_fft_stage_c2c")?;
        let transpose_pipeline = pipeline(&device, &library, "wvf_fft_transpose_c2c")?;
        let multiply_pipeline = pipeline(&device, &library, "wvf_fft_multiply_spectra")?;
        let postprocess_pipeline = pipeline(&device, &library, "wvf_fft_postprocess_complex_dense")?;
        Ok(Self {
            device,
            queue,
            pad_pipeline,
            stage_pipeline,
            transpose_pipeline,
            multiply_pipeline,
            postprocess_pipeline,
            image_plans: HashMap::new(),
            kernel_spectra: HashMap::new(),
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
        let spectrum_buffer = self.kernel_spectra(key, kernels)?;
        let plan = self
            .image_plans
            .get(&key)
            .ok_or_else(|| "GPU FFT image plan cache insertion failed".to_string())?;

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
        encode_c2c_2d(
            &self.stage_pipeline,
            &self.transpose_pipeline,
            &command_buffer,
            &plan.width_forward_single,
            &plan.height_forward_single,
            &plan.single_padded,
            &plan.single_a,
            &plan.single_b,
            &plan.single_transpose,
            &plan.single_output,
            &plan.transpose_forward_single_buffer,
            &plan.transpose_reverse_single_buffer,
            1,
        );
        encode_multiply(
            &self.multiply_pipeline,
            &command_buffer,
            &plan.single_output,
            &spectrum_buffer,
            &plan.double_input,
            &plan.complex_count_buffer,
            plan.plane_len,
        );
        encode_c2c_2d(
            &self.stage_pipeline,
            &self.transpose_pipeline,
            &command_buffer,
            &plan.width_inverse_double,
            &plan.height_inverse_double,
            &plan.double_input,
            &plan.double_a,
            &plan.double_b,
            &plan.double_transpose,
            &plan.double_output,
            &plan.transpose_forward_double_buffer,
            &plan.transpose_reverse_double_buffer,
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
    ) -> Result<Buffer, String> {
        let kernel_hash = hash_f32_bytes(&kernels.kernel_x, 14695981039346656037);
        let kernel_hash = hash_f32_bytes(&kernels.kernel_y, kernel_hash);
        let key = KernelKey {
            fft_w: plan_key.fft_w,
            fft_h: plan_key.fft_h,
            radius: plan_key.radius,
            kernel_width: kernels.kernel_width,
            kernel_hash,
        };
        if !self.kernel_spectra.contains_key(&key) {
            let plan = self
                .image_plans
                .get(&plan_key)
                .ok_or_else(|| "GPU FFT image plan was missing".to_string())?;
            let values = build_complex_kernel_values(plan, kernels)?;
            let input_bytes = crate::checked_len(
                values.len(),
                size_of::<Complex32>(),
                "complex kernel upload",
            )?;
            let upload = self.device.new_buffer_with_data(
                values.as_ptr().cast(),
                input_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let spectrum_buffer = self.device.new_buffer(
                input_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let command_buffer = self.queue.new_command_buffer();
            encode_c2c_2d(
                &self.stage_pipeline,
                &self.transpose_pipeline,
                &command_buffer,
                &plan.width_forward_double,
                &plan.height_forward_double,
                &upload,
                &plan.double_a,
                &plan.double_b,
                &plan.double_transpose,
                &spectrum_buffer,
                &plan.transpose_forward_double_buffer,
                &plan.transpose_reverse_double_buffer,
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
        let plane_len = crate::checked_len(fft_w, fft_h, "FFT plane")?;
        let padded_width = key
            .width
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded width exceeded uint32".to_string())?;
        let padded_height = key
            .height
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded height exceeded uint32".to_string())?;

        let complex_plane_bytes = crate::checked_len(
            plane_len,
            size_of::<Complex32>(),
            "complex FFT plane",
        )?;
        let complex_double_bytes = crate::checked_len(complex_plane_bytes, 2, "double complex plane")?;
        let complex_count = c_uint::try_from(plane_len).map_err(|_| "FFT plane exceeded uint32".to_string())?;

        let width_radices = factor_fft_length(fft_w)?;
        let height_radices = factor_fft_length(fft_h)?;
        let width_forward_single =
            build_row_plan(&self.device, fft_w, fft_h, 1, &width_radices, false)?;
        let width_forward_double =
            build_row_plan(&self.device, fft_w, fft_h, 2, &width_radices, false)?;
        let width_inverse_double =
            build_row_plan(&self.device, fft_w, fft_h, 2, &width_radices, true)?;
        let height_forward_single =
            build_row_plan(&self.device, fft_h, fft_w, 1, &height_radices, false)?;
        let height_forward_double =
            build_row_plan(&self.device, fft_h, fft_w, 2, &height_radices, false)?;
        let height_inverse_double =
            build_row_plan(&self.device, fft_h, fft_w, 2, &height_radices, true)?;

        let private = MTLResourceOptions::StorageModePrivate;
        let single_padded = self.device.new_buffer(complex_plane_bytes as u64, private);
        let single_a = self.device.new_buffer(complex_plane_bytes as u64, private);
        let single_b = self.device.new_buffer(complex_plane_bytes as u64, private);
        let single_transpose = self.device.new_buffer(complex_plane_bytes as u64, private);
        let single_output = self.device.new_buffer(complex_plane_bytes as u64, private);
        let double_input = self.device.new_buffer(complex_double_bytes as u64, private);
        let double_a = self.device.new_buffer(complex_double_bytes as u64, private);
        let double_b = self.device.new_buffer(complex_double_bytes as u64, private);
        let double_transpose = self.device.new_buffer(complex_double_bytes as u64, private);
        let double_output = self.device.new_buffer(complex_double_bytes as u64, private);

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
            plane_stride: complex_count,
            scale: 1.0 / plane_len as f32,
        };
        let transpose_forward_single = WvfFftTransposeParams {
            width: key.fft_w,
            height: key.fft_h,
            batch_count: 1,
        };
        let transpose_reverse_single = WvfFftTransposeParams {
            width: key.fft_h,
            height: key.fft_w,
            batch_count: 1,
        };
        let transpose_forward_double = WvfFftTransposeParams {
            width: key.fft_w,
            height: key.fft_h,
            batch_count: 2,
        };
        let transpose_reverse_double = WvfFftTransposeParams {
            width: key.fft_h,
            height: key.fft_w,
            batch_count: 2,
        };

        Ok(ImagePlan {
            fft_w,
            fft_h,
            plane_len,
            single_padded,
            single_a,
            single_b,
            single_transpose,
            single_output,
            double_input,
            double_a,
            double_b,
            double_transpose,
            double_output,
            pad_params_buffer: param_buffer(&self.device, &pad_params),
            postprocess_params_buffer: param_buffer(&self.device, &postprocess_params),
            complex_count_buffer: param_buffer(&self.device, &complex_count),
            transpose_forward_single_buffer: param_buffer(&self.device, &transpose_forward_single),
            transpose_reverse_single_buffer: param_buffer(&self.device, &transpose_reverse_single),
            transpose_forward_double_buffer: param_buffer(&self.device, &transpose_forward_double),
            transpose_reverse_double_buffer: param_buffer(&self.device, &transpose_reverse_double),
            width_forward_single,
            width_forward_double,
            width_inverse_double,
            height_forward_single,
            height_forward_double,
            height_inverse_double,
        })
    }
}

fn build_complex_kernel_values(
    plan: &ImagePlan,
    kernels: &DenseConvolutionKernels,
) -> Result<Vec<Complex32>, String> {
    let mut values = vec![Complex32::new(0.0, 0.0); plan.plane_len * 2];
    let kernel_width = kernels.kernel_width as usize;
    for y in 0..kernel_width {
        for x in 0..kernel_width {
            let src = y * kernel_width + x;
            let dst = y * plan.fft_w + x;
            values[dst] = Complex32::new(kernels.kernel_x[src], 0.0);
            values[plan.plane_len + dst] = Complex32::new(kernels.kernel_y[src], 0.0);
        }
    }
    Ok(values)
}

fn build_row_plan(
    device: &Device,
    len: usize,
    rows: usize,
    batch_count: usize,
    radices: &[u32],
    inverse: bool,
) -> Result<RowPlan, String> {
    let mut stages = Vec::with_capacity(radices.len());
    let mut prev = 1usize;
    for &radix in radices {
        let stride = len
            .checked_div(prev * radix as usize)
            .ok_or_else(|| "invalid FFT stage factorization".to_string())?;
        let params = WvfFftStageParams {
            row_len: len as c_uint,
            row_count: (rows * batch_count) as c_uint,
            stride: stride as c_uint,
            prev: prev as c_uint,
            radix,
        };
        let weights = build_stage_weights(radix as usize, prev, inverse)?;
        stages.push(StagePlan {
            params,
            params_buffer: param_buffer(device, &params),
            weights_buffer: device.new_buffer_with_data(
                weights.as_ptr().cast(),
                (weights.len() * size_of::<Complex32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        });
        prev *= radix as usize;
    }
    Ok(RowPlan { len, stages })
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
    pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    plan: &RowPlan,
    input: &Buffer,
    scratch_a: &Buffer,
    scratch_b: &Buffer,
) -> bool {
    let mut src = input;
    for (index, stage) in plan.stages.iter().enumerate() {
        let dst = if index % 2 == 0 { scratch_a } else { scratch_b };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        encoder.set_buffer(2, Some(&stage.params_buffer), 0);
        encoder.set_buffer(3, Some(&stage.weights_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: plan.len as u64,
                height: stage.params.row_count as u64,
                depth: 1,
            },
            threadgroup_1d(pipeline),
        );
        encoder.end_encoding();
        src = dst;
    }
    plan.stages.len() % 2 == 1
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
    encoder.dispatch_threads(
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: batch_count as u64,
        },
        threadgroup_2d(pipeline),
    );
    encoder.end_encoding();
}

fn encode_c2c_2d(
    stage_pipeline: &ComputePipelineState,
    transpose_pipeline: &ComputePipelineState,
    command_buffer: &metal::CommandBufferRef,
    width_plan: &RowPlan,
    height_plan: &RowPlan,
    input: &Buffer,
    scratch_a: &Buffer,
    scratch_b: &Buffer,
    transpose: &Buffer,
    output: &Buffer,
    transpose_forward_params: &Buffer,
    transpose_reverse_params: &Buffer,
    batch_count: usize,
) {
    let width_in_a = encode_row_fft(stage_pipeline, command_buffer, width_plan, input, scratch_a, scratch_b);
    let width_output = if width_in_a { scratch_a } else { scratch_b };
    encode_transpose(
        transpose_pipeline,
        command_buffer,
        width_output,
        transpose,
        transpose_forward_params,
        width_plan.len,
        height_plan.len,
        batch_count,
    );
    let height_in_a = encode_row_fft(stage_pipeline, command_buffer, height_plan, transpose, scratch_a, scratch_b);
    let height_output = if height_in_a { scratch_a } else { scratch_b };
    encode_transpose(
        transpose_pipeline,
        command_buffer,
        height_output,
        output,
        transpose_reverse_params,
        height_plan.len,
        width_plan.len,
        batch_count,
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
