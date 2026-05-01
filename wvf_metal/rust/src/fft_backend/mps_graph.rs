use super::interop;
use super::DenseConvolutionKernels;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal_performance_shaders::{MPSCommandBuffer, MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphDevice, MPSGraphFFTDescriptor, MPSGraphFFTScalingMode,
    MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
    MPSGraphTensorShapedTypeDictionary, MPSGraphExecutable,
};
use std::collections::HashMap;
use std::mem::size_of;
use std::os::raw::{c_float, c_uint};
use std::time::Instant;

const FFT_SHADER_SOURCE: &str = include_str!("../wvf.metal");

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
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImagePlanKey {
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    fft_w: c_uint,
    fft_h: c_uint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct KernelPlanKey {
    fft_w: c_uint,
    fft_h: c_uint,
    kernel_width: c_uint,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct KernelSpectrumKey {
    fft_w: c_uint,
    fft_h: c_uint,
    radius: c_uint,
    kernel_width: c_uint,
    kernel_hash: u64,
}

#[derive(Clone)]
struct ImagePlan {
    fft_w: c_uint,
    fft_h: c_uint,
    complex_count: c_uint,
    padded_buffer: Buffer,
    image_spectrum_buffer: Buffer,
    multiplied_spectra_buffer: Buffer,
    inverse_real_buffer: Buffer,
    pad_params_buffer: Buffer,
    postprocess_params_buffer: Buffer,
    complex_count_buffer: Buffer,
    forward_executable: Retained<MPSGraphExecutable>,
    inverse_executable: Retained<MPSGraphExecutable>,
    forward_inputs: Retained<NSArray<MPSGraphTensorData>>,
    forward_results: Retained<NSArray<MPSGraphTensorData>>,
    inverse_inputs: Retained<NSArray<MPSGraphTensorData>>,
    inverse_results: Retained<NSArray<MPSGraphTensorData>>,
}

#[derive(Clone)]
struct KernelPlan {
    executable: Retained<MPSGraphExecutable>,
    input_shape: Retained<MPSShape>,
    output_shape: Retained<MPSShape>,
    input_row_bytes: usize,
    output_row_bytes: usize,
    output_bytes: usize,
}

#[derive(Clone)]
struct KernelSpectra {
    buffer: Buffer,
}

pub(super) struct MpsGraphFftBackend {
    device: Device,
    queue: CommandQueue,
    graph_device: Retained<MPSGraphDevice>,
    pad_pipeline: ComputePipelineState,
    multiply_pipeline: ComputePipelineState,
    postprocess_pipeline: ComputePipelineState,
    image_plans: HashMap<ImagePlanKey, ImagePlan>,
    kernel_plans: HashMap<KernelPlanKey, KernelPlan>,
    kernel_spectra: HashMap<KernelSpectrumKey, KernelSpectra>,
}

struct ProfileRun {
    enabled: bool,
    radius: c_uint,
    width: c_uint,
    height: c_uint,
    start: Instant,
    last: Instant,
    steps: Vec<(&'static str, f64)>,
}

impl ProfileRun {
    fn new(width: c_uint, height: c_uint, radius: c_uint) -> Self {
        let enabled = std::env::var("WVF_METAL_FFT_PROFILE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !normalized.is_empty() && normalized != "0" && normalized != "false"
            })
            .unwrap_or(false);
        let start = Instant::now();
        Self {
            enabled,
            radius,
            width,
            height,
            start,
            last: start,
            steps: Vec::new(),
        }
    }

    fn mark(&mut self, label: &'static str) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        self.steps
            .push((label, now.duration_since(self.last).as_secs_f64() * 1_000.0));
        self.last = now;
    }

    fn finish(mut self) {
        if !self.enabled {
            return;
        }
        self.mark("finish");
        let total_ms = self.start.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "wvf_metal experimental fft profile size={}x{} radius={} total_ms={total_ms:.3} steps={:?}",
            self.height,
            self.width,
            self.radius,
            self.steps
        );
    }
}

impl MpsGraphFftBackend {
    pub(super) fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let queue = device.new_command_queue();
        let graph_device = unsafe { MPSGraphDevice::deviceWithMTLDevice(interop::device_ref(&device)) };
        let library = compile_library(&device, FFT_SHADER_SOURCE)?;
        let pad_pipeline = pipeline(&device, &library, "wvf_fft_reflect_pad_real_dense")?;
        let multiply_pipeline = pipeline(&device, &library, "wvf_fft_multiply_spectra")?;
        let postprocess_pipeline = pipeline(&device, &library, "wvf_fft_postprocess_dense")?;
        Ok(Self {
            device,
            queue,
            graph_device,
            pad_pipeline,
            multiply_pipeline,
            postprocess_pipeline,
            image_plans: HashMap::new(),
            kernel_plans: HashMap::new(),
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
        let mut profile = ProfileRun::new(width, height, radius);
        if image.is_null() || out_x.is_null() || out_y.is_null() || magnitude.is_null() || angle.is_null() {
            return Err("null pointer passed to FFT backend".to_string());
        }

        let fft_w_u64 = next_smooth_fft_size(u64::from(width) + 4 * u64::from(radius));
        let fft_h_u64 = next_smooth_fft_size(u64::from(height) + 4 * u64::from(radius));
        let fft_w = c_uint::try_from(fft_w_u64)
            .map_err(|_| "FFT width exceeded uint32".to_string())?;
        let fft_h = c_uint::try_from(fft_h_u64)
            .map_err(|_| "FFT height exceeded uint32".to_string())?;

        let plan = self.image_plan(ImagePlanKey {
            width,
            height,
            radius,
            fft_w,
            fft_h,
        })?;
        let spectra = self.kernel_spectra(radius, fft_w, fft_h, kernels)?;
        profile.mark("plan_and_kernel_cache");

        let image_pixels = crate::checked_image_pixels(width, height)?;
        let image_bytes = crate::checked_len(image_pixels, size_of::<c_float>(), "image")?;
        let output_bytes = image_bytes;
        let shared = MTLResourceOptions::StorageModeShared;
        let image_buffer =
            self.device
                .new_buffer_with_bytes_no_copy(image.cast(), image_bytes as u64, shared, None);
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
        profile.mark("buffer_wrap");

        let mps_command_buffer = unsafe {
            MPSCommandBuffer::commandBufferFromCommandQueue(interop::command_queue_ref(&self.queue))
        };
        let mut root = unsafe { mps_command_buffer.rootCommandBuffer() };
        let mut command_buffer = unsafe { interop::metal_command_buffer_ref(&root) };

        self.encode_pad(command_buffer, &image_buffer, &plan)?;
        profile.mark("pad");

        unsafe {
            let _ = plan
                .forward_executable
                .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                    &mps_command_buffer,
                    &plan.forward_inputs,
                    Some(&plan.forward_results),
                    None,
                );
        }
        profile.mark("forward_fft");

        root = unsafe { mps_command_buffer.rootCommandBuffer() };
        command_buffer = unsafe { interop::metal_command_buffer_ref(&root) };
        self.encode_multiply(command_buffer, &plan, &spectra)?;
        profile.mark("multiply");

        unsafe {
            let _ = plan
                .inverse_executable
                .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                    &mps_command_buffer,
                    &plan.inverse_inputs,
                    Some(&plan.inverse_results),
                    None,
                );
        }
        profile.mark("inverse_fft");

        root = unsafe { mps_command_buffer.rootCommandBuffer() };
        command_buffer = unsafe { interop::metal_command_buffer_ref(&root) };
        self.encode_postprocess(
            command_buffer,
            &plan,
            &out_x_buffer,
            &out_y_buffer,
            &magnitude_buffer,
            &angle_buffer,
        )?;
        profile.mark("postprocess_encode");

        command_buffer.commit();
        command_buffer.wait_until_completed();
        profile.mark("wait");
        profile.finish();
        Ok(())
    }

    fn image_plan(&mut self, key: ImagePlanKey) -> Result<ImagePlan, String> {
        if !self.image_plans.contains_key(&key) {
            let plan = self.build_image_plan(key)?;
            self.image_plans.insert(key, plan);
        }
        self.image_plans
            .get(&key)
            .cloned()
            .ok_or_else(|| "image FFT plan cache insertion failed".to_string())
    }

    fn kernel_plan(&mut self, key: KernelPlanKey) -> Result<KernelPlan, String> {
        if !self.kernel_plans.contains_key(&key) {
            let plan = self.build_kernel_plan(key)?;
            self.kernel_plans.insert(key, plan);
        }
        self.kernel_plans
            .get(&key)
            .cloned()
            .ok_or_else(|| "kernel FFT plan cache insertion failed".to_string())
    }

    fn kernel_spectra(
        &mut self,
        radius: c_uint,
        fft_w: c_uint,
        fft_h: c_uint,
        kernels: &DenseConvolutionKernels,
    ) -> Result<KernelSpectra, String> {
        let kernel_hash = hash_bytes(&kernels.kernel_x, 14695981039346656037);
        let kernel_hash = hash_bytes(&kernels.kernel_y, kernel_hash);
        let key = KernelSpectrumKey {
            fft_w,
            fft_h,
            radius,
            kernel_width: kernels.kernel_width,
            kernel_hash,
        };
        if !self.kernel_spectra.contains_key(&key) {
            let plan = self.kernel_plan(KernelPlanKey {
                fft_w,
                fft_h,
                kernel_width: kernels.kernel_width,
            })?;
            let kernel_values = stacked_kernel_values(kernels);
            let input_bytes = crate::checked_len(
                kernel_values.len(),
                size_of::<c_float>(),
                "kernel FFT input",
            )?;
            let input_buffer = self.device.new_buffer_with_data(
                kernel_values.as_ptr().cast(),
                input_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let spectrum_buffer = self.device.new_buffer(
                plan.output_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let input_data = tensor_data(
                &input_buffer,
                &plan.input_shape,
                MPSDataType::Float32,
                Some(plan.input_row_bytes),
            );
            let output_data = tensor_data(
                &spectrum_buffer,
                &plan.output_shape,
                MPSDataType::ComplexFloat32,
                Some(plan.output_row_bytes),
            );
            let inputs = NSArray::from_slice(&[&*input_data]);
            let results = NSArray::from_slice(&[&*output_data]);
            unsafe {
                let _ = plan
                    .executable
                    .runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
                        interop::command_queue_ref(&self.queue),
                        &inputs,
                        Some(&results),
                        None,
                    );
            }
            self.kernel_spectra
                .insert(key, KernelSpectra { buffer: spectrum_buffer });
        }
        self.kernel_spectra
            .get(&key)
            .cloned()
            .ok_or_else(|| "kernel spectrum cache insertion failed".to_string())
    }

    fn build_image_plan(&self, key: ImagePlanKey) -> Result<ImagePlan, String> {
        let complex_w = key.fft_w as usize / 2 + 1;
        let complex_count = crate::checked_len(key.fft_h as usize, complex_w, "complex plane")?;
        let complex_count_u32 =
            c_uint::try_from(complex_count).map_err(|_| "complex plane exceeded uint32".to_string())?;
        let plane_stride = crate::checked_len(
            key.fft_w as usize,
            key.fft_h as usize,
            "real plane",
        )?;
        let plane_stride_u32 =
            c_uint::try_from(plane_stride).map_err(|_| "real plane exceeded uint32".to_string())?;

        let forward_input_shape = shape(&[key.fft_h as usize, key.fft_w as usize]);
        let forward_output_shape = shape(&[key.fft_h as usize, complex_w]);
        let inverse_input_shape = shape(&[2, key.fft_h as usize, complex_w]);
        let inverse_output_shape = shape(&[2, key.fft_h as usize, key.fft_w as usize]);

        let forward_executable = build_forward_fft_executable(
            &self.graph_device,
            &forward_input_shape,
            &forward_output_shape,
        );
        let inverse_executable = build_inverse_fft_executable(
            &self.graph_device,
            &inverse_input_shape,
            &inverse_output_shape,
        );

        let fft_row_bytes = crate::checked_len(key.fft_w as usize, size_of::<c_float>(), "FFT row")?;
        let padded_bytes = crate::checked_len(fft_row_bytes, key.fft_h as usize, "padded buffer")?;
        let complex_row_bytes =
            crate::checked_len(complex_w, size_of::<[f32; 2]>(), "complex spectrum row")?;
        let image_spectrum_bytes =
            crate::checked_len(complex_row_bytes, key.fft_h as usize, "image spectrum buffer")?;
        let multiplied_spectra_bytes =
            crate::checked_len(image_spectrum_bytes, 2, "multiplied spectrum buffer")?;
        let inverse_real_bytes =
            crate::checked_len(padded_bytes, 2, "inverse real buffer")?;

        let private = MTLResourceOptions::StorageModePrivate;
        let padded_buffer = self.device.new_buffer(padded_bytes as u64, private);
        let image_spectrum_buffer = self.device.new_buffer(image_spectrum_bytes as u64, private);
        let multiplied_spectra_buffer =
            self.device.new_buffer(multiplied_spectra_bytes as u64, private);
        let inverse_real_buffer = self.device.new_buffer(inverse_real_bytes as u64, private);

        let padded_input = tensor_data(
            &padded_buffer,
            &forward_input_shape,
            MPSDataType::Float32,
            Some(fft_row_bytes),
        );
        let image_spectrum_output = tensor_data(
            &image_spectrum_buffer,
            &forward_output_shape,
            MPSDataType::ComplexFloat32,
            Some(complex_row_bytes),
        );
        let multiplied_spectra_input = tensor_data(
            &multiplied_spectra_buffer,
            &inverse_input_shape,
            MPSDataType::ComplexFloat32,
            Some(complex_row_bytes),
        );
        let inverse_real_output = tensor_data(
            &inverse_real_buffer,
            &inverse_output_shape,
            MPSDataType::Float32,
            Some(fft_row_bytes),
        );

        let forward_inputs = NSArray::from_slice(&[&*padded_input]);
        let forward_results = NSArray::from_slice(&[&*image_spectrum_output]);
        let inverse_inputs = NSArray::from_slice(&[&*multiplied_spectra_input]);
        let inverse_results = NSArray::from_slice(&[&*inverse_real_output]);

        let padded_width = key
            .width
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded width exceeded uint32".to_string())?;
        let padded_height = key
            .height
            .checked_add(key.radius.saturating_mul(2))
            .ok_or_else(|| "padded height exceeded uint32".to_string())?;
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
            plane_stride: plane_stride_u32,
        };
        let pad_params_buffer = self.device.new_buffer_with_data(
            (&pad_params as *const WvfFftPadParams).cast(),
            size_of::<WvfFftPadParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let postprocess_params_buffer = self.device.new_buffer_with_data(
            (&postprocess_params as *const WvfFftPostprocessParams).cast(),
            size_of::<WvfFftPostprocessParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let complex_count_buffer = self.device.new_buffer_with_data(
            (&complex_count_u32 as *const c_uint).cast(),
            size_of::<c_uint>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(ImagePlan {
            fft_w: key.fft_w,
            fft_h: key.fft_h,
            complex_count: complex_count_u32,
            padded_buffer,
            image_spectrum_buffer,
            multiplied_spectra_buffer,
            inverse_real_buffer,
            pad_params_buffer,
            postprocess_params_buffer,
            complex_count_buffer,
            forward_executable,
            inverse_executable,
            forward_inputs,
            forward_results,
            inverse_inputs,
            inverse_results,
        })
    }

    fn build_kernel_plan(&self, key: KernelPlanKey) -> Result<KernelPlan, String> {
        let input_shape = shape(&[2, key.kernel_width as usize, key.kernel_width as usize]);
        let complex_w = key.fft_w as usize / 2 + 1;
        let output_shape = shape(&[2, key.fft_h as usize, complex_w]);
        let executable = unsafe {
            let graph = MPSGraph::new();
            let kernel_tensor = graph.placeholderWithShape_dataType_name(
                Some(&input_shape),
                MPSDataType::Float32,
                None,
            );
            let zero_left = shape(&[0, 0, 0]);
            let zero_right = shape(&[
                0,
                key.fft_h as usize - key.kernel_width as usize,
                key.fft_w as usize - key.kernel_width as usize,
            ]);
            let padded = graph.padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
                &kernel_tensor,
                objc2_metal_performance_shaders_graph::MPSGraphPaddingMode::Constant,
                &zero_left,
                &zero_right,
                0.0,
                None,
            );
            let fft_axes = index_array(&[1, 2]);
            let fft_desc = fft_descriptor(false, MPSGraphFFTScalingMode::None);
            let spectrum = graph.realToHermiteanFFTWithTensor_axes_descriptor_name(
                &padded,
                &fft_axes,
                &fft_desc,
                None,
            );
            let feed_types =
                shaped_type_dictionary(&[(&kernel_tensor, &input_shape, MPSDataType::Float32)]);
            let targets = NSArray::from_slice(&[&*spectrum]);
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                Some(&self.graph_device),
                &feed_types,
                &targets,
                None,
                None,
            )
        };
        let output_row_bytes =
            crate::checked_len(complex_w, size_of::<[f32; 2]>(), "kernel spectrum row")?;
        let output_elems = crate::checked_len(
            2,
            crate::checked_len(key.fft_h as usize, complex_w, "kernel spectrum shape")?,
            "kernel spectrum shape",
        )?;
        let output_bytes = crate::checked_len(
            output_elems,
            size_of::<[f32; 2]>(),
            "kernel spectrum buffer",
        )?;

        Ok(KernelPlan {
            executable,
            input_shape,
            output_shape,
            input_row_bytes: key.kernel_width as usize * size_of::<c_float>(),
            output_row_bytes,
            output_bytes,
        })
    }

    fn encode_pad(
        &self,
        command_buffer: &metal::CommandBufferRef,
        image_buffer: &Buffer,
        plan: &ImagePlan,
    ) -> Result<(), String> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pad_pipeline);
        encoder.set_buffer(0, Some(image_buffer), 0);
        encoder.set_buffer(1, Some(&plan.padded_buffer), 0);
        encoder.set_buffer(2, Some(&plan.pad_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: plan.fft_w as u64,
                height: plan.fft_h as u64,
                depth: 1,
            },
            threadgroup_2d(&self.pad_pipeline),
        );
        encoder.end_encoding();
        Ok(())
    }

    fn encode_multiply(
        &self,
        command_buffer: &metal::CommandBufferRef,
        plan: &ImagePlan,
        spectra: &KernelSpectra,
    ) -> Result<(), String> {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.multiply_pipeline);
        encoder.set_buffer(0, Some(&plan.image_spectrum_buffer), 0);
        encoder.set_buffer(1, Some(&spectra.buffer), 0);
        encoder.set_buffer(2, Some(&plan.multiplied_spectra_buffer), 0);
        encoder.set_buffer(3, Some(&plan.complex_count_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(plan.complex_count) * 2,
                height: 1,
                depth: 1,
            },
            threadgroup_1d(&self.multiply_pipeline),
        );
        encoder.end_encoding();
        Ok(())
    }

    fn encode_postprocess(
        &self,
        command_buffer: &metal::CommandBufferRef,
        plan: &ImagePlan,
        out_x: &Buffer,
        out_y: &Buffer,
        magnitude: &Buffer,
        angle: &Buffer,
    ) -> Result<(), String> {
        let params: WvfFftPostprocessParams = unsafe {
            *(plan.postprocess_params_buffer.contents() as *const WvfFftPostprocessParams)
        };
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.postprocess_pipeline);
        encoder.set_buffer(0, Some(&plan.inverse_real_buffer), 0);
        encoder.set_buffer(1, Some(out_x), 0);
        encoder.set_buffer(2, Some(out_y), 0);
        encoder.set_buffer(3, Some(magnitude), 0);
        encoder.set_buffer(4, Some(angle), 0);
        encoder.set_buffer(5, Some(&plan.postprocess_params_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: params.width as u64,
                height: params.height as u64,
                depth: 1,
            },
            threadgroup_2d(&self.postprocess_pipeline),
        );
        encoder.end_encoding();
        Ok(())
    }
}

fn compile_library(device: &Device, source: &str) -> Result<Library, String> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    device
        .new_library_with_source(source, &options)
        .map_err(|err| format!("failed to compile Metal FFT utility shaders: {err}"))
}

fn pipeline(device: &Device, library: &Library, name: &str) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|err| format!("failed to load Metal FFT function {name}: {err}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| format!("failed to create Metal FFT pipeline {name}: {err}"))
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

fn threadgroup_1d(pipeline: &ComputePipelineState) -> MTLSize {
    MTLSize {
        width: pipeline.max_total_threads_per_threadgroup().max(1),
        height: 1,
        depth: 1,
    }
}

fn build_forward_fft_executable(
    graph_device: &MPSGraphDevice,
    input_shape: &MPSShape,
    _output_shape: &MPSShape,
) -> Retained<MPSGraphExecutable> {
    unsafe {
        let graph = MPSGraph::new();
        let input_tensor = graph.placeholderWithShape_dataType_name(
            Some(input_shape),
            MPSDataType::Float32,
            None,
        );
        let axes = index_array(&[0, 1]);
        let descriptor = fft_descriptor(false, MPSGraphFFTScalingMode::None);
        let spectrum = graph.realToHermiteanFFTWithTensor_axes_descriptor_name(
            &input_tensor,
            &axes,
            &descriptor,
            None,
        );
        let feed_types =
            shaped_type_dictionary(&[(&input_tensor, input_shape, MPSDataType::Float32)]);
        let targets = NSArray::from_slice(&[&*spectrum]);
        graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
            Some(graph_device),
            &feed_types,
            &targets,
            None,
            None,
        )
    }
}

fn build_inverse_fft_executable(
    graph_device: &MPSGraphDevice,
    input_shape: &MPSShape,
    _output_shape: &MPSShape,
) -> Retained<MPSGraphExecutable> {
    unsafe {
        let graph = MPSGraph::new();
        let input_tensor = graph.placeholderWithShape_dataType_name(
            Some(input_shape),
            MPSDataType::ComplexFloat32,
            None,
        );
        let axes = index_array(&[1, 2]);
        let descriptor = fft_descriptor(true, MPSGraphFFTScalingMode::Size);
        let planes = graph.HermiteanToRealFFTWithTensor_axes_descriptor_name(
            &input_tensor,
            &axes,
            &descriptor,
            None,
        );
        let feed_types =
            shaped_type_dictionary(&[(&input_tensor, input_shape, MPSDataType::ComplexFloat32)]);
        let targets = NSArray::from_slice(&[&*planes]);
        graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
            Some(graph_device),
            &feed_types,
            &targets,
            None,
            None,
        )
    }
}

fn fft_descriptor(
    inverse: bool,
    scaling_mode: MPSGraphFFTScalingMode,
) -> Retained<MPSGraphFFTDescriptor> {
    let descriptor = unsafe { MPSGraphFFTDescriptor::new() };
    unsafe {
        descriptor.setInverse(inverse);
        descriptor.setScalingMode(scaling_mode);
        descriptor.setRoundToOddHermitean(false);
    }
    descriptor
}

fn shape(dims: &[usize]) -> Retained<MPSShape> {
    let values: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|&dim| NSNumber::new_usize(dim))
        .collect();
    NSArray::from_retained_slice(&values)
}

fn index_array(values: &[i64]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values
        .iter()
        .map(|&value| NSNumber::new_i64(value))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

fn shaped_type_dictionary(
    entries: &[(&MPSGraphTensor, &MPSShape, MPSDataType)],
) -> Retained<MPSGraphTensorShapedTypeDictionary> {
    let mut keys = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (tensor, shape, data_type) in entries {
        keys.push(*tensor);
        values.push(shaped_type(shape, *data_type));
    }
    NSDictionary::from_retained_objects(&keys, &values)
}

fn shaped_type(shape: &MPSShape, data_type: MPSDataType) -> Retained<MPSGraphShapedType> {
    unsafe {
        MPSGraphShapedType::initWithShape_dataType(
            MPSGraphShapedType::alloc(),
            Some(shape),
            data_type,
        )
    }
}

fn tensor_data(
    buffer: &Buffer,
    shape: &MPSShape,
    data_type: MPSDataType,
    row_bytes: Option<usize>,
) -> Retained<MPSGraphTensorData> {
    unsafe {
        match row_bytes {
            Some(row_bytes) => MPSGraphTensorData::initWithMTLBuffer_shape_dataType_rowBytes(
                MPSGraphTensorData::alloc(),
                interop::buffer_ref(buffer),
                shape,
                data_type,
                row_bytes,
            ),
            None => MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                interop::buffer_ref(buffer),
                shape,
                data_type,
            ),
        }
    }
}

fn stacked_kernel_values(kernels: &DenseConvolutionKernels) -> Vec<c_float> {
    let mut values = Vec::with_capacity(kernels.kernel_x.len() + kernels.kernel_y.len());
    values.extend_from_slice(&kernels.kernel_x);
    values.extend_from_slice(&kernels.kernel_y);
    values
}

fn hash_bytes(values: &[c_float], seed: u64) -> u64 {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let mut hash = seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
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
