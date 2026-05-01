use super::interop;
use super::DenseConvolutionKernels;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphDevice, MPSGraphFFTDescriptor, MPSGraphFFTScalingMode, MPSGraphPaddingMode,
    MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData, MPSGraphTensorShapedTypeDictionary,
    MPSGraphExecutable,
};
use std::collections::HashMap;
use std::mem::size_of;
use std::os::raw::{c_float, c_uint};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImagePlanKey {
    width: c_uint,
    height: c_uint,
    radius: c_uint,
    fft_w: c_uint,
    fft_h: c_uint,
    kernel_width: c_uint,
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
    executable: Retained<MPSGraphExecutable>,
    image_shape: Retained<MPSShape>,
    kernel_shape: Retained<MPSShape>,
    output_shape: Retained<MPSShape>,
    kernel_row_bytes: usize,
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
    image_plans: HashMap<ImagePlanKey, ImagePlan>,
    kernel_plans: HashMap<KernelPlanKey, KernelPlan>,
    kernel_spectra: HashMap<KernelSpectrumKey, KernelSpectra>,
}

impl MpsGraphFftBackend {
    pub(super) fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "no Metal device is available".to_string())?;
        let queue = device.new_command_queue();
        let graph_device = unsafe { MPSGraphDevice::deviceWithMTLDevice(interop::device_ref(&device)) };
        Ok(Self {
            device,
            queue,
            graph_device,
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
        if image.is_null() || out_x.is_null() || out_y.is_null() || magnitude.is_null() || angle.is_null() {
            return Err("null pointer passed to FFT backend".to_string());
        }

        let fft_w_u64 = next_smooth_fft_size(u64::from(width) + 4 * u64::from(radius));
        let fft_h_u64 = next_smooth_fft_size(u64::from(height) + 4 * u64::from(radius));
        let fft_w = c_uint::try_from(fft_w_u64)
            .map_err(|_| "FFT width exceeded uint32".to_string())?;
        let fft_h = c_uint::try_from(fft_h_u64)
            .map_err(|_| "FFT height exceeded uint32".to_string())?;
        let plan_key = ImagePlanKey {
            width,
            height,
            radius,
            fft_w,
            fft_h,
            kernel_width: kernels.kernel_width,
        };

        let plan = self.image_plan(plan_key)?;
        let spectra = self.kernel_spectra(radius, fft_w, fft_h, kernels)?;

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

        let image_row_bytes = width as usize * size_of::<c_float>();
        let output_row_bytes = image_row_bytes;
        let image_data = tensor_data(
            &image_buffer,
            &plan.image_shape,
            MPSDataType::Float32,
            Some(image_row_bytes),
        );
        let kernel_data = tensor_data(
            &spectra.buffer,
            &plan.kernel_shape,
            MPSDataType::ComplexFloat32,
            Some(plan.kernel_row_bytes),
        );
        let out_x_data = tensor_data(
            &out_x_buffer,
            &plan.output_shape,
            MPSDataType::Float32,
            Some(output_row_bytes),
        );
        let out_y_data = tensor_data(
            &out_y_buffer,
            &plan.output_shape,
            MPSDataType::Float32,
            Some(output_row_bytes),
        );
        let magnitude_data = tensor_data(
            &magnitude_buffer,
            &plan.output_shape,
            MPSDataType::Float32,
            Some(output_row_bytes),
        );
        let angle_data = tensor_data(
            &angle_buffer,
            &plan.output_shape,
            MPSDataType::Float32,
            Some(output_row_bytes),
        );

        let inputs = NSArray::from_slice(&[&*image_data, &*kernel_data]);
        let results = NSArray::from_slice(&[
            &*out_x_data,
            &*out_y_data,
            &*magnitude_data,
            &*angle_data,
        ]);
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
                MPSGraphPaddingMode::Constant,
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
        let output_row_bytes = crate::checked_len(complex_w, size_of::<[f32; 2]>(), "kernel spectrum row")?;
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

    fn build_image_plan(&self, key: ImagePlanKey) -> Result<ImagePlan, String> {
        let image_shape = shape(&[key.height as usize, key.width as usize]);
        let complex_w = key.fft_w as usize / 2 + 1;
        let kernel_shape = shape(&[2, key.fft_h as usize, complex_w]);
        let output_shape = shape(&[key.height as usize, key.width as usize]);
        let executable = unsafe {
            let graph = MPSGraph::new();
            let image_tensor = graph.placeholderWithShape_dataType_name(
                Some(&image_shape),
                MPSDataType::Float32,
                None,
            );
            let kernel_tensor = graph.placeholderWithShape_dataType_name(
                Some(&kernel_shape),
                MPSDataType::ComplexFloat32,
                None,
            );
            let reflect_pad = shape(&[key.radius as usize, key.radius as usize]);
            let reflected = graph.padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
                &image_tensor,
                MPSGraphPaddingMode::Symmetric,
                &reflect_pad,
                &reflect_pad,
                0.0,
                None,
            );
            let zero_left = shape(&[0, 0]);
            let zero_right = shape(&[
                key.fft_h as usize - (key.height as usize + 2 * key.radius as usize),
                key.fft_w as usize - (key.width as usize + 2 * key.radius as usize),
            ]);
            let padded = graph.padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
                &reflected,
                MPSGraphPaddingMode::Constant,
                &zero_left,
                &zero_right,
                0.0,
                None,
            );
            let image_axes = index_array(&[0, 1]);
            let batched_axes = index_array(&[1, 2]);
            let forward_desc = fft_descriptor(false, MPSGraphFFTScalingMode::None);
            let inverse_desc = fft_descriptor(true, MPSGraphFFTScalingMode::Size);
            let spectrum = graph.realToHermiteanFFTWithTensor_axes_descriptor_name(
                &padded,
                &image_axes,
                &forward_desc,
                None,
            );
            let spectrum = graph.expandDimsOfTensor_axis_name(&spectrum, 0, None);
            let filtered = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                &spectrum,
                &kernel_tensor,
                None,
            );
            let planes = graph.HermiteanToRealFFTWithTensor_axes_descriptor_name(
                &filtered,
                &batched_axes,
                &inverse_desc,
                None,
            );
            let split = graph.splitTensor_splitSizes_axis_name(
                &planes,
                &index_array(&[1, 1]),
                0,
                None,
            );
            let gx_plane = split.objectAtIndex(0);
            let gy_plane = split.objectAtIndex(1);
            let gx_plane = graph.squeezeTensor_axis_name(&gx_plane, 0, None);
            let gy_plane = graph.squeezeTensor_axis_name(&gy_plane, 0, None);
            let crop = usize::try_from(key.radius)
                .map_err(|_| "radius is too large".to_string())?
                .saturating_mul(2);
            let gx = crop_tensor(
                &graph,
                &gx_plane,
                crop,
                crop,
                key.height as usize,
                key.width as usize,
            );
            let gy = crop_tensor(
                &graph,
                &gy_plane,
                crop,
                crop,
                key.height as usize,
                key.width as usize,
            );
            let gx_sq = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&gx, &gx, None);
            let gy_sq = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&gy, &gy, None);
            let mag_sq =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&gx_sq, &gy_sq, None);
            let magnitude = graph.squareRootWithTensor_name(&mag_sq, None);
            let zero = graph.constantWithScalar_dataType(0.0, MPSDataType::Float32);
            let pi = graph.constantWithScalar_dataType(std::f64::consts::PI, MPSDataType::Float32);
            let angle_raw = graph.atan2WithPrimaryTensor_secondaryTensor_name(&gy, &gx, None);
            let angle_plus_pi =
                graph.additionWithPrimaryTensor_secondaryTensor_name(&angle_raw, &pi, None);
            let angle_non_negative = graph
                .selectWithPredicateTensor_truePredicateTensor_falsePredicateTensor_name(
                    &graph.lessThanWithPrimaryTensor_secondaryTensor_name(&angle_raw, &zero, None),
                    &angle_plus_pi,
                    &angle_raw,
                    None,
                );
            let angle_minus_pi = graph.subtractionWithPrimaryTensor_secondaryTensor_name(
                &angle_non_negative,
                &pi,
                None,
            );
            let angle = graph
                .selectWithPredicateTensor_truePredicateTensor_falsePredicateTensor_name(
                    &graph.greaterThanOrEqualToWithPrimaryTensor_secondaryTensor_name(
                        &angle_non_negative,
                        &pi,
                        None,
                    ),
                    &angle_minus_pi,
                    &angle_non_negative,
                    None,
                );
            let feed_types = shaped_type_dictionary(&[
                (&image_tensor, &image_shape, MPSDataType::Float32),
                (&kernel_tensor, &kernel_shape, MPSDataType::ComplexFloat32),
            ]);
            let targets = NSArray::from_slice(&[&*gx, &*gy, &*magnitude, &*angle]);
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                Some(&self.graph_device),
                &feed_types,
                &targets,
                None,
                None,
            )
        };

        Ok(ImagePlan {
            executable,
            image_shape,
            kernel_shape,
            output_shape,
            kernel_row_bytes: crate::checked_len(complex_w, size_of::<[f32; 2]>(), "kernel spectrum row")?,
        })
    }
}

fn fft_descriptor(inverse: bool, scaling_mode: MPSGraphFFTScalingMode) -> Retained<MPSGraphFFTDescriptor> {
    let descriptor = unsafe { MPSGraphFFTDescriptor::new() };
    unsafe {
        descriptor.setInverse(inverse);
        descriptor.setScalingMode(scaling_mode);
        descriptor.setRoundToOddHermitean(false);
    }
    descriptor
}

fn crop_tensor(
    graph: &MPSGraph,
    tensor: &MPSGraphTensor,
    start_y: usize,
    start_x: usize,
    height: usize,
    width: usize,
) -> Retained<MPSGraphTensor> {
    unsafe {
        graph.sliceTensor_starts_ends_strides_name(
            tensor,
            &index_array(&[start_y as i64, start_x as i64]),
            &index_array(&[(start_y + height) as i64, (start_x + width) as i64]),
            &index_array(&[1, 1]),
            None,
        )
    }
}

fn shape(dims: &[usize]) -> Retained<MPSShape> {
    let values: Vec<Retained<NSNumber>> = dims.iter().map(|&dim| NSNumber::new_usize(dim)).collect();
    NSArray::from_retained_slice(&values)
}

fn index_array(values: &[i64]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = values.iter().map(|&value| NSNumber::new_i64(value)).collect();
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
