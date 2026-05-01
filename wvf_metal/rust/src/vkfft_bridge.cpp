#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "vkFFT.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct MetalContext {
    NS::AutoreleasePool* pool = nullptr;
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;

    ~MetalContext() {
        if (queue) {
            queue->release();
        }
        if (device) {
            device->release();
        }
        if (pool) {
            pool->release();
        }
    }
};

void write_error(char* error_out, size_t error_len, const std::string& message) {
    if (!error_out || error_len == 0) {
        return;
    }
    const size_t copy_len = std::min(error_len - 1, message.size());
    std::memcpy(error_out, message.data(), copy_len);
    error_out[copy_len] = '\0';
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

uint64_t next_power_of_two(uint64_t value) {
    if (value <= 2) {
        return 2;
    }
    --value;
    for (uint64_t shift = 1; shift < 64; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

int reflect_index(int64_t value, int64_t limit) {
    if (limit <= 1) {
        return 0;
    }
    const int64_t period = 2 * limit;
    value %= period;
    if (value < 0) {
        value += period;
    }
    if (value >= limit) {
        value = period - value - 1;
    }
    return static_cast<int>(value);
}

float unsigned_angle(float y, float x) {
    float theta = std::atan2(y, x);
    if (theta < 0.0f) {
        theta += kPi;
    }
    if (theta >= kPi) {
        theta -= kPi;
    }
    return theta;
}

VkFFTResult append_vkfft(MTL::CommandQueue* queue, VkFFTApplication* app, int inverse) {
    VkFFTLaunchParams launch_params = {};
    MTL::CommandBuffer* command_buffer = queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    launch_params.commandBuffer = command_buffer;

    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    launch_params.commandEncoder = encoder;

    VkFFTResult result = VkFFTAppend(app, inverse, &launch_params);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    return result;
}

VkFFTResult copy_host_to_buffer(
    MetalContext& metal,
    const void* source,
    uint64_t byte_count,
    MTL::Buffer* destination
) {
    MTL::Buffer* staging = metal.device->newBuffer(
        source,
        static_cast<NS::UInteger>(byte_count),
        MTL::ResourceStorageModeShared
    );
    if (!staging) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    MTL::CommandBuffer* command_buffer = metal.queue->commandBuffer();
    if (!command_buffer) {
        staging->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::BlitCommandEncoder* encoder = command_buffer->blitCommandEncoder();
    if (!encoder) {
        staging->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->copyFromBuffer(staging, 0, destination, 0, static_cast<NS::UInteger>(byte_count));
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    staging->release();
    return VKFFT_SUCCESS;
}

VkFFTResult copy_buffer_to_host(
    MetalContext& metal,
    MTL::Buffer* source,
    void* destination,
    uint64_t byte_count
) {
    MTL::Buffer* staging = metal.device->newBuffer(
        static_cast<NS::UInteger>(byte_count),
        MTL::ResourceStorageModeShared
    );
    if (!staging) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    MTL::CommandBuffer* command_buffer = metal.queue->commandBuffer();
    if (!command_buffer) {
        staging->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::BlitCommandEncoder* encoder = command_buffer->blitCommandEncoder();
    if (!encoder) {
        staging->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->copyFromBuffer(source, 0, staging, 0, static_cast<NS::UInteger>(byte_count));
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    std::memcpy(destination, staging->contents(), static_cast<size_t>(byte_count));
    staging->release();
    return VKFFT_SUCCESS;
}

VkFFTResult multiply_spectra(
    MetalContext& metal,
    MTL::Buffer* input,
    MTL::Buffer* kernels,
    MTL::Buffer* output,
    uint64_t complex_count
) {
    static constexpr const char* shader = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void wvf_multiply_spectra(
            device const float2* input [[buffer(0)]],
            device const float2* kernels [[buffer(1)]],
            device float2* output [[buffer(2)]],
            constant uint& n_complex [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint total = n_complex * 2u;
            if (id >= total) {
                return;
            }
            uint plane = id / n_complex;
            uint idx = id - plane * n_complex;
            float2 a = input[idx];
            float2 b = kernels[id];
            output[id] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
    )";

    if (complex_count > std::numeric_limits<uint32_t>::max() / 2u) {
        return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    }

    NS::Error* error = nullptr;
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    if (!options) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    options->setFastMathEnabled(true);
    NS::String* source = NS::String::string(shader, NS::UTF8StringEncoding);
    MTL::Library* library = metal.device->newLibrary(source, options, &error);
    options->release();
    if (!library) {
        return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
    }
    NS::String* name = NS::String::string("wvf_multiply_spectra", NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(name);
    if (!function) {
        library->release();
        return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
    }
    MTL::ComputePipelineState* pipeline = metal.device->newComputePipelineState(function, &error);
    function->release();
    library->release();
    if (!pipeline) {
        return VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
    }

    uint32_t n_complex = static_cast<uint32_t>(complex_count);
    MTL::Buffer* params = metal.device->newBuffer(
        &n_complex,
        sizeof(n_complex),
        MTL::ResourceStorageModeShared
    );
    if (!params) {
        pipeline->release();
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }

    MTL::CommandBuffer* command_buffer = metal.queue->commandBuffer();
    if (!command_buffer) {
        params->release();
        pipeline->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        params->release();
        pipeline->release();
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }

    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input, 0, 0);
    encoder->setBuffer(kernels, 0, 1);
    encoder->setBuffer(output, 0, 2);
    encoder->setBuffer(params, 0, 3);
    const NS::UInteger total = static_cast<NS::UInteger>(complex_count * 2ull);
    const NS::UInteger group_width = std::min<NS::UInteger>(
        pipeline->maxTotalThreadsPerThreadgroup(),
        std::max<NS::UInteger>(pipeline->threadExecutionWidth(), 1)
    );
    encoder->dispatchThreads(
        MTL::Size::Make(total, 1, 1),
        MTL::Size::Make(group_width, 1, 1)
    );
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    params->release();
    pipeline->release();
    return VKFFT_SUCCESS;
}

int run_wvf_vkfft(
    const float* image,
    uint32_t width,
    uint32_t height,
    uint32_t radius,
    const float* kernel_x,
    const float* kernel_y,
    uint32_t kernel_width,
    float* out_x,
    float* out_y,
    float* magnitude,
    float* angle,
    char* error_out,
    size_t error_len
) {
    if (!image || !kernel_x || !kernel_y || !out_x || !out_y || !magnitude || !angle) {
        write_error(error_out, error_len, "null pointer passed to VkFFT WVF backend");
        return 1;
    }
    if (width == 0 || height == 0 || radius == 0 || kernel_width != 2 * radius + 1) {
        write_error(error_out, error_len, "invalid dimensions for VkFFT WVF backend");
        return 1;
    }

    const uint64_t min_fft_w = static_cast<uint64_t>(width) + 4ull * radius;
    const uint64_t min_fft_h = static_cast<uint64_t>(height) + 4ull * radius;
    const uint64_t fft_w = next_power_of_two(min_fft_w);
    const uint64_t fft_h = next_power_of_two(min_fft_h);
    const uint64_t padded_w = static_cast<uint64_t>(width) + 2ull * radius;
    const uint64_t padded_h = static_cast<uint64_t>(height) + 2ull * radius;
    const uint64_t real_pitch = 2ull * (fft_w / 2ull + 1ull);

    uint64_t real_plane_count = 0;
    uint64_t complex_count = 0;
    if (!checked_mul(real_pitch, fft_h, &real_plane_count) ||
        !checked_mul(fft_w / 2ull + 1ull, fft_h, &complex_count)) {
        write_error(error_out, error_len, "VkFFT WVF buffer dimensions overflowed");
        return 1;
    }

    const uint64_t one_plane_bytes = real_plane_count * sizeof(float);
    const uint64_t two_plane_bytes = 2ull * real_plane_count * sizeof(float);
    uint64_t input_buffer_size = one_plane_bytes;
    uint64_t two_plane_buffer_size = two_plane_bytes;

    MetalContext metal;
    metal.pool = NS::AutoreleasePool::alloc()->init();
    metal.device = MTL::CreateSystemDefaultDevice();
    if (!metal.device) {
        write_error(error_out, error_len, "no Metal device is available for VkFFT");
        return 1;
    }
    metal.queue = metal.device->newCommandQueue();
    if (!metal.queue) {
        write_error(error_out, error_len, "failed to create Metal command queue for VkFFT");
        return 1;
    }

    std::vector<float> input_data(static_cast<size_t>(real_plane_count), 0.0f);
    std::vector<float> kernel_data(static_cast<size_t>(2ull * real_plane_count), 0.0f);

    const volatile uint64_t padded_h_limit = padded_h;
    const volatile uint64_t padded_w_limit = padded_w;
    for (uint64_t y = 0; y < padded_h_limit; ++y) {
        const int src_y = reflect_index(
            static_cast<int64_t>(y) - static_cast<int64_t>(radius),
            static_cast<int64_t>(height)
        );
        for (uint64_t x = 0; x < padded_w_limit; ++x) {
            const int src_x = reflect_index(
                static_cast<int64_t>(x) - static_cast<int64_t>(radius),
                static_cast<int64_t>(width)
            );
            input_data[static_cast<size_t>(y * real_pitch + x)] =
                image[static_cast<uint64_t>(src_y) * width + static_cast<uint64_t>(src_x)];
        }
    }

    for (uint64_t y = 0; y < kernel_width; ++y) {
        for (uint64_t x = 0; x < kernel_width; ++x) {
            const uint64_t src = y * kernel_width + x;
            const uint64_t dst = y * real_pitch + x;
            kernel_data[static_cast<size_t>(dst)] = kernel_x[src];
            kernel_data[static_cast<size_t>(real_plane_count + dst)] = kernel_y[src];
        }
    }

    MTL::Buffer* input_buffer = metal.device->newBuffer(one_plane_bytes, MTL::ResourceStorageModePrivate);
    MTL::Buffer* kernel_buffer = metal.device->newBuffer(two_plane_bytes, MTL::ResourceStorageModePrivate);
    MTL::Buffer* output_buffer = metal.device->newBuffer(two_plane_bytes, MTL::ResourceStorageModePrivate);
    if (!input_buffer || !kernel_buffer || !output_buffer) {
        if (input_buffer) input_buffer->release();
        if (kernel_buffer) kernel_buffer->release();
        if (output_buffer) output_buffer->release();
        write_error(error_out, error_len, "failed to allocate Metal buffers for VkFFT");
        return 1;
    }

    VkFFTResult result = copy_host_to_buffer(
        metal,
        input_data.data(),
        one_plane_bytes,
        input_buffer
    );
    if (result != VKFFT_SUCCESS) {
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to upload WVF image to VkFFT");
        return static_cast<int>(result);
    }
    result = copy_host_to_buffer(
        metal,
        kernel_data.data(),
        two_plane_bytes,
        kernel_buffer
    );
    if (result != VKFFT_SUCCESS) {
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to upload WVF kernels to VkFFT");
        return static_cast<int>(result);
    }

    VkFFTConfiguration kernel_config = {};
    kernel_config.FFTdim = 2;
    kernel_config.size[0] = fft_w;
    kernel_config.size[1] = fft_h;
    kernel_config.size[2] = 1;
    kernel_config.performR2C = true;
    kernel_config.disableMergeSequencesR2C = 1;
    kernel_config.normalize = 1;
    kernel_config.coordinateFeatures = 1;
    kernel_config.numberBatches = 2;
    kernel_config.makeForwardPlanOnly = 1;
    kernel_config.device = metal.device;
    kernel_config.queue = metal.queue;
    kernel_config.buffer = &kernel_buffer;
    kernel_config.bufferSize = &two_plane_buffer_size;

    VkFFTApplication kernel_app = {};
    result = initializeVkFFT(&kernel_app, kernel_config);
    if (result != VKFFT_SUCCESS) {
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to initialize VkFFT kernel plan");
        return static_cast<int>(result);
    }
    result = append_vkfft(metal.queue, &kernel_app, -1);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to transform WVF kernels with VkFFT");
        return static_cast<int>(result);
    }

    VkFFTConfiguration input_config = kernel_config;
    input_config.numberBatches = 1;
    input_config.buffer = &input_buffer;
    input_config.bufferSize = &input_buffer_size;

    VkFFTApplication input_app = {};
    result = initializeVkFFT(&input_app, input_config);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to initialize VkFFT image plan");
        return static_cast<int>(result);
    }
    result = append_vkfft(metal.queue, &input_app, -1);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&input_app);
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to transform WVF image with VkFFT");
        return static_cast<int>(result);
    }

    result = multiply_spectra(metal, input_buffer, kernel_buffer, output_buffer, complex_count);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&input_app);
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to multiply WVF spectra");
        return static_cast<int>(result);
    }

    VkFFTConfiguration inverse_config = kernel_config;
    inverse_config.makeForwardPlanOnly = 0;
    inverse_config.makeInversePlanOnly = 1;
    inverse_config.buffer = &output_buffer;
    inverse_config.bufferSize = &two_plane_buffer_size;

    VkFFTApplication inverse_app = {};
    result = initializeVkFFT(&inverse_app, inverse_config);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&input_app);
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to initialize VkFFT inverse plan");
        return static_cast<int>(result);
    }
    result = append_vkfft(metal.queue, &inverse_app, 1);
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&inverse_app);
        deleteVkFFT(&input_app);
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to invert WVF spectra with VkFFT");
        return static_cast<int>(result);
    }

    std::vector<float> output_data(static_cast<size_t>(2ull * real_plane_count), 0.0f);
    result = copy_buffer_to_host(
        metal,
        output_buffer,
        output_data.data(),
        two_plane_bytes
    );
    if (result != VKFFT_SUCCESS) {
        deleteVkFFT(&inverse_app);
        deleteVkFFT(&input_app);
        deleteVkFFT(&kernel_app);
        input_buffer->release();
        kernel_buffer->release();
        output_buffer->release();
        write_error(error_out, error_len, "failed to download WVF output from VkFFT");
        return static_cast<int>(result);
    }
    const uint64_t crop = 2ull * radius;
    for (uint64_t y = 0; y < height; ++y) {
        const uint64_t src_row = (y + crop) * real_pitch;
        const uint64_t dst_row = y * width;
        for (uint64_t x = 0; x < width; ++x) {
            const uint64_t src_idx = src_row + x + crop;
            const uint64_t dst_idx = dst_row + x;
            const float gx = output_data[static_cast<size_t>(src_idx)];
            const float gy = output_data[static_cast<size_t>(real_plane_count + src_idx)];
            out_x[dst_idx] = gx;
            out_y[dst_idx] = gy;
            magnitude[dst_idx] = std::sqrt(gx * gx + gy * gy);
            angle[dst_idx] = unsigned_angle(gy, gx);
        }
    }

    deleteVkFFT(&inverse_app);
    deleteVkFFT(&input_app);
    deleteVkFFT(&kernel_app);
    input_buffer->release();
    kernel_buffer->release();
    output_buffer->release();
    return 0;
}

} // namespace

extern "C" int wvf_vkfft_magnitude_angle(
    const float* image,
    uint32_t width,
    uint32_t height,
    uint32_t radius,
    const float* kernel_x,
    const float* kernel_y,
    uint32_t kernel_width,
    float* out_x,
    float* out_y,
    float* magnitude,
    float* angle,
    char* error_out,
    size_t error_len
) {
    return run_wvf_vkfft(
        image,
        width,
        height,
        radius,
        kernel_x,
        kernel_y,
        kernel_width,
        out_x,
        out_y,
        magnitude,
        angle,
        error_out,
        error_len
    );
}
