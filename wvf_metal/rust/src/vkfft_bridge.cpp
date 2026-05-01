#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "vkFFT.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr char kMultiplyShader[] = R"(
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

struct ScopedAutoreleasePool {
    NS::AutoreleasePool* pool = nullptr;

    ScopedAutoreleasePool() : pool(NS::AutoreleasePool::alloc()->init()) {}

    ~ScopedAutoreleasePool() {
        if (pool) {
            pool->release();
        }
    }
};

struct VkFFTPlanHandle {
    VkFFTApplication app = {};
    bool initialized = false;

    ~VkFFTPlanHandle() {
        if (initialized) {
            deleteVkFFT(&app);
        }
    }
};

struct PlanKey {
    uint64_t fft_w = 0;
    uint64_t fft_h = 0;
    uint64_t one_plane_bytes = 0;
    uint64_t two_plane_bytes = 0;

    bool operator==(const PlanKey& other) const {
        return fft_w == other.fft_w &&
               fft_h == other.fft_h &&
               one_plane_bytes == other.one_plane_bytes &&
               two_plane_bytes == other.two_plane_bytes;
    }
};

struct KernelKey {
    uint64_t fft_w = 0;
    uint64_t fft_h = 0;
    uint32_t radius = 0;
    uint32_t kernel_width = 0;
    uint64_t kernel_hash = 0;

    bool operator==(const KernelKey& other) const {
        return fft_w == other.fft_w &&
               fft_h == other.fft_h &&
               radius == other.radius &&
               kernel_width == other.kernel_width &&
               kernel_hash == other.kernel_hash;
    }
};

template <typename T>
void hash_combine(size_t& seed, const T& value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

struct PlanKeyHash {
    size_t operator()(const PlanKey& key) const {
        size_t seed = 0;
        hash_combine(seed, key.fft_w);
        hash_combine(seed, key.fft_h);
        hash_combine(seed, key.one_plane_bytes);
        hash_combine(seed, key.two_plane_bytes);
        return seed;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& key) const {
        size_t seed = 0;
        hash_combine(seed, key.fft_w);
        hash_combine(seed, key.fft_h);
        hash_combine(seed, key.radius);
        hash_combine(seed, key.kernel_width);
        hash_combine(seed, key.kernel_hash);
        return seed;
    }
};

struct PlanBundle {
    uint64_t real_pitch = 0;
    uint64_t real_plane_count = 0;
    uint64_t complex_count = 0;
    pfUINT one_plane_bytes = 0;
    pfUINT two_plane_bytes = 0;
    std::array<MTL::Buffer*, 1> placeholder_one = {nullptr};
    std::array<MTL::Buffer*, 1> placeholder_two = {nullptr};
    std::array<MTL::Buffer*, 1> scratch_input = {nullptr};
    std::array<MTL::Buffer*, 1> scratch_output = {nullptr};
    MTL::Buffer* shared_one = nullptr;
    MTL::Buffer* shared_two = nullptr;
    VkFFTPlanHandle input_plan;
    VkFFTPlanHandle kernel_plan;
    VkFFTPlanHandle inverse_plan;

    ~PlanBundle() {
        if (shared_two) {
            shared_two->release();
        }
        if (shared_one) {
            shared_one->release();
        }
        if (scratch_output[0]) {
            scratch_output[0]->release();
        }
        if (scratch_input[0]) {
            scratch_input[0]->release();
        }
        if (placeholder_two[0]) {
            placeholder_two[0]->release();
        }
        if (placeholder_one[0]) {
            placeholder_one[0]->release();
        }
    }
};

struct KernelSpectra {
    MTL::Buffer* buffer = nullptr;

    ~KernelSpectra() {
        if (buffer) {
            buffer->release();
        }
    }
};

struct MetalRuntime {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::ComputePipelineState* multiply_pipeline = nullptr;
    std::mutex mutex;
    std::unordered_map<PlanKey, std::unique_ptr<PlanBundle>, PlanKeyHash> plan_cache;
    std::unordered_map<KernelKey, std::unique_ptr<KernelSpectra>, KernelKeyHash> kernel_cache;

    ~MetalRuntime() {
        kernel_cache.clear();
        plan_cache.clear();
        if (multiply_pipeline) {
            multiply_pipeline->release();
        }
        if (queue) {
            queue->release();
        }
        if (device) {
            device->release();
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

bool is_smooth_fft_size(uint64_t value) {
    if (value < 2) {
        return false;
    }
    for (uint64_t factor : {2ull, 3ull, 5ull, 7ull}) {
        while ((value % factor) == 0) {
            value /= factor;
        }
    }
    return value == 1;
}

uint64_t next_smooth_fft_size(uint64_t value) {
    value = std::max<uint64_t>(value, 2);
    while (!is_smooth_fft_size(value)) {
        ++value;
    }
    return value;
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

std::string vkfft_error(const std::string& prefix, VkFFTResult result) {
    return prefix + ": " + std::string(getVkFFTErrorString(result));
}

uint64_t hash_bytes(const void* data, size_t byte_count, uint64_t seed = 14695981039346656037ull) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = seed;
    for (size_t i = 0; i < byte_count; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

MTL::ComputePipelineState* build_multiply_pipeline(MTL::Device* device, std::string* error_out) {
    NS::Error* error = nullptr;
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    if (!options) {
        if (error_out) {
            *error_out = "failed to allocate Metal compile options for VkFFT";
        }
        return nullptr;
    }
    options->setFastMathEnabled(true);
    NS::String* source = NS::String::string(kMultiplyShader, NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(source, options, &error);
    options->release();
    if (!library) {
        if (error_out) {
            *error_out = "failed to compile Metal spectrum multiply shader";
        }
        return nullptr;
    }
    NS::String* name = NS::String::string("wvf_multiply_spectra", NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(name);
    if (!function) {
        library->release();
        if (error_out) {
            *error_out = "failed to load Metal spectrum multiply function";
        }
        return nullptr;
    }
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    function->release();
    library->release();
    if (!pipeline) {
        if (error_out) {
            *error_out = "failed to create Metal spectrum multiply pipeline";
        }
        return nullptr;
    }
    return pipeline;
}

MetalRuntime* get_runtime(std::string* error_out) {
    static std::unique_ptr<MetalRuntime> runtime;
    static std::string init_error;
    static bool initialized = false;

    if (!initialized) {
        initialized = true;
        ScopedAutoreleasePool pool;
        auto created = std::make_unique<MetalRuntime>();
        created->device = MTL::CreateSystemDefaultDevice();
        if (!created->device) {
            init_error = "no Metal device is available for VkFFT";
        } else {
            created->queue = created->device->newCommandQueue();
            if (!created->queue) {
                init_error = "failed to create Metal command queue for VkFFT";
            } else {
                created->multiply_pipeline = build_multiply_pipeline(created->device, &init_error);
                if (created->multiply_pipeline) {
                    runtime = std::move(created);
                }
            }
        }
    }

    if (!runtime) {
        if (error_out) {
            *error_out = init_error.empty() ? "failed to initialize VkFFT runtime" : init_error;
        }
        return nullptr;
    }
    return runtime.get();
}

VkFFTResult append_vkfft(
    MetalRuntime& runtime,
    VkFFTApplication* app,
    int inverse,
    MTL::Buffer** buffer
) {
    VkFFTLaunchParams launch_params = {};
    MTL::CommandBuffer* command_buffer = runtime.queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    launch_params.commandBuffer = command_buffer;
    launch_params.buffer = buffer;
    launch_params.inputBuffer = buffer;
    launch_params.outputBuffer = buffer;

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
    MetalRuntime& runtime,
    const void* source,
    uint64_t byte_count,
    MTL::Buffer* staging,
    MTL::Buffer* destination
) {
    if (!staging || !destination) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    std::memcpy(staging->contents(), source, static_cast<size_t>(byte_count));
    staging->didModifyRange(NS::Range::Make(0, static_cast<NS::UInteger>(byte_count)));

    MTL::CommandBuffer* command_buffer = runtime.queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::BlitCommandEncoder* encoder = command_buffer->blitCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->copyFromBuffer(staging, 0, destination, 0, static_cast<NS::UInteger>(byte_count));
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    return VKFFT_SUCCESS;
}

VkFFTResult copy_buffer_to_host(
    MetalRuntime& runtime,
    MTL::Buffer* source,
    MTL::Buffer* staging,
    void* destination,
    uint64_t byte_count
) {
    if (!source || !staging) {
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    MTL::CommandBuffer* command_buffer = runtime.queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::BlitCommandEncoder* encoder = command_buffer->blitCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->copyFromBuffer(source, 0, staging, 0, static_cast<NS::UInteger>(byte_count));
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    std::memcpy(destination, staging->contents(), static_cast<size_t>(byte_count));
    return VKFFT_SUCCESS;
}

bool initialize_plan(
    VkFFTPlanHandle* handle,
    VkFFTConfiguration config,
    const std::string& label,
    std::string* error_out
) {
    VkFFTResult result = initializeVkFFT(&handle->app, config);
    if (result != VKFFT_SUCCESS) {
        if (error_out) {
            *error_out = vkfft_error(label, result);
        }
        return false;
    }
    handle->initialized = true;
    return true;
}

PlanBundle* get_or_create_plan_bundle(
    MetalRuntime& runtime,
    const PlanKey& key,
    uint64_t real_pitch,
    uint64_t real_plane_count,
    uint64_t complex_count,
    std::string* error_out
) {
    auto existing = runtime.plan_cache.find(key);
    if (existing != runtime.plan_cache.end()) {
        return existing->second.get();
    }

    auto plan = std::make_unique<PlanBundle>();
    plan->real_pitch = real_pitch;
    plan->real_plane_count = real_plane_count;
    plan->complex_count = complex_count;
    plan->one_plane_bytes = static_cast<pfUINT>(key.one_plane_bytes);
    plan->two_plane_bytes = static_cast<pfUINT>(key.two_plane_bytes);

    plan->placeholder_one[0] = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->one_plane_bytes),
        MTL::ResourceStorageModePrivate
    );
    plan->placeholder_two[0] = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->two_plane_bytes),
        MTL::ResourceStorageModePrivate
    );
    plan->scratch_input[0] = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->one_plane_bytes),
        MTL::ResourceStorageModePrivate
    );
    plan->scratch_output[0] = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->two_plane_bytes),
        MTL::ResourceStorageModePrivate
    );
    plan->shared_one = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->one_plane_bytes),
        MTL::ResourceStorageModeShared
    );
    plan->shared_two = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->two_plane_bytes),
        MTL::ResourceStorageModeShared
    );
    if (!plan->placeholder_one[0] || !plan->placeholder_two[0] ||
        !plan->scratch_input[0] || !plan->scratch_output[0] ||
        !plan->shared_one || !plan->shared_two) {
        if (error_out) {
            *error_out = "failed to allocate cached Metal buffers for VkFFT";
        }
        return nullptr;
    }

    VkFFTConfiguration base = {};
    base.FFTdim = 2;
    base.size[0] = key.fft_w;
    base.size[1] = key.fft_h;
    base.size[2] = 1;
    base.performR2C = true;
    base.disableMergeSequencesR2C = 1;
    base.normalize = 1;
    base.coordinateFeatures = 1;
    base.device = runtime.device;
    base.queue = runtime.queue;

    VkFFTConfiguration kernel_config = base;
    kernel_config.numberBatches = 2;
    kernel_config.makeForwardPlanOnly = 1;
    kernel_config.buffer = plan->placeholder_two.data();
    kernel_config.bufferSize = &plan->two_plane_bytes;
    if (!initialize_plan(&plan->kernel_plan, kernel_config, "failed to initialize VkFFT kernel plan", error_out)) {
        return nullptr;
    }

    VkFFTConfiguration input_config = base;
    input_config.numberBatches = 1;
    input_config.makeForwardPlanOnly = 1;
    input_config.buffer = plan->placeholder_one.data();
    input_config.bufferSize = &plan->one_plane_bytes;
    if (!initialize_plan(&plan->input_plan, input_config, "failed to initialize VkFFT image plan", error_out)) {
        return nullptr;
    }

    VkFFTConfiguration inverse_config = base;
    inverse_config.numberBatches = 2;
    inverse_config.makeInversePlanOnly = 1;
    inverse_config.buffer = plan->placeholder_two.data();
    inverse_config.bufferSize = &plan->two_plane_bytes;
    if (!initialize_plan(&plan->inverse_plan, inverse_config, "failed to initialize VkFFT inverse plan", error_out)) {
        return nullptr;
    }

    auto inserted = runtime.plan_cache.emplace(key, std::move(plan));
    return inserted.first->second.get();
}

KernelSpectra* get_or_create_kernel_spectra(
    MetalRuntime& runtime,
    PlanBundle& plan,
    const PlanKey& plan_key,
    uint32_t radius,
    const float* kernel_x,
    const float* kernel_y,
    uint32_t kernel_width,
    std::string* error_out
) {
    const size_t kernel_elements =
        static_cast<size_t>(kernel_width) * static_cast<size_t>(kernel_width);
    uint64_t kernel_hash = hash_bytes(kernel_x, kernel_elements * sizeof(float));
    kernel_hash = hash_bytes(kernel_y, kernel_elements * sizeof(float), kernel_hash);
    const KernelKey key = {
        plan_key.fft_w,
        plan_key.fft_h,
        radius,
        kernel_width,
        kernel_hash,
    };
    auto existing = runtime.kernel_cache.find(key);
    if (existing != runtime.kernel_cache.end()) {
        return existing->second.get();
    }

    std::vector<float> kernel_data(static_cast<size_t>(2ull * plan.real_plane_count), 0.0f);
    for (uint64_t y = 0; y < kernel_width; ++y) {
        for (uint64_t x = 0; x < kernel_width; ++x) {
            const uint64_t src = y * kernel_width + x;
            const uint64_t dst = y * plan.real_pitch + x;
            kernel_data[static_cast<size_t>(dst)] = kernel_x[src];
            kernel_data[static_cast<size_t>(plan.real_plane_count + dst)] = kernel_y[src];
        }
    }

    auto spectra = std::make_unique<KernelSpectra>();
    spectra->buffer = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan.two_plane_bytes),
        MTL::ResourceStorageModePrivate
    );
    if (!spectra->buffer) {
        if (error_out) {
            *error_out = "failed to allocate cached VkFFT kernel spectrum buffer";
        }
        return nullptr;
    }

    VkFFTResult result = copy_host_to_buffer(
        runtime,
        kernel_data.data(),
        plan.two_plane_bytes,
        plan.shared_two,
        spectra->buffer
    );
    if (result != VKFFT_SUCCESS) {
        if (error_out) {
            *error_out = vkfft_error("failed to upload WVF kernels to VkFFT", result);
        }
        return nullptr;
    }

    MTL::Buffer* kernel_buffer_array[1] = {spectra->buffer};
    result = append_vkfft(runtime, &plan.kernel_plan.app, -1, kernel_buffer_array);
    if (result != VKFFT_SUCCESS) {
        if (error_out) {
            *error_out = vkfft_error("failed to transform WVF kernels with VkFFT", result);
        }
        return nullptr;
    }

    auto inserted = runtime.kernel_cache.emplace(key, std::move(spectra));
    return inserted.first->second.get();
}

VkFFTResult multiply_spectra(
    MetalRuntime& runtime,
    MTL::Buffer* input,
    MTL::Buffer* kernels,
    MTL::Buffer* output,
    uint64_t complex_count
) {
    if (complex_count > std::numeric_limits<uint32_t>::max() / 2u) {
        return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    }

    uint32_t n_complex = static_cast<uint32_t>(complex_count);
    MTL::CommandBuffer* command_buffer = runtime.queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }

    encoder->setComputePipelineState(runtime.multiply_pipeline);
    encoder->setBuffer(input, 0, 0);
    encoder->setBuffer(kernels, 0, 1);
    encoder->setBuffer(output, 0, 2);
    encoder->setBytes(&n_complex, sizeof(n_complex), 3);
    const NS::UInteger total = static_cast<NS::UInteger>(complex_count * 2ull);
    const NS::UInteger group_width = std::min<NS::UInteger>(
        runtime.multiply_pipeline->maxTotalThreadsPerThreadgroup(),
        std::max<NS::UInteger>(runtime.multiply_pipeline->threadExecutionWidth(), 1)
    );
    encoder->dispatchThreads(
        MTL::Size::Make(total, 1, 1),
        MTL::Size::Make(group_width, 1, 1)
    );
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
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
    const uint64_t fft_w = next_smooth_fft_size(min_fft_w);
    const uint64_t fft_h = next_smooth_fft_size(min_fft_h);
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
    const PlanKey plan_key = {
        fft_w,
        fft_h,
        one_plane_bytes,
        two_plane_bytes,
    };

    ScopedAutoreleasePool pool;
    std::string runtime_error;
    MetalRuntime* runtime = get_runtime(&runtime_error);
    if (!runtime) {
        write_error(error_out, error_len, runtime_error);
        return 1;
    }
    std::lock_guard<std::mutex> lock(runtime->mutex);

    std::string cache_error;
    PlanBundle* plan = get_or_create_plan_bundle(
        *runtime,
        plan_key,
        real_pitch,
        real_plane_count,
        complex_count,
        &cache_error
    );
    if (!plan) {
        write_error(error_out, error_len, cache_error);
        return 1;
    }

    KernelSpectra* kernel_spectra = get_or_create_kernel_spectra(
        *runtime,
        *plan,
        plan_key,
        radius,
        kernel_x,
        kernel_y,
        kernel_width,
        &cache_error
    );
    if (!kernel_spectra) {
        write_error(error_out, error_len, cache_error);
        return 1;
    }

    std::vector<float> input_data(static_cast<size_t>(plan->real_plane_count), 0.0f);
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
            input_data[static_cast<size_t>(y * plan->real_pitch + x)] =
                image[static_cast<uint64_t>(src_y) * width + static_cast<uint64_t>(src_x)];
        }
    }

    VkFFTResult result = copy_host_to_buffer(
        *runtime,
        input_data.data(),
        one_plane_bytes,
        plan->shared_one,
        plan->scratch_input[0]
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to upload WVF image to VkFFT", result));
        return static_cast<int>(result);
    }

    result = append_vkfft(*runtime, &plan->input_plan.app, -1, plan->scratch_input.data());
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to transform WVF image with VkFFT", result));
        return static_cast<int>(result);
    }

    result = multiply_spectra(
        *runtime,
        plan->scratch_input[0],
        kernel_spectra->buffer,
        plan->scratch_output[0],
        plan->complex_count
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to multiply WVF spectra", result));
        return static_cast<int>(result);
    }

    result = append_vkfft(*runtime, &plan->inverse_plan.app, 1, plan->scratch_output.data());
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to invert WVF spectra with VkFFT", result));
        return static_cast<int>(result);
    }

    std::vector<float> output_data(static_cast<size_t>(2ull * plan->real_plane_count), 0.0f);
    result = copy_buffer_to_host(
        *runtime,
        plan->scratch_output[0],
        plan->shared_two,
        output_data.data(),
        two_plane_bytes
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to download WVF output from VkFFT", result));
        return static_cast<int>(result);
    }

    const uint64_t crop = 2ull * radius;
    for (uint64_t y = 0; y < height; ++y) {
        const uint64_t src_row = (y + crop) * plan->real_pitch;
        const uint64_t dst_row = y * width;
        for (uint64_t x = 0; x < width; ++x) {
            const uint64_t src_idx = src_row + x + crop;
            const uint64_t dst_idx = dst_row + x;
            const float gx = output_data[static_cast<size_t>(src_idx)];
            const float gy = output_data[static_cast<size_t>(plan->real_plane_count + src_idx)];
            out_x[dst_idx] = gx;
            out_y[dst_idx] = gy;
            magnitude[dst_idx] = std::sqrt(gx * gx + gy * gy);
            angle[dst_idx] = unsigned_angle(gy, gx);
        }
    }

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
