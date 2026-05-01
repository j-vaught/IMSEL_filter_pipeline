#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "vkFFT.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
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
constexpr char kUtilityShaders[] = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct WVFPadParams {
        uint image_width;
        uint image_height;
        uint padded_width;
        uint padded_height;
        uint real_pitch;
        uint fft_height;
        uint radius;
    };

    struct WVFPostprocessParams {
        uint width;
        uint height;
        uint crop;
        uint real_pitch;
        uint real_plane_count;
    };

    inline int wvf_reflect_index(int value, int limit) {
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

    inline float wvf_unsigned_angle(float y, float x) {
        float theta = atan2(y, x);
        if (theta < 0.0f) {
            theta += M_PI_F;
        }
        if (theta >= M_PI_F) {
            theta -= M_PI_F;
        }
        return theta;
    }

    kernel void wvf_reflect_pad_real(
        device const float* image [[buffer(0)]],
        device float* padded [[buffer(1)]],
        constant WVFPadParams& params [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= params.real_pitch || gid.y >= params.fft_height) {
            return;
        }

        const uint dst_idx = gid.y * params.real_pitch + gid.x;
        if (gid.x >= params.padded_width || gid.y >= params.padded_height) {
            padded[dst_idx] = 0.0f;
            return;
        }

        const int src_x =
            wvf_reflect_index(int(gid.x) - int(params.radius), int(params.image_width));
        const int src_y =
            wvf_reflect_index(int(gid.y) - int(params.radius), int(params.image_height));
        padded[dst_idx] = image[uint(src_y) * params.image_width + uint(src_x)];
    }

    kernel void wvf_multiply_spectra(
        device const float2* input [[buffer(0)]],
        device const float2* kernels [[buffer(1)]],
        device float2* output [[buffer(2)]],
        constant uint& n_complex [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint total = n_complex * 2u;
        if (id >= total) {
            return;
        }
        const uint plane = id / n_complex;
        const uint idx = id - plane * n_complex;
        const float2 a = input[idx];
        const float2 b = kernels[id];
        output[id] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    kernel void wvf_fft_postprocess(
        device const float* planes [[buffer(0)]],
        device float* out_x [[buffer(1)]],
        device float* out_y [[buffer(2)]],
        device float* magnitude [[buffer(3)]],
        device float* angle [[buffer(4)]],
        constant WVFPostprocessParams& params [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= params.width || gid.y >= params.height) {
            return;
        }

        const uint out_idx = gid.y * params.width + gid.x;
        const uint src_idx =
            (gid.y + params.crop) * params.real_pitch + gid.x + params.crop;
        const float gx = planes[src_idx];
        const float gy = planes[params.real_plane_count + src_idx];
        out_x[out_idx] = gx;
        out_y[out_idx] = gy;
        magnitude[out_idx] = sqrt(gx * gx + gy * gy);
        angle[out_idx] = wvf_unsigned_angle(gy, gx);
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
    MTL::Buffer* shared_two = nullptr;
    VkFFTPlanHandle input_plan;
    VkFFTPlanHandle kernel_plan;
    VkFFTPlanHandle inverse_plan;

    ~PlanBundle() {
        if (shared_two) {
            shared_two->release();
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
    MTL::ComputePipelineState* pad_pipeline = nullptr;
    MTL::ComputePipelineState* multiply_pipeline = nullptr;
    MTL::ComputePipelineState* postprocess_pipeline = nullptr;
    std::mutex mutex;
    std::unordered_map<PlanKey, std::unique_ptr<PlanBundle>, PlanKeyHash> plan_cache;
    std::unordered_map<KernelKey, std::unique_ptr<KernelSpectra>, KernelKeyHash> kernel_cache;

    ~MetalRuntime() {
        kernel_cache.clear();
        plan_cache.clear();
        if (postprocess_pipeline) {
            postprocess_pipeline->release();
        }
        if (multiply_pipeline) {
            multiply_pipeline->release();
        }
        if (pad_pipeline) {
            pad_pipeline->release();
        }
        if (queue) {
            queue->release();
        }
        if (device) {
            device->release();
        }
    }
};

struct ExternalBuffers {
    MTL::Buffer* image = nullptr;
    MTL::Buffer* out_x = nullptr;
    MTL::Buffer* out_y = nullptr;
    MTL::Buffer* magnitude = nullptr;
    MTL::Buffer* angle = nullptr;

    ~ExternalBuffers() {
        if (angle) {
            angle->release();
        }
        if (magnitude) {
            magnitude->release();
        }
        if (out_y) {
            out_y->release();
        }
        if (out_x) {
            out_x->release();
        }
        if (image) {
            image->release();
        }
    }
};

struct WVFPadParams {
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t padded_width = 0;
    uint32_t padded_height = 0;
    uint32_t real_pitch = 0;
    uint32_t fft_height = 0;
    uint32_t radius = 0;
};

struct WVFPostprocessParams {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t crop = 0;
    uint32_t real_pitch = 0;
    uint32_t real_plane_count = 0;
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

MTL::ComputePipelineState* build_compute_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const char* function_name,
    std::string* error_out
) {
    NS::Error* error = nullptr;
    NS::String* name = NS::String::string(function_name, NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(name);
    if (!function) {
        if (error_out) {
            *error_out = std::string("failed to load Metal function ") + function_name;
        }
        return nullptr;
    }
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (!pipeline) {
        if (error_out) {
            *error_out = std::string("failed to create Metal pipeline ") + function_name;
        }
        return nullptr;
    }
    return pipeline;
}

bool initialize_runtime_pipelines(MetalRuntime* runtime, std::string* error_out) {
    NS::Error* error = nullptr;
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    if (!options) {
        if (error_out) {
            *error_out = "failed to allocate Metal compile options for VkFFT";
        }
        return false;
    }
    options->setFastMathEnabled(true);
    NS::String* source = NS::String::string(kUtilityShaders, NS::UTF8StringEncoding);
    MTL::Library* library = runtime->device->newLibrary(source, options, &error);
    options->release();
    if (!library) {
        if (error_out) {
            *error_out = "failed to compile Metal VkFFT utility shaders";
        }
        return false;
    }

    runtime->pad_pipeline =
        build_compute_pipeline(runtime->device, library, "wvf_reflect_pad_real", error_out);
    runtime->multiply_pipeline =
        runtime->pad_pipeline
            ? build_compute_pipeline(runtime->device, library, "wvf_multiply_spectra", error_out)
            : nullptr;
    runtime->postprocess_pipeline =
        runtime->multiply_pipeline
            ? build_compute_pipeline(runtime->device, library, "wvf_fft_postprocess", error_out)
            : nullptr;
    library->release();
    return runtime->pad_pipeline && runtime->multiply_pipeline && runtime->postprocess_pipeline;
}

int selected_device_index(std::string* error_out) {
    const char* value = std::getenv("WVF_METAL_DEVICE_INDEX");
    if (!value || !value[0]) {
        return -1;
    }

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || (end && *end != '\0') || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        if (error_out) {
            *error_out = "WVF_METAL_DEVICE_INDEX must be a non-negative integer";
        }
        return -2;
    }
    return static_cast<int>(parsed);
}

MTL::Device* create_selected_device(int device_index, std::string* error_out) {
    if (device_index < 0) {
        return MTL::CreateSystemDefaultDevice();
    }

    NS::Array* devices = MTL::CopyAllDevices();
    if (!devices) {
        if (error_out) {
            *error_out = "failed to enumerate Metal devices for VkFFT";
        }
        return nullptr;
    }

    const int count = static_cast<int>(devices->count());
    if (device_index >= count) {
        if (error_out) {
            *error_out =
                "WVF_METAL_DEVICE_INDEX " + std::to_string(device_index) +
                " is out of range for " + std::to_string(count) + " Metal device(s)";
        }
        devices->release();
        return nullptr;
    }

    auto* device = static_cast<MTL::Device*>(
        devices->object(static_cast<NS::UInteger>(device_index))
    );
    if (device) {
        device->retain();
    }
    devices->release();
    return device;
}

MetalRuntime* get_runtime(std::string* error_out) {
    static std::mutex runtime_mutex;
    static std::unordered_map<int, std::unique_ptr<MetalRuntime>> runtimes;
    static std::unordered_map<int, std::string> init_errors;

    std::string selection_error;
    const int device_index = selected_device_index(&selection_error);
    if (device_index == -2) {
        if (error_out) {
            *error_out = selection_error;
        }
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (runtimes.find(device_index) == runtimes.end() &&
        init_errors.find(device_index) == init_errors.end()) {
        ScopedAutoreleasePool pool;
        std::string init_error;
        auto created = std::make_unique<MetalRuntime>();
        created->device = create_selected_device(device_index, &init_error);
        if (!created->device) {
            if (init_error.empty()) {
                init_error = "no Metal device is available for VkFFT";
            }
        } else {
            created->queue = created->device->newCommandQueue();
            if (!created->queue) {
                init_error = "failed to create Metal command queue for VkFFT";
            } else if (initialize_runtime_pipelines(created.get(), &init_error)) {
                runtimes.emplace(device_index, std::move(created));
            }
        }
        if (runtimes.find(device_index) == runtimes.end()) {
            init_errors.emplace(device_index, init_error);
        }
    }

    auto runtime_it = runtimes.find(device_index);
    if (runtime_it == runtimes.end()) {
        if (error_out) {
            const auto error_it = init_errors.find(device_index);
            if (error_it == init_errors.end() || error_it->second.empty()) {
                *error_out = "failed to initialize VkFFT runtime";
            } else {
                *error_out = error_it->second;
            }
        }
        return nullptr;
    }
    return runtime_it->second.get();
}

MTL::Size threadgroup_1d(MTL::ComputePipelineState* pipeline) {
    const NS::UInteger width = std::min<NS::UInteger>(
        pipeline->maxTotalThreadsPerThreadgroup(),
        std::max<NS::UInteger>(pipeline->threadExecutionWidth(), 1)
    );
    return MTL::Size::Make(width, 1, 1);
}

MTL::Size threadgroup_2d(MTL::ComputePipelineState* pipeline) {
    const NS::UInteger width = std::min<NS::UInteger>(
        pipeline->maxTotalThreadsPerThreadgroup(),
        std::max<NS::UInteger>(pipeline->threadExecutionWidth(), 1)
    );
    const NS::UInteger height =
        std::max<NS::UInteger>(1, std::min<NS::UInteger>(16, pipeline->maxTotalThreadsPerThreadgroup() / width));
    return MTL::Size::Make(width, height, 1);
}

VkFFTResult append_vkfft(
    MTL::CommandBuffer* command_buffer,
    VkFFTApplication* app,
    int inverse,
    MTL::Buffer** buffer
) {
    VkFFTLaunchParams launch_params = {};
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
    return result;
}

VkFFTResult run_vkfft_once(
    MetalRuntime& runtime,
    VkFFTApplication* app,
    int inverse,
    MTL::Buffer** buffer
) {
    MTL::CommandBuffer* command_buffer = runtime.queue->commandBuffer();
    if (!command_buffer) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    const VkFFTResult result = append_vkfft(command_buffer, app, inverse, buffer);
    if (result != VKFFT_SUCCESS) {
        return result;
    }
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    return VKFFT_SUCCESS;
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

bool initialize_plan(
    VkFFTPlanHandle* handle,
    VkFFTConfiguration config,
    const std::string& label,
    std::string* error_out
) {
    const VkFFTResult result = initializeVkFFT(&handle->app, config);
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
    plan->shared_two = runtime.device->newBuffer(
        static_cast<NS::UInteger>(plan->two_plane_bytes),
        MTL::ResourceStorageModeShared
    );
    if (!plan->placeholder_one[0] || !plan->placeholder_two[0] ||
        !plan->scratch_input[0] || !plan->scratch_output[0] ||
        !plan->shared_two) {
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
    result = run_vkfft_once(runtime, &plan.kernel_plan.app, -1, kernel_buffer_array);
    if (result != VKFFT_SUCCESS) {
        if (error_out) {
            *error_out = vkfft_error("failed to transform WVF kernels with VkFFT", result);
        }
        return nullptr;
    }

    auto inserted = runtime.kernel_cache.emplace(key, std::move(spectra));
    return inserted.first->second.get();
}

ExternalBuffers wrap_external_buffers(
    MetalRuntime& runtime,
    const float* image,
    uint64_t image_bytes,
    float* out_x,
    float* out_y,
    float* magnitude,
    float* angle,
    uint64_t output_bytes
) {
    ExternalBuffers buffers;
    const auto shared = MTL::ResourceStorageModeShared;
    buffers.image =
        runtime.device->newBuffer(image, static_cast<NS::UInteger>(image_bytes), shared, nullptr);
    buffers.out_x = runtime.device->newBuffer(
        static_cast<const void*>(out_x),
        static_cast<NS::UInteger>(output_bytes),
        shared,
        nullptr
    );
    buffers.out_y = runtime.device->newBuffer(
        static_cast<const void*>(out_y),
        static_cast<NS::UInteger>(output_bytes),
        shared,
        nullptr
    );
    buffers.magnitude = runtime.device->newBuffer(
        static_cast<const void*>(magnitude),
        static_cast<NS::UInteger>(output_bytes),
        shared,
        nullptr
    );
    buffers.angle = runtime.device->newBuffer(
        static_cast<const void*>(angle),
        static_cast<NS::UInteger>(output_bytes),
        shared,
        nullptr
    );
    if (buffers.image) {
        buffers.image->didModifyRange(NS::Range::Make(0, static_cast<NS::UInteger>(image_bytes)));
    }
    return buffers;
}

VkFFTResult encode_reflect_pad(
    MetalRuntime& runtime,
    MTL::CommandBuffer* command_buffer,
    MTL::Buffer* source_image,
    MTL::Buffer* destination,
    const WVFPadParams& params
) {
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->setComputePipelineState(runtime.pad_pipeline);
    encoder->setBuffer(source_image, 0, 0);
    encoder->setBuffer(destination, 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    encoder->dispatchThreads(
        MTL::Size::Make(params.real_pitch, params.fft_height, 1),
        threadgroup_2d(runtime.pad_pipeline)
    );
    encoder->endEncoding();
    return VKFFT_SUCCESS;
}

VkFFTResult encode_multiply_spectra(
    MetalRuntime& runtime,
    MTL::CommandBuffer* command_buffer,
    MTL::Buffer* input,
    MTL::Buffer* kernels,
    MTL::Buffer* output,
    uint64_t complex_count
) {
    if (complex_count > std::numeric_limits<uint32_t>::max() / 2u) {
        return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    }

    const uint32_t n_complex = static_cast<uint32_t>(complex_count);
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->setComputePipelineState(runtime.multiply_pipeline);
    encoder->setBuffer(input, 0, 0);
    encoder->setBuffer(kernels, 0, 1);
    encoder->setBuffer(output, 0, 2);
    encoder->setBytes(&n_complex, sizeof(n_complex), 3);
    encoder->dispatchThreads(
        MTL::Size::Make(static_cast<NS::UInteger>(complex_count * 2ull), 1, 1),
        threadgroup_1d(runtime.multiply_pipeline)
    );
    encoder->endEncoding();
    return VKFFT_SUCCESS;
}

VkFFTResult encode_postprocess(
    MetalRuntime& runtime,
    MTL::CommandBuffer* command_buffer,
    MTL::Buffer* source_planes,
    const ExternalBuffers& outputs,
    const WVFPostprocessParams& params
) {
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
    }
    encoder->setComputePipelineState(runtime.postprocess_pipeline);
    encoder->setBuffer(source_planes, 0, 0);
    encoder->setBuffer(outputs.out_x, 0, 1);
    encoder->setBuffer(outputs.out_y, 0, 2);
    encoder->setBuffer(outputs.magnitude, 0, 3);
    encoder->setBuffer(outputs.angle, 0, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    encoder->dispatchThreads(
        MTL::Size::Make(params.width, params.height, 1),
        threadgroup_2d(runtime.postprocess_pipeline)
    );
    encoder->endEncoding();
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
    if (real_pitch > std::numeric_limits<uint32_t>::max() ||
        fft_h > std::numeric_limits<uint32_t>::max() ||
        padded_w > std::numeric_limits<uint32_t>::max() ||
        padded_h > std::numeric_limits<uint32_t>::max() ||
        real_plane_count > std::numeric_limits<uint32_t>::max()) {
        write_error(error_out, error_len, "VkFFT WVF dimensions exceed uint32 limits");
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

    const uint64_t image_bytes = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * sizeof(float);
    const uint64_t output_bytes = image_bytes;
    ExternalBuffers external = wrap_external_buffers(
        *runtime,
        image,
        image_bytes,
        out_x,
        out_y,
        magnitude,
        angle,
        output_bytes
    );
    if (!external.image || !external.out_x || !external.out_y || !external.magnitude || !external.angle) {
        write_error(error_out, error_len, "failed to wrap host buffers for VkFFT");
        return 1;
    }

    WVFPadParams pad_params = {};
    pad_params.image_width = width;
    pad_params.image_height = height;
    pad_params.padded_width = static_cast<uint32_t>(padded_w);
    pad_params.padded_height = static_cast<uint32_t>(padded_h);
    pad_params.real_pitch = static_cast<uint32_t>(real_pitch);
    pad_params.fft_height = static_cast<uint32_t>(fft_h);
    pad_params.radius = radius;

    WVFPostprocessParams post_params = {};
    post_params.width = width;
    post_params.height = height;
    post_params.crop = radius * 2u;
    post_params.real_pitch = static_cast<uint32_t>(real_pitch);
    post_params.real_plane_count = static_cast<uint32_t>(real_plane_count);

    MTL::CommandBuffer* command_buffer = runtime->queue->commandBuffer();
    if (!command_buffer) {
        write_error(error_out, error_len, "failed to create Metal command buffer for VkFFT");
        return 1;
    }

    VkFFTResult result = encode_reflect_pad(
        *runtime,
        command_buffer,
        external.image,
        plan->scratch_input[0],
        pad_params
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to reflect-pad WVF image on Metal", result));
        return static_cast<int>(result);
    }

    result = append_vkfft(command_buffer, &plan->input_plan.app, -1, plan->scratch_input.data());
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to transform WVF image with VkFFT", result));
        return static_cast<int>(result);
    }

    result = encode_multiply_spectra(
        *runtime,
        command_buffer,
        plan->scratch_input[0],
        kernel_spectra->buffer,
        plan->scratch_output[0],
        plan->complex_count
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to multiply WVF spectra", result));
        return static_cast<int>(result);
    }

    result = append_vkfft(command_buffer, &plan->inverse_plan.app, 1, plan->scratch_output.data());
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to invert WVF spectra with VkFFT", result));
        return static_cast<int>(result);
    }

    result = encode_postprocess(
        *runtime,
        command_buffer,
        plan->scratch_output[0],
        external,
        post_params
    );
    if (result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to postprocess WVF FFT output", result));
        return static_cast<int>(result);
    }

    command_buffer->commit();
    command_buffer->waitUntilCompleted();
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
