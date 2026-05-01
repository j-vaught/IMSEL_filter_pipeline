#include "vkFFT.h"

#include <algorithm>
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
    void* placeholder_one[1] = {nullptr};
    void* placeholder_two[1] = {nullptr};
    void* scratch_input[1] = {nullptr};
    void* scratch_output[1] = {nullptr};
    VkFFTPlanHandle input_plan;
    VkFFTPlanHandle kernel_plan;
    VkFFTPlanHandle inverse_plan;

    ~PlanBundle() {
        if (scratch_output[0]) {
            cudaFree(scratch_output[0]);
        }
        if (scratch_input[0]) {
            cudaFree(scratch_input[0]);
        }
        if (placeholder_two[0]) {
            cudaFree(placeholder_two[0]);
        }
        if (placeholder_one[0]) {
            cudaFree(placeholder_one[0]);
        }
    }
};

struct KernelSpectra {
    void* buffer = nullptr;

    ~KernelSpectra() {
        if (buffer) {
            cudaFree(buffer);
        }
    }
};

struct CudaRuntime {
    int device_index = 0;
    CUdevice device = 0;
    CUcontext context = nullptr;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
    std::unordered_map<PlanKey, std::unique_ptr<PlanBundle>, PlanKeyHash> plan_cache;
    std::unordered_map<KernelKey, std::unique_ptr<KernelSpectra>, KernelKeyHash> kernel_cache;
    struct ExternalBuffers* reusable_io = nullptr;

    ~CudaRuntime();
};

struct ExternalBuffers {
    float* image = nullptr;
    float* out_x = nullptr;
    float* out_y = nullptr;
    float* magnitude = nullptr;
    float* angle = nullptr;
    uint64_t image_capacity = 0;
    uint64_t output_capacity = 0;

    void release_image() {
        if (image) {
            cudaFree(image);
            image = nullptr;
        }
        image_capacity = 0;
    }

    void release_outputs() {
        if (angle) {
            cudaFree(angle);
            angle = nullptr;
        }
        if (magnitude) {
            cudaFree(magnitude);
            magnitude = nullptr;
        }
        if (out_y) {
            cudaFree(out_y);
            out_y = nullptr;
        }
        if (out_x) {
            cudaFree(out_x);
            out_x = nullptr;
        }
        output_capacity = 0;
    }

    ~ExternalBuffers() {
        release_outputs();
        release_image();
    }
};

CudaRuntime::~CudaRuntime() {
    if (reusable_io) {
        delete reusable_io;
    }
    if (context) {
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == context) {
            cuCtxSetCurrent(nullptr);
        }
        cuDevicePrimaryCtxRelease(device);
    }
}

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

int selected_device_index(std::string* error_out) {
    const char* raw_value = std::getenv("WVF_GPU_DEVICE_INDEX");
    if (!raw_value || std::strlen(raw_value) == 0) {
        raw_value = std::getenv("WVF_METAL_DEVICE_INDEX");
    }
    if (!raw_value || std::strlen(raw_value) == 0) {
        return 0;
    }

    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(raw_value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        if (error_out) {
            *error_out = "GPU device index must be a non-negative integer";
        }
        return -1;
    }
    return static_cast<int>(value);
}

bool initialize_runtime(CudaRuntime* runtime, std::string* error_out) {
    if (!runtime) {
        if (error_out) {
            *error_out = "internal CUDA runtime allocation failed";
        }
        return false;
    }

    const int requested_index = selected_device_index(error_out);
    if (requested_index < 0) {
        return false;
    }

    cudaError_t cuda_result = cudaSuccess;
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        if (error_out) {
            *error_out = "failed to initialize CUDA driver";
        }
        return false;
    }

    int device_count = 0;
    cuda_result = cudaGetDeviceCount(&device_count);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to enumerate CUDA devices: ") + cudaGetErrorString(cuda_result);
        }
        return false;
    }
    if (device_count <= 0) {
        if (error_out) {
            *error_out = "no CUDA device is available for VkFFT";
        }
        return false;
    }
    if (requested_index >= device_count) {
        if (error_out) {
            *error_out =
                "GPU device index " + std::to_string(requested_index) +
                " is out of range for " + std::to_string(device_count) + " CUDA device(s)";
        }
        return false;
    }

    cuda_result = cudaSetDevice(requested_index);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to select CUDA device: ") + cudaGetErrorString(cuda_result);
        }
        return false;
    }
    cuda_result = cudaFree(nullptr);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to initialize CUDA runtime: ") + cudaGetErrorString(cuda_result);
        }
        return false;
    }

    cu_result = cuDeviceGet(&runtime->device, requested_index);
    if (cu_result != CUDA_SUCCESS) {
        if (error_out) {
            *error_out = "failed to acquire CUDA device handle";
        }
        return false;
    }

    cu_result = cuDevicePrimaryCtxRetain(&runtime->context, runtime->device);
    if (cu_result != CUDA_SUCCESS) {
        if (error_out) {
            *error_out = "failed to retain CUDA primary context";
        }
        return false;
    }
    cu_result = cuCtxSetCurrent(runtime->context);
    if (cu_result != CUDA_SUCCESS) {
        if (error_out) {
            *error_out = "failed to activate CUDA primary context";
        }
        return false;
    }

    runtime->device_index = requested_index;
    return true;
}

bool activate_runtime(CudaRuntime& runtime, std::string* error_out) {
    cudaError_t cuda_result = cudaSetDevice(runtime.device_index);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to select CUDA device: ") + cudaGetErrorString(cuda_result);
        }
        return false;
    }

    const CUresult cu_result = cuCtxSetCurrent(runtime.context);
    if (cu_result != CUDA_SUCCESS) {
        if (error_out) {
            *error_out = "failed to activate CUDA primary context";
        }
        return false;
    }
    return true;
}

CudaRuntime* get_runtime(std::string* error_out) {
    static std::mutex runtime_mutex;
    static std::unordered_map<int, std::unique_ptr<CudaRuntime>> runtimes;

    const int requested_index = selected_device_index(error_out);
    if (requested_index < 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(runtime_mutex);
    auto existing = runtimes.find(requested_index);
    if (existing != runtimes.end()) {
        return existing->second.get();
    }

    auto runtime = std::make_unique<CudaRuntime>();
    std::string init_error;
    if (!initialize_runtime(runtime.get(), &init_error)) {
        if (error_out) {
            *error_out = init_error;
        }
        return nullptr;
    }

    auto inserted = runtimes.emplace(requested_index, std::move(runtime));
    return inserted.first->second.get();
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

__device__ inline int reflect_index_device(int value, int limit) {
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

__device__ inline float unsigned_angle_device(float y, float x) {
    float theta = atan2f(y, x);
    if (theta < 0.0f) {
        theta += kPi;
    }
    if (theta >= kPi) {
        theta -= kPi;
    }
    return theta;
}

__global__ void wvf_reflect_pad_real(
    const float* image,
    float* padded,
    WVFPadParams params
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.real_pitch || y >= params.fft_height) {
        return;
    }

    const uint32_t dst_idx = y * params.real_pitch + x;
    if (x >= params.padded_width || y >= params.padded_height) {
        padded[dst_idx] = 0.0f;
        return;
    }

    const int src_x = reflect_index_device(static_cast<int>(x) - static_cast<int>(params.radius),
                                           static_cast<int>(params.image_width));
    const int src_y = reflect_index_device(static_cast<int>(y) - static_cast<int>(params.radius),
                                           static_cast<int>(params.image_height));
    padded[dst_idx] = image[static_cast<uint32_t>(src_y) * params.image_width + static_cast<uint32_t>(src_x)];
}

__global__ void wvf_multiply_spectra(
    const float2* input,
    const float2* kernels,
    float2* output,
    uint32_t n_complex
) {
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_complex * 2u;
    if (id >= total) {
        return;
    }
    const uint32_t plane = id / n_complex;
    const uint32_t idx = id - plane * n_complex;
    const float2 a = input[idx];
    const float2 b = kernels[id];
    output[id] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void wvf_fft_postprocess(
    const float* planes,
    float* out_x,
    float* out_y,
    float* magnitude,
    float* angle,
    WVFPostprocessParams params
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    const uint32_t out_idx = y * params.width + x;
    const uint32_t src_idx = (y + params.crop) * params.real_pitch + x + params.crop;
    const float gx = planes[src_idx];
    const float gy = planes[params.real_plane_count + src_idx];
    out_x[out_idx] = gx;
    out_y[out_idx] = gy;
    magnitude[out_idx] = sqrtf(gx * gx + gy * gy);
    angle[out_idx] = unsigned_angle_device(gy, gx);
}

VkFFTResult run_vkfft_once(
    VkFFTApplication* app,
    int inverse,
    void** buffer
) {
    VkFFTLaunchParams launch_params = {};
    launch_params.buffer = buffer;
    launch_params.inputBuffer = buffer;
    launch_params.outputBuffer = buffer;
    return VkFFTAppend(app, inverse, &launch_params);
}

PlanBundle* get_or_create_plan_bundle(
    CudaRuntime& runtime,
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

    cudaError_t cuda_result = cudaSuccess;
    cuda_result = cudaMalloc(&plan->placeholder_one[0], plan->one_plane_bytes);
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMalloc(&plan->placeholder_two[0], plan->two_plane_bytes);
    }
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMalloc(&plan->scratch_input[0], plan->one_plane_bytes);
    }
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMalloc(&plan->scratch_output[0], plan->two_plane_bytes);
    }
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to allocate CUDA buffers for VkFFT: ") +
                         cudaGetErrorString(cuda_result);
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
    base.device = &runtime.device;

    VkFFTConfiguration kernel_config = base;
    kernel_config.numberBatches = 2;
    kernel_config.makeForwardPlanOnly = 1;
    kernel_config.buffer = plan->placeholder_two;
    kernel_config.bufferSize = &plan->two_plane_bytes;
    if (!initialize_plan(&plan->kernel_plan, kernel_config, "failed to initialize VkFFT kernel plan", error_out)) {
        return nullptr;
    }

    VkFFTConfiguration input_config = base;
    input_config.numberBatches = 1;
    input_config.makeForwardPlanOnly = 1;
    input_config.buffer = plan->placeholder_one;
    input_config.bufferSize = &plan->one_plane_bytes;
    if (!initialize_plan(&plan->input_plan, input_config, "failed to initialize VkFFT image plan", error_out)) {
        return nullptr;
    }

    VkFFTConfiguration inverse_config = base;
    inverse_config.numberBatches = 2;
    inverse_config.makeInversePlanOnly = 1;
    inverse_config.buffer = plan->placeholder_two;
    inverse_config.bufferSize = &plan->two_plane_bytes;
    if (!initialize_plan(&plan->inverse_plan, inverse_config, "failed to initialize VkFFT inverse plan", error_out)) {
        return nullptr;
    }

    auto inserted = runtime.plan_cache.emplace(key, std::move(plan));
    return inserted.first->second.get();
}

KernelSpectra* get_or_create_kernel_spectra(
    CudaRuntime& runtime,
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
    cudaError_t cuda_result = cudaMalloc(&spectra->buffer, plan.two_plane_bytes);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to allocate cached CUDA kernel spectrum buffer: ") +
                         cudaGetErrorString(cuda_result);
        }
        return nullptr;
    }

    cuda_result = cudaMemcpyAsync(
        spectra->buffer,
        kernel_data.data(),
        plan.two_plane_bytes,
        cudaMemcpyHostToDevice,
        runtime.stream
    );
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to upload WVF kernels to CUDA: ") +
                         cudaGetErrorString(cuda_result);
        }
        return nullptr;
    }

    void* kernel_buffer_array[1] = {spectra->buffer};
    const VkFFTResult result = run_vkfft_once(&plan.kernel_plan.app, -1, kernel_buffer_array);
    if (result != VKFFT_SUCCESS) {
        if (error_out) {
            *error_out = vkfft_error("failed to transform WVF kernels with VkFFT", result);
        }
        return nullptr;
    }
    cuda_result = cudaStreamSynchronize(runtime.stream);
    if (cuda_result != cudaSuccess) {
        if (error_out) {
            *error_out = std::string("failed to synchronize CUDA stream after kernel FFT: ") +
                         cudaGetErrorString(cuda_result);
        }
        return nullptr;
    }

    auto inserted = runtime.kernel_cache.emplace(key, std::move(spectra));
    return inserted.first->second.get();
}

bool ensure_reusable_io_buffers(
    CudaRuntime& runtime,
    uint64_t image_bytes,
    uint64_t output_bytes,
    std::string* error_out
) {
    if (!runtime.reusable_io) {
        runtime.reusable_io = new ExternalBuffers();
    }
    ExternalBuffers* buffers = runtime.reusable_io;
    if (!buffers) {
        if (error_out) {
            *error_out = "internal CUDA buffer allocation failed";
        }
        return false;
    }

    if (buffers->image_capacity < image_bytes) {
        buffers->release_image();
    }
    if (buffers->output_capacity < output_bytes) {
        buffers->release_outputs();
    }

    cudaError_t cuda_result = cudaSuccess;
    if (!buffers->image) {
        cuda_result = cudaMalloc(reinterpret_cast<void**>(&buffers->image), image_bytes);
        if (cuda_result == cudaSuccess) {
            buffers->image_capacity = image_bytes;
        }
    }
    if (cuda_result == cudaSuccess && !buffers->out_x) {
        cuda_result = cudaMalloc(reinterpret_cast<void**>(&buffers->out_x), output_bytes);
    }
    if (cuda_result == cudaSuccess && !buffers->out_y) {
        cuda_result = cudaMalloc(reinterpret_cast<void**>(&buffers->out_y), output_bytes);
    }
    if (cuda_result == cudaSuccess && !buffers->magnitude) {
        cuda_result = cudaMalloc(reinterpret_cast<void**>(&buffers->magnitude), output_bytes);
    }
    if (cuda_result == cudaSuccess && !buffers->angle) {
        cuda_result = cudaMalloc(reinterpret_cast<void**>(&buffers->angle), output_bytes);
    }
    if (cuda_result == cudaSuccess) {
        buffers->output_capacity = output_bytes;
    }
    if (cuda_result != cudaSuccess) {
        if (!buffers->image) {
            buffers->release_image();
        }
        if (!buffers->out_x || !buffers->out_y || !buffers->magnitude || !buffers->angle) {
            buffers->release_outputs();
        }
        if (error_out) {
            *error_out = std::string("failed to allocate CUDA IO buffers: ") +
                         cudaGetErrorString(cuda_result);
        }
        return false;
    }
    return true;
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

    std::string runtime_error;
    CudaRuntime* runtime = get_runtime(&runtime_error);
    if (!runtime) {
        write_error(error_out, error_len, runtime_error);
        return 1;
    }

    if (!activate_runtime(*runtime, &runtime_error)) {
        write_error(error_out, error_len, runtime_error);
        return 1;
    }

    std::lock_guard<std::mutex> lock(runtime->mutex);

    cudaError_t cuda_result = cudaSuccess;

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

    const uint64_t image_bytes =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * sizeof(float);
    const uint64_t output_bytes = image_bytes;
    if (!ensure_reusable_io_buffers(*runtime, image_bytes, output_bytes, &cache_error)) {
        write_error(error_out, error_len, cache_error);
        return 1;
    }
    ExternalBuffers& external = *runtime->reusable_io;

    cuda_result = cudaMemcpyAsync(
        external.image,
        image,
        image_bytes,
        cudaMemcpyHostToDevice,
        runtime->stream
    );
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to upload WVF image to CUDA: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
    }

    const dim3 threads_2d(16, 16, 1);
    const dim3 grid_pad(
        static_cast<unsigned int>((real_pitch + threads_2d.x - 1ull) / threads_2d.x),
        static_cast<unsigned int>((fft_h + threads_2d.y - 1ull) / threads_2d.y),
        1
    );
    WVFPadParams pad_params = {};
    pad_params.image_width = width;
    pad_params.image_height = height;
    pad_params.padded_width = static_cast<uint32_t>(padded_w);
    pad_params.padded_height = static_cast<uint32_t>(padded_h);
    pad_params.real_pitch = static_cast<uint32_t>(real_pitch);
    pad_params.fft_height = static_cast<uint32_t>(fft_h);
    pad_params.radius = radius;
    wvf_reflect_pad_real<<<grid_pad, threads_2d, 0, runtime->stream>>>(
        external.image,
        static_cast<float*>(plan->scratch_input[0]),
        pad_params
    );
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to reflect-pad WVF image on CUDA: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
    }

    const VkFFTResult forward_result =
        run_vkfft_once(&plan->input_plan.app, -1, plan->scratch_input);
    if (forward_result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to transform WVF image with VkFFT", forward_result));
        return static_cast<int>(forward_result);
    }

    if (complex_count > std::numeric_limits<uint32_t>::max() / 2u) {
        write_error(error_out, error_len, "VkFFT complex spectrum length exceeded uint32");
        return 1;
    }
    const uint32_t n_complex = static_cast<uint32_t>(complex_count);
    const uint32_t total_complex = n_complex * 2u;
    const uint32_t threads_1d = 256u;
    const uint32_t blocks_1d = (total_complex + threads_1d - 1u) / threads_1d;
    wvf_multiply_spectra<<<blocks_1d, threads_1d, 0, runtime->stream>>>(
        static_cast<const float2*>(plan->scratch_input[0]),
        static_cast<const float2*>(kernel_spectra->buffer),
        static_cast<float2*>(plan->scratch_output[0]),
        n_complex
    );
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to multiply WVF spectra on CUDA: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
    }

    const VkFFTResult inverse_result =
        run_vkfft_once(&plan->inverse_plan.app, 1, plan->scratch_output);
    if (inverse_result != VKFFT_SUCCESS) {
        write_error(error_out, error_len, vkfft_error("failed to invert WVF spectra with VkFFT", inverse_result));
        return static_cast<int>(inverse_result);
    }

    WVFPostprocessParams post_params = {};
    post_params.width = width;
    post_params.height = height;
    post_params.crop = radius * 2u;
    post_params.real_pitch = static_cast<uint32_t>(real_pitch);
    post_params.real_plane_count = static_cast<uint32_t>(real_plane_count);
    const dim3 grid_post(
        static_cast<unsigned int>((width + threads_2d.x - 1u) / threads_2d.x),
        static_cast<unsigned int>((height + threads_2d.y - 1u) / threads_2d.y),
        1
    );
    wvf_fft_postprocess<<<grid_post, threads_2d, 0, runtime->stream>>>(
        static_cast<const float*>(plan->scratch_output[0]),
        external.out_x,
        external.out_y,
        external.magnitude,
        external.angle,
        post_params
    );
    cuda_result = cudaGetLastError();
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to postprocess WVF FFT output on CUDA: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
    }

    cuda_result = cudaMemcpyAsync(out_x, external.out_x, output_bytes, cudaMemcpyDeviceToHost, runtime->stream);
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMemcpyAsync(out_y, external.out_y, output_bytes, cudaMemcpyDeviceToHost, runtime->stream);
    }
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMemcpyAsync(
            magnitude,
            external.magnitude,
            output_bytes,
            cudaMemcpyDeviceToHost,
            runtime->stream
        );
    }
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMemcpyAsync(angle, external.angle, output_bytes, cudaMemcpyDeviceToHost, runtime->stream);
    }
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to download WVF CUDA outputs: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
    }

    cuda_result = cudaStreamSynchronize(runtime->stream);
    if (cuda_result != cudaSuccess) {
        write_error(
            error_out,
            error_len,
            std::string("failed to synchronize CUDA stream: ") + cudaGetErrorString(cuda_result)
        );
        return 1;
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
