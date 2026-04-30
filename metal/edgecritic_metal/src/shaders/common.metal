#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH_MS 32
#define CGMM_N_MAX 64
#define CGMM_K 3
#define CGMM_EPS 1.0e-12f
#define CGMM_PI 3.14159265358979323846f
#define CGMM_TWO_PI 6.28318530717958647692f
#define CGMM_KAPPA_MAX 700.0f
#define RECOVERY_TRIG_ORDER 4
#define RECOVERY_TRIG_COEFFS 9

struct KernelParams {
    uint width;
    uint height;
    uint n_offsets;
};

struct LfParams {
    uint width;
    uint height;
    uint n_pixels;
    uint n_offsets;
    float cos_t;
    float sin_t;
};

struct LfBatchParams {
    uint width;
    uint height;
    uint n_pixels;
    uint n_thetas;
    uint n_ms;
    uint max_m;
    uint n_samples;
};

struct LfStackParams {
    uint width;
    uint height;
    uint n_orientations;
    uint n_samples;
    uint border;
    float weight_sum;
};

struct LfProjectParams {
    uint width;
    uint height;
    float cos_t;
    float sin_t;
};

struct LfPlaneParams {
    uint width;
    uint height;
    uint n_samples;
    uint theta_idx;
    uint border;
    float weight_sum;
};

struct LfBoxSeedParams {
    uint width;
    uint height;
    float cos_t;
    float sin_t;
};

struct LfBoxPassParams {
    uint width;
    uint height;
    uint radius;
    int key_min;
    uint line_count;
};

struct LfBoxFinalizeParams {
    uint width;
    uint height;
    uint theta_idx;
};

struct LfBoxMultiParams {
    uint width;
    uint height;
    uint n_ms;
    uint output_layout;
    int key_min;
    uint line_count;
    uint theta_idx;
};

struct LfScanlineParams {
    uint width;
    uint height;
    uint n_samples;
    uint radius;
    int key_min;
    uint line_count;
    uint theta_idx;
    uint chunk_len;
};

struct RecoveryParams {
    uint n_rows;
    uint k;
    uint dense_n;
    uint sep;
    float tau_sec_floor;
    float h;
    float h2_over6;
    float rhs_scale;
    float pi_over_dense;
    float gamma_inv;
    float cyclic_denom_inv;
    uint response_layout;
    uint plane_size;
    uint trig_order;
    uint trig_coeffs;
};

struct RecoveryReduceParams {
    uint count;
    uint group_size;
};

struct CgmmParams {
    uint P;
    uint N;
    uint n_iters;
    float init_kappa;
    float tau_M_rel;
    float theta_min_phi;
};

inline int reflect_index(int value, int limit) {
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
