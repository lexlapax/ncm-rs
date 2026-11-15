#include <math_constants.h>
// FP32 layout copies only: no arithmetic on tensor values. The Rust caller
// validates batch <= 128, sequence <= 768, heads == 8 and head width == 64.
extern "C" __global__ void ncm_pack_qkv_8x64(const float* input, float* output,
                                             unsigned int batch, unsigned int seq) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    if (idx >= seq * 64 || group >= batch * 24) return;
    const unsigned int t = idx / 64, d = idx % 64;
    const unsigned int h = group % 8, q = (group / 8) % 3, b = group / 24;
    const unsigned int src = ((b * seq + t) * 24 + q * 8 + h) * 64 + d;
    const unsigned int dst = (((q * batch + b) * 8 + h) * seq + t) * 64 + d;
    output[dst] = input[src];
}
extern "C" __global__ void ncm_merge_heads_8x64(const float* input, float* output,
                                                 unsigned int batch, unsigned int seq) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    if (idx >= 512 || group >= batch * seq) return;
    const unsigned int h = idx / 64, d = idx % 64, b = group / seq, t = group % seq;
    const unsigned int src = ((b * 8 + h) * seq + t) * 64 + d;
    const unsigned int dst = (b * seq + t) * 512 + idx;
    output[dst] = input[src];
}

// Experimental categorical draw for one row of 4096 finite FP32 logits per
// block. Supplied uniforms use all 32 bits. This is an inverse-CDF draw;
// it does not claim to reproduce PyTorch's random-number stream.
extern "C" __global__ void ncm_categorical_4096(const float* logits,
                                                const unsigned int* uniform_bits,
                                                unsigned int* output) {
    const unsigned int row = blockIdx.x, t = threadIdx.x;
    const float* values = logits + row * 4096;
    __shared__ float work[256];
    __shared__ unsigned int bad;
    if (t == 0) { bad = 0; output[row] = 0xffffffffU; }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (unsigned int j = 0; j < 16; ++j) {
        float v = values[t * 16 + j];
        if (isnan(v) || v == CUDART_INF_F) atomicOr(&bad, 1);
        maximum = fmaxf(maximum, v);
    }
    work[t] = maximum;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) work[t] = fmaxf(work[t], work[t + stride]);
        __syncthreads();
    }
    maximum = work[0];
    if (bad || maximum == -CUDART_INF_F) {
        if (t == 0) output[row] = 0xffffffffU;
        return;
    }
    __syncthreads(); // All threads must read the reduction result before reuse.
    float local = 0;
    for (unsigned int j = 0; j < 16; ++j) local += expf(values[t * 16 + j] - maximum);
    work[t] = local;
    __syncthreads();
    // A fixed-order scan guarantees monotone section boundaries. Parallel
    // reassociation of FP32 prefix sums can produce overlapping boundaries,
    // multiple writers, or an unwritten output near a rounding boundary.
    if (t == 0) {
        float prefix = 0.0f;
        for (unsigned int i = 0; i < 256; ++i) {
            prefix += work[i];
            work[i] = prefix;
        }
    }
    __syncthreads();
    __shared__ double threshold;
    if (t == 0) threshold = double(uniform_bits[row]) * (1.0 / 4294967296.0) * double(work[255]);
    __syncthreads();
    float partial = t ? work[t - 1] : 0.0f;
    if (threshold >= partial && threshold < work[t]) {
        unsigned int choice = t * 16;
        for (unsigned int j = 0; j < 16; ++j) {
            float weight = expf(values[t * 16 + j] - maximum);
            if (weight > 0.0f) choice = t * 16 + j;
            partial += weight;
            if (threshold < partial) break;
        }
        output[row] = choice;
    }
}

// Exact layout copies for the TinyStories 12-head, 64-wide generator.
extern "C" __global__ void ncm_pack_qkv_12x64(const float* input, float* output,
                                              unsigned int batch, unsigned int seq) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    if (idx >= seq * 64 || group >= batch * 36) return;
    const unsigned int t = idx / 64, d = idx % 64;
    const unsigned int h = group % 12, q = (group / 12) % 3, b = group / 36;
    const unsigned int src = ((b * seq + t) * 36 + q * 12 + h) * 64 + d;
    const unsigned int dst = (((q * batch + b) * 12 + h) * seq + t) * 64 + d;
    output[dst] = input[src];
}
extern "C" __global__ void ncm_merge_heads_12x64(const float* input, float* output,
                                                  unsigned int batch, unsigned int seq) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    if (idx >= 768 || group >= batch * seq) return;
    const unsigned int h = idx / 64, d = idx % 64, b = group / seq, t = group % seq;
    const unsigned int src = ((b * 12 + h) * seq + t) * 64 + d;
    const unsigned int dst = (b * seq + t) * 768 + idx;
    output[dst] = input[src];
}
