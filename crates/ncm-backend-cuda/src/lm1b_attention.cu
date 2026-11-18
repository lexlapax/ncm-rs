// Ampere FP32 attention experiment for the released LM1B-small head geometry.
// Uses the public PTX matrix fragment layout and a three-product TF32 expansion.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

struct ResidualPair {
    uint32_t base;
    uint32_t residual;
};

__device__ __forceinline__ ResidualPair decompose(float value) {
    const uint32_t base = __float_as_uint(value) & 0xffffe000u;
    const float remainder = __fsub_rn(value, __uint_as_float(base));
    return {base, (__float_as_uint(remainder) + 0x1000u) & 0xffffe000u};
}

__device__ __forceinline__ void multiply(float (&out)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template<bool transpose_b>
__device__ __forceinline__ void accumulate_product(
        float (&out)[4], const float* a, const float* b,
        const int rows, const int cols, const int shared,
        const int a_stride, const int b_stride, const int quad, const int lane) {
    float stage[4] = {};
    for (int offset = 0; offset < (shared + 31) / 32 * 32; offset += 8) {
        uint32_t ah[4], ar[4], bh[2], br[2];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = quad + 8 * (i & 1);
            const int c = offset + lane + 4 * (i / 2);
            const auto x = decompose(r < rows && c < shared ? a[r * a_stride + c] : 0.0f);
            ah[i] = x.base;
            ar[i] = x.residual;
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = offset + lane + 4 * i;
            const int c = quad;
            const auto x = decompose(r < shared && c < cols ?
                (transpose_b ? b[c * b_stride + r] : b[r * b_stride + c]) : 0.0f);
            bh[i] = x.base;
            br[i] = x.residual;
        }
        multiply(stage, ar, bh);
        multiply(stage, ah, br);
        multiply(stage, ah, bh);
        if (offset % 32 == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = __fadd_rn(out[i], stage[i]);
                stage[i] = 0.0f;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fadd_rn(out[i], stage[i]);
}

__device__ __forceinline__ void update_weights(float* weights, const float* mask,
        const int keys, float& maximum, float& denominator, float& rescale) {
    const float before = maximum;
    for (int i = 0; i < keys; ++i) {
        const float score = __fadd_rn(__fmul_rn(weights[i], 0.125f), mask ? mask[i] : 0.0f);
        weights[i] = __fmul_rn(score, 1.4426950408889634f);
        maximum = fmaxf(maximum, weights[i]);
    }
    rescale = before < maximum ? exp2f(__fsub_rn(before, maximum)) : 1.0f;
    denominator = __fmul_rn(denominator, rescale);
    const float center = maximum == -INFINITY ? 0.0f : maximum;
    for (int offset = 0; offset < keys; offset += 32) {
        float pair[4] = {};
        for (int i = offset; i < min(keys, offset + 32); ++i) {
            const float w = exp2f(__fsub_rn(weights[i], center));
            weights[i] = w;
            const int lane = (i % 8) / 2;
            pair[lane] = __fadd_rn(pair[lane], w);
        }
        const float sum = __fadd_rn(__fadd_rn(pair[0], pair[1]), __fadd_rn(pair[2], pair[3]));
        denominator = __fadd_rn(denominator, sum);
    }
}

} // namespace

extern "C" __global__ void ncm_lm1b_efficient_attention(
        const float* query, const float* key, const float* value, const float* mask,
        float* result, int length, int heads, int mask_batches) {
    constexpr int width = 64;
    constexpr int query_tile = 16;
    constexpr int key_tile = 64;
    const int bh = blockIdx.y;
    const int start_row = blockIdx.x * query_tile;
    const int rows = min(query_tile, length - start_row);
    const int thread = threadIdx.x;
    const int warp_column = (thread / 32) * 8;
    const int quad = (thread % 32) / 4;
    const int lane = thread % 4;

    const auto q = query + (bh * length + start_row) * width;
    const auto k = key + bh * length * width;
    const auto v = value + bh * length * width;
    const auto out = result + (bh * length + start_row) * width;
    if (mask) mask += ((mask_batches == 1 ? 0 : bh / heads) * length + start_row) * length;

    __shared__ float weights[query_tile][key_tile];
    __shared__ float maximum[query_tile], denominator[query_tile], rescale[query_tile];
    if (thread < rows) {
        maximum[thread] = -INFINITY;
        denominator[thread] = 0.0f;
    }
    float numerators[4] = {};
    for (int start_key = 0; start_key < length; start_key += key_tile) {
        const int keys = min(key_tile, length - start_key);
        if (warp_column < keys) {
            float scores[4] = {};
            accumulate_product<true>(scores, q, k + (start_key + warp_column) * width,
                rows, min(8, keys - warp_column), width, width, width, quad, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = quad + 8 * (i / 2);
                const int col = warp_column + 2 * lane + (i & 1);
                if (row < rows && col < keys) weights[row][col] = scores[i];
            }
        }
        __syncthreads();
        if (thread < rows) {
            const float* bias = mask ? mask + thread * length + start_key : nullptr;
            update_weights(weights[thread], bias, keys, maximum[thread],
                           denominator[thread], rescale[thread]);
        }
        __syncthreads();
        if (start_key > 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = quad + 8 * (i / 2);
                if (row < rows) numerators[i] = __fmul_rn(numerators[i], rescale[row]);
            }
        }
        accumulate_product<false>(numerators, &weights[0][0], v + start_key * width + warp_column,
            rows, 8, keys, key_tile, width, quad, lane);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = quad + 8 * (i / 2);
        const int col = warp_column + 2 * lane + (i & 1);
        if (row < rows) {
            const float sum = denominator[row];
            out[row * width + col] = __fmul_rn(numerators[i], sum == 0.0f ? 1.0f : __frcp_rn(sum));
        }
    }
}
