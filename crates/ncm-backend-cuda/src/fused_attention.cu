// Ampere (SM80+) FlashAttention-2 fused Tensor-Core kernel for BF16 and FP16 (head_dim = 64).
// Uses mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16} with FP32 online softmax and zero O(S^2)
// global memory materialization. Supports arbitrary seq_q / seq_k (including prefix KV cache)
// and optional additive mask [mask_batches, 1, seq_q, seq_k].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int HEAD_DIM = 64;
constexpr int BR = 64;        // 4 warps * 16 query rows per warp
constexpr int BC = 32;        // 32 key/value rows per KV tile
constexpr int SMEM_STRIDE = 72; // 64 + 8 half padding to eliminate bank conflicts

template <typename T>
__device__ __forceinline__ uint16_t to_bits(T v);

template <>
__device__ __forceinline__ uint16_t to_bits<__half>(__half v) {
    return __half_as_ushort(v);
}

template <>
__device__ __forceinline__ uint16_t to_bits<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat16_as_ushort(v);
}

template <typename T>
__device__ __forceinline__ float bits_to_float(uint16_t b);

template <>
__device__ __forceinline__ float bits_to_float<__half>(uint16_t b) {
    return __half2float(__ushort_as_half(b));
}

template <>
__device__ __forceinline__ float bits_to_float<__nv_bfloat16>(uint16_t b) {
    return __bfloat162float(__ushort_as_bfloat16(b));
}

template <typename T>
__device__ __forceinline__ uint16_t float_to_bits(float v);

template <>
__device__ __forceinline__ uint16_t float_to_bits<__half>(float v) {
    return __half_as_ushort(__float2half_rn(v));
}

template <>
__device__ __forceinline__ uint16_t float_to_bits<__nv_bfloat16>(float v) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <typename T>
__device__ __forceinline__ uint32_t pack2_from_floats(float lo, float hi) {
    return pack2(float_to_bits<T>(lo), float_to_bits<T>(hi));
}

template <typename T>
__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]);

template <>
__device__ __forceinline__ void mma_m16n8k16<__half>(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <>
__device__ __forceinline__ void mma_m16n8k16<__nv_bfloat16>(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

template <typename T, int MaskMode>
__device__ __forceinline__ void apply_scale_mask_and_max(
    float (&s_acc)[4][4],
    const uint16_t* __restrict__ mrow0,
    const uint16_t* __restrict__ mrow1,
    const float* __restrict__ sMask,
    int r0,
    int r1,
    int seq_q,
    int seq_k,
    int k_start,
    int tid,
    float scale_log2,
    float& tile_m0,
    float& tile_m1) {
    constexpr float LOG2E = 1.4426950408889634f;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int lcol0 = nt * 8 + tid * 2;
        const int lcol1 = lcol0 + 1;
        const int col0 = k_start + lcol0;
        const int col1 = col0 + 1;

        if (r0 < seq_q && col0 < seq_k) {
            float v = s_acc[nt][0] * scale_log2;
            if constexpr (MaskMode == 1) {
                v += sMask[lcol0];
            } else if constexpr (MaskMode == 2) {
                v += bits_to_float<T>(mrow0[col0]) * LOG2E;
            }
            s_acc[nt][0] = v;
            tile_m0 = fmaxf(tile_m0, v);
        } else {
            s_acc[nt][0] = -INFINITY;
        }

        if (r0 < seq_q && col1 < seq_k) {
            float v = s_acc[nt][1] * scale_log2;
            if constexpr (MaskMode == 1) {
                v += sMask[lcol1];
            } else if constexpr (MaskMode == 2) {
                v += bits_to_float<T>(mrow0[col1]) * LOG2E;
            }
            s_acc[nt][1] = v;
            tile_m0 = fmaxf(tile_m0, v);
        } else {
            s_acc[nt][1] = -INFINITY;
        }

        if (r1 < seq_q && col0 < seq_k) {
            float v = s_acc[nt][2] * scale_log2;
            if constexpr (MaskMode == 1) {
                v += sMask[lcol0];
            } else if constexpr (MaskMode == 2) {
                v += bits_to_float<T>(mrow1[col0]) * LOG2E;
            }
            s_acc[nt][2] = v;
            tile_m1 = fmaxf(tile_m1, v);
        } else {
            s_acc[nt][2] = -INFINITY;
        }

        if (r1 < seq_q && col1 < seq_k) {
            float v = s_acc[nt][3] * scale_log2;
            if constexpr (MaskMode == 1) {
                v += sMask[lcol1];
            } else if constexpr (MaskMode == 2) {
                v += bits_to_float<T>(mrow1[col1]) * LOG2E;
            }
            s_acc[nt][3] = v;
            tile_m1 = fmaxf(tile_m1, v);
        } else {
            s_acc[nt][3] = -INFINITY;
        }
    }
}

template <typename T>
__device__ void flash_attention_d64_impl(
    const uint16_t* __restrict__ q_ptr,
    const uint16_t* __restrict__ k_ptr,
    const uint16_t* __restrict__ v_ptr,
    const uint16_t* __restrict__ mask_ptr,
    uint16_t* __restrict__ out_ptr,
    int seq_q,
    int seq_k,
    int heads,
    int mask_batches,
    int mask_rows,
    float scale) {

    const int bh = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2; // 0..7
    const int tid = lane & 3;    // 0..3

    const int q_block_start = blockIdx.x * BR;
    const int q_warp_start = q_block_start + warp_id * 16;

    const uint16_t* q_head = q_ptr + static_cast<size_t>(bh) * seq_q * HEAD_DIM;
    const uint16_t* k_head = k_ptr + static_cast<size_t>(bh) * seq_k * HEAD_DIM;
    const uint16_t* v_head = v_ptr + static_cast<size_t>(bh) * seq_k * HEAD_DIM;
    uint16_t* o_head = out_ptr + static_cast<size_t>(bh) * seq_q * HEAD_DIM;

    const uint16_t* mask_batch_ptr = nullptr;
    if (mask_ptr != nullptr) {
        const int mb = (mask_batches == 1) ? 0 : (bh / heads);
        mask_batch_ptr = mask_ptr + static_cast<size_t>(mb) * mask_rows * seq_k;
    }

    constexpr float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;
    const bool warp_active = (q_warp_start < seq_q);

    // Load this warp's 16x64 Q tile directly into mma.m16n8k16 A-fragment registers!
    // 4 k-chunks of 16 -> 4 * 4 = 16 u32 registers per thread.
    uint32_t q_frag[4][4];
    const int r0 = q_warp_start + group;
    const int r1 = q_warp_start + group + 8;
    if (warp_active) {
        #pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            const int c0 = kc * 16 + tid * 2;
            const int c1 = c0 + 8;
            if (r0 < seq_q) {
                const uint16_t* qr0 = q_head + static_cast<size_t>(r0) * HEAD_DIM;
                q_frag[kc][0] = *reinterpret_cast<const uint32_t*>(qr0 + c0);
                q_frag[kc][2] = *reinterpret_cast<const uint32_t*>(qr0 + c1);
            } else {
                q_frag[kc][0] = 0u;
                q_frag[kc][2] = 0u;
            }
            if (r1 < seq_q) {
                const uint16_t* qr1 = q_head + static_cast<size_t>(r1) * HEAD_DIM;
                q_frag[kc][1] = *reinterpret_cast<const uint32_t*>(qr1 + c0);
                q_frag[kc][3] = *reinterpret_cast<const uint32_t*>(qr1 + c1);
            } else {
                q_frag[kc][1] = 0u;
                q_frag[kc][3] = 0u;
            }
        }
    }

    // Output accumulator: 8 tiles of n=8 covering HEAD_DIM=64, 4 floats each = 32 floats.
    float o_acc[8][4];
    #pragma unroll
    for (int d_tile = 0; d_tile < 8; ++d_tile) {
        o_acc[d_tile][0] = 0.0f;
        o_acc[d_tile][1] = 0.0f;
        o_acc[d_tile][2] = 0.0f;
        o_acc[d_tile][3] = 0.0f;
    }

    // Running softmax stats for row0 (r0) and row1 (r1).
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;

    // Shared memory for one KV tile [BC=32][SMEM_STRIDE=72] each (~9.3 KiB total).
    __shared__ alignas(16) uint16_t sK[BC][SMEM_STRIDE];
    __shared__ alignas(16) uint16_t sV[BC][SMEM_STRIDE];
    __shared__ float sMask[BC];

    for (int k_start = 0; k_start < seq_k; k_start += BC) {
        // Cooperatively load K[k_start..k_start+BC, 0..64] and V[k_start..k_start+BC, 0..64].
        // 128 threads, 32*64 = 2048 elements -> 16 elements per thread (2 rows of 8 halfs via uint4).
        #pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            const int idx = iter * 128 + threadIdx.x; // 0..255
            const int row = idx >> 3;                 // 0..31
            const int col = (idx & 7) << 3;           // 0, 8, ..., 56
            const int g_row = k_start + row;
            if (g_row < seq_k) {
                const uint4 kv = *reinterpret_cast<const uint4*>(
                    k_head + static_cast<size_t>(g_row) * HEAD_DIM + col);
                const uint4 vv = *reinterpret_cast<const uint4*>(
                    v_head + static_cast<size_t>(g_row) * HEAD_DIM + col);
                *reinterpret_cast<uint4*>(&sK[row][col]) = kv;
                *reinterpret_cast<uint4*>(&sV[row][col]) = vv;
            } else {
                const uint4 z = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(&sK[row][col]) = z;
                *reinterpret_cast<uint4*>(&sV[row][col]) = z;
            }
        }
        if (mask_batch_ptr != nullptr && mask_rows == 1 && threadIdx.x < BC) {
            const int g_col = k_start + threadIdx.x;
            sMask[threadIdx.x] = (g_col < seq_k)
                ? (bits_to_float<T>(mask_batch_ptr[g_col]) * LOG2E)
                : -INFINITY;
        }
        __syncthreads();

        if (warp_active) {
            // 1. Compute S = Q @ K^T -> 4 n-tiles of 8 covering BC=32.
            float s_acc[4][4];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                s_acc[nt][0] = 0.0f;
                s_acc[nt][1] = 0.0f;
                s_acc[nt][2] = 0.0f;
                s_acc[nt][3] = 0.0f;
                const int k_row = nt * 8 + group;
                #pragma unroll
                for (int kc = 0; kc < 4; ++kc) {
                    const int c0 = kc * 16 + tid * 2;
                    const int c1 = c0 + 8;
                    uint32_t b_frag[2];
                    b_frag[0] = *reinterpret_cast<const uint32_t*>(&sK[k_row][c0]);
                    b_frag[1] = *reinterpret_cast<const uint32_t*>(&sK[k_row][c1]);
                    mma_m16n8k16<T>(s_acc[nt], q_frag[kc], b_frag);
                }
            }

            // 2. Apply scale*LOG2E, optional additive mask, out-of-bounds -inf, and online softmax.
            float tile_m0 = -INFINITY;
            float tile_m1 = -INFINITY;
            if (mask_batch_ptr == nullptr) {
                apply_scale_mask_and_max<T, 0>(
                    s_acc, nullptr, nullptr, sMask,
                    r0, r1, seq_q, seq_k, k_start, tid, scale_log2, tile_m0, tile_m1);
            } else if (mask_rows == 1) {
                apply_scale_mask_and_max<T, 1>(
                    s_acc, nullptr, nullptr, sMask,
                    r0, r1, seq_q, seq_k, k_start, tid, scale_log2, tile_m0, tile_m1);
            } else {
                const uint16_t* mrow0 = mask_batch_ptr + static_cast<size_t>(r0) * seq_k;
                const uint16_t* mrow1 = mask_batch_ptr + static_cast<size_t>(r1) * seq_k;
                apply_scale_mask_and_max<T, 2>(
                    s_acc, mrow0, mrow1, sMask,
                    r0, r1, seq_q, seq_k, k_start, tid, scale_log2, tile_m0, tile_m1);
            }

            tile_m0 = quad_max(tile_m0);
            tile_m1 = quad_max(tile_m1);

            const float m0_new = fmaxf(m0, tile_m0);
            const float m1_new = fmaxf(m1, tile_m1);
            const float alpha0 = (m0 == -INFINITY || m0_new == -INFINITY) ? 0.0f : exp2f(m0 - m0_new);
            const float alpha1 = (m1 == -INFINITY || m1_new == -INFINITY) ? 0.0f : exp2f(m1 - m1_new);

            l0 *= alpha0;
            l1 *= alpha1;
            #pragma unroll
            for (int d_tile = 0; d_tile < 8; ++d_tile) {
                o_acc[d_tile][0] *= alpha0;
                o_acc[d_tile][1] *= alpha0;
                o_acc[d_tile][2] *= alpha1;
                o_acc[d_tile][3] *= alpha1;
            }

            float tile_l0 = 0.0f;
            float tile_l1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float p00 = (s_acc[nt][0] == -INFINITY) ? 0.0f : exp2f(s_acc[nt][0] - m0_new);
                const float p01 = (s_acc[nt][1] == -INFINITY) ? 0.0f : exp2f(s_acc[nt][1] - m0_new);
                const float p10 = (s_acc[nt][2] == -INFINITY) ? 0.0f : exp2f(s_acc[nt][2] - m1_new);
                const float p11 = (s_acc[nt][3] == -INFINITY) ? 0.0f : exp2f(s_acc[nt][3] - m1_new);
                s_acc[nt][0] = p00;
                s_acc[nt][1] = p01;
                s_acc[nt][2] = p10;
                s_acc[nt][3] = p11;
                tile_l0 += p00 + p01;
                tile_l1 += p10 + p11;
            }
            l0 += quad_sum(tile_l0);
            l1 += quad_sum(tile_l1);
            m0 = m0_new;
            m1 = m1_new;

            // 3. Pack P directly from registers into mma.m16n8k16 A-fragments (2 k-chunks of 16)
            // and accumulate O += P @ V across all 8 d-tiles (n=8 each, HEAD_DIM=64).
            uint32_t p_frag[2][4];
            #pragma unroll
            for (int pc = 0; pc < 2; ++pc) {
                const int nt0 = pc * 2;
                const int nt1 = nt0 + 1;
                p_frag[pc][0] = pack2_from_floats<T>(s_acc[nt0][0], s_acc[nt0][1]);
                p_frag[pc][1] = pack2_from_floats<T>(s_acc[nt0][2], s_acc[nt0][3]);
                p_frag[pc][2] = pack2_from_floats<T>(s_acc[nt1][0], s_acc[nt1][1]);
                p_frag[pc][3] = pack2_from_floats<T>(s_acc[nt1][2], s_acc[nt1][3]);
            }

            #pragma unroll
            for (int d_tile = 0; d_tile < 8; ++d_tile) {
                const int d_col = d_tile * 8 + group;
                #pragma unroll
                for (int pc = 0; pc < 2; ++pc) {
                    const int vk0 = pc * 16 + tid * 2;
                    const int vk1 = vk0 + 8;
                    uint32_t v_frag[2];
                    v_frag[0] = pack2(sV[vk0][d_col], sV[vk0 + 1][d_col]);
                    v_frag[1] = pack2(sV[vk1][d_col], sV[vk1 + 1][d_col]);
                    mma_m16n8k16<T>(o_acc[d_tile], p_frag[pc], v_frag);
                }
            }
        }
        __syncthreads();
    }

    // 4. Normalize by l and store directly to global output in BF16 or FP16.
    if (!warp_active) {
        return;
    }
    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    if (r0 < seq_q) {
        uint16_t* dst0 = o_head + static_cast<size_t>(r0) * HEAD_DIM;
        #pragma unroll
        for (int d_tile = 0; d_tile < 8; ++d_tile) {
            const int c = d_tile * 8 + tid * 2;
            *reinterpret_cast<uint32_t*>(dst0 + c) =
                pack2_from_floats<T>(o_acc[d_tile][0] * inv_l0, o_acc[d_tile][1] * inv_l0);
        }
    }
    if (r1 < seq_q) {
        uint16_t* dst1 = o_head + static_cast<size_t>(r1) * HEAD_DIM;
        #pragma unroll
        for (int d_tile = 0; d_tile < 8; ++d_tile) {
            const int c = d_tile * 8 + tid * 2;
            *reinterpret_cast<uint32_t*>(dst1 + c) =
                pack2_from_floats<T>(o_acc[d_tile][2] * inv_l1, o_acc[d_tile][3] * inv_l1);
        }
    }
}

} // namespace

extern "C" __global__ void ncm_flash_attention_bf16_d64(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ mask,
    uint16_t* __restrict__ out,
    int seq_q,
    int seq_k,
    int heads,
    int mask_batches,
    int mask_rows,
    float scale) {
    flash_attention_d64_impl<__nv_bfloat16>(
        q, k, v, mask, out, seq_q, seq_k, heads, mask_batches, mask_rows, scale);
}

extern "C" __global__ void ncm_flash_attention_f16_d64(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ mask,
    uint16_t* __restrict__ out,
    int seq_q,
    int seq_k,
    int heads,
    int mask_batches,
    int mask_rows,
    float scale) {
    flash_attention_d64_impl<__half>(
        q, k, v, mask, out, seq_q, seq_k, heads, mask_batches, mask_rows, scale);
}
