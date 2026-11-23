#include <cuda_fp16.h>
#include <math_constants.h>

// Unique finite maximum per 4096-way row. Exact ties and nonfinite values are
// rejected with a sentinel; no ordering claim is made for upstream tied sorts.
extern "C" __global__ void ncm_unique_max_4096(const float* logits, unsigned int* output) {
    const unsigned int t = threadIdx.x, row = blockIdx.x;
    const float* values = logits + row * 4096;
    __shared__ float maxima[256];
    __shared__ unsigned int indices[256], counts[256], bad;
    if (t == 0) bad = 0;
    __syncthreads();
    float best = -CUDART_INF_F;
    unsigned int index = 0, count = 0;
    for (unsigned int j = 0; j < 16; ++j) {
        unsigned int i = t * 16 + j;
        float value = values[i];
        if (!isfinite(value)) atomicOr(&bad, 1);
        if (value > best) { best = value; index = i; count = 1; }
        else if (value == best) { ++count; }
    }
    maxima[t] = best; indices[t] = index; counts[t] = count;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (maxima[t + stride] > maxima[t]) {
                maxima[t] = maxima[t + stride];
                indices[t] = indices[t + stride];
                counts[t] = counts[t + stride];
            } else if (maxima[t + stride] == maxima[t]) {
                counts[t] += counts[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) output[row] = bad || counts[0] != 1 ? 0xffffffffU : indices[0];
}

// First finite maximum for the pinned 4096-way PyTorch CUDA radix-sort path.
// It preserves the lowest original index on equal keys, including signed zeros.
extern "C" __global__ void ncm_xsum_reference_top1_4096(const float* logits, unsigned int* output) {
    const unsigned int t = threadIdx.x, row = blockIdx.x;
    const float* values = logits + row * 4096;
    __shared__ float maxima[256];
    __shared__ unsigned int indices[256], bad;
    if (t == 0) bad = 0;
    __syncthreads();
    float best = -CUDART_INF_F;
    unsigned int index = t * 16;
    for (unsigned int j = 0; j < 16; ++j) {
        unsigned int i = t * 16 + j;
        float value = values[i];
        if (!isfinite(value)) atomicOr(&bad, 1);
        if (value > best) { best = value; index = i; }
    }
    maxima[t] = best; indices[t] = index;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (maxima[t + stride] > maxima[t]) {
                maxima[t] = maxima[t + stride];
                indices[t] = indices[t + stride];
            } else if (maxima[t + stride] == maxima[t]) {
                indices[t] = min(indices[t], indices[t + stride]);
            }
        }
        __syncthreads();
    }
    if (t == 0) output[row] = bad ? 0xffffffffU : indices[0];
}
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

// Explicit LM1B FP64 inverse-CDF draw. One 256-thread block owns one row;
// a monotone section scan gives each valid uniform a single writer.
extern "C" __global__ void ncm_categorical_8192(const float* logits,
                                                const unsigned int* uniform_bits,
                                                unsigned int* output) {
    const unsigned int row = blockIdx.x, t = threadIdx.x;
    const float* values = logits + row * 8192;
    __shared__ double work[256];
    __shared__ unsigned int bad;
    if (t == 0) { bad = 0; output[row] = 0xffffffffU; }
    __syncthreads();
    float maximum = -CUDART_INF_F;
    for (unsigned int j = 0; j < 32; ++j) {
        float v = values[t * 32 + j];
        if (!isfinite(v)) atomicOr(&bad, 1);
        maximum = fmaxf(maximum, v);
    }
    work[t] = double(maximum);
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) work[t] = fmax(work[t], work[t + stride]);
        __syncthreads();
    }
    const double max_value = work[0];
    if (bad) return;
    __syncthreads();
    double local = 0.0;
    for (unsigned int j = 0; j < 32; ++j) local += exp(double(values[t * 32 + j]) - max_value);
    work[t] = local;
    __syncthreads();
    if (t == 0) {
        double prefix = 0.0;
        for (unsigned int i = 0; i < 256; ++i) {
            prefix += work[i];
            work[i] = prefix;
        }
    }
    __syncthreads();
    __shared__ double threshold;
    if (t == 0) threshold = (double(uniform_bits[row]) + 0.5) * (1.0 / 4294967296.0) * work[255];
    __syncthreads();
    double partial = t ? work[t - 1] : 0.0;
    if (threshold >= partial && threshold < work[t]) {
        unsigned int choice = t * 32;
        for (unsigned int j = 0; j < 32; ++j) {
            double weight = exp(double(values[t * 32 + j]) - max_value);
            if (weight > 0.0) choice = t * 32 + j;
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

// Exact FP64 XSum penalties. Each time step uses a tiled reduction followed
// by one row-wise reduction; launches serialize sequential history updates.
extern "C" __global__ void ncm_xsum_penalty_init(unsigned* counts, unsigned* bad,
                                                  unsigned batch, unsigned vocab) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<batch*vocab) counts[i]=0;
    if(i<batch) bad[i]=0;
}
extern "C" __global__ void ncm_xsum_penalty_step(const float* logits,const float* allow,const unsigned* counts,
                     const unsigned* history,double* partial,unsigned* partial_id,
                     unsigned* bad,int vocab,int time,int t,int tiles) {
    unsigned tid=threadIdx.x,row=blockIdx.y,v=blockIdx.x*256+tid;
    __shared__ double scores[256];__shared__ unsigned ids[256];
    double score=-CUDART_INF;unsigned id=0xffffffffU;
    if(v<unsigned(vocab)) {
        float x=logits[((size_t)row*time+t)*vocab+v],a=allow[v];
        if(!isfinite(x)||!isfinite(a)||a<0) atomicOr(&bad[row],1);
        else {
            double allowed=__dmul_rn(double(a),double(time));
            double count=double(counts[(size_t)row*vocab+v]);
            double excess=fmax(__dsub_rn(count,allowed),0.0);
            double deficit=fmax(__dsub_rn(allowed,count),0.0);
            double remaining=__ddiv_rn(double(time-t),double(time));
            score=__dsub_rn(double(x),__dmul_rn(2.0,excess));
            score=__dadd_rn(score,__dmul_rn(remaining,deficit));
            const double penalties[4]={-5.0,-2.5,-1.25,-0.625};
            for(int j=0;j<4&&j<t;j++)
                if(history[row*time+t-j-1]==v)score=__dadd_rn(score,penalties[j]);
            id=v;
        }
    }
    scores[tid]=score;ids[tid]=id;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(tid<stride&&(scores[tid+stride]>scores[tid]||
         (scores[tid+stride]==scores[tid]&&ids[tid+stride]<ids[tid]))) {
            scores[tid]=scores[tid+stride];ids[tid]=ids[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){partial[row*tiles+blockIdx.x]=scores[0];partial_id[row*tiles+blockIdx.x]=ids[0];}
}
extern "C" __global__ void ncm_lm1b_penalty_step(const float* logits,const float* allow,const unsigned* counts,
                     const unsigned* history,double* partial,unsigned* partial_id,
                     unsigned* bad,int vocab,int time,int t,int tiles) {
    unsigned tid=threadIdx.x,row=blockIdx.y,v=blockIdx.x*256+tid;
    __shared__ double scores[256];__shared__ unsigned ids[256];
    double score=-CUDART_INF;unsigned id=0xffffffffU;
    if(v<unsigned(vocab)) {
        float x=logits[((size_t)row*time+t)*vocab+v],a=allow[v];
        if(!isfinite(x)||!isfinite(a)||a<0) atomicOr(&bad[row],1);
        else {
            double allowed=__dmul_rn(double(a),double(time));
            double count=double(counts[(size_t)row*vocab+v]);
            double excess=fmax(__dsub_rn(count,allowed),0.0);
            double deficit=fmax(__dsub_rn(allowed,count),0.0);
            double remaining=__ddiv_rn(double(time-t),double(time));
            score=__dsub_rn(double(x),__dmul_rn(11.5,excess));
            score=__dadd_rn(score,__dmul_rn(__dmul_rn(2.0,remaining),deficit));
            const double penalties[4]={-10.0,-5.0,-2.5,-1.25};
            for(int j=0;j<4&&j<t;j++)
                if(history[row*time+t-j-1]==v)score=__dadd_rn(score,penalties[j]);
            id=v;
        }
    }
    scores[tid]=score;ids[tid]=id;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(tid<stride&&(scores[tid+stride]>scores[tid]||
         (scores[tid+stride]==scores[tid]&&ids[tid+stride]<ids[tid]))) {
            scores[tid]=scores[tid+stride];ids[tid]=ids[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){partial[row*tiles+blockIdx.x]=scores[0];partial_id[row*tiles+blockIdx.x]=ids[0];}
}
extern "C" __global__ void ncm_xsum_penalty_finish(const double* partial,const unsigned* partial_id,unsigned* counts,
                       unsigned* out,unsigned* bad,int vocab,int time,int t,int tiles) {
    unsigned tid=threadIdx.x,row=blockIdx.x;
    __shared__ double scores[256];__shared__ unsigned ids[256];
    double best=-CUDART_INF;unsigned id=0xffffffffU;
    for(int tile=tid;tile<tiles;tile+=256){
        double s=partial[row*tiles+tile];unsigned i=partial_id[row*tiles+tile];
        if(s>best||(s==best&&i<id)){best=s;id=i;}
    }
    scores[tid]=best;ids[tid]=id;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1){
        if(tid<stride&&(scores[tid+stride]>scores[tid]||
         (scores[tid+stride]==scores[tid]&&ids[tid+stride]<ids[tid]))){
            scores[tid]=scores[tid+stride];ids[tid]=ids[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        unsigned chosen=ids[0];if(chosen==0xffffffffU){bad[row]=1;chosen=0;}
        out[row*time+t]=bad[row]?0xffffffffU:chosen;
        counts[(size_t)row*vocab+chosen]++;
    }
}

// Warp-wise softmax follows the reduction order of candle-kernels 0.11.1,
// which credits ggml's llama.cpp ggml-cuda.cu at d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318.
// Upstream MIT notices are retained in third-party/candle-MIT.txt and
// third-party/llama-cpp-MIT.txt.
// Scale and broadcast-mask addition are fused into the two input reads.
extern "C" __global__ void ncm_masked_softmax_64(const float* scores,const float* mask,
                                                float* output,unsigned heads,unsigned seq,
                                                unsigned mask_batch) {
    const unsigned row=blockIdx.x,lane=threadIdx.x;
    const unsigned batch=row/(heads*seq), query=row%seq;
    const unsigned mask_row=((mask_batch==1?0:batch)*seq+query)*seq;
    const unsigned score_row=row*seq;
    float maximum=-CUDART_INF_F;
    for(unsigned col=lane;col<seq;col+=32) {
        float scaled=__fadd_rn(__fmul_rn(scores[score_row+col],0.125f),0.0f);
        float value=__fadd_rn(scaled,mask[mask_row+col]);
        maximum=fmaxf(maximum,value);
    }
    for(int shift=16;shift>0;shift>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,shift,32));
    float sum=0.0f;
    for(unsigned col=lane;col<seq;col+=32) {
        float scaled=__fadd_rn(__fmul_rn(scores[score_row+col],0.125f),0.0f);
        float value=__fadd_rn(scaled,mask[mask_row+col]);
        float e=expf(value-maximum);
        sum+=e;
        output[score_row+col]=e;
    }
    for(int shift=16;shift>0;shift>>=1)
        sum+=__shfl_xor_sync(0xffffffff,sum,shift,32);
    float inverse=float(1.0/double(sum));
    for(unsigned col=lane;col<seq;col+=32) output[score_row+col]*=inverse;
}

// Keep a row's masked scores and exponentials in registers. Four independent
// warps share a block without changing the original per-row reduction order.
template <unsigned Iterations>
__device__ void masked_softmax_64_registers(const float* scores, const float* mask,
                                            float* output, unsigned heads, unsigned seq,
                                            unsigned mask_batch) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    if (row >= heads * seq) return;
    const unsigned score_row = (blockIdx.y * heads * seq + row) * seq;
    const unsigned mask_row = ((mask_batch == 1 ? 0 : blockIdx.y) * seq + row % seq) * seq;
    float values[Iterations];
    float maximum = -CUDART_INF_F;
    #pragma unroll
    for (unsigned i = 0; i < Iterations; ++i) {
        const unsigned col = lane + i * 32;
        if (col < seq) {
            const float scaled = __fadd_rn(__fmul_rn(scores[score_row + col], 0.125f), 0.0f);
            values[i] = __fadd_rn(scaled, mask[mask_row + col]);
            maximum = fmaxf(maximum, values[i]);
        }
    }
    #pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, shift, 32));
    }
    float sum = 0.0f;
    #pragma unroll
    for (unsigned i = 0; i < Iterations; ++i) {
        if (lane + i * 32 < seq) {
            values[i] = expf(values[i] - maximum);
            sum = __fadd_rn(sum, values[i]);
        }
    }
    #pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, shift, 32));
    }
    const float inverse = float(1.0 / double(sum));
    #pragma unroll
    for (unsigned i = 0; i < Iterations; ++i) {
        const unsigned col = lane + i * 32;
        if (col < seq) output[score_row + col] = values[i] * inverse;
    }
}

extern "C" __global__ void ncm_masked_softmax_64_registers_17(
    const float* scores, const float* mask, float* output, unsigned heads,
    unsigned seq, unsigned mask_batch) {
    masked_softmax_64_registers<17>(scores, mask, output, heads, seq, mask_batch);
}

extern "C" __global__ void ncm_masked_softmax_64_registers_18(
    const float* scores, const float* mask, float* output, unsigned heads,
    unsigned seq, unsigned mask_batch) {
    masked_softmax_64_registers<18>(scores, mask, output, heads, seq, mask_batch);
}

// Normalization arithmetic adapted from PyTorch v2.15.0; see third-party/pytorch-BSD.txt.
#include <stddef.h>
struct Stats { float mean, sigma, count; };

__device__ Stats online(float value, Stats s) {
    float delta = value - s.mean;
    float count = s.count + 1.0f;
    float mean = s.mean + delta * (1.0f / count);
    return {mean, s.sigma + delta * (value - mean), count};
}

__device__ Stats combine(Stats b, Stats a) {
    float delta = b.mean - a.mean;
    float count = a.count + b.count;
    if (count <= 0.0f) return {0.0f, 0.0f, 0.0f};
    float coef = 1.0f / count;
    float na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean,
            a.sigma + b.sigma + delta * delta * a.count * nb, count};
}

extern "C" __global__ void ncm_pytorch_layernorm_768(const float* input, const float* gamma, const float* beta, float* output) {
    constexpr int width = 768;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = lane + 32 * warp;
    const float* row = input + size_t(blockIdx.x) * width;
    __shared__ float shared[6];
    Stats s{0.0f, 0.0f, 0.0f};
    for (int vector = tid; vector < width / 4; vector += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) s = online(row[4 * vector + j], s);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        Stats other{__shfl_down_sync(0xffffffff, s.mean, offset),
                    __shfl_down_sync(0xffffffff, s.sigma, offset),
                    __shfl_down_sync(0xffffffff, s.count, offset)};
        s = combine(s, other);
    }
    for (int offset = 2; offset > 0; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            int target = warp - offset;
            shared[2 * target] = s.mean; shared[2 * target + 1] = s.sigma; shared[4 + target] = s.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) {
            Stats other{shared[2 * warp], shared[2 * warp + 1], shared[4 + warp]};
            s = combine(s, other);
        }
        __syncthreads();
    }
    if (lane == 0 && warp == 0) { shared[0] = s.mean; shared[1] = s.sigma / float(width); }
    __syncthreads();
    float mean = shared[0], inverse = rsqrtf(shared[1] + 1e-6f);
    for (int vector = tid; vector < width / 4; vector += 128) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = 4 * vector + j;
            output[size_t(blockIdx.x) * width + col] = gamma[col] * (inverse * (row[col] - mean)) + beta[col];
        }
    }
}


// Upstream-order RMS normalization, with separately rounded products.
extern "C" __global__ void ncm_pytorch_rms64(const float* input, const float* weight, float* output, unsigned rows) {
    unsigned lane = threadIdx.x, row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    float a = input[row * 64 + lane], b = input[row * 64 + lane + 32];
    float sum = __fadd_rn(__fmul_rn(a,a), __fmul_rn(b,b));
    for (unsigned offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset, 32));
    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    float scale = rsqrtf(__fadd_rn(__fmul_rn(sum, 0.015625f), 1e-6f));
    output[row * 64 + lane] = __fmul_rn(__fmul_rn(a,scale),weight[lane]);
    output[row * 64 + lane + 32] = __fmul_rn(__fmul_rn(b,scale),weight[lane+32]);
}


struct GroupStats { float mean, m2, count; };

__device__ GroupStats group_add(GroupStats a, float x) {
    float count = a.count + 1, delta = x - a.mean, mean = a.mean + delta / count;
    return {mean, a.m2 + delta * (x - mean), count};
}

__device__ GroupStats group_combine(GroupStats a, GroupStats b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    float delta = b.mean - a.mean, count = a.count + b.count, ratio = b.count / count;
    return {a.mean + delta * ratio, a.m2 + b.m2 + delta * delta * a.count * ratio, count};
}

__device__ GroupStats group_reduce(GroupStats a) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        GroupStats b{__shfl_down_sync(0xffffffff, a.mean, offset),
                     __shfl_down_sync(0xffffffff, a.m2, offset),
                     __shfl_down_sync(0xffffffff, a.count, offset)};
        a = group_combine(a, b);
    }
    return a;
}

extern "C" __global__ void ncm_pytorch_groupnorm(const float* x, float* y) {
    constexpr int width = 576;
    int t = threadIdx.x;
    const float* input = x + size_t(blockIdx.x) * width;
    GroupStats value{0, 0, 0};
    for (int j = t; j < width / 4; j += blockDim.x)
        for (int k = 0; k < 4; ++k) value = group_add(value, input[j * 4 + k]);
    value = group_reduce(value);
    __shared__ GroupStats warps[16];
    if (t % 32 == 0) warps[t / 32] = value;
    __syncthreads();
    value = t < 16 ? warps[t] : GroupStats{0, 0, 0};
    if (t < 32) value = group_reduce(value);
    __shared__ float mean, inverse;
    if (t == 0) {
        mean = value.mean;
        inverse = rsqrtf(value.m2 / value.count + 1e-5f);
    }
    __syncthreads();
    for (int i = t; i < width; i += blockDim.x)
        y[size_t(blockIdx.x) * width + i] = (input[i] - mean) * inverse;
}

// WMT14 initial-cascade widths 48*4 and 48*16. The thread dispatch follows
// pinned PyTorch group_norm_kernel.cu: one warp below 512 elements, else 512.
extern "C" __global__ void ncm_wmt14_groupnorm(const float* x, float* y, int width) {
    int t = threadIdx.x;
    const float* input = x + size_t(blockIdx.x) * width;
    GroupStats value{0, 0, 0};
    for (int j = t; j < width / 4; j += blockDim.x)
        for (int k = 0; k < 4; ++k) value = group_add(value, input[j * 4 + k]);
    value = group_reduce(value);
    __shared__ GroupStats warps[16];
    if (blockDim.x > 32) {
        if (t % 32 == 0) warps[t / 32] = value;
        __syncthreads();
        value = t < 16 ? warps[t] : GroupStats{0, 0, 0};
        if (t < 32) value = group_reduce(value);
    }
    __shared__ float mean, inverse;
    if (t == 0) {
        mean = value.mean;
        inverse = rsqrtf(value.m2 / value.count + 1e-5f);
    }
    __syncthreads();
    for (int i = t; i < width; i += blockDim.x)
        y[size_t(blockIdx.x) * width + i] = (input[i] - mean) * inverse;
}

extern "C" __global__ void ncm_pytorch_gelu(const float* x, float* y, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v * 0.5f * (1.f + erff(v * 0.7071067811865475244f)); }
}
extern "C" __global__ void ncm_xsum_reference_rope(const float* x, const float* c,
                                                     const float* s, float* y,
                                                     unsigned pairs, unsigned seq) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    unsigned row = i / 32, d = i % 32, pos = row % seq, k = row * 64 + d;
    float a = x[k], b = x[k + 32], cv = c[pos * 32 + d], sv = s[pos * 32 + d];
    y[k] = __fadd_rn(__fmul_rn(a, cv), __fmul_rn(-b, sv));
    y[k + 32] = __fadd_rn(__fmul_rn(b, cv), __fmul_rn(a, sv));
}

// WMT14 diagnostic: first index of the maximum in a 151680-way FP32 row.
// NaNs are ignored, matching Candle's fmax reduction followed by equality;
// an all-NaN row returns the same u32::MAX sentinel as the generic path.
extern "C" __global__ void ncm_wmt14_first_max_151680(const float* logits, unsigned int* output) {
    constexpr unsigned width = 151680;
    const unsigned tid = threadIdx.x, row = blockIdx.x;
    const float* input = logits + size_t(row) * width;
    __shared__ float maxima[256];
    __shared__ unsigned indices[256];
    float best = -CUDART_INF_F;
    unsigned index = 0xffffffffu;
    for (unsigned i = tid; i < width; i += 256) {
        float value = input[i];
        if (!isnan(value) && (value > best || (value == best && i < index))) {
            best = value; index = i;
        }
    }
    maxima[tid] = best; indices[tid] = index;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = maxima[tid + stride];
            unsigned other_index = indices[tid + stride];
            if (other > maxima[tid] || (other == maxima[tid] && other_index < indices[tid])) {
                maxima[tid] = other; indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) output[row] = indices[0];
}

extern "C" __global__ void ncm_lm1b_linear_fma(const float* input, float* output,
                                               unsigned batch, unsigned length,
                                               unsigned output_length, unsigned channels) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batch * output_length * channels;
    if (i >= total) return;
    unsigned channel = i % channels;
    unsigned position = (i / channels) % output_length;
    unsigned row = i / (output_length * channels);
    float scale = float(length) / float(output_length);
    float source = fmaxf(__fsub_rn(__fmul_rn(scale, float(position) + 0.5f), 0.5f), 0.0f);
    unsigned left = min(unsigned(source), length - 1);
    unsigned right = min(left + 1, length - 1);
    float weight = fminf(fmaxf(source - float(left), 0.0f), 1.0f);
    float a = input[(row * length + left) * channels + channel];
    float b = input[(row * length + right) * channels + channel];
    float first = __fmul_rn(1.0f - weight, a);
    output[i] = fmaf(weight, b, first);
}

// Generic 16-bit (BF16/FP16) layout copies: no arithmetic on tensor values.
extern "C" __global__ void ncm_pack_qkv_u16(const unsigned short* input, unsigned short* output,
                                            unsigned int batch, unsigned int seq,
                                            unsigned int heads, unsigned int head) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    const unsigned int per_head = seq * head;
    const unsigned int q_heads = 3 * heads;
    if (idx >= per_head || group >= batch * q_heads) return;
    const unsigned int t = idx / head, d = idx % head;
    const unsigned int h = group % heads, q = (group / heads) % 3, b = group / q_heads;
    const unsigned int src = ((b * seq + t) * q_heads + q * heads + h) * head + d;
    const unsigned int dst = (((q * batch + b) * heads + h) * seq + t) * head + d;
    output[dst] = input[src];
}

extern "C" __global__ void ncm_merge_heads_u16(const unsigned short* input, unsigned short* output,
                                               unsigned int batch, unsigned int seq,
                                               unsigned int heads, unsigned int head) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int group = blockIdx.y;
    const unsigned int width = heads * head;
    if (idx >= width || group >= batch * seq) return;
    const unsigned int h = idx / head, d = idx % head, b = group / seq, t = group % seq;
    const unsigned int src = ((b * heads + h) * seq + t) * head + d;
    const unsigned int dst = (b * seq + t) * width + idx;
    output[dst] = input[src];
}

__device__ __forceinline__ float ncm_bf16_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short ncm_f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (unsigned short)((bits >> 16) | 0x0040U);
    }
    unsigned int lsb = (bits >> 16) & 1U;
    return (unsigned short)((bits + 0x7fffU + lsb) >> 16);
}

__device__ __forceinline__ float ncm_f16_to_f32(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ unsigned short ncm_f32_to_f16(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

// Fused scale + optional additive mask + FP32-accumulating warp softmax for BF16/FP16.
// Eliminates [B, H, Q, K] FP32 staging tensors and per-layer mask casts while preserving
// full FP32 softmax precision.
template <bool IsBf16>
__device__ void ncm_fused_softmax_16(const unsigned short* scores, const unsigned short* mask,
                                     unsigned short* output, float scale,
                                     unsigned int total_rows, unsigned int heads,
                                     unsigned int q_len, unsigned int k_len,
                                     unsigned int mask_b, unsigned int mask_q,
                                     unsigned int has_mask) {
    const unsigned int row = blockIdx.x * 4U + (threadIdx.x >> 5);
    const unsigned int lane = threadIdx.x & 31U;
    if (row >= total_rows) return;
    const unsigned int q_idx = row % q_len;
    const unsigned int b_idx = row / (heads * q_len);
    const size_t score_row = (size_t)row * (size_t)k_len;
    const size_t mask_row = has_mask
        ? ((size_t)(mask_b == 1U ? 0U : b_idx) * (size_t)mask_q + (size_t)(mask_q == 1U ? 0U : q_idx)) * (size_t)k_len
        : 0;
    auto load_f32 = [](unsigned short v) -> float {
        return IsBf16 ? ncm_bf16_to_f32(v) : ncm_f16_to_f32(v);
    };
    auto store_16 = [](float v) -> unsigned short {
        return IsBf16 ? ncm_f32_to_bf16(v) : ncm_f32_to_f16(v);
    };

    if (k_len <= 768U) {
        float vals[24];
        float maximum = -CUDART_INF_F;
        #pragma unroll
        for (unsigned int i = 0; i < 24U; ++i) {
            const unsigned int col = lane + i * 32U;
            if (col < k_len) {
                float v = load_f32(scores[score_row + col]) * scale;
                if (has_mask) v += load_f32(mask[mask_row + col]);
                vals[i] = v;
                maximum = fmaxf(maximum, v);
            }
        }
        #pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, shift, 32));
        }
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < 24U; ++i) {
            if (lane + i * 32U < k_len) {
                vals[i] = expf(vals[i] - maximum);
                sum += vals[i];
            }
        }
        #pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) {
            sum += __shfl_xor_sync(0xffffffffU, sum, shift, 32);
        }
        const float inv_sum = 1.0f / sum;
        #pragma unroll
        for (unsigned int i = 0; i < 24U; ++i) {
            const unsigned int col = lane + i * 32U;
            if (col < k_len) {
                output[score_row + col] = store_16(vals[i] * inv_sum);
            }
        }
        return;
    }

    float maximum = -CUDART_INF_F;
    for (unsigned int col = lane; col < k_len; col += 32U) {
        float v = load_f32(scores[score_row + col]) * scale;
        if (has_mask) v += load_f32(mask[mask_row + col]);
        maximum = fmaxf(maximum, v);
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, shift, 32));
    }
    float sum = 0.0f;
    for (unsigned int col = lane; col < k_len; col += 32U) {
        float v = load_f32(scores[score_row + col]) * scale;
        if (has_mask) v += load_f32(mask[mask_row + col]);
        sum += expf(v - maximum);
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        sum += __shfl_xor_sync(0xffffffffU, sum, shift, 32);
    }
    const float inv_sum = 1.0f / sum;
    for (unsigned int col = lane; col < k_len; col += 32U) {
        float v = load_f32(scores[score_row + col]) * scale;
        if (has_mask) v += load_f32(mask[mask_row + col]);
        output[score_row + col] = store_16(expf(v - maximum) * inv_sum);
    }
}

extern "C" __global__ void ncm_fused_softmax_bf16(const unsigned short* scores,
                                                  const unsigned short* mask,
                                                  unsigned short* output, float scale,
                                                  unsigned int total_rows, unsigned int heads,
                                                  unsigned int q_len, unsigned int k_len,
                                                  unsigned int mask_b, unsigned int mask_q,
                                                  unsigned int has_mask) {
    ncm_fused_softmax_16<true>(scores, mask, output, scale, total_rows, heads, q_len, k_len,
                               mask_b, mask_q, has_mask);
}

extern "C" __global__ void ncm_fused_softmax_f16(const unsigned short* scores,
                                                 const unsigned short* mask,
                                                 unsigned short* output, float scale,
                                                 unsigned int total_rows, unsigned int heads,
                                                 unsigned int q_len, unsigned int k_len,
                                                 unsigned int mask_b, unsigned int mask_q,
                                                 unsigned int has_mask) {
    ncm_fused_softmax_16<false>(scores, mask, output, scale, total_rows, heads, q_len, k_len,
                                mask_b, mask_q, has_mask);
}
