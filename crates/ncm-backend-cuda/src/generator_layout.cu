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
