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
