// Standalone strict-FP32 cuBLASLt feasibility diagnostic; not production code.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_ok(cudaError_t s, const char* where) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(s));
}
static void blas_ok(cublasStatus_t s, const char* where) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + ": cublas status " + std::to_string(s));
}
template<class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(std::size_t count) { cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc"); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { cuda_ok(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
};
struct LtHandle {
    cublasLtHandle_t value{};
    LtHandle() { blas_ok(cublasLtCreate(&value), "cublasLtCreate"); }
    ~LtHandle() { if (value) cublasLtDestroy(value); }
    LtHandle(const LtHandle&) = delete;
};
struct Layout {
    cublasLtMatrixLayout_t value{};
    Layout(uint64_t rows, uint64_t cols, int64_t ld) {
        blas_ok(cublasLtMatrixLayoutCreate(&value, CUDA_R_32F, rows, cols, ld), "layout create");
    }
    ~Layout() { if (value) cublasLtMatrixLayoutDestroy(value); }
    Layout(const Layout&) = delete;
};
struct Descriptor {
    cublasLtMatmulDesc_t value{};
    explicit Descriptor(const float* bias) {
        blas_ok(cublasLtMatmulDescCreate(&value, CUBLAS_COMPUTE_32F, CUDA_R_32F), "descriptor create");
        try {
            cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            blas_ok(cublasLtMatmulDescSetAttribute(value, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "transpose A");
            blas_ok(cublasLtMatmulDescSetAttribute(value, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "transpose B");
            blas_ok(cublasLtMatmulDescSetAttribute(value, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "bias epilogue");
            blas_ok(cublasLtMatmulDescSetAttribute(value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)), "bias pointer");
        } catch (...) { cublasLtMatmulDescDestroy(value); value = nullptr; throw; }
    }
    ~Descriptor() { if (value) cublasLtMatmulDescDestroy(value); }
    Descriptor(const Descriptor&) = delete;
};
struct Preference {
    cublasLtMatmulPreference_t value{};
    explicit Preference(std::size_t bytes) {
        blas_ok(cublasLtMatmulPreferenceCreate(&value), "preference create");
        try { blas_ok(cublasLtMatmulPreferenceSetAttribute(value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)), "workspace preference"); }
        catch (...) { cublasLtMatmulPreferenceDestroy(value); value = nullptr; throw; }
    }
    ~Preference() { if (value) cublasLtMatmulPreferenceDestroy(value); }
    Preference(const Preference&) = delete;
};
struct Candidate {
    bool available = false;
    std::size_t requested = 0;
    cublasLtMatmulHeuristicResult_t heuristic{};
    double heuristic_ms = 0;
    double first_gpu_ms = 0;
    double first_wall_ms = 0;
    std::vector<float> gpu_ms;
    std::vector<double> wall_ms;
};
static Candidate select(LtHandle& handle, Descriptor& desc, Layout& a, Layout& b, Layout& c, std::size_t limit) {
    Preference pref(limit);
    Candidate out; out.requested = limit;
    int returned = 0;
    auto start = std::chrono::steady_clock::now();
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle.value, desc.value, a.value, b.value, c.value, c.value, pref.value, 1, &out.heuristic, &returned);
    out.heuristic_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || (status == CUBLAS_STATUS_SUCCESS && returned == 0)) return out;
    blas_ok(status, "heuristic");
    blas_ok(out.heuristic.state, "heuristic result");
    if (returned != 1 || out.heuristic.workspaceSize > limit) throw std::runtime_error("invalid heuristic result");
    out.available = true;
    return out;
}
static void launch(LtHandle& h, Descriptor& d, Layout& a, Layout& b, Layout& c,
                   Candidate& choice, const float* w, const float* x, float* y, void* workspace,
                   Event& begin, Event& end, float& gpu_ms, double& wall_ms) {
    constexpr float alpha = 1.0f, beta = 0.0f;
    cuda_ok(cudaEventRecord(begin.value), "event start");
    auto start = std::chrono::steady_clock::now();
    blas_ok(cublasLtMatmul(h.value, d.value, &alpha, w, a.value, x, b.value, &beta, y, c.value, y, c.value,
                           &choice.heuristic.algo, workspace, choice.requested, nullptr), "matmul");
    cuda_ok(cudaEventRecord(end.value), "event end");
    cuda_ok(cudaEventSynchronize(end.value), "event synchronize");
    wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    cuda_ok(cudaEventElapsedTime(&gpu_ms, begin.value, end.value), "event elapsed");
}
static void array(const std::vector<float>& v) {
    std::cout << '[';
    for (std::size_t i=0;i<v.size();++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}
static void array(const std::vector<double>& v) {
    std::cout << '[';
    for (std::size_t i=0;i<v.size();++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}
static void result(const Candidate& c) {
    std::cout << "{\"available\":" << (c.available ? "true" : "false") << ",\"requested_workspace_bytes\":" << c.requested
              << ",\"heuristic_ms\":" << c.heuristic_ms;
    if (c.available) {
        std::cout << ",\"algorithm_workspace_bytes\":" << c.heuristic.workspaceSize << ",\"first_gpu_ms\":" << c.first_gpu_ms
                  << ",\"first_wall_ms\":" << c.first_wall_ms << ",\"gpu_ms\":";
        array(c.gpu_ms); std::cout << ",\"wall_ms\":"; array(c.wall_ms);
    }
    std::cout << '}';
}
static void one_case(LtHandle& h, Buffer<std::uint8_t>& workspace, int batch, int k, int m) {
    const int n = batch * 89;
    const std::size_t xn = std::size_t(n)*k, wn = std::size_t(m)*k, yn = std::size_t(n)*m;
    std::vector<float> hx(xn), hw(wn), hb(m), baseline(yn), candidate(yn);
    for (std::size_t i=0;i<xn;++i) hx[i] = (int(i*17%251)-125)/251.0f;
    for (std::size_t i=0;i<wn;++i) hw[i] = (int(i*73%509)-254)/509.0f;
    for (int i=0;i<m;++i) hb[i] = (i%11-5)/11.0f;
    Buffer<float> dx(xn), dw(wn), db(m), dy(yn);
    cuda_ok(cudaMemcpy(dx.ptr,hx.data(),xn*sizeof(float),cudaMemcpyHostToDevice), "copy x");
    cuda_ok(cudaMemcpy(dw.ptr,hw.data(),wn*sizeof(float),cudaMemcpyHostToDevice), "copy w");
    cuda_ok(cudaMemcpy(db.ptr,hb.data(),m*sizeof(float),cudaMemcpyHostToDevice), "copy bias");
    Layout a(k,m,k), b(k,n,k), c(m,n,m);
    Descriptor desc(db.ptr);
    Candidate zero = select(h,desc,a,b,c,0), four = select(h,desc,a,b,c,4*1024*1024);
    if (!four.available) throw std::runtime_error("4 MiB baseline unavailable");
    Event begin, end;
    float gpu = 0; double wall = 0;
    launch(h,desc,a,b,c,four,dw.ptr,dx.ptr,dy.ptr,workspace.ptr,begin,end,gpu,wall);
    four.first_gpu_ms=gpu; four.first_wall_ms=wall;
    cuda_ok(cudaMemcpy(baseline.data(),dy.ptr,yn*sizeof(float),cudaMemcpyDeviceToHost), "copy baseline");
    double max_ref_error = 0;
    for (int ni : {0,n/2,n-1}) for (int mi : {0,m/2,m-1}) {
        double expected = hb[mi];
        for (int j=0;j<k;++j) expected += double(hx[std::size_t(ni)*k+j])*double(hw[std::size_t(mi)*k+j]);
        const double actual = baseline[std::size_t(ni)*m+mi], error = std::abs(actual-expected);
        if (!std::isfinite(actual) || error > 1e-4 + 2e-4*std::abs(expected)) throw std::runtime_error("host arithmetic mismatch");
        max_ref_error = std::max(max_ref_error,error);
    }
    double max_abs = 0; std::size_t differing = 0;
    if (zero.available) {
        launch(h,desc,a,b,c,zero,dw.ptr,dx.ptr,dy.ptr,nullptr,begin,end,gpu,wall);
        zero.first_gpu_ms=gpu; zero.first_wall_ms=wall;
        cuda_ok(cudaMemcpy(candidate.data(),dy.ptr,yn*sizeof(float),cudaMemcpyDeviceToHost), "copy candidate");
        for (std::size_t i=0;i<yn;++i) {
            const double a0=baseline[i], a1=candidate[i], error=std::abs(a0-a1);
            if (!std::isfinite(a0) || !std::isfinite(a1) || error > 1e-4 + 2e-4*std::abs(a0)) throw std::runtime_error("workspace result mismatch");
            if (a0 != a1) ++differing;
            max_abs=std::max(max_abs,error);
        }
    }
    for (int i=0;i<35;++i) {
        for (int j=0;j<2;++j) {
            Candidate& choice = ((i+j)%2==0) ? four : zero;
            if (!choice.available) continue;
            launch(h,desc,a,b,c,choice,dw.ptr,dx.ptr,dy.ptr,choice.requested ? workspace.ptr : nullptr,begin,end,gpu,wall);
            if (i>=5) { choice.gpu_ms.push_back(gpu); choice.wall_ms.push_back(wall); }
        }
    }
    std::cout << "{\"batch\":" << batch << ",\"n\":" << n << ",\"input_width\":" << k << ",\"output_width\":" << m
              << ",\"reference_elements\":9,\"max_reference_abs\":" << max_ref_error
              << ",\"output_elements\":" << yn << ",\"differing_values\":" << differing << ",\"max_abs_difference\":" << max_abs
              << ",\"zero\":"; result(zero); std::cout << ",\"four_mib\":"; result(four); std::cout << '}';
}
int main() {
    try {
        cuda_ok(cudaSetDevice(0), "device");
        cudaDeviceProp props{}; cuda_ok(cudaGetDeviceProperties(&props,0), "device properties");
        int runtime=0, driver=0; cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime version"); cuda_ok(cudaDriverGetVersion(&driver), "driver version");
        LtHandle handle; Buffer<std::uint8_t> workspace(4*1024*1024);
        std::cout << std::setprecision(12) << "{\"schema_version\":1,\"compute_capability\":[" << props.major << ',' << props.minor
                  << "],\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver << ",\"warmups\":5,\"trials\":30,\"cases\":[";
        bool first=true;
        for (int batch : {1,2,4}) for (auto shape : {std::pair<int,int>{256,768}, {256,256}, {256,1024}, {1024,256}}) {
            if (!first) std::cout << ','; first=false; one_case(handle,workspace,batch,shape.first,shape.second);
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "diagnostic failed: " << e.what() << '\n'; return 1; }
}
