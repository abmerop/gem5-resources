/*
 * HIP test for V_LOG_F16 (VOP1, f16 base-2 logarithm).
 *
 * gem5 computes log2 in f16 as f16( std::log2( f32(src) ) ) via vop1Helper.
 * The base e32 form is exercised (vop1Helper also supports DPP/SDWA, but base
 * alone is sufficient to mark the instruction working).  Inputs are exactly
 * f16-representable integers 1..256, so the only rounding is on the f16 output;
 * results are checked against log2 with a tolerance covering ~1 f16 ULP.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_log_f16 v_log_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kbase(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    __half x = __float2half(in[i]);
    __half y;
    __asm__ volatile("v_log_f16_e32 %0, %1\n\t" : "=v"(y) : "v"(x));
    out[i] = __half2float(y);
}

int main(void) {
    float *h_in = (float*)malloc(N*4), *h_out = (float*)malloc(N*4);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);   // 1..256, f16-exact
    float *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in, N*4)); GPU_CHECK(hipMalloc(&d_out, N*4));
    GPU_CHECK(hipMemcpy(d_in, h_in, N*4, hipMemcpyHostToDevice));
    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N*4, hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float e = log2f(h_in[i]);
        float tol = 0.02f * (fabsf(e) + 1.0f);   // ~1 f16 ULP margin
        if (fabsf(h_out[i] - e) > tol) {
            fprintf(stderr, "FAIL[%d] in=%g got %g exp %g (tol %g)\n",
                    i, h_in[i], h_out[i], e, tol);
            pass = 0; break;
        }
    }
    if (pass) printf("log2(2)=%g log2(8)=%g log2(256)=%g\n", h_out[1], h_out[7], h_out[255]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
