/*
 * HIP test for V_TRUNC_F32 (ENC_VOP1, round toward zero (truncate) f32).
 *
 * D.f32 = trunc(S0.f32); inputs are exact so the result is bit-exact.
 *
 * DPP/SDWA note: gem5 (vop1.cc) guards v_trunc_f32 with
 *   panic_if(isSDWAInst()/isDPPInst(), "... not implemented")
 * so exercising those encodings would abort the simulator.  Following the
 * existing v_frexp_mant_f32 precedent, this test covers the base encoding
 * only; the DB keeps dpp/sdwa marked not-working, matching gem5 support.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_trunc_f32 v_trunc_f32.cpp
 * Run:     gem5.opt config.py -- ./v_trunc_f32
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_trunc_f32(const float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float va = a[i], vd;
    __asm__ volatile(
        "v_trunc_f32_e32 %0, %1\n\t"
        : "=v"(vd)
        : "v"(va));
    out[i] = vd;
}

int
main(void)
{
    float *h_a   = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i - 128) * 0.5f + ((i % 3) * 0.25f);
    }

    float *d_a, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_trunc_f32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float va = h_a[i];
        float expected = (truncf(va));
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]\n", i);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("ok\n");
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_out);
    return pass ? 0 : 1;
}
