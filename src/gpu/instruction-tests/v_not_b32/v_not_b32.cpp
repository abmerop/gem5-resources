/*
 * HIP test for V_NOT_B32 (ENC_VOP1, bitwise NOT of a 32-bit value).
 *
 * D.u32 = ~S0.u32.
 *
 * DPP/SDWA note: gem5 (vop1.cc) guards v_not_b32 with
 *   panic_if(isSDWAInst()/isDPPInst(), "... not implemented")
 * so exercising those encodings would abort the simulator.  Following the
 * existing v_frexp_mant_f32 precedent, this test covers the base encoding
 * only; the DB keeps dpp/sdwa marked not-working, matching gem5 support.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_not_b32 v_not_b32.cpp
 * Run:     gem5.opt config.py -- ./v_not_b32
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
kernel_v_not_b32(const uint32_t *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i], vd;
    __asm__ volatile(
        "v_not_b32_e32 %0, %1\n\t"
        : "=v"(vd)
        : "v"(va));
    out[i] = vd;
}

int
main(void)
{
    uint32_t *h_a   = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint32_t)i * 2654435761u ^ 0x5a5a5a5au;
    }

    uint32_t *d_a, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_not_b32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t va = h_a[i];
        uint32_t expected = (~va);
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
