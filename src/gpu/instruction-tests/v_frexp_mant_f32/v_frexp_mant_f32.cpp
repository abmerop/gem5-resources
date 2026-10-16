/*
 * HIP test for V_FREXP_MANT_F32 (ENC_VOP1, frexp mantissa of an f32).
 *
 * V_FREXP_MANT_F32 returns the normalized mantissa of frexp (per gem5
 * src/arch/amdgpu/common/insts/vop1.cc):
 *
 *   if isinf(S0) || isnan(S0):  VDST = S0           (passthrough)
 *   else:                       VDST = frexp(S0)     (mantissa in [0.5,1))
 *
 * which matches the C library std::frexp mantissa exactly (frexp only adjusts
 * the exponent, so the result is bit-exact and verified with ==).
 *
 * DPP / SDWA variant note: although the ISA defines DPP and SDWA encodings for
 * this instruction, gem5's implementation explicitly rejects them with
 * panic_if(isSDWAInst()/isDPPInst(), "... not implemented for v_frexp_mant_f32").
 * Exercising those forms would abort the simulator rather than validate
 * anything, so this test covers the base encoding only; the DB therefore
 * keeps the dpp/sdwa variants marked not-working, which accurately reflects
 * gem5's current support.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_frexp_mant_f32 v_frexp_mant_f32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_frexp_mant_f32
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
kernel_v_frexp_mant_f32(const float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float va = a[i];
    float vd;
    __asm__ volatile(
        "v_frexp_mant_f32_e32 %0, %1\n\t"
        : "=v"(vd)
        : "v"(va));

    out[i] = vd;
}

static uint32_t
fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

int
main(void)
{
    float *h_a   = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        /* normal positive/negative values across magnitudes */
        float v = (float)((i - 128)) * 0.5f + 0.125f;
        if (i % 37 == 0)
            v = (float)(1 << (i % 20));    /* exact powers of two */
        h_a[i] = v;
    }

    float *d_a, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_frexp_mant_f32,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected;
        if (isinf(h_a[i]) || isnan(h_a[i])) {
            expected = h_a[i];
        } else {
            int e;
            expected = frexpf(h_a[i], &e);
        }
        if (fbits(h_out[i]) != fbits(expected)) {
            fprintf(stderr, "FAIL out[%d]: got %g (0x%08x) expected %g "
                    "(0x%08x) a=%g\n", i, h_out[i], fbits(h_out[i]),
                    expected, fbits(expected), h_a[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%g out[%d]=%g\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_out));
    free(h_a);
    free(h_out);

    return pass ? 0 : 1;
}
