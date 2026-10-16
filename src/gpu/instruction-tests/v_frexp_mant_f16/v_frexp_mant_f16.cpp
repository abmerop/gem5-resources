/*
 * HIP test for V_FREXP_MANT_F16 (ENC_VOP1, frexp mantissa of an f16).
 *
 * V_FREXP_MANT_F16 returns the normalized frexp mantissa of S0.f16:
 *   if isinf(S0) || isnan(S0):  D = S0          (passthrough)
 *   else:                       D = frexp(S0)    (mantissa in [0.5,1))
 *
 * This mirrors the established V_FREXP_MANT_F32 semantics (which gem5
 * implements via the C library frexp and which the v_frexp_mant_f32 test
 * confirms).  Inputs here are normal, exactly-representable f16 values (no
 * zero/inf/nan/subnormal), so the frexp mantissa is itself exactly f16
 * representable and the comparison is bit-exact.
 *
 * Variants exercised:
 *   base : v_frexp_mant_f16_e32 vd, va                D = frexp_mant(a[i])
 *   dpp  : v_frexp_mant_f16_dpp ... quad_perm:[1,0,3,2]
 *          D = frexp_mant(a[swizzled]) within each group of 4 lanes
 *   sdwa : v_frexp_mant_f16_sdwa ... src0_sel:WORD_0  D = frexp_mant(a[i])
 *          (low 16 bits selected; same result as base)
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_frexp_mant_f16 v_frexp_mant_f16.cpp
 * Run:     gem5.opt config.py -- ./v_frexp_mant_f16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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

/* base: v_frexp_mant_f16_e32 */
__global__ void
kbase(const __half *a, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint16_t ha;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    uint32_t va = ha, vd;
    __asm__ volatile("v_frexp_mant_f16_e32 %0, %1\n\t"
                     : "=v"(vd) : "v"(va));
    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

/* dpp: quad_perm:[1,0,3,2] swizzles lanes within each group of 4 */
__global__ void
kdpp(const __half *a, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint16_t ha;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    uint32_t va = ha, vd;
    __asm__ volatile("v_frexp_mant_f16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(va));
    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

/* sdwa: src0_sel:WORD_0 selects the low 16 bits (== base result) */
__global__ void
ksdwa(const __half *a, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint16_t ha;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    uint32_t va = ha, vd;
    __asm__ volatile("v_frexp_mant_f16_sdwa %0, %1 dst_sel:DWORD "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t"
                     : "=v"(vd) : "v"(va));
    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

static uint16_t
half_bits(__half h)
{
    uint16_t b;
    __builtin_memcpy(&b, &h, sizeof(b));
    return b;
}

int
main(void)
{
    __half *h_a    = (__half *)malloc(N * sizeof(__half));
    __half *h_base = (__half *)malloc(N * sizeof(__half));
    __half *h_dpp  = (__half *)malloc(N * sizeof(__half));
    __half *h_sdwa = (__half *)malloc(N * sizeof(__half));

    /* Normal, exactly-representable f16 values, mixed sign, no 0/inf/nan. */
    for (int i = 0; i < N; i++) {
        float v = (float)((i % 31) + 1) * 0.5f;     /* 0.5 .. 16.0 */
        if (i & 1)
            v = -v;
        h_a[i] = __float2half(v);
    }

    __half *d_a, *d_base, *d_dpp, *d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_base, N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_dpp,  N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_sdwa, N * sizeof(__half)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_a, d_base, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,  dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_a, d_dpp, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_a, d_sdwa, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_base, d_base, N * sizeof(__half), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,  d_dpp,  N * sizeof(__half), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa, d_sdwa, N * sizeof(__half), hipMemcpyDeviceToHost));

    /* quad_perm:[1,0,3,2] source-lane mapping within each group of 4 */
    const int qp[4] = {1, 0, 3, 2};

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int e;
        /* base */
        float m = frexpf(__half2float(h_a[i]), &e);
        uint16_t eb = half_bits(__float2half(m));
        if (half_bits(h_base[i]) != eb) {
            fprintf(stderr, "FAIL base[%d]: got 0x%04x expected 0x%04x\n",
                    i, half_bits(h_base[i]), eb);
            pass = 0; break;
        }
        /* sdwa == base */
        if (half_bits(h_sdwa[i]) != eb) {
            fprintf(stderr, "FAIL sdwa[%d]: got 0x%04x expected 0x%04x\n",
                    i, half_bits(h_sdwa[i]), eb);
            pass = 0; break;
        }
        /* dpp: source lane swizzled within group of 4 */
        int src = (i & ~3) + qp[i & 3];
        float md = frexpf(__half2float(h_a[src]), &e);
        uint16_t ed = half_bits(__float2half(md));
        if (half_bits(h_dpp[i]) != ed) {
            fprintf(stderr, "FAIL dpp[%d]: got 0x%04x expected 0x%04x "
                    "(src=%d)\n", i, half_bits(h_dpp[i]), ed, src);
            pass = 0; break;
        }
    }

    if (pass)
        printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",
               half_bits(h_base[1]), half_bits(h_dpp[1]), half_bits(h_sdwa[1]));
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_dpp)); GPU_CHECK(hipFree(d_sdwa));
    free(h_a); free(h_base); free(h_dpp); free(h_sdwa);
    return pass ? 0 : 1;
}
