/*
 * HIP test for V_MAD_MIX_F32 (VOP3P mixed-precision fused multiply-add).
 *
 * V_MAD_MIX_F32 computes:
 *
 *   VDST.f32 = fma(SRC0, SRC1, SRC2)
 *
 * where each source can independently be a full f32 register value or
 * either the low or high f16 half of a register, controlled by two
 * instruction-encoded bit-fields:
 *
 *   OPSEL_HI[i] = 0 → SRC[i] is f32 (raw 32-bit float)
 *   OPSEL_HI[i] = 1 → SRC[i] is f16 (converted to f32 before the fma)
 *                      OPSEL[i] = 0 → use bits[15:0]  (low  half)
 *                      OPSEL[i] = 1 → use bits[31:16] (high half)
 *
 * The mnemonic emitted by the LLVM assembler for GFX9 targets is
 * v_fma_mix_f32 (an alias).  gem5's Vega decoder maps both to the same
 * Inst_VOP3P__V_MAD_MIX_F32 implementation.
 *
 * This test exercises three source configurations that the compiler
 * naturally maps to V_MAD_MIX_F32:
 *
 *   Kernel A — f16-lo × f16-lo + f32   op_sel_hi:[1,1,0]
 *     Both multiplicands come from the low half of a packed __half2
 *     register; the addend is a full f32.  This is the classic
 *     "dot product into f32 accumulator" pattern used in mixed-precision
 *     GEMM.
 *
 *   Kernel B — f16-hi × f16-lo + f32   op_sel:[1,0,0] op_sel_hi:[1,1,0]
 *     The first multiplicand comes from the high half of a packed __half2
 *     (OPSEL[0]=1, OPSEL_HI[0]=1); the second from the low half of
 *     another __half2 (OPSEL[1]=0, OPSEL_HI[1]=1); the addend is f32.
 *
 * Build:
 *   hipcc -O2 -o v_mad_mix_f32 v_mad_mix_f32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_mad_mix_f32
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
/* Tolerance for f16→f32 round-trip comparisons. */
static const float EPS = 1e-2f;

/*
 * Kernel A: v_fma_mix_f32  op_sel_hi:[1,1,0]
 *
 * VDST = fma(f16_lo(SRC0), f16_lo(SRC1), SRC2.f32)
 *
 * Each a[i] and b[i] carry two packed f16 values.  The compiler selects
 * the low half of each for the multiply operands and uses c[i] as the
 * full-precision addend.
 */
__global__ void
kernel_a(const __half2 *a, const __half2 *b, const float *c,
         float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Extract the low f16 from each packed pair and widen to f32. */
    float fa = __half2float(__low2half(a[i]));
    float fb = __half2float(__low2half(b[i]));

    /* __fmaf_rn maps directly to v_fma_mix_f32 op_sel_hi:[1,1,0] when
     * the inputs were obtained by half-to-float conversion. */
    out[i] = __fmaf_rn(fa, fb, c[i]);
}

/*
 * Kernel B: v_fma_mix_f32  op_sel:[1,0,0] op_sel_hi:[1,1,0]
 *
 * VDST = fma(f16_hi(SRC0), f16_lo(SRC1), SRC2.f32)
 *
 * The first multiplicand comes from the HIGH half of a[i] (OPSEL[0]=1)
 * while the second comes from the LOW half of b[i] (OPSEL[1]=0).
 */
__global__ void
kernel_b(const __half2 *a, const __half2 *b, const float *c,
         float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float fa = __half2float(__high2half(a[i]));
    float fb = __half2float(__low2half(b[i]));

    out[i] = __fmaf_rn(fa, fb, c[i]);
}

int
main(void)
{
    /* ----- host buffers ----- */
    __half2 *h_a  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_b  = (__half2 *)malloc(N * sizeof(__half2));
    float   *h_c  = (float   *)malloc(N * sizeof(float));
    float   *h_oA = (float   *)malloc(N * sizeof(float));
    float   *h_oB = (float   *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        /* Pack two distinct f16 values into each __half2 so both halves
         * carry meaningful data for kernel B's hi-half path. */
        float lo_a = (float)(i + 1) * 0.5f;        /* low  half of a */
        float hi_a = (float)(i + 1) * 0.25f;        /* high half of a */
        float lo_b = (float)(N - i) * 0.5f;         /* low  half of b */
        float hi_b = (float)(N - i) * 0.125f;       /* high half of b */
        h_a[i] = __halves2half2(__float2half(lo_a), __float2half(hi_a));
        h_b[i] = __halves2half2(__float2half(lo_b), __float2half(hi_b));
        h_c[i] = (float)(i % 16);
    }

    /* ----- device buffers ----- */
    __half2 *d_a;
    __half2 *d_b;
    float   *d_c, *d_oA, *d_oB;
    GPU_CHECK(hipMalloc(&d_a,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_b,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_c,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oA, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oB, N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(float),   hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_a, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_c, d_oA, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_b, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_c, d_oB, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_oA, d_oA, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB, d_oB, N * sizeof(float), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        /* Kernel A: fma(f16_lo(a), f16_lo(b), c) */
        float fa_lo = __half2float(__low2half(h_a[i]));
        float fb_lo = __half2float(__low2half(h_b[i]));
        float expected_a = fmaf(fa_lo, fb_lo, h_c[i]);

        if (fabsf(h_oA[i] - expected_a) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_a[%d]: got %.6f expected %.6f "
                    "(fa=%.4f fb=%.4f c=%.4f)\n",
                    i, h_oA[i], expected_a, fa_lo, fb_lo, h_c[i]);
            pass = 0;
            break;
        }

        /* Kernel B: fma(f16_hi(a), f16_lo(b), c) */
        float fa_hi = __half2float(__high2half(h_a[i]));
        float expected_b = fmaf(fa_hi, fb_lo, h_c[i]);

        if (fabsf(h_oB[i] - expected_b) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_b[%d]: got %.6f expected %.6f "
                    "(fa_hi=%.4f fb=%.4f c=%.4f)\n",
                    i, h_oB[i], expected_b, fa_hi, fb_lo, h_c[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("kernel_a out[0]=%.4f out[%d]=%.4f\n",
               h_oA[0], N - 1, h_oA[N - 1]);
        printf("kernel_b out[0]=%.4f out[%d]=%.4f\n",
               h_oB[0], N - 1, h_oB[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c));
    GPU_CHECK(hipFree(d_oA));
    GPU_CHECK(hipFree(d_oB));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_oA);
    free(h_oB);

    return pass ? 0 : 1;
}
