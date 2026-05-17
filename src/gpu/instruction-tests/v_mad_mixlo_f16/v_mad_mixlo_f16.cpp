/*
 * HIP test for V_MAD_MIXLO_F16 (VOP3P mixed-precision fused multiply-add,
 * f16 result written to the low 16 bits of the destination register).
 *
 * V_MAD_MIXLO_F16 computes:
 *
 *   tmp.f32  = fma(SRC0, SRC1, SRC2)          (same source selection as V_MAD_MIX_F32)
 *   VDST[15:0] = f32_to_f16(tmp)              (narrow and write to low half)
 *   VDST[31:16] unchanged                      (high half of destination preserved)
 *
 * Source operand selection is identical to V_MAD_MIX_F32:
 *   OPSEL_HI[i] = 0 → SRC[i] is f32 (raw 32-bit float)
 *   OPSEL_HI[i] = 1 → SRC[i] is f16 (converted to f32 before the fma)
 *                      OPSEL[i] = 0 → bits[15:0]  (low  half)
 *                      OPSEL[i] = 1 → bits[31:16] (high half)
 *
 * The distinguishing characteristic vs V_MAD_MIX_F32:
 *   - Result is converted back to f16 and stored in the destination's LOW 16 bits.
 *   - This is the building block for packed f16 output: MIXLO writes the
 *     first element and MIXHI writes the second element into bits[31:16],
 *     together producing a packed __half2 result.
 *
 * The mnemonic emitted by the LLVM assembler for GFX9 targets is
 * v_fma_mixlo_f16 (an alias).  gem5 decodes this as
 * Inst_VOP3P__V_MAD_MIXLO_F16.
 *
 * This test exercises two source configurations:
 *
 *   Kernel A — f32 × f16-lo + f16-lo → f16-lo   op_sel_hi:[0,1,1]
 *     SRC0 is a full f32; SRC1 and SRC2 are the low f16 halves of their
 *     registers.  The natural pattern for accumulating a full-precision
 *     scale factor (SRC0) into a mixed-precision multiply-add, storing the
 *     result as f16.
 *
 *   Kernel B — f16-hi × f16-lo + f32 → f16-lo   op_sel:[1,0,0] op_sel_hi:[1,1,0]
 *     SRC0 comes from the HIGH half of a packed __half2 (OPSEL[0]=1,
 *     OPSEL_HI[0]=1); SRC1 from the LOW half of another __half2; SRC2 is
 *     a full f32 accumulator.  Demonstrates independent half-selection per
 *     source operand.
 *
 * Build:
 *   hipcc -O2 -o v_mad_mixlo_f16 v_mad_mixlo_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_mad_mixlo_f16
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
/* Tolerance: f16 has ~3 decimal digits of precision. */
static const float EPS = 1e-2f;

/*
 * Kernel A: v_fma_mixlo_f16  op_sel_hi:[0,1,1]
 *
 * VDST[15:0] = f32_to_f16( fma(SRC0.f32, f16_lo(SRC1), f16_lo(SRC2)) )
 *
 * SRC0 is a full f32 scale factor; SRC1 and SRC2 are low-half f16 values
 * loaded from __half arrays.  The compiler emits v_fma_mixlo_f16 because
 * SRC1 and SRC2 are f16 (OPSEL_HI[1:2]=1) while SRC0 is f32 (OPSEL_HI[0]=0),
 * and the result must be stored as f16.
 */
__global__ void
kernel_a(const float *scale, const __half *b, const __half *c,
         __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float fs  = scale[i];
    float fb  = __half2float(b[i]);
    float fc  = __half2float(c[i]);

    /* __fmaf_rn then __float2half → compiler fuses into v_fma_mixlo_f16
     * op_sel_hi:[0,1,1]: SRC0=fs (f32), SRC1=b[i] (f16 lo), SRC2=c[i] (f16 lo) */
    out[i] = __float2half(__fmaf_rn(fs, fb, fc));
}

/*
 * Kernel B: v_fma_mixlo_f16  op_sel:[1,0,0] op_sel_hi:[1,1,0]
 *
 * VDST[15:0] = f32_to_f16( fma(f16_hi(SRC0), f16_lo(SRC1), SRC2.f32) )
 *
 * SRC0's HIGH half (OPSEL[0]=1, OPSEL_HI[0]=1) and SRC1's LOW half
 * (OPSEL[1]=0, OPSEL_HI[1]=1) are the f16 multiplicands; SRC2 is a full
 * f32 addend (OPSEL_HI[2]=0).
 */
__global__ void
kernel_b(const __half2 *a, const __half *b, const float *acc,
         __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float fa  = __half2float(__high2half(a[i]));
    float fb  = __half2float(b[i]);
    float fc  = acc[i];

    /* op_sel:[1,0,0] op_sel_hi:[1,1,0]: SRC0=hi(a) (f16 hi), SRC1=b (f16 lo),
     * SRC2=acc (f32) */
    out[i] = __float2half(__fmaf_rn(fa, fb, fc));
}

int
main(void)
{
    /* ----- host buffers ----- */
    float   *h_scale = (float   *)malloc(N * sizeof(float));
    __half2 *h_a2    = (__half2 *)malloc(N * sizeof(__half2));
    __half  *h_b     = (__half  *)malloc(N * sizeof(__half));
    __half  *h_c     = (__half  *)malloc(N * sizeof(__half));
    float   *h_acc   = (float   *)malloc(N * sizeof(float));
    __half  *h_oA    = (__half  *)malloc(N * sizeof(__half));
    __half  *h_oB    = (__half  *)malloc(N * sizeof(__half));

    for (int i = 0; i < N; i++) {
        /* Keep values small to avoid f16 overflow (max ~65504). */
        h_scale[i] = (float)(i % 8 + 1) * 0.5f;
        float lo_a  = (float)(i % 16 + 1) * 0.25f;
        float hi_a  = (float)(i % 16 + 1) * 0.125f;
        h_a2[i] = __halves2half2(__float2half(lo_a), __float2half(hi_a));
        h_b[i]  = __float2half((float)(N - i) * 0.5f);
        h_c[i]  = __float2half((float)(i % 8));
        h_acc[i] = (float)(i % 16);
    }

    /* ----- device buffers ----- */
    float   *d_scale;
    __half2 *d_a2;
    __half  *d_b, *d_c, *d_oA, *d_oB;
    float   *d_acc;

    GPU_CHECK(hipMalloc(&d_scale, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_a2,   N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_b,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_c,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_acc,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oA,   N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_oB,   N * sizeof(__half)));

    GPU_CHECK(hipMemcpy(d_scale, h_scale, N * sizeof(float),   hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_a2,    h_a2,    N * sizeof(__half2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,     h_b,     N * sizeof(__half),  hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,     h_c,     N * sizeof(__half),  hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_acc,   h_acc,   N * sizeof(float),   hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_a, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_scale, d_b, d_c, d_oA, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_b, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a2, d_b, d_acc, d_oB, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_oA, d_oA, N * sizeof(__half), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB, d_oB, N * sizeof(__half), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        /* Kernel A: fma(scale, b, c) -> f16 */
        float fb  = __half2float(h_b[i]);
        float fc  = __half2float(h_c[i]);
        float exp_a = __half2float(
            __float2half(fmaf(h_scale[i], fb, fc)));

        if (fabsf(__half2float(h_oA[i]) - exp_a) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_a[%d]: got %.4f expected %.4f "
                    "(scale=%.3f b=%.3f c=%.3f)\n",
                    i, __half2float(h_oA[i]), exp_a,
                    h_scale[i], fb, fc);
            pass = 0;
            break;
        }

        /* Kernel B: fma(hi(a), b, acc) -> f16 */
        float fa_hi = __half2float(__high2half(h_a2[i]));
        float exp_b = __half2float(
            __float2half(fmaf(fa_hi, fb, h_acc[i])));

        if (fabsf(__half2float(h_oB[i]) - exp_b) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_b[%d]: got %.4f expected %.4f "
                    "(fa_hi=%.3f b=%.3f acc=%.3f)\n",
                    i, __half2float(h_oB[i]), exp_b,
                    fa_hi, fb, h_acc[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("kernel_a out[0]=%.4f out[%d]=%.4f\n",
               __half2float(h_oA[0]), N - 1, __half2float(h_oA[N - 1]));
        printf("kernel_b out[0]=%.4f out[%d]=%.4f\n",
               __half2float(h_oB[0]), N - 1, __half2float(h_oB[N - 1]));
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_scale));
    GPU_CHECK(hipFree(d_a2));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c));
    GPU_CHECK(hipFree(d_acc));
    GPU_CHECK(hipFree(d_oA));
    GPU_CHECK(hipFree(d_oB));
    free(h_scale);
    free(h_a2);
    free(h_b);
    free(h_c);
    free(h_acc);
    free(h_oA);
    free(h_oB);

    return pass ? 0 : 1;
}
