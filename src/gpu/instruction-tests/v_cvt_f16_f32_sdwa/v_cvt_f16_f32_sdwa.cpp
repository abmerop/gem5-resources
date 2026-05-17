/*
 * HIP test for V_CVT_F16_F32 with SDWA dst_sel:WORD_1.
 *
 * V_CVT_F16_F32 converts a 32-bit float to a 16-bit float:
 *
 *   VDST.f16 = f32_to_f16(SRC0.f32)
 *
 * Standard (_e32) form — writes the f16 result into bits[15:0]:
 *
 *   v_cvt_f16_f32_e32  vdst, vsrc
 *   vdst[31:16] = 0 (zeroed), vdst[15:0] = f16(vsrc)
 *
 * SDWA form with dst_sel:WORD_1 — writes the f16 result into bits[31:16],
 * leaving bits[15:0] from the prior content of vdst (UNUSED_PRESERVE) or
 * zeroed (UNUSED_PAD):
 *
 *   v_cvt_f16_f32_sdwa  vdst, vsrc  dst_sel:WORD_1 dst_unused:UNUSED_PAD \
 *                                    src0_sel:DWORD
 *   vdst[31:16] = f16(vsrc),  vdst[15:0] = 0 (UNUSED_PAD)
 *
 * This is the compiler's preferred sequence for building a packed __half2
 * from two f32 values without a separate pack instruction:
 *
 *   lo_half → v_cvt_f16_f32_e32   vdst, src_lo    (writes bits[15:0])
 *   hi_half → v_cvt_f16_f32_sdwa  vdst, src_hi  dst_sel:WORD_1  (bits[31:16])
 *
 * The two instructions share the same destination VGPR; by the time the
 * second one completes, vdst holds a fully packed __half2.  This is more
 * efficient than converting separately and then using v_pack_b32_f16.
 *
 * SDWA DST_SEL encoding:
 *   4 (WORD_0): write result to bits[15:0]
 *   5 (WORD_1): write result to bits[31:16]   ← used here
 *
 * SDWA DST_UNUSED encoding (what happens to the other word):
 *   0 (UNUSED_PAD):      pad with zeros
 *   2 (UNUSED_PRESERVE): keep the previous value
 *
 * Build:
 *   hipcc -O2 -o v_cvt_f16_f32_sdwa v_cvt_f16_f32_sdwa.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_f16_f32_sdwa
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
static const float EPS      = 1e-3f;

/*
 * Kernel A: pack two f32 values into one __half2 using the SDWA pattern.
 *
 * __halves2half2(lo, hi) is compiled to:
 *
 *   v_cvt_f16_f32_e32   vdst, src_lo      ; vdst[15:0]  = f16(lo)
 *   v_cvt_f16_f32_sdwa  vdst, src_hi      ; vdst[31:16] = f16(hi)
 *       dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
 *
 * The second instruction writes f16(hi) into the HIGH 16 bits of the same
 * destination register that already holds f16(lo) in its low 16 bits.
 */
__global__ void
kernel_pack(const float *lo_in, const float *hi_in, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __half lo = __float2half(lo_in[i]);   /* → v_cvt_f16_f32_e32       */
    __half hi = __float2half(hi_in[i]);   /* → v_cvt_f16_f32_sdwa WORD_1 */
    out[i] = __halves2half2(lo, hi);
}

/*
 * Kernel B: same operation via __floats2half2_rn for contrast.
 *
 * The compiler also emits v_cvt_f16_f32_sdwa dst_sel:WORD_1 here.
 */
__global__ void
kernel_f2h2(const float *a, const float *b, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    out[i] = __floats2half2_rn(a[i], b[i]);
}

int
main(void)
{
    /* ----- host buffers ----- */
    float   *h_lo  = (float   *)malloc(N * sizeof(float));
    float   *h_hi  = (float   *)malloc(N * sizeof(float));
    __half2 *h_oA  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_oB  = (__half2 *)malloc(N * sizeof(__half2));

    for (int i = 0; i < N; i++) {
        /* Use distinct lo and hi values so a mis-placement is detectable. */
        h_lo[i] = (float)(i + 1) * 0.5f;
        h_hi[i] = (float)(i + 1) * 2.0f;
    }

    /* ----- device buffers ----- */
    float   *d_lo, *d_hi;
    __half2 *d_oA, *d_oB;

    GPU_CHECK(hipMalloc(&d_lo,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_hi,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oA,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oB,  N * sizeof(__half2)));

    GPU_CHECK(hipMemcpy(d_lo, h_lo, N * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_hi, h_hi, N * sizeof(float), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_pack, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_lo, d_hi, d_oA, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_f2h2, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_lo, d_hi, d_oB, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_oA, d_oA, N * sizeof(__half2), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB, d_oB, N * sizeof(__half2), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float exp_lo = __half2float(__float2half(h_lo[i]));
        float exp_hi = __half2float(__float2half(h_hi[i]));

        float got_lo_A = __half2float(__low2half(h_oA[i]));
        float got_hi_A = __half2float(__high2half(h_oA[i]));
        float got_lo_B = __half2float(__low2half(h_oB[i]));
        float got_hi_B = __half2float(__high2half(h_oB[i]));

        if (fabsf(got_lo_A - exp_lo) > EPS || fabsf(got_hi_A - exp_hi) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_pack[%d]: lo=%g(exp %g) hi=%g(exp %g)\n",
                    i, got_lo_A, exp_lo, got_hi_A, exp_hi);
            pass = 0; break;
        }
        if (fabsf(got_lo_B - exp_lo) > EPS || fabsf(got_hi_B - exp_hi) > EPS) {
            fprintf(stderr,
                    "FAIL kernel_f2h2[%d]: lo=%g(exp %g) hi=%g(exp %g)\n",
                    i, got_lo_B, exp_lo, got_hi_B, exp_hi);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("kernel_pack[0]: lo=%.3f hi=%.3f\n",
               __half2float(__low2half(h_oA[0])),
               __half2float(__high2half(h_oA[0])));
        printf("kernel_f2h2[0]: lo=%.3f hi=%.3f\n",
               __half2float(__low2half(h_oB[0])),
               __half2float(__high2half(h_oB[0])));
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_lo));  GPU_CHECK(hipFree(d_hi));
    GPU_CHECK(hipFree(d_oA));  GPU_CHECK(hipFree(d_oB));
    free(h_lo); free(h_hi); free(h_oA); free(h_oB);

    return pass ? 0 : 1;
}
