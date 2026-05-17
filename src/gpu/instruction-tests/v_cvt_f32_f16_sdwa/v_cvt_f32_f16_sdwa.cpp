/*
 * HIP test for V_CVT_F32_F16 with SDWA src0_sel:WORD_1.
 *
 * V_CVT_F32_F16 converts a 16-bit float to a 32-bit float:
 *
 *   VDST.f32 = f16_to_f32(SRC0[15:0])   (standard VOP1 form, _e32 suffix)
 *
 * With the SDWA (Sub-DWORD Addressing) modifier the source word can be
 * selected independently of the register allocation:
 *
 *   src0_sel:WORD_0 → use bits[15:0]  of SRC0  (low  f16)  ← same as _e32
 *   src0_sel:WORD_1 → use bits[31:16] of SRC0  (high f16)
 *
 * The SDWA form is emitted when the compiler needs to extract the HIGH half
 * of a packed __half2 register and convert it to f32.  Without SDWA the
 * compiler would need a separate shift instruction (v_lshrrev_b32) to move
 * bits[31:16] into bits[15:0] before the conversion; SDWA eliminates that
 * extra instruction by performing the selection inside the converter itself.
 *
 * Assembly forms produced by this test:
 *
 *   v_cvt_f32_f16_sdwa vdst, vsrc  dst_sel:DWORD dst_unused:UNUSED_PAD \
 *                                   src0_sel:WORD_1
 *     ↑ emitted for __half2float(__high2half(x))
 *
 *   v_cvt_f32_f16_e32 vdst, vsrc
 *     ↑ emitted for __half2float(__low2half(x))   (shown for contrast)
 *
 * Three kernels are included to exercise different SDWA use cases:
 *
 *   Kernel A — single high-half extraction:
 *     out[i] = __half2float(__high2half(a[i]))
 *     Produces exactly one v_cvt_f32_f16_sdwa WORD_1.
 *
 *   Kernel B — parallel lo/hi extraction:
 *     lo[i] = __half2float(__low2half(a[i]))    → v_cvt_f32_f16_e32
 *     hi[i] = __half2float(__high2half(a[i]))   → v_cvt_f32_f16_sdwa WORD_1
 *     Shows both encodings side-by-side.
 *
 *   Kernel C — arithmetic on both halves:
 *     out[i] = (__half2float(__low2half(a[i]))  + __half2float(__low2half(b[i])))
 *            + (__half2float(__high2half(a[i])) + __half2float(__high2half(b[i])))
 *     Produces two v_cvt_f32_f16_sdwa WORD_1 (one per __half2 operand).
 *
 * Build:
 *   hipcc -O2 -o v_cvt_f32_f16_sdwa v_cvt_f32_f16_sdwa.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_f32_f16_sdwa
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
 * Kernel A: v_cvt_f32_f16_sdwa  src0_sel:WORD_1
 *
 * Extracts the HIGH f16 from each packed __half2 and widens it to f32.
 * The compiler uses SDWA with src0_sel:WORD_1 to select bits[31:16] of
 * the source VGPR directly, without a preceding shift instruction.
 */
__global__ void
kernel_a(const __half2 *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* __high2half selects bits[31:16] of the __half2 register.
     * The compiler fuses __high2half + __half2float into a single
     * v_cvt_f32_f16_sdwa ... src0_sel:WORD_1. */
    out[i] = __half2float(__high2half(a[i]));
}

/*
 * Kernel B: both v_cvt_f32_f16_e32 (low) and v_cvt_f32_f16_sdwa WORD_1 (high)
 *
 * Processing the low half uses the standard _e32 encoding (no SDWA needed
 * since bits[15:0] are the natural source word).  Processing the high half
 * requires SDWA to avoid an extra shift.  Both instructions appear in the
 * same kernel body, demonstrating the encoding contrast.
 */
__global__ void
kernel_b(const __half2 *a, float *lo_out, float *hi_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    lo_out[i] = __half2float(__low2half(a[i]));   /* → v_cvt_f32_f16_e32  */
    hi_out[i] = __half2float(__high2half(a[i]));  /* → v_cvt_f32_f16_sdwa WORD_1 */
}

/*
 * Kernel C: two v_cvt_f32_f16_sdwa WORD_1 instructions
 *
 * Both a[i] and b[i] have their high halves extracted with SDWA.
 * The loop body then accumulates all four halves (lo_a, hi_a, lo_b, hi_b)
 * into a single f32 sum.
 */
__global__ void
kernel_c(const __half2 *a, const __half2 *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float lo_a = __half2float(__low2half(a[i]));
    float hi_a = __half2float(__high2half(a[i])); /* → v_cvt_f32_f16_sdwa WORD_1 */
    float lo_b = __half2float(__low2half(b[i]));
    float hi_b = __half2float(__high2half(b[i])); /* → v_cvt_f32_f16_sdwa WORD_1 */

    out[i] = (lo_a + lo_b) + (hi_a + hi_b);
}

int
main(void)
{
    /* ----- host buffers ----- */
    __half2 *h_a     = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_b     = (__half2 *)malloc(N * sizeof(__half2));
    float   *h_oA    = (float   *)malloc(N * sizeof(float));
    float   *h_oB_lo = (float   *)malloc(N * sizeof(float));
    float   *h_oB_hi = (float   *)malloc(N * sizeof(float));
    float   *h_oC    = (float   *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        /* Use distinct lo and hi values so that lo/hi mix-ups are detectable. */
        float lo_a = (float)(i + 1) * 0.5f;
        float hi_a = (float)(i + 1) * 2.0f;
        float lo_b = (float)(N - i) * 0.25f;
        float hi_b = (float)(N - i) * 1.5f;
        h_a[i] = __halves2half2(__float2half(lo_a), __float2half(hi_a));
        h_b[i] = __halves2half2(__float2half(lo_b), __float2half(hi_b));
    }

    /* ----- device buffers ----- */
    __half2 *d_a;
    __half2 *d_b;
    float   *d_oA, *d_oB_lo, *d_oB_hi, *d_oC;

    GPU_CHECK(hipMalloc(&d_a,     N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_b,     N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oA,    N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oB_lo, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oB_hi, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_oC,    N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half2), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_a, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_oA, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_b, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_oB_lo, d_oB_hi, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_c, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_oC, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_oA,    d_oA,    N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB_lo, d_oB_lo, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB_hi, d_oB_hi, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oC,    d_oC,    N * sizeof(float), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float lo_a = __half2float(__low2half(h_a[i]));
        float hi_a = __half2float(__high2half(h_a[i]));
        float lo_b = __half2float(__low2half(h_b[i]));
        float hi_b = __half2float(__high2half(h_b[i]));

        /* Kernel A: high-half only */
        if (fabsf(h_oA[i] - hi_a) > EPS) {
            fprintf(stderr, "FAIL kernel_a[%d]: got %.4f expected %.4f\n",
                    i, h_oA[i], hi_a);
            pass = 0; break;
        }

        /* Kernel B: lo and hi separately */
        if (fabsf(h_oB_lo[i] - lo_a) > EPS) {
            fprintf(stderr, "FAIL kernel_b lo[%d]: got %.4f expected %.4f\n",
                    i, h_oB_lo[i], lo_a);
            pass = 0; break;
        }
        if (fabsf(h_oB_hi[i] - hi_a) > EPS) {
            fprintf(stderr, "FAIL kernel_b hi[%d]: got %.4f expected %.4f\n",
                    i, h_oB_hi[i], hi_a);
            pass = 0; break;
        }

        /* Kernel C: sum of all four halves */
        float expected_c = (lo_a + lo_b) + (hi_a + hi_b);
        if (fabsf(h_oC[i] - expected_c) > EPS) {
            fprintf(stderr, "FAIL kernel_c[%d]: got %.4f expected %.4f\n",
                    i, h_oC[i], expected_c);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("kernel_a  hi[0]=%.4f  hi[%d]=%.4f\n",
               h_oA[0], N - 1, h_oA[N - 1]);
        printf("kernel_b  lo[0]=%.4f  hi[0]=%.4f\n",
               h_oB_lo[0], h_oB_hi[0]);
        printf("kernel_c out[0]=%.4f out[%d]=%.4f\n",
               h_oC[0], N - 1, h_oC[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_oA));
    GPU_CHECK(hipFree(d_oB_lo));
    GPU_CHECK(hipFree(d_oB_hi));
    GPU_CHECK(hipFree(d_oC));
    free(h_a);
    free(h_b);
    free(h_oA);
    free(h_oB_lo);
    free(h_oB_hi);
    free(h_oC);

    return pass ? 0 : 1;
}
