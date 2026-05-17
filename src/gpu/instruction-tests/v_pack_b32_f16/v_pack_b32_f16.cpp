/*
 * HIP test for V_PACK_B32_F16 (VOP3A pack two f16 values into one 32-bit
 * register).
 *
 * V_PACK_B32_F16 assembles a packed __half2 from two 16-bit sources:
 *
 *   VDST[15:0]  = selected 16-bit slice of SRC0
 *   VDST[31:16] = selected 16-bit slice of SRC1
 *
 * The OPSEL field independently selects the low (bits[15:0]) or high
 * (bits[31:16]) half of each source register:
 *
 *   OPSEL[0] = 0 → SRC0 low  half  (default)
 *   OPSEL[0] = 1 → SRC0 high half
 *   OPSEL[1] = 0 → SRC1 low  half  (default)
 *   OPSEL[1] = 1 → SRC1 high half
 *   OPSEL[2]     → (unused for VDST, always 0)
 *
 * This gives four distinct instruction variants, all exercised here:
 *
 *   Kernel A — opsel:[0,0,0]  (default, no op_sel in assembly)
 *     VDST = { SRC1[15:0] : SRC0[15:0] }
 *     Natural output of __floats2half2_rn(a, b):
 *       v_cvt_f16_f32 → v_cvt_f16_f32 → v_pack_b32_f16
 *
 *   Kernel B — op_sel:[1,0,0]
 *     VDST = { SRC1[15:0] : SRC0[31:16] }
 *     Pack the HIGH half of one register with the LOW half of another.
 *     Emitted via inline assembly.
 *
 *   Kernel C — op_sel:[0,1,0]
 *     VDST = { SRC1[31:16] : SRC0[15:0] }
 *     Emitted via inline assembly.
 *
 *   Kernel D — op_sel:[1,1,0]
 *     VDST = { SRC1[31:16] : SRC0[31:16] }
 *     Pack both HIGH halves.  Without OPSEL this would require two
 *     v_lshrrev_b32 shifts; OPSEL eliminates them.
 *     Emitted via inline assembly.
 *
 * Build:
 *   hipcc -O2 -o v_pack_b32_f16 v_pack_b32_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_pack_b32_f16
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
 * Kernel A: v_pack_b32_f16 (no op_sel — default opsel:[0,0,0])
 *
 * VDST = { f16(b[i]) : f16(a[i]) }
 *
 * __floats2half2_rn converts two f32 values to f16 and packs them.
 * The compiler emits: v_cvt_f16_f32 + v_cvt_f16_f32 + v_pack_b32_f16.
 */
__global__ void
kernel_a(const float *a, const float *b, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Compiler fuses the two f32→f16 conversions with a pack:
     *   v_pack_b32_f16 vdst, va_f16, vb_f16  (opsel:[0,0,0]) */
    out[i] = __floats2half2_rn(a[i], b[i]);
}

/*
 * Helper: load a packed __half2 as a raw uint32 for use in inline asm.
 * The __half2 layout on AMD GPUs is: bits[15:0]=lo, bits[31:16]=hi.
 */
__device__ __forceinline__ uint32_t
half2_as_u32(const __half2 *p)
{
    uint32_t v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

/*
 * Kernel B: v_pack_b32_f16 op_sel:[1,0,0]
 *
 * VDST[15:0]  = SRC0[31:16]  (high f16 of a[i])
 * VDST[31:16] = SRC1[15:0]   (low  f16 of b[i])
 *
 * Inline assembly with op_sel:[1,0,0] selects the HIGH half of a[i]
 * as the low element of the output, and the LOW half of b[i] as the
 * high element, without any shift instructions.
 */
__global__ void
kernel_b(const __half2 *a, const __half2 *b, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = half2_as_u32(a + i);
    uint32_t vb = half2_as_u32(b + i);
    uint32_t vd;

    /* op_sel:[1,0,0]: VDST[15:0]=SRC0[31:16]  VDST[31:16]=SRC1[15:0] */
    __asm__ volatile(
        "v_pack_b32_f16 %0, %1, %2 op_sel:[1,0,0]\n\t"
        : "=v"(vd) : "v"(va), "v"(vb));

    __builtin_memcpy(out + i, &vd, sizeof(vd));
}

/*
 * Kernel C: v_pack_b32_f16 op_sel:[0,1,0]
 *
 * VDST[15:0]  = SRC0[15:0]   (low  f16 of a[i])
 * VDST[31:16] = SRC1[31:16]  (high f16 of b[i])
 */
__global__ void
kernel_c(const __half2 *a, const __half2 *b, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = half2_as_u32(a + i);
    uint32_t vb = half2_as_u32(b + i);
    uint32_t vd;

    /* op_sel:[0,1,0]: VDST[15:0]=SRC0[15:0]  VDST[31:16]=SRC1[31:16] */
    __asm__ volatile(
        "v_pack_b32_f16 %0, %1, %2 op_sel:[0,1,0]\n\t"
        : "=v"(vd) : "v"(va), "v"(vb));

    __builtin_memcpy(out + i, &vd, sizeof(vd));
}

/*
 * Kernel D: v_pack_b32_f16 op_sel:[1,1,0]
 *
 * VDST[15:0]  = SRC0[31:16]  (high f16 of a[i])
 * VDST[31:16] = SRC1[31:16]  (high f16 of b[i])
 *
 * Without OPSEL this would require:
 *   v_lshrrev_b32 tmp0, 16, a[i]
 *   v_lshrrev_b32 tmp1, 16, b[i]
 *   v_pack_b32_f16 out, tmp0, tmp1
 * OPSEL eliminates both shifts.
 */
__global__ void
kernel_d(const __half2 *a, const __half2 *b, __half2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = half2_as_u32(a + i);
    uint32_t vb = half2_as_u32(b + i);
    uint32_t vd;

    /* op_sel:[1,1,0]: VDST[15:0]=SRC0[31:16]  VDST[31:16]=SRC1[31:16] */
    __asm__ volatile(
        "v_pack_b32_f16 %0, %1, %2 op_sel:[1,1,0]\n\t"
        : "=v"(vd) : "v"(va), "v"(vb));

    __builtin_memcpy(out + i, &vd, sizeof(vd));
}

int
main(void)
{
    /* ----- host buffers ----- */
    float   *h_fa  = (float   *)malloc(N * sizeof(float));
    float   *h_fb  = (float   *)malloc(N * sizeof(float));
    __half2 *h_a2  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_b2  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_oA  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_oB  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_oC  = (__half2 *)malloc(N * sizeof(__half2));
    __half2 *h_oD  = (__half2 *)malloc(N * sizeof(__half2));

    for (int i = 0; i < N; i++) {
        /* Keep values small and distinct: lo and hi halves differ. */
        float lo_a = (float)(i + 1) * 0.5f;
        float hi_a = (float)(i + 1) * 2.0f;
        float lo_b = (float)(N - i) * 0.25f;
        float hi_b = (float)(N - i) * 3.0f;
        h_fa[i] = lo_a;
        h_fb[i] = lo_b;
        h_a2[i] = __halves2half2(__float2half(lo_a), __float2half(hi_a));
        h_b2[i] = __halves2half2(__float2half(lo_b), __float2half(hi_b));
    }

    /* ----- device buffers ----- */
    float   *d_fa, *d_fb;
    __half2 *d_a2, *d_b2, *d_oA, *d_oB, *d_oC, *d_oD;

    GPU_CHECK(hipMalloc(&d_fa,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_fb,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_a2,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_b2,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oA,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oB,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oC,  N * sizeof(__half2)));
    GPU_CHECK(hipMalloc(&d_oD,  N * sizeof(__half2)));

    GPU_CHECK(hipMemcpy(d_fa, h_fa, N * sizeof(float),   hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_fb, h_fb, N * sizeof(float),   hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_a2, h_a2, N * sizeof(__half2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b2, h_b2, N * sizeof(__half2), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_a, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_fa, d_fb, d_oA, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_b, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a2, d_b2, d_oB, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_c, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a2, d_b2, d_oC, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_d, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a2, d_b2, d_oD, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_oA, d_oA, N * sizeof(__half2), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oB, d_oB, N * sizeof(__half2), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oC, d_oC, N * sizeof(__half2), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_oD, d_oD, N * sizeof(__half2), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        __half lo_a = __low2half(h_a2[i]);
        __half hi_a = __high2half(h_a2[i]);
        __half lo_b = __low2half(h_b2[i]);
        __half hi_b = __high2half(h_b2[i]);

        /* Kernel A: { f16(fb) : f16(fa) } */
        __half exp_a_lo = __float2half(h_fa[i]);
        __half exp_a_hi = __float2half(h_fb[i]);
        if (__half2float(__low2half(h_oA[i]))  != __half2float(exp_a_lo) ||
            __half2float(__high2half(h_oA[i])) != __half2float(exp_a_hi)) {
            fprintf(stderr, "FAIL kernel_a[%d]: lo=%.3f(exp %.3f) "
                    "hi=%.3f(exp %.3f)\n", i,
                    __half2float(__low2half(h_oA[i])),  __half2float(exp_a_lo),
                    __half2float(__high2half(h_oA[i])), __half2float(exp_a_hi));
            pass = 0; break;
        }

        /* Kernel B op_sel:[1,0,0]: lo=hi(a), hi=lo(b) */
        if (__half2float(__low2half(h_oB[i]))  != __half2float(hi_a) ||
            __half2float(__high2half(h_oB[i])) != __half2float(lo_b)) {
            fprintf(stderr, "FAIL kernel_b[%d]: lo=%.3f(exp %.3f) "
                    "hi=%.3f(exp %.3f)\n", i,
                    __half2float(__low2half(h_oB[i])),  __half2float(hi_a),
                    __half2float(__high2half(h_oB[i])), __half2float(lo_b));
            pass = 0; break;
        }

        /* Kernel C op_sel:[0,1,0]: lo=lo(a), hi=hi(b) */
        if (__half2float(__low2half(h_oC[i]))  != __half2float(lo_a) ||
            __half2float(__high2half(h_oC[i])) != __half2float(hi_b)) {
            fprintf(stderr, "FAIL kernel_c[%d]: lo=%.3f(exp %.3f) "
                    "hi=%.3f(exp %.3f)\n", i,
                    __half2float(__low2half(h_oC[i])),  __half2float(lo_a),
                    __half2float(__high2half(h_oC[i])), __half2float(hi_b));
            pass = 0; break;
        }

        /* Kernel D op_sel:[1,1,0]: lo=hi(a), hi=hi(b) */
        if (__half2float(__low2half(h_oD[i]))  != __half2float(hi_a) ||
            __half2float(__high2half(h_oD[i])) != __half2float(hi_b)) {
            fprintf(stderr, "FAIL kernel_d[%d]: lo=%.3f(exp %.3f) "
                    "hi=%.3f(exp %.3f)\n", i,
                    __half2float(__low2half(h_oD[i])),  __half2float(hi_a),
                    __half2float(__high2half(h_oD[i])), __half2float(hi_b));
            pass = 0; break;
        }
    }

    if (pass) {
        printf("kernel_a[0]: lo=%.3f hi=%.3f\n",
               __half2float(__low2half(h_oA[0])),
               __half2float(__high2half(h_oA[0])));
        printf("kernel_b[0]: lo=%.3f hi=%.3f  (hi(a), lo(b))\n",
               __half2float(__low2half(h_oB[0])),
               __half2float(__high2half(h_oB[0])));
        printf("kernel_c[0]: lo=%.3f hi=%.3f  (lo(a), hi(b))\n",
               __half2float(__low2half(h_oC[0])),
               __half2float(__high2half(h_oC[0])));
        printf("kernel_d[0]: lo=%.3f hi=%.3f  (hi(a), hi(b))\n",
               __half2float(__low2half(h_oD[0])),
               __half2float(__high2half(h_oD[0])));
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_fa));  GPU_CHECK(hipFree(d_fb));
    GPU_CHECK(hipFree(d_a2));  GPU_CHECK(hipFree(d_b2));
    GPU_CHECK(hipFree(d_oA));  GPU_CHECK(hipFree(d_oB));
    GPU_CHECK(hipFree(d_oC));  GPU_CHECK(hipFree(d_oD));
    free(h_fa); free(h_fb); free(h_a2); free(h_b2);
    free(h_oA); free(h_oB); free(h_oC); free(h_oD);

    return pass ? 0 : 1;
}
