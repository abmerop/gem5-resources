/*
 * HIP test for V_CMP_EQ_F16 (VOPC / VOP3A half-precision equality comparison).
 *
 * V_CMP_EQ_F16 tests whether two f16 operands are equal:
 *
 *   D.u64[threadID] = (S0 == S1)
 *
 * IEEE 754 semantics: NaN != NaN, NaN != any finite, +0.0 == -0.0.
 * The comparison is ordered: if either operand is NaN the result is 0.
 *
 * Two hardware encodings:
 *
 *   v_cmp_eq_f16_e32  vcc,  vsrc0, vsrc1   (VOPC, 32-bit)
 *     Generated naturally by __heq(a, b).  Writes result into VCC.
 *
 *   v_cmp_eq_f16_e64  sdst, vsrc0, vsrc1   (VOP3A, 64-bit)
 *     Explicit 64-bit SGPR pair destination; allows abs/neg modifiers.
 *     Emitted via inline assembly.
 *
 * Test cases cover:
 *   A — equal finite values        (a == b, both non-NaN)    → 1
 *   B — unequal finite values      (a != b, both non-NaN)    → 0
 *   C — NaN vs finite              (NaN != finite)           → 0
 *   D — +0.0 vs -0.0               (+0 == -0 in IEEE 754)   → 1
 *
 * Kernel A uses __heq (VOPC, _e32); Kernel B uses inline asm (VOP3A, _e64).
 * Both paths are exercised across all four test cases.
 *
 * Build:
 *   hipcc -O2 -o v_cmp_eq_f16 v_cmp_eq_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmp_eq_f16
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

/*
 * Kernel A: v_cmp_eq_f16_e32  vcc, vsrc0, vsrc1   (VOPC natural form)
 *
 * __heq(a, b) maps directly to v_cmp_eq_f16_e32; the compiler reads the
 * VCC result into a VGPR via v_cndmask_b32 and stores it as an int.
 */
__global__ void
kernel_vopc(const __half *a, const __half *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* __heq returns true iff a == b (ordered, returns false for NaN). */
    out[i] = __heq(a[i], b[i]) ? 1 : 0;
}

/*
 * Kernel B: v_cmp_eq_f16_e64  sdst, vsrc0, vsrc1   (VOP3A explicit form)
 *
 * The inline assembly writes the wavefront-wide comparison bitmask into an
 * explicit 64-bit SGPR pair (sdst).  Each thread then extracts its own lane
 * bit by shifting sdst by threadIdx.x positions.
 */
__global__ void
kernel_vop3(const __half *a, const __half *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = 0, vb = 0;
    uint16_t ha, hb;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    __builtin_memcpy(&hb, b + i, sizeof(hb));
    va = ha;
    vb = hb;

    uint64_t sdst;

    __asm__ volatile(
        "v_cmp_eq_f16_e64 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "v"(va), "v"(vb));

    out[i] = (int)((sdst >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    /* ----- host buffers ----- */
    __half *h_a  = (__half *)malloc(N * sizeof(__half));
    __half *h_b  = (__half *)malloc(N * sizeof(__half));
    int *h_vopc  = (int    *)malloc(N * sizeof(int));
    int *h_vop3  = (int    *)malloc(N * sizeof(int));

    /* IEEE 754 f16 quiet NaN: exponent=0x1F (all ones), mantissa non-zero. */
    const uint16_t f16_nan   = 0x7E00;
    const uint16_t f16_neg0  = 0x8000; /* -0.0 in f16 */
    __half nan_val, neg0_val;
    __builtin_memcpy(&nan_val,  &f16_nan,  sizeof(nan_val));
    __builtin_memcpy(&neg0_val, &f16_neg0, sizeof(neg0_val));

    for (int i = 0; i < N; i++) {
        switch (i % 4) {
        case 0:
            /* Case A: equal finite values → expect 1 */
            h_a[i] = __float2half((float)(i / 4 + 1) * 0.5f);
            h_b[i] = h_a[i];
            break;
        case 1:
            /* Case B: unequal finite values → expect 0 */
            h_a[i] = __float2half(1.0f);
            h_b[i] = __float2half(2.0f);
            break;
        case 2:
            /* Case C: NaN vs finite → expect 0 (NaN != anything) */
            h_a[i] = nan_val;
            h_b[i] = __float2half(1.0f);
            break;
        case 3:
            /* Case D: +0.0 vs -0.0 → expect 1 (IEEE 754: +0 == -0) */
            h_a[i] = __float2half(0.0f);
            h_b[i] = neg0_val;
            break;
        }
    }

    /* ----- device buffers ----- */
    __half *d_a, *d_b;
    int    *d_vopc, *d_vop3;

    GPU_CHECK(hipMalloc(&d_a,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_b,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_vopc, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_vop3, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;

    hipLaunchKernelGGL(kernel_vopc, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_vopc, N);
    GPU_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_vop3, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_vop3, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_vopc, d_vopc, N * sizeof(int), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_vop3, d_vop3, N * sizeof(int), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    /* Expected results for each case mod 4. */
    static const int expected[4] = {
        1,  /* case A: equal finite    */
        0,  /* case B: unequal finite  */
        0,  /* case C: NaN vs finite   */
        1,  /* case D: +0.0 == -0.0   */
    };

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int exp = expected[i % 4];

        if (h_vopc[i] != exp) {
            fprintf(stderr, "FAIL vopc[%d] case %d: got %d expected %d\n",
                    i, i % 4, h_vopc[i], exp);
            pass = 0; break;
        }
        if (h_vop3[i] != exp) {
            fprintf(stderr, "FAIL vop3[%d] case %d: got %d expected %d\n",
                    i, i % 4, h_vop3[i], exp);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("VOPC (_e32): [A]=%d [B]=%d [C]=%d [D]=%d\n",
               h_vopc[0], h_vopc[1], h_vopc[2], h_vopc[3]);
        printf("VOP3 (_e64): [A]=%d [B]=%d [C]=%d [D]=%d\n",
               h_vop3[0], h_vop3[1], h_vop3[2], h_vop3[3]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_a));   GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_vopc)); GPU_CHECK(hipFree(d_vop3));
    free(h_a); free(h_b); free(h_vopc); free(h_vop3);

    return pass ? 0 : 1;
}
