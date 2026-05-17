/*
 * HIP test for V_CMP_O_F16 (VOPC / VOP3A ordered half-precision comparison).
 *
 * V_CMP_O_F16 tests whether both operands are ordered (i.e. neither is NaN):
 *
 *   D.u64[threadID] = (!isNan(SRC0) && !isNan(SRC1))
 *
 * It is the f16 counterpart of V_CMP_O_F32.  Two hardware encodings exist:
 *
 *   v_cmp_o_f16_e32  vcc,  vsrc0, vsrc1   (VOPC, 32-bit)
 *     Writes result bits into VCC; implicit destination.
 *
 *   v_cmp_o_f16_e64  sdst, vsrc0, vsrc1   (VOP3A, 64-bit)
 *     Writes result bits into an explicit 64-bit SGPR pair (sdst).
 *     Allows abs/neg modifiers and SGPR/literal sources.
 *
 * This test uses the VOP3A (_e64) form via inline assembly because it
 * produces an explicit output register that can be read back without
 * manipulating VCC directly, and because it is the form decoded by gem5's
 * Inst_VOP3__V_CMP_O_F16 class.
 *
 * The test exercises four lane categories:
 *   1. Both operands are normal finite values    → ordered (bit = 1)
 *   2. SRC0 is NaN, SRC1 is finite              → unordered (bit = 0)
 *   3. SRC0 is finite, SRC1 is NaN              → unordered (bit = 0)
 *   4. Both operands are NaN                     → unordered (bit = 0)
 *
 * Build:
 *   hipcc -O2 -o v_cmp_o_f16 v_cmp_o_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmp_o_f16
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
 * v_cmp_o_f16_e64  sdst, vsrc0, vsrc1   (VOP3A form)
 *
 * Semantics: sdst[lane] = (!isNan(vsrc0[lane]) && !isNan(vsrc1[lane]))
 *
 * The "=s" constraint allocates a 64-bit SGPR pair for the result.  Lane i
 * of the wavefront maps to bit i of sdst.  A non-zero result means at least
 * one active lane saw ordered operands; bit 0 is the result for lane 0.
 *
 * Each thread reads one __half pair, issues the VOP3A comparison, and
 * extracts its own lane bit by reading sdst back through a scalar register
 * and shifting.
 */
__global__ void
kernel_cmp_o_f16(const __half *a, const __half *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Load the f16 values into 32-bit VGPRs so the assembler can allocate
     * standard VGPR constraints.  Only the low 16 bits are consumed by the
     * comparison. */
    uint32_t va = 0, vb = 0;
    uint16_t ha, hb;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    __builtin_memcpy(&hb, b + i, sizeof(hb));
    va = ha;
    vb = hb;

    uint64_t sdst;

    /* VOP3A encoding: explicit 64-bit SGPR destination.
     * sdst bit[lane] = 1 if both src0[lane] and src1[lane] are not NaN. */
    __asm__ volatile(
        "v_cmp_o_f16_e64 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "v"(va), "v"(vb));

    /* Extract this thread's lane bit from the wavefront-wide result.
     * threadIdx.x gives the lane index within the 64-thread wavefront. */
    int lane = threadIdx.x;
    out[i] = (int)((sdst >> lane) & 1ULL);
}

int
main(void)
{
    /* ----- host buffers ----- */
    __half *h_a  = (__half *)malloc(N * sizeof(__half));
    __half *h_b  = (__half *)malloc(N * sizeof(__half));
    int    *h_out = (int   *)malloc(N * sizeof(int));

    /* IEEE 754 quiet NaN for f16: exponent all-ones, mantissa non-zero. */
    const uint16_t f16_nan = 0x7E00;
    __half nan_val;
    __builtin_memcpy(&nan_val, &f16_nan, sizeof(nan_val));

    for (int i = 0; i < N; i++) {
        /* Distribute the four test categories across lanes. */
        switch (i % 4) {
        case 0:
            /* Both finite: ordered → expect 1 */
            h_a[i] = __float2half((float)(i + 1) * 0.5f);
            h_b[i] = __float2half((float)(N - i) * 0.25f);
            break;
        case 1:
            /* SRC0 = NaN, SRC1 finite: unordered → expect 0 */
            h_a[i] = nan_val;
            h_b[i] = __float2half(1.0f);
            break;
        case 2:
            /* SRC0 finite, SRC1 = NaN: unordered → expect 0 */
            h_a[i] = __float2half(1.0f);
            h_b[i] = nan_val;
            break;
        case 3:
            /* Both NaN: unordered → expect 0 */
            h_a[i] = nan_val;
            h_b[i] = nan_val;
            break;
        }
    }

    /* ----- device buffers ----- */
    __half *d_a, *d_b;
    int    *d_out;

    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_cmp_o_f16, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        /* Compute expected result on host using isnan on widened f32. */
        float fa = __half2float(h_a[i]);
        float fb = __half2float(h_b[i]);
        int expected = (!isnan(fa) && !isnan(fb)) ? 1 : 0;

        if (h_out[i] != expected) {
            fprintf(stderr,
                    "FAIL[%d]: got %d expected %d  "
                    "(fa=%g isnan=%d  fb=%g isnan=%d)\n",
                    i, h_out[i], expected,
                    fa, (int)isnan(fa), fb, (int)isnan(fb));
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("case 0 (both finite):  out[0]=%d (expected 1)\n", h_out[0]);
        printf("case 1 (src0 NaN):     out[1]=%d (expected 0)\n", h_out[1]);
        printf("case 2 (src1 NaN):     out[2]=%d (expected 0)\n", h_out[2]);
        printf("case 3 (both NaN):     out[3]=%d (expected 0)\n", h_out[3]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_out));
    free(h_a);
    free(h_b);
    free(h_out);

    return pass ? 0 : 1;
}
