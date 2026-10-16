/*
 * HIP test for v_mfma_f32_16x16x128_f8f6f4
 * (VOP3P-MAI mixed FP8/FP6/FP4 matrix instruction, gfx950 / CDNA4).
 *
 * Semantics (M=16, N=16, K=128, single block):
 *   D[i][j] = C[i][j] + sum_{k=0..127} A[i][k] * B[k][j]
 * The A and B element formats are selected by the CBSZ / BLGP encoding
 * fields:
 *   000 = FP8 E4M3, 001 = BF8 E5M2, 010 = FP6 E2M3, 011 = BF6 E3M2,
 *   100 = FP4 E2M1.  The accumulator C/D is always f32.
 *
 * The bare `v_mfma_f32_16x16x128_f8f6f4` mnemonic assembles with CBSZ=0
 * and BLGP=0, i.e. both A and B are read as the FP8 E4M3 family.  In gem5
 * this decodes to Inst_VOP3P_MAI__V_MFMA_F8F6F4<16,16,128,...>; the
 * FNUZ vs non-FNUZ choice is a runtime decode-control flag (fp8Mode),
 * not an operand-encoding difference, so the same mnemonic covers both.
 *
 * Operand widths (verified with llvm-mc -mcpu=gfx950):
 *   A = 8 VGPRs, B = 8 VGPRs, C/D = 4 f32 regs (AGPRs via "+a").
 *   For K=128 FP8 packed 4-per-dword: gprs_a = 16*128/(64*4) = 8, same for B.
 *
 * Self-checking trick (mapping- and format-independent):
 *   Set every A element and every B element to 1.0, and C = 0.  The value
 *   1.0 is exactly representable as 0x38 in E4M3 (and E4M3FNUZ: ebits=4,
 *   mbits=3, bias=7 -> 0 0111 000 = 0x38), so each dword of A and B is
 *   0x38383838.  Every output element is then exactly the sum of K unit
 *   products = K = 128.0, regardless of the A<->B lane mapping.  Because
 *   1.0 and 128.0 are exactly representable in f32, the comparison is exact.
 *
 * Build:
 *   make AMDGPU_TARGET=gfx950
 *   (or: hipcc -O2 --offload-arch=gfx950 -o v_mfma_f32_16x16x128_f8f6f4 \
 *          v_mfma_f32_16x16x128_f8f6f4.cpp)
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_mfma_f32_16x16x128_f8f6f4
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Per-lane VGPR/AGPR counts for the 16x16x128 FP8 form. */
static const int ADW = 8;  /* A: 8 VGPRs */
static const int BDW = 8;  /* B: 8 VGPRs */
static const int CD  = 4;  /* C/D: 4 f32 AGPRs */

typedef __attribute__((ext_vector_type(8))) int   int8_v;
typedef __attribute__((ext_vector_type(4))) float float4_v;

/*
 * mfma_kernel
 *
 * One wavefront (64 lanes) computes one 16x16 output tile.  Each lane loads
 * its 8 A dwords, 8 B dwords, and 4 f32 accumulator words, executes the
 * mixed-format MFMA (CBSZ=0/BLGP=0 => FP8 E4M3 inputs), and writes the 4
 * result words back.  C/D use the "+a" (AGPR) accumulator constraint.
 */
__global__ void
mfma_kernel(const uint32_t *a_in, const uint32_t *b_in,
            const float *c_in, float *d_out, int num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves) return;
    int base = wf_id * 64 + lane;

    int8_v va;
    for (int i = 0; i < ADW; i++) va[i] = (int)a_in[base * ADW + i];
    int8_v vb;
    for (int i = 0; i < BDW; i++) vb[i] = (int)b_in[base * BDW + i];
    float4_v vd;
    for (int i = 0; i < CD; i++) vd[i] = c_in[base * CD + i];

    // s_nop after the MFMA: the compiler does NOT insert the mandatory MFMA
    // write -> memory-read hazard waits for an inline-asm MFMA. This is a long
    // (K=128) MFMA, so without the waits the store below reads a[0:3] before
    // even reg0 commits and gets the pre-MFMA C value (0) on silicon (gem5 has
    // no latency model so it passes either way). 128 waits covers this op.
    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x128_f8f6f4 %0, %1, %2, %0\n\t"
        "s_nop 15\n\t s_nop 15\n\t s_nop 15\n\t s_nop 15\n\t"
        "s_nop 15\n\t s_nop 15\n\t s_nop 15\n\t s_nop 15\n\t"
        : "+a"(vd)
        : "v"(va), "v"(vb));

    for (int i = 0; i < CD; i++) d_out[base * CD + i] = vd[i];
}

int
main(void)
{
    const int num_waves = 4;
    const int threads   = num_waves * 64;

    uint32_t *h_a = (uint32_t *)malloc((size_t)threads * ADW * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc((size_t)threads * BDW * sizeof(uint32_t));
    float    *h_c = (float *)malloc((size_t)threads * CD * sizeof(float));
    float    *h_d = (float *)malloc((size_t)threads * CD * sizeof(float));

    /* Every FP8 element = 1.0 (0x38); accumulator C = 0. */
    for (int t = 0; t < threads; t++) {
        for (int i = 0; i < ADW; i++) h_a[t * ADW + i] = 0x38383838u;
        for (int i = 0; i < BDW; i++) h_b[t * BDW + i] = 0x38383838u;
        for (int i = 0; i < CD;  i++) h_c[t * CD  + i] = 0.0f;
    }

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, (size_t)threads * ADW * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, (size_t)threads * BDW * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, (size_t)threads * CD  * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, (size_t)threads * CD  * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a, h_a, (size_t)threads * ADW * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, (size_t)threads * BDW * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, (size_t)threads * CD * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, (size_t)threads * CD * sizeof(float)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(64), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d, d_d, (size_t)threads * CD * sizeof(float),
                        hipMemcpyDeviceToHost));

    /* K = 128 unit products, exact in f32. */
    const float EXPECTED = 128.0f;
    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int i = 0; i < CD; i++) {
            float got = h_d[t * CD + i];
            if (got != EXPECTED) {
                fprintf(stderr, "FAIL t=%d [%d]: got %.3f expected %.3f\n",
                        t, i, got, EXPECTED);
                pass = 0;
                break;
            }
            checked++;
        }
    }
    printf("v_mfma_f32_16x16x128_f8f6f4: checked %d, expected %.3f\n",
           checked, EXPECTED);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);
    return pass ? 0 : 1;
}
