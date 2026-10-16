/*
 * HIP test for v_mfma_f32_32x32x8_f16 (VOP3P-MAI matrix FMA, gfx950).
 *
 * Computes a 32x32 output tile (B=1) with inner dimension K=8 of f16 inputs,
 * accumulating into f32:
 *
 *   D[i][j] = C[i][j] + sum_{k=0}^{7} A[i][k] * B[k][j]
 *
 * Operand register layout (per wavefront, 64 lanes), from llvm-mc -mcpu=gfx950:
 *   v_mfma_f32_32x32x8_f16 a[0:15], v[s0:s0+1], v[s1:s1+1], a[0:15]
 *     SRC0 (A): 2 VGPRs = 4 packed f16 per lane
 *     SRC1 (B): 2 VGPRs = 4 packed f16 per lane
 *     SRC2/VDST (C/D): 16 AGPRs of f32 per lane
 *
 * Mapping-independent verification trick: fill the ENTIRE A matrix (32x8) and
 * the ENTIRE B matrix (8x32) with 1.0 by setting every f16 input slot on every
 * lane to 1.0.  Then, regardless of how the hardware distributes matrix
 * elements across lanes/registers, every output element is exactly
 *
 *   D[i][j] = 0 + sum_{k=0}^{7} 1.0 * 1.0 = 8.0
 *
 * Build (gfx950 only):
 *   hipcc -O2 --offload-arch=gfx950 -o v_mfma_f32_32x32x8_f16 \
 *         v_mfma_f32_32x32x8_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_mfma_f32_32x32x8_f16
 */

#include <hip/hip_runtime.h>
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

/* M=32, N=32, K=8, B=1.  Accumulator = 32*32/64 = 16 f32 per lane. */
static const int CD_REGS = 16;

typedef __attribute__((ext_vector_type(2)))  short short2_vec;
typedef __attribute__((ext_vector_type(4)))  short short4_vec;
typedef __attribute__((ext_vector_type(16))) float float16_vec;

union F16x2 { uint32_t u32; short2_vec s16; };

/*
 * One block = one wavefront (64 threads).  Every lane loads 4 f16 = 1.0 for A,
 * 4 f16 = 1.0 for B, and a zero 16-wide f32 accumulator, issues the MFMA, then
 * writes the 16 accumulator values back out.
 */
__global__ void
mfma_kernel(const uint32_t *src0_packed,   /* A: 2 x uint32 per thread */
            const uint32_t *src1_packed,   /* B: 2 x uint32 per thread */
            const float    *c_init,        /* C: 16 x f32 per thread   */
            float          *d_out,         /* D: 16 x f32 per thread   */
            int             num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base = wf_id * 64 + lane;

    short2_vec va0 = ((F16x2){.u32 = src0_packed[base * 2 + 0]}).s16;
    short2_vec va1 = ((F16x2){.u32 = src0_packed[base * 2 + 1]}).s16;
    short2_vec vb0 = ((F16x2){.u32 = src1_packed[base * 2 + 0]}).s16;
    short2_vec vb1 = ((F16x2){.u32 = src1_packed[base * 2 + 1]}).s16;

    short4_vec va, vb;
    va.lo = va0;  va.hi = va1;
    vb.lo = vb0;  vb.hi = vb1;

    float16_vec vd;
    for (int i = 0; i < CD_REGS; i++)
        vd[i] = c_init[base * CD_REGS + i];

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_32x32x8_f16 %0, %1, %2, %0\n\t"
        : "+a"(vd)
        : "v"(va), "v"(vb));

    for (int i = 0; i < CD_REGS; i++)
        d_out[base * CD_REGS + i] = vd[i];
}

int
main(void)
{
    const int num_waves = 4;
    const int wf_size   = 64;
    const int threads   = num_waves * wf_size;

    size_t ab_words = (size_t)threads * 2;        /* 2 uint32 per thread */
    size_t cd_flts  = (size_t)threads * CD_REGS;  /* 16 f32 per thread   */

    uint32_t *h_a = (uint32_t *)calloc(ab_words, sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)calloc(ab_words, sizeof(uint32_t));
    float    *h_c = (float *)calloc(cd_flts, sizeof(float));
    float    *h_d = (float *)calloc(cd_flts, sizeof(float));

    /* f16 1.0 = 0x3C00; pack two per uint32 -> all input elements = 1.0. */
    const uint32_t ONE_ONE = 0x3C003C00u;
    for (int t = 0; t < threads; t++) {
        h_a[t * 2 + 0] = ONE_ONE;
        h_a[t * 2 + 1] = ONE_ONE;
        h_b[t * 2 + 0] = ONE_ONE;
        h_b[t * 2 + 1] = ONE_ONE;
        for (int i = 0; i < CD_REGS; i++)
            h_c[t * CD_REGS + i] = 0.0f;
    }

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, ab_words * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, ab_words * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_flts * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, cd_flts * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, ab_words * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, ab_words * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_flts * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, cd_flts * sizeof(float)));

    hipLaunchKernelGGL(mfma_kernel,
                       dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_flts * sizeof(float),
                        hipMemcpyDeviceToHost));

    const float EXPECTED = 8.0f;   /* K=8, all-ones A and B, C=0 */
    const float EPS      = 0.01f;

    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int i = 0; i < CD_REGS; i++) {
            float got = h_d[t * CD_REGS + i];
            if (fabsf(got - EXPECTED) > EPS) {
                fprintf(stderr, "FAIL t=%d accum[%d]: got %.3f expected %.3f\n",
                        t, i, got, EXPECTED);
                pass = 0;
                break;
            }
            checked++;
        }
    }

    printf("v_mfma_f32_32x32x8_f16: checked %d values, expected %.1f\n",
           checked, EXPECTED);
    if (pass)
        printf("  D[0]=%.1f D[last]=%.1f\n", h_d[0], h_d[cd_flts - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);

    return pass ? 0 : 1;
}
