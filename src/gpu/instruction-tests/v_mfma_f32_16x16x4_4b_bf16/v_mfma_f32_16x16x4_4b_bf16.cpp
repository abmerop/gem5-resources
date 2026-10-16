/*
 * HIP test for V_MFMA_F32_16X16X4_4B_BF16 (VOP3P-MAI matrix FMA).
 *
 * Computes B=4 batched 16x16 output tiles with K=4 inner dimension from bf16
 * inputs, accumulating into f32:
 *
 *   D[i][j] = C[i][j] + sum_{k=0}^{K-1} A[i][k] * B[k][j]      (per block)
 *
 * gem5 implements this via the Inst_VOP3P_MAI__V_MFMA_MXFP<16,16,4,4,...>
 * template (src/arch/amdgpu/vega/insts/vop3p_mai.hh).
 *
 * Layout-independent verification trick
 * -------------------------------------
 * The exact mapping of A/B/C elements onto wavefront lanes is intricate.  To
 * avoid depending on it, every A bf16 element is set to the same constant CA,
 * every B bf16 element to CB, and every C f32 element to C0.  Then for every
 * output element, regardless of lane/item mapping:
 *
 *   D = C0 + K * CA * CB
 *
 * With K=4, CA=2.0, CB=3.0, C0=5.0 -> D = 5 + 4*6 = 29.0 (exact in bf16/f32).
 * Per-lane register counts (from the gem5 template):
 *   A: 2 dwords (4 bf16) ; B: 2 dwords (4 bf16) ; C/D: 16 f32.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mfma_f32_16x16x4_4b_bf16 \
 *        v_mfma_f32_16x16x4_4b_bf16.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* M=16, N=16, K=4, B=4 */
static const int K = 4;

static const int CD_REGS = 16;   /* M*N*B/64 = 16*16*4/64 */

typedef __attribute__((ext_vector_type(4)))  short short4_vec;
typedef __attribute__((ext_vector_type(16))) float float16_vec;

static uint16_t
f32_to_bf16_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);   /* exact for our power-of-two-ish values */
}

__global__ void
mfma_kernel(const uint32_t *a_packed,   /* 2 dwords per thread (4 bf16) */
            const uint32_t *b_packed,   /* 2 dwords per thread (4 bf16) */
            const float    *c_init,     /* 16 f32 per thread            */
            float          *d_out,      /* 16 f32 per thread            */
            int             num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base = wf_id * 64 + lane;

    union { uint32_t u32[2]; short4_vec s; } a, b;
    a.u32[0] = a_packed[base * 2 + 0];
    a.u32[1] = a_packed[base * 2 + 1];
    b.u32[0] = b_packed[base * 2 + 0];
    b.u32[1] = b_packed[base * 2 + 1];
    short4_vec va = a.s, vb = b.s;

    float16_vec vd;
    for (int i = 0; i < CD_REGS; i++)
        vd[i] = c_init[base * CD_REGS + i];

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x4_4b_bf16 %0, %1, %2, %0\n\t"
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

    const float CA = 2.0f, CB = 3.0f, C0 = 5.0f;
    const float EXPECTED = C0 + (float)K * CA * CB;   /* 29.0 */

    size_t ab_dwords = (size_t)threads * 2;
    size_t cd_floats = (size_t)threads * CD_REGS;

    uint32_t *h_a = (uint32_t *)calloc(ab_dwords, sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)calloc(ab_dwords, sizeof(uint32_t));
    float    *h_c = (float    *)calloc(cd_floats, sizeof(float));
    float    *h_d = (float    *)calloc(cd_floats, sizeof(float));

    uint16_t ca = f32_to_bf16_bits(CA);
    uint16_t cb = f32_to_bf16_bits(CB);
    uint32_t a_dw = (uint32_t)ca | ((uint32_t)ca << 16);
    uint32_t b_dw = (uint32_t)cb | ((uint32_t)cb << 16);

    for (int t = 0; t < threads; t++) {
        h_a[t * 2 + 0] = a_dw;
        h_a[t * 2 + 1] = a_dw;
        h_b[t * 2 + 0] = b_dw;
        h_b[t * 2 + 1] = b_dw;
        for (int i = 0; i < CD_REGS; i++)
            h_c[t * CD_REGS + i] = C0;
    }

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, ab_dwords * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, ab_dwords * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_floats * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, cd_floats * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, ab_dwords * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, ab_dwords * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_floats * sizeof(float),    hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0,   cd_floats * sizeof(float)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_floats * sizeof(float), hipMemcpyDeviceToHost));

    const float EPS = 0.5f;
    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int i = 0; i < CD_REGS; i++) {
            float got = h_d[t * CD_REGS + i];
            if (fabsf(got - EXPECTED) > EPS) {
                fprintf(stderr, "FAIL t=%d accum[%d]: got %.3f expected %.3f\n",
                        t, i, got, EXPECTED);
                pass = 0; break;
            }
            checked++;
        }
    }

    printf("v_mfma_f32_16x16x4_4b_bf16: expected %.1f, checked %d values\n",
           EXPECTED, checked);
    if (pass)
        printf("D[0]=%.1f D[last]=%.1f\n", h_d[0], h_d[cd_floats - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);
    return pass ? 0 : 1;
}
