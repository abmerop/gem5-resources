/*
 * HIP test for V_MFMA_I32_16X16X4_4B_I8 (VOP3P-MAI integer matrix FMA).
 *
 * Computes B=4 batched 16x16 output tiles with K=4 inner dimension from int8
 * inputs, accumulating into int32:
 *
 *   D[i][j] = C[i][j] + sum_{k=0}^{K-1} A[i][k] * B[k][j]      (per block)
 *
 * gem5 implements this via the Inst_VOP3P_MAI__V_MFMA_I8<16,16,4,4,...>
 * template (src/arch/amdgpu/vega/insts/vop3p_mai.hh).
 *
 * Layout-independent verification: every A int8 element = CA, every B int8
 * element = CB, every C int32 element = C0.  Then for every output element,
 * regardless of lane/item mapping:
 *
 *   D = C0 + K * CA * CB
 *
 * With K=4, CA=2, CB=3, C0=5 -> D = 5 + 4*6 = 29 (exact, integer).
 * Per-lane register counts (gem5 template): A 1 dword (4 int8), B 1 dword,
 * C/D 16 int32.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mfma_i32_16x16x4_4b_i8 \
 *        v_mfma_i32_16x16x4_4b_i8.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int K        = 4;
static const int CD_REGS  = 16;   /* M*N*B/64 = 16*16*4/64 */

typedef __attribute__((ext_vector_type(16))) int int16i_vec;

__global__ void
mfma_kernel(const uint32_t *a_packed,   /* 1 dword (4 int8) per thread */
            const uint32_t *b_packed,   /* 1 dword (4 int8) per thread */
            const int      *c_init,     /* 16 int32 per thread         */
            int            *d_out,      /* 16 int32 per thread         */
            int             num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base = wf_id * 64 + lane;

    uint32_t va = a_packed[base];
    uint32_t vb = b_packed[base];

    int16i_vec vd;
    for (int i = 0; i < CD_REGS; i++)
        vd[i] = c_init[base * CD_REGS + i];

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_i32_16x16x4_4b_i8 %0, %1, %2, %0\n\t"
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

    const int CA = 2, CB = 3, C0 = 5;
    const int EXPECTED = C0 + K * CA * CB;   /* 29 */

    size_t ab_dwords = (size_t)threads;
    size_t cd_ints   = (size_t)threads * CD_REGS;

    uint32_t *h_a = (uint32_t *)malloc(ab_dwords * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(ab_dwords * sizeof(uint32_t));
    int      *h_c = (int      *)malloc(cd_ints   * sizeof(int));
    int      *h_d = (int      *)malloc(cd_ints   * sizeof(int));

    uint8_t ca = (uint8_t)CA, cb = (uint8_t)CB;
    uint32_t a_dw = (uint32_t)ca | ((uint32_t)ca << 8) |
                    ((uint32_t)ca << 16) | ((uint32_t)ca << 24);
    uint32_t b_dw = (uint32_t)cb | ((uint32_t)cb << 8) |
                    ((uint32_t)cb << 16) | ((uint32_t)cb << 24);

    for (int t = 0; t < threads; t++) {
        h_a[t] = a_dw;
        h_b[t] = b_dw;
        for (int i = 0; i < CD_REGS; i++)
            h_c[t * CD_REGS + i] = C0;
    }

    uint32_t *d_a, *d_b;
    int      *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, ab_dwords * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, ab_dwords * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_ints   * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_d, cd_ints   * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, ab_dwords * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, ab_dwords * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_ints   * sizeof(int),      hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0,   cd_ints   * sizeof(int)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_ints * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int i = 0; i < CD_REGS; i++) {
            int got = h_d[t * CD_REGS + i];
            if (got != EXPECTED) {
                fprintf(stderr, "FAIL t=%d accum[%d]: got %d expected %d\n",
                        t, i, got, EXPECTED);
                pass = 0; break;
            }
            checked++;
        }
    }

    printf("v_mfma_i32_16x16x4_4b_i8: expected %d, checked %d values\n",
           EXPECTED, checked);
    if (pass)
        printf("D[0]=%d D[last]=%d\n", h_d[0], h_d[cd_ints - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);
    return pass ? 0 : 1;
}
