/*
 * HIP test for v_smfmac_i32_16x16x64_i8 (sparse MFMA, i8 inputs, i32 acc,
 * gfx950).
 *
 * SMFMAC = Sparse Matrix Fused Multiply-ACcumulate, 4:2 structured sparsity:
 *
 *   D.i32 = A_sparse.i8 * B.i8 + C.i32
 *
 * Operand register layout (per wavefront, 64 lanes), from llvm-mc -mcpu=gfx950:
 *   v_smfmac_i32_16x16x64_i8 v[0:3], v[4:5], v[8:11], v20
 *     SRC0 (A, sparse): 2 VGPRs = 8 packed i8 per lane (K/2 non-zero bytes)
 *     SRC1 (B, dense) : 4 VGPRs = 16 packed i8 per lane
 *     SRC2 (index)    : 1 VGPR  = 4:2 sparsity index
 *     SRC2/VDST (C/D) : 4 VGPRs = 4 i32 accumulators per lane
 *
 * Mapping-independent verification trick: set EVERY A byte = 1 and EVERY B
 * byte = 1, with C = 0.  Each output element accumulates exactly K/2 = 32
 * non-zero products, each equal to 1*1 = 1, regardless of how lanes/registers
 * are mapped and regardless of which positions the sparsity index selects
 * (every B byte is 1).  Therefore every output element is exactly 32.
 *
 * Build (gfx950 only):
 *   hipcc -O2 --offload-arch=gfx950 -o v_smfmac_i32_16x16x64_i8 \
 *         v_smfmac_i32_16x16x64_i8.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_smfmac_i32_16x16x64_i8
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

/* M=16, N=16, K=64.  Accumulator = 16*16/64 = 4 i32 per lane. */
static const int GPRS_A = 2;   /* 8 i8  per lane */
static const int GPRS_B = 4;   /* 16 i8 per lane */
static const int GPRS_D = 4;   /* 4 i32 per lane */

typedef __attribute__((ext_vector_type(2))) int int2_v;
typedef __attribute__((ext_vector_type(4))) int int4_v;

__global__ void
smfmac_kernel(const uint32_t *a_in, const uint32_t *b_in,
              const int32_t *c_in, int32_t *d_out, int num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base_a  = (wf_id * 64 + lane) * GPRS_A;
    int base_b  = (wf_id * 64 + lane) * GPRS_B;
    int base_cd = (wf_id * 64 + lane) * GPRS_D;

    int2_v va;   /* A: 2 VGPRs */
    va[0] = (int)a_in[base_a + 0];
    va[1] = (int)a_in[base_a + 1];

    int4_v vb;   /* B: 4 VGPRs */
    for (int i = 0; i < 4; i++)
        vb[i] = (int)b_in[base_b + i];

    int4_v vc;   /* C/D accumulator: 4 VGPRs */
    for (int i = 0; i < 4; i++)
        vc[i] = c_in[base_cd + i];

    uint32_t idx = 0x44444444u;   /* result is index-independent (all B == 1) */

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_i32_16x16x64_i8 %0, %1, %2, %3\n\t"
        : "+v"(vc)
        : "v"(va), "v"(vb), "v"(idx));

    for (int i = 0; i < 4; i++)
        d_out[base_cd + i] = vc[i];
}

int
main(void)
{
    const int num_waves = 4;
    const int wf_size   = 64;
    const int threads   = num_waves * wf_size;

    size_t a_count  = (size_t)threads * GPRS_A;
    size_t b_count  = (size_t)threads * GPRS_B;
    size_t cd_count = (size_t)threads * GPRS_D;

    uint32_t *h_a = (uint32_t *)calloc(a_count, sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)calloc(b_count, sizeof(uint32_t));
    int32_t  *h_c = (int32_t *)calloc(cd_count, sizeof(int32_t));
    int32_t  *h_d = (int32_t *)calloc(cd_count, sizeof(int32_t));

    /* All A bytes = 1, all B bytes = 1 -> 0x01010101 per dword. C = 0. */
    for (int t = 0; t < threads; t++) {
        for (int r = 0; r < GPRS_A; r++) h_a[t * GPRS_A + r] = 0x01010101u;
        for (int r = 0; r < GPRS_B; r++) h_b[t * GPRS_B + r] = 0x01010101u;
        for (int r = 0; r < GPRS_D; r++) h_c[t * GPRS_D + r] = 0;
    }

    uint32_t *d_a, *d_b;
    int32_t  *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, a_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, b_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_count * sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_d, cd_count * sizeof(int32_t)));

    GPU_CHECK(hipMemcpy(d_a, h_a, a_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, b_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_count * sizeof(int32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, cd_count * sizeof(int32_t)));

    hipLaunchKernelGGL(smfmac_kernel,
                       dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_count * sizeof(int32_t), hipMemcpyDeviceToHost));

    const int32_t EXPECTED = 32;   /* K/2 non-zero products, all 1*1 */

    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int r = 0; r < GPRS_D; r++) {
            int32_t got = h_d[t * GPRS_D + r];
            if (got != EXPECTED) {
                fprintf(stderr, "FAIL t=%d vgpr=%d: got %d expected %d\n",
                        t, r, got, EXPECTED);
                pass = 0;
                break;
            }
            checked++;
        }
    }

    printf("v_smfmac_i32_16x16x64_i8: checked %d values, expected %d\n",
           checked, EXPECTED);
    if (pass)
        printf("  D[0]=%d D[last]=%d\n", h_d[0], h_d[cd_count - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);

    return pass ? 0 : 1;
}
