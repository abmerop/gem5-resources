/*
 * HIP test for v_smfmac_f32_16x16x128_bf8_bf8 (sparse MFMA, bf8 A / bf8 B, f32
 * accumulator, gfx950 / CDNA4).  Pattern D.
 *
 *   D.f32 = A_sparse.bf8 * B.bf8 + C.f32   (4:2 structured sparsity, K=64)
 *
 * gem5 class:
 *   Inst_VOP3P_MAI__V_SMFMAC<16, 16, 64, VecElemF32, VecOperandF32,
 *                            AMDGPU::MXFP::fp8_e5m2, AMDGPU::MXFP::fp8_e5m2, ...>
 * (bf8 == E5M2).  See src/arch/amdgpu/vega/insts/vop3p_mai.hh.
 *
 * Operand register layout (per wavefront, 64 lanes), from
 * `llvm-mc -mcpu=gfx950 --show-encoding`:
 *   v_smfmac_f32_16x16x128_bf8_bf8 a[0:3], v[0:1], v[4:7], v8
 *     SRC0 (A, sparse bf8): 2 VGPRs = 8 packed bf8 per lane
 *     SRC1 (B, dense  bf8): 4 VGPRs = 16 packed bf8 per lane
 *     SRC2 (index)        : 1 VGPR  = 4:2 sparsity index
 *     C/D  (acc)          : 4 AGPRs of f32 per lane (16*16/64 = 4)
 *
 * BF8 / E5M2 encoding (gem5 src/arch/amdgpu/common/dtype/fp8_e5m2.hh):
 *   sign(1) exp(5, bias 15) mant(2).  1.0 = exp=15, mant=0, sign=0
 *   = 0b0_01111_00 = 0x3C.  So a packed dword of 1.0 bf8 = 0x3C3C3C3C.
 *
 * Mapping-independent verification: every A and B bf8 byte = 1.0, C = 0.
 * gem5's execute() (vop3p_mai.hh) accumulates, for each output element,
 * exactly K/2 = 64 non-zero products (sparse: half of K contributes).  With
 * every A = 1.0 and every B = 1.0, each product is 1.0*1.0 = 1.0 regardless
 * of which lane/register/index positions are selected (all B are 1.0, so
 * whichever sparse partner is chosen contributes 1.0).  1.0 and 64.0 are
 * exactly representable in f32, so every output == 64.0 exactly (strict ==).
 *
 * Build (gfx950 only):
 *   hipcc -O2 --offload-arch=gfx950 -o v_smfmac_f32_16x16x128_bf8_bf8 \
 *         v_smfmac_f32_16x16x128_bf8_bf8.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_smfmac_f32_16x16x128_bf8_bf8
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

static const int CD_REGS = 4;    /* 16*16/64 f32 per lane */
static const int GPRS_A  = 4;    /* 16 bf8 per lane (sparse A, K=128) */
static const int GPRS_B  = 8;    /* 32 bf8 per lane (dense  B, K=128) */

/* BF8 (E5M2) encoding of 1.0: sign=0 exp=15 mant=0 = 0x3C. */
static const uint32_t BF8_ONE_x4 = 0x3C3C3C3Cu;

typedef __attribute__((ext_vector_type(4)))  int   int4a_v;
typedef __attribute__((ext_vector_type(8)))  int   int8_v;
typedef __attribute__((ext_vector_type(4)))  float float4_v;

__global__ void
smfmac_kernel(const uint32_t *a_in, const uint32_t *b_in,
              const float *c_in, float *d_out, int num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base_a  = (wf_id * 64 + lane) * GPRS_A;
    int base_b  = (wf_id * 64 + lane) * GPRS_B;
    int base_cd = (wf_id * 64 + lane) * CD_REGS;

    int4a_v va;
    for (int i = 0; i < GPRS_A; i++)
        va[i] = (int)a_in[base_a + i];

    int8_v vb;
    for (int i = 0; i < GPRS_B; i++)
        vb[i] = (int)b_in[base_b + i];

    float4_v vc;
    for (int i = 0; i < CD_REGS; i++)
        vc[i] = c_in[base_cd + i];

    uint32_t idx = 0x44444444u;   /* index-independent (all B == 1.0) */

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_f32_16x16x128_bf8_bf8 %0, %1, %2, %3\n\t"
        : "+a"(vc)
        : "v"(va), "v"(vb), "v"(idx));

    for (int i = 0; i < CD_REGS; i++)
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
    size_t cd_count = (size_t)threads * CD_REGS;

    uint32_t *h_a = (uint32_t *)calloc(a_count, sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)calloc(b_count, sizeof(uint32_t));
    float    *h_c = (float *)calloc(cd_count, sizeof(float));
    float    *h_d = (float *)calloc(cd_count, sizeof(float));

    /* bf8 1.0 = 0x3C -> 0x3C3C3C3C for both A and B; C = 0. */
    for (int t = 0; t < threads; t++) {
        for (int r = 0; r < GPRS_A; r++) h_a[t * GPRS_A + r] = BF8_ONE_x4;
        for (int r = 0; r < GPRS_B; r++) h_b[t * GPRS_B + r] = BF8_ONE_x4;
        for (int r = 0; r < CD_REGS; r++) h_c[t * CD_REGS + r] = 0.0f;
    }

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, a_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, b_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_count * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, cd_count * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, a_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, b_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_count * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, cd_count * sizeof(float)));

    hipLaunchKernelGGL(smfmac_kernel,
                       dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_count * sizeof(float), hipMemcpyDeviceToHost));

    /* K/2 = 64 products of exactly 1.0*1.0; 64.0 is exact in f32. */
    const float EXPECTED = 64.0f;

    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++) {
        for (int i = 0; i < CD_REGS; i++) {
            float got = h_d[t * CD_REGS + i];
            if (got != EXPECTED) {
                fprintf(stderr, "FAIL t=%d acc[%d]: got %.6f expected %.6f\n",
                        t, i, got, EXPECTED);
                pass = 0; break;
            }
            checked++;
        }
    }

    printf("v_smfmac_f32_16x16x128_bf8_bf8: checked %d values, expected %.1f\n",
           checked, EXPECTED);
    if (pass)
        printf("  D[0]=%.1f D[last]=%.1f\n", h_d[0], h_d[cd_count - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);
    return pass ? 0 : 1;
}
