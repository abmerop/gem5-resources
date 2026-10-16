/*
 * HIP test for v_smfmac_f32_16x16x64_bf8_fp8 (sparse MFMA, BF8/E5M2 A / FP8/E4M3 B, f32
 * accumulator, gfx950 / CDNA4).  Pattern D.
 *
 *   D.f32 = A_sparse.bf8 * B.fp8 + C.f32   (4:2 structured sparsity, K=64)
 *
 * Operand register layout (per wavefront, 64 lanes), from
 * `llvm-mc -mcpu=gfx950 --show-encoding`:
 *   SRC0 (A, sparse): 2 VGPRs   SRC1 (B, dense): 4 VGPRs
 *   SRC2 (index): 1 VGPR           C/D (acc): 4 AGPRs of f32 (16*16/64 = 4)
 *
 * Encodings of 1.0:  FP8/E4M3 = 0x38,  BF8/E5M2 = 0x3C.
 * Every A and B byte = 1.0, C = 0.  Sparse MFMA accumulates K/2 = 32
 * non-zero products, each 1.0*1.0 = 1.0; 32.0 is exact in f32, so every
 * output == 32.0 (strict ==), independent of the sparsity index.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_smfmac_f32_16x16x64_bf8_fp8 v_smfmac_f32_16x16x64_bf8_fp8.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                   \
        hipError_t e = (cmd);                                              \
        if (e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                                 \
            exit(1);                                                       \
        }                                                                 \
    } while (0)

static const int CD_REGS = 4;
static const int GPRS_A  = 2;
static const int GPRS_B  = 4;

static const uint32_t A_ONE_x4 = 0x3C3C3C3Cu;
static const uint32_t B_ONE_x4 = 0x38383838u;

typedef __attribute__((ext_vector_type(2)))  int   intA_v;
typedef __attribute__((ext_vector_type(4)))  int   intB_v;
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

    intA_v va;
    for (int i = 0; i < GPRS_A; i++) va[i] = (int)a_in[base_a + i];
    intB_v vb;
    for (int i = 0; i < GPRS_B; i++) vb[i] = (int)b_in[base_b + i];
    float4_v vc;
    for (int i = 0; i < CD_REGS; i++) vc[i] = c_in[base_cd + i];

    uint32_t idx = 0x44444444u;

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_f32_16x16x64_bf8_fp8 %0, %1, %2, %3\n\t"
        : "+a"(vc)
        : "v"(va), "v"(vb), "v"(idx));

    for (int i = 0; i < CD_REGS; i++) d_out[base_cd + i] = vc[i];
}

int
main(void)
{
    const int num_waves = 4, wf_size = 64;
    const int threads = num_waves * wf_size;
    size_t a_count = (size_t)threads * GPRS_A;
    size_t b_count = (size_t)threads * GPRS_B;
    size_t cd_count = (size_t)threads * CD_REGS;

    uint32_t *h_a = (uint32_t *)calloc(a_count, sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)calloc(b_count, sizeof(uint32_t));
    float *h_c = (float *)calloc(cd_count, sizeof(float));
    float *h_d = (float *)calloc(cd_count, sizeof(float));

    for (int t = 0; t < threads; t++) {
        for (int r = 0; r < GPRS_A; r++) h_a[t * GPRS_A + r] = A_ONE_x4;
        for (int r = 0; r < GPRS_B; r++) h_b[t * GPRS_B + r] = B_ONE_x4;
        for (int r = 0; r < CD_REGS; r++) h_c[t * CD_REGS + r] = 0.0f;
    }

    uint32_t *d_a, *d_b;
    float *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, a_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, b_count * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, cd_count * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, cd_count * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a, h_a, a_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, b_count * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_count * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, cd_count * sizeof(float)));

    hipLaunchKernelGGL(smfmac_kernel, dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d, d_d, cd_count * sizeof(float), hipMemcpyDeviceToHost));

    const float EXPECTED = 32.0f;
    int pass = 1, checked = 0;
    for (int t = 0; t < threads && pass; t++)
        for (int i = 0; i < CD_REGS; i++) {
            float got = h_d[t * CD_REGS + i];
            if (got != EXPECTED) {
                fprintf(stderr, "FAIL t=%d acc[%d]: got %.6f expected %.6f\n",
                        t, i, got, EXPECTED);
                pass = 0; break;
            }
            checked++;
        }
    printf("v_smfmac_f32_16x16x64_bf8_fp8: checked %d values, expected %.1f\n", checked, EXPECTED);
    if (pass) printf("  D[0]=%.1f D[last]=%.1f\n", h_d[0], h_d[cd_count - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_d));
    free(h_a); free(h_b); free(h_c); free(h_d);
    return pass ? 0 : 1;
}
