/*
 * HIP test for V_DOT2C_F32_BF16 (VOP2) on CDNA4 (gfx950), plus its DPP variant.
 *
 * V_DOT2C_F32_BF16 is a 2-element bf16 dot product that ACCUMULATES into the
 * destination:
 *
 *   VDST.f32 += bf16_to_f32(SRC0[15:0])  * bf16_to_f32(VSRC1[15:0])
 *   VDST.f32 += bf16_to_f32(SRC0[31:16]) * bf16_to_f32(VSRC1[31:16])
 *
 * (matches gem5 Inst_VOP2__V_DOT2C_F32_BF16::execute: two separate f32 adds.)
 * SDWA is not supported for this opcode; only base and DPP are exercised.
 *
 * To keep the self-check bit-exact, all bf16 lanes carry small integers
 * (<= 8), which are represented exactly by bf16 (8-bit mantissa) and whose
 * products/sums are exact in f32.
 *
 * base : v_dot2c_f32_bf16 %0, %1, %2                 (%0 accumulator)
 * dpp  : v_dot2c_f32_bf16 %0, %1, %2 quad_perm:[1,0,3,2] ...
 *          SRC0 is taken from the quad-permuted lane.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_dot2c_f32_bf16 v_dot2c_f32_bf16.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;
static const int QP[4]      = {1, 0, 3, 2};

/* bf16 bits of an exactly-representable small integer. */
static inline uint16_t
bf16_of_int(int k)
{
    float f = (float)k;
    uint32_t fb;
    memcpy(&fb, &f, 4);
    return (uint16_t)(fb >> 16);
}
static inline float
bf16_to_f32(uint16_t b)
{
    uint32_t fb = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &fb, 4);
    return f;
}

__global__ void
kernel_base(const uint32_t *s0, const uint32_t *s1, const float *acc,
            float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t a = s0[i], b = s1[i];
    float d = acc[i];
    __asm__ volatile("v_dot2c_f32_bf16 %0, %1, %2\n\t"
                     : "+v"(d)
                     : "v"(a), "v"(b));
    out[i] = d;
}

__global__ void
kernel_dpp(const uint32_t *s0, const uint32_t *s1, const float *acc,
           float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t a = s0[i], b = s1[i];
    float d = acc[i];
    __asm__ volatile(
        "v_dot2c_f32_bf16 %0, %1, %2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf\n\t"
        : "+v"(d)
        : "v"(a), "v"(b));
    out[i] = d;
}

int
main(void)
{
    uint32_t *h_s0 = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_s1 = (uint32_t *)malloc(N * sizeof(uint32_t));
    float    *h_ac = (float *)malloc(N * sizeof(float));
    float    *h_ob = (float *)malloc(N * sizeof(float));
    float    *h_od = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        uint16_t a1 = bf16_of_int((i % 7) + 1);
        uint16_t a2 = bf16_of_int((i % 5) + 1);
        uint16_t b1 = bf16_of_int((i % 3) + 1);
        uint16_t b2 = bf16_of_int((i % 4) + 1);
        h_s0[i] = ((uint32_t)a2 << 16) | a1;
        h_s1[i] = ((uint32_t)b2 << 16) | b1;
        h_ac[i] = (float)(i % 9);
    }

    uint32_t *d_s0, *d_s1;
    float    *d_ac, *d_ob, *d_od;
    GPU_CHECK(hipMalloc(&d_s0, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_s1, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_ac, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_ob, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_od, N * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_s0, h_s0, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s1, h_s1, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_ac, h_ac, N * sizeof(float), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_base, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_s0, d_s1, d_ac, d_ob, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_dpp, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_s0, d_s1, d_ac, d_od, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_ob, d_ob, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_od, d_od, N * sizeof(float), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        /* base: own lane */
        float e_b = h_ac[i];
        e_b += bf16_to_f32(h_s0[i] & 0xFFFF) * bf16_to_f32(h_s1[i] & 0xFFFF);
        e_b += bf16_to_f32(h_s0[i] >> 16) * bf16_to_f32(h_s1[i] >> 16);
        if (h_ob[i] != e_b) {
            fprintf(stderr, "FAIL base[%d]: got %f expected %f\n",
                    i, h_ob[i], e_b);
            pass = 0; break;
        }
        /* dpp: SRC0 from permuted lane */
        int block = i / BLOCK_SIZE, lane = i % BLOCK_SIZE, j = lane & 3;
        int srcidx = block * BLOCK_SIZE + ((lane & ~3) + QP[j]);
        uint32_t s0p = h_s0[srcidx];
        float e_d = h_ac[i];
        e_d += bf16_to_f32(s0p & 0xFFFF) * bf16_to_f32(h_s1[i] & 0xFFFF);
        e_d += bf16_to_f32(s0p >> 16) * bf16_to_f32(h_s1[i] >> 16);
        if (h_od[i] != e_d) {
            fprintf(stderr, "FAIL dpp[%d]: got %f expected %f\n",
                    i, h_od[i], e_d);
            pass = 0; break;
        }
    }

    if (pass)
        printf("base[0]=%f dpp[0]=%f base[255]=%f\n",
               h_ob[0], h_od[0], h_ob[255]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_s0)); GPU_CHECK(hipFree(d_s1));
    GPU_CHECK(hipFree(d_ac)); GPU_CHECK(hipFree(d_ob)); GPU_CHECK(hipFree(d_od));
    free(h_s0); free(h_s1); free(h_ac); free(h_ob); free(h_od);
    return pass ? 0 : 1;
}
