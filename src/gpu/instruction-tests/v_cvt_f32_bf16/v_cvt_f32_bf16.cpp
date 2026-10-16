/*
 * HIP test for V_CVT_F32_BF16 (VOP1) on CDNA4 (gfx950), including its
 * DPP and SDWA variants.
 *
 * V_CVT_F32_BF16 widens a 16-bit brain-float (bf16) to a 32-bit float:
 *
 *   VDST.f32 = bf16_to_f32(SRC0[15:0])
 *
 * bf16 is simply the upper 16 bits of an IEEE-754 f32, so the conversion is
 * exact and loss-less:  f32_bits = ((uint32)bf16_bits) << 16.  This makes the
 * self-check bit-exact (no epsilon needed).
 *
 * Three encodings are exercised, one kernel each:
 *
 *   base : v_cvt_f32_bf16     %0, %1
 *            reads bf16 from SRC0[15:0]
 *
 *   sdwa : v_cvt_f32_bf16_sdwa %0, %1 src0_sel:WORD_1
 *            reads bf16 from SRC0[31:16] (the high packed word); dst is a full
 *            DWORD so dst_sel:DWORD / UNUSED_PRESERVE defaults are legal.
 *
 *   dpp  : v_cvt_f32_bf16_dpp  %0, %1 quad_perm:[1,0,3,2] row_mask:0xf \
 *                                       bank_mask:0xf
 *            reads bf16 from SRC0[15:0] of the quad-permuted source lane; the
 *            quad_perm [1,0,3,2] swaps lanes within each pair of a 4-lane quad.
 *
 * The source operands are packed uint32 values holding two bf16 words
 * (lo = bits[15:0], hi = bits[31:16]) so the same input array feeds all three
 * encodings.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cvt_f32_bf16 v_cvt_f32_bf16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_f32_bf16
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

/* Quad-permute pattern used by the DPP kernel. */
static const int QP[4] = {1, 0, 3, 2};

static inline uint32_t
bf16_to_f32_bits(uint16_t bf)
{
    return ((uint32_t)bf) << 16;
}

/*
 * Kernel base: v_cvt_f32_bf16 from SRC0[15:0].
 */
__global__ void
kernel_base(const uint32_t *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t src = in[i];
    float vd;
    __asm__ volatile("v_cvt_f32_bf16 %0, %1\n\t"
                     : "=v"(vd)
                     : "v"(src));
    out[i] = vd;
}

/*
 * Kernel sdwa: v_cvt_f32_bf16_sdwa ... src0_sel:WORD_1 reads SRC0[31:16].
 */
__global__ void
kernel_sdwa(const uint32_t *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t src = in[i];
    float vd;
    __asm__ volatile("v_cvt_f32_bf16_sdwa %0, %1 src0_sel:WORD_1\n\t"
                     : "=v"(vd)
                     : "v"(src));
    out[i] = vd;
}

/*
 * Kernel dpp: v_cvt_f32_bf16_dpp ... quad_perm:[1,0,3,2].  The source bf16
 * (SRC0[15:0]) is taken from the quad-permuted lane.
 */
__global__ void
kernel_dpp(const uint32_t *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t src = in[i];
    float vd;
    __asm__ volatile(
        "v_cvt_f32_bf16_dpp %0, %1 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf\n\t"
        : "=v"(vd)
        : "v"(src));
    out[i] = vd;
}

int
main(void)
{
    uint32_t *h_in  = (uint32_t *)malloc(N * sizeof(uint32_t));
    float    *h_ob  = (float *)malloc(N * sizeof(float));
    float    *h_os  = (float *)malloc(N * sizeof(float));
    float    *h_od  = (float *)malloc(N * sizeof(float));

    /* Pack two distinct bf16 words per element so lo/hi mix-ups are caught. */
    for (int i = 0; i < N; i++) {
        uint16_t lo = (uint16_t)(0x3F80 + i);       /* ~1.0 .. increasing   */
        uint16_t hi = (uint16_t)(0xC000 + (i * 7)); /* negative-ish spread  */
        h_in[i] = ((uint32_t)hi << 16) | lo;
    }

    uint32_t *d_in;
    float    *d_ob, *d_os, *d_od;
    GPU_CHECK(hipMalloc(&d_in, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_ob, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_os, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_od, N * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_in, h_in, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_base, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_ob, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_sdwa, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_os, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_dpp, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_od, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_ob, d_ob, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_os, d_os, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_od, d_od, N * sizeof(float), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint16_t lo = (uint16_t)(h_in[i] & 0xFFFF);
        uint16_t hi = (uint16_t)(h_in[i] >> 16);

        /* base: low bf16 */
        uint32_t exp_b = bf16_to_f32_bits(lo);
        uint32_t got_b;
        memcpy(&got_b, &h_ob[i], 4);
        if (got_b != exp_b) {
            fprintf(stderr, "FAIL base[%d]: got 0x%08x expected 0x%08x\n",
                    i, got_b, exp_b);
            pass = 0; break;
        }

        /* sdwa WORD_1: high bf16 */
        uint32_t exp_s = bf16_to_f32_bits(hi);
        uint32_t got_s;
        memcpy(&got_s, &h_os[i], 4);
        if (got_s != exp_s) {
            fprintf(stderr, "FAIL sdwa[%d]: got 0x%08x expected 0x%08x\n",
                    i, got_s, exp_s);
            pass = 0; break;
        }

        /* dpp quad_perm:[1,0,3,2]: low bf16 from permuted lane */
        int block = i / BLOCK_SIZE;
        int lane  = i % BLOCK_SIZE;
        int j     = lane & 3;
        int srclane = (lane & ~3) + QP[j];
        int srcidx  = block * BLOCK_SIZE + srclane;
        uint16_t src_lo = (uint16_t)(h_in[srcidx] & 0xFFFF);
        uint32_t exp_d  = bf16_to_f32_bits(src_lo);
        uint32_t got_d;
        memcpy(&got_d, &h_od[i], 4);
        if (got_d != exp_d) {
            fprintf(stderr, "FAIL dpp[%d]: got 0x%08x expected 0x%08x\n",
                    i, got_d, exp_d);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("base[0]=0x%08x sdwa[0]=0x%08x dpp[0]=0x%08x\n",
               *(uint32_t *)&h_ob[0], *(uint32_t *)&h_os[0],
               *(uint32_t *)&h_od[0]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in));
    GPU_CHECK(hipFree(d_ob));
    GPU_CHECK(hipFree(d_os));
    GPU_CHECK(hipFree(d_od));
    free(h_in);
    free(h_ob);
    free(h_os);
    free(h_od);

    return pass ? 0 : 1;
}
