/*
 * HIP test for V_MQSAD_U32_U8 (VOP3) on CDNA4 (gfx950).
 *
 * Masked Quad Sum-of-Absolute-Differences with 32-bit accumulators:
 *
 *   d[k] = msad_u8( bits(SRC0.u64, 24+8k .. 55+8k), SRC1.u32, SRC2[k].u32 )
 *          for k = 0..3   (four sliding 32-bit windows over the 64-bit SRC0)
 *
 * where msad_u8(s0,s1,s2) sums, per byte, |s0.byte - s1.byte| but only for
 * bytes whose s1 (reference) byte is non-zero, then adds s2:
 *
 *   tmp = s2; for b in 0..3: if s1.byte[b]!=0: tmp += |s0.byte[b]-s1.byte[b]|
 *
 * (matches gem5 Inst_VOP3__V_MQSAD_U32_U8::execute exactly).
 * Result is 128 bits -> 4 VGPRs.  SDWA/DPP are not supported for this opcode.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_mqsad_u32_u8 v_mqsad_u32_u8.cpp
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

typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

static inline uint32_t
byte(uint32_t v, int b)
{
    return (v >> (8 * b)) & 0xFF;
}
static uint32_t
msad_u8(uint32_t s0, uint32_t s1, uint32_t s2)
{
    uint32_t tmp = s2;
    for (int b = 0; b < 4; b++) {
        uint32_t s1b = byte(s1, b), s0b = byte(s0, b);
        if (s1b)
            tmp += (s0b > s1b) ? (s0b - s1b) : (s1b - s0b);
    }
    return tmp;
}

__global__ void
kernel(const u32x2 *s0, const uint32_t *s1, const u32x4 *s2, u32x4 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    u32x2 a = s0[i];
    uint32_t b = s1[i];
    u32x4 c = s2[i];
    u32x4 d;
    __asm__ volatile("v_mqsad_u32_u8 %0, %1, %2, %3\n\t"
                     : "=v"(d)
                     : "v"(a), "v"(b), "v"(c));
    out[i] = d;
}

int
main(void)
{
    u32x2 *h_s0 = (u32x2 *)malloc(N * sizeof(u32x2));
    uint32_t *h_s1 = (uint32_t *)malloc(N * sizeof(uint32_t));
    u32x4 *h_s2 = (u32x4 *)malloc(N * sizeof(u32x4));
    u32x4 *h_out = (u32x4 *)malloc(N * sizeof(u32x4));

    for (int i = 0; i < N; i++) {
        h_s0[i].x = 0x11223344u + i * 0x01010101u;
        h_s0[i].y = 0x55667788u ^ (i * 7u);
        /* include some zero reference bytes to exercise the mask */
        h_s1[i] = (i & 1) ? 0x00203040u : 0x0a0b0c0du;
        h_s2[i].x = i;
        h_s2[i].y = 2 * i;
        h_s2[i].z = 3 * i;
        h_s2[i].w = 4 * i;
    }

    u32x2 *d_s0; uint32_t *d_s1; u32x4 *d_s2, *d_out;
    GPU_CHECK(hipMalloc(&d_s0, N * sizeof(u32x2)));
    GPU_CHECK(hipMalloc(&d_s1, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_s2, N * sizeof(u32x4)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(u32x4)));
    GPU_CHECK(hipMemcpy(d_s0, h_s0, N * sizeof(u32x2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s1, h_s1, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s2, h_s2, N * sizeof(u32x4), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_s0, d_s1, d_s2, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(u32x4), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint64_t src0 = ((uint64_t)h_s0[i].y << 32) | h_s0[i].x;
        uint32_t s1 = h_s1[i];
        uint32_t w0 = (uint32_t)(src0 & 0xffffffffu);
        uint32_t w1 = (uint32_t)((src0 >> 8) & 0xffffffffu);
        uint32_t w2 = (uint32_t)((src0 >> 16) & 0xffffffffu);
        uint32_t w3 = (uint32_t)((src0 >> 24) & 0xffffffffu);
        uint32_t e0 = msad_u8(w0, s1, h_s2[i].x);
        uint32_t e1 = msad_u8(w1, s1, h_s2[i].y);
        uint32_t e2 = msad_u8(w2, s1, h_s2[i].z);
        uint32_t e3 = msad_u8(w3, s1, h_s2[i].w);
        if (h_out[i].x != e0 || h_out[i].y != e1 ||
            h_out[i].z != e2 || h_out[i].w != e3) {
            fprintf(stderr,
                "FAIL[%d]: got %u,%u,%u,%u exp %u,%u,%u,%u\n", i,
                h_out[i].x, h_out[i].y, h_out[i].z, h_out[i].w,
                e0, e1, e2, e3);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%u,%u,%u,%u\n",
               h_out[0].x, h_out[0].y, h_out[0].z, h_out[0].w);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_s0)); GPU_CHECK(hipFree(d_s1));
    GPU_CHECK(hipFree(d_s2)); GPU_CHECK(hipFree(d_out));
    free(h_s0); free(h_s1); free(h_s2); free(h_out);
    return pass ? 0 : 1;
}
