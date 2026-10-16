/*
 * HIP test for V_MQSAD_PK_U16_U8 (VOP3) on CDNA4 (gfx950).
 *
 * Masked Quad SAD, packed into 16-bit accumulators (64-bit result):
 *
 *   d[k] = (uint16) msad_u8( bits(SRC0.u64, 24+8k .. 55+8k),
 *                            SRC1.u32, bits(SRC2.u64, 16k .. 16k+15) )
 *          for k = 0..3, packed as  D = d3:d2:d1:d0
 *
 * msad_u8 sums per-byte |s0-s1| for reference (s1) bytes that are non-zero,
 * plus the (16-bit) accumulator (matches gem5
 * Inst_VOP3__V_MQSAD_PK_U16_U8::execute exactly).  SDWA/DPP not supported.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_mqsad_pk_u16_u8 v_mqsad_pk_u16_u8.cpp
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
kernel(const u32x2 *s0, const uint32_t *s1, const u32x2 *s2, u32x2 *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    u32x2 a = s0[i];
    uint32_t b = s1[i];
    u32x2 c = s2[i];
    u32x2 d;
    __asm__ volatile("v_mqsad_pk_u16_u8 %0, %1, %2, %3\n\t"
                     : "=v"(d)
                     : "v"(a), "v"(b), "v"(c));
    out[i] = d;
}

int
main(void)
{
    u32x2 *h_s0 = (u32x2 *)malloc(N * sizeof(u32x2));
    uint32_t *h_s1 = (uint32_t *)malloc(N * sizeof(uint32_t));
    u32x2 *h_s2 = (u32x2 *)malloc(N * sizeof(u32x2));
    u32x2 *h_out = (u32x2 *)malloc(N * sizeof(u32x2));

    for (int i = 0; i < N; i++) {
        h_s0[i].x = 0x11223344u + i * 0x01010101u;
        h_s0[i].y = 0x55667788u ^ (i * 7u);
        h_s1[i] = (i & 1) ? 0x00203040u : 0x0a0b0c0du;
        /* keep accumulators small so 16-bit packing does not overflow oddly */
        h_s2[i].x = ((uint32_t)(i & 0xF) << 16) | (i & 0x1F);
        h_s2[i].y = ((uint32_t)(i & 0x7) << 16) | (i & 0x3);
    }

    u32x2 *d_s0; uint32_t *d_s1; u32x2 *d_s2, *d_out;
    GPU_CHECK(hipMalloc(&d_s0, N * sizeof(u32x2)));
    GPU_CHECK(hipMalloc(&d_s1, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_s2, N * sizeof(u32x2)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(u32x2)));
    GPU_CHECK(hipMemcpy(d_s0, h_s0, N * sizeof(u32x2), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s1, h_s1, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s2, h_s2, N * sizeof(u32x2), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_s0, d_s1, d_s2, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(u32x2), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint64_t src0 = ((uint64_t)h_s0[i].y << 32) | h_s0[i].x;
        uint64_t src2 = ((uint64_t)h_s2[i].y << 32) | h_s2[i].x;
        uint32_t s1 = h_s1[i];
        uint32_t w0 = (uint32_t)(src0 & 0xffffffffu);
        uint32_t w1 = (uint32_t)((src0 >> 8) & 0xffffffffu);
        uint32_t w2 = (uint32_t)((src0 >> 16) & 0xffffffffu);
        uint32_t w3 = (uint32_t)((src0 >> 24) & 0xffffffffu);
        uint16_t d0 = (uint16_t)msad_u8(w0, s1, (uint32_t)((src2 >> 0) & 0xffff));
        uint16_t d1 = (uint16_t)msad_u8(w1, s1, (uint32_t)((src2 >> 16) & 0xffff));
        uint16_t d2 = (uint16_t)msad_u8(w2, s1, (uint32_t)((src2 >> 32) & 0xffff));
        uint16_t d3 = (uint16_t)msad_u8(w3, s1, (uint32_t)((src2 >> 48) & 0xffff));
        uint64_t exp = ((uint64_t)d3 << 48) | ((uint64_t)d2 << 32) |
                       ((uint64_t)d1 << 16) | (uint64_t)d0;
        uint64_t got = ((uint64_t)h_out[i].y << 32) | h_out[i].x;
        if (got != exp) {
            fprintf(stderr, "FAIL[%d]: got 0x%016llx exp 0x%016llx\n",
                    i, (unsigned long long)got, (unsigned long long)exp);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x%08x\n", h_out[0].y, h_out[0].x);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_s0)); GPU_CHECK(hipFree(d_s1));
    GPU_CHECK(hipFree(d_s2)); GPU_CHECK(hipFree(d_out));
    free(h_s0); free(h_s1); free(h_s2); free(h_out);
    return pass ? 0 : 1;
}
