/*
 * HIP test for V_BITOP3_B32 (VOP3 programmable 3-input bitwise LUT).
 *
 * V_BITOP3_B32 computes, per bit, D = LUT(S0, S1, S2) where the 8-bit truth
 * table selects, for each (s0,s1,s2) bit combination, the output bit:
 *   out_bit = (lut >> ((s0<<2) | (s1<<1) | s2)) & 1
 *
 * On gfx950 the 8-bit truth table is split across the VOP3 NEG[2:0],
 * ABS[5:3] and OMOD[7:6] fields; the assembler packs `bitop3:0xNN` into them
 * and gem5 (vop3.cc) reconstructs the same table.  We use bitop3:0x96, which
 * is the XOR3 truth table, so the reference is simply S0 ^ S1 ^ S2.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_bitop3_b32 v_bitop3_b32.cpp
 * Run:     gem5.opt config.py -- ./v_bitop3_b32
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_bitop3_b32(const uint32_t *a, const uint32_t *b, const uint32_t *c,
                    uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i], vb = b[i], vc = c[i], vd;
    __asm__ volatile(
        "v_bitop3_b32 %0, %1, %2, %3 bitop3:0x96\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb), "v"(vc));

    out[i] = vd;
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_c = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint32_t)i * 2654435761u;
        h_b[i] = (uint32_t)(i * 40503u) ^ 0x5a5a5a5au;
        h_c[i] = (uint32_t)(~i * 2246822519u) ^ 0x0f0f0f0fu;
    }

    uint32_t *d_a, *d_b, *d_c, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_bitop3_b32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_c, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = h_a[i] ^ h_b[i] ^ h_c[i];   /* bitop3:0x96 */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_c); free(h_out);
    return pass ? 0 : 1;
}
