/*
 * HIP test for S_SUB_I32 (SOP2 signed 32-bit subtract (D = S0 - S1, sets SCC)).
 *
 * Verified against two's-complement wrapping subtraction.
 *
 * Pattern B (scalar): operands are kernel arguments (already in SGPRs); one
 * wavefront per block computes one result, broadcast via SDST; lane 0 writes
 * it.  One block is launched per test case.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o s_sub_i32 s_sub_i32.cpp
 * Run:     gem5.opt config.py -- ./s_sub_i32
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

static const int BLOCK_SIZE = 64;

__global__ void
kernel_s_sub_i32(int32_t s0, int32_t s1, int32_t *out)
{
    int32_t sdst;
    __asm__ volatile(
        "s_sub_i32 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { int32_t s0, s1; } cases[] = {
        {0,0},{5,3},{-4,7},{100,250},{-2147483647-1,1},{12345,-678},{0,-1},{2000000000,-2000000000},
    };
    const int nc = (int)(sizeof(cases) / sizeof(cases[0]));

    int32_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, nc * sizeof(int32_t)));

    for (int c = 0; c < nc; c++) {
        kernel_s_sub_i32<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out + c);
        GPU_CHECK(hipGetLastError());
    }
    GPU_CHECK(hipDeviceSynchronize());

    int32_t *h_out = (int32_t *)malloc(nc * sizeof(int32_t));
    GPU_CHECK(hipMemcpy(h_out, d_out, nc * sizeof(int32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < nc; c++) {
        int32_t s0 = cases[c].s0, s1 = cases[c].s1;
        int32_t expected = ((int32_t)((uint32_t)s0 - (uint32_t)s1));
        if (h_out[c] != expected) {
            fprintf(stderr, "FAIL case %d: got 0x%llx expected 0x%llx\n",
                    c, (unsigned long long)h_out[c],
                    (unsigned long long)expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%llx\n", (unsigned long long)h_out[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);
    return pass ? 0 : 1;
}
