/*
 * HIP test for S_ORN2_B64 (SOP2 64-bit OR-NOT2 (D = S0 | ~S1)).
 *
 * Bitwise OR of S0 with the complement of S1, full 64 bits.
 *
 * Pattern B (scalar): operands are kernel arguments (already in SGPRs); one
 * wavefront per block computes one result, broadcast via SDST; lane 0 writes
 * it.  One block is launched per test case.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o s_orn2_b64 s_orn2_b64.cpp
 * Run:     gem5.opt config.py -- ./s_orn2_b64
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
kernel_s_orn2_b64(uint64_t s0, uint64_t s1, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile(
        "s_orn2_b64 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { uint64_t s0, s1; } cases[] = {
        {0,0},{0xffffffffffffffffULL,0},{0,0xffffffffffffffffULL},{0x0123456789abcdefULL,0xfedcba9876543210ULL},{0xaaaaaaaaaaaaaaaaULL,0x5555555555555555ULL},{0xdeadbeefULL,0xf0f0f0f0ULL},
    };
    const int nc = (int)(sizeof(cases) / sizeof(cases[0]));

    uint64_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, nc * sizeof(uint64_t)));

    for (int c = 0; c < nc; c++) {
        kernel_s_orn2_b64<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out + c);
        GPU_CHECK(hipGetLastError());
    }
    GPU_CHECK(hipDeviceSynchronize());

    uint64_t *h_out = (uint64_t *)malloc(nc * sizeof(uint64_t));
    GPU_CHECK(hipMemcpy(h_out, d_out, nc * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint64_t s0 = cases[c].s0, s1 = cases[c].s1;
        uint64_t expected = ((uint64_t)(s0 | ~s1));
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
