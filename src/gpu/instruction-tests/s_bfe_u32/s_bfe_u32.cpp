/*
 * HIP test for S_BFE_U32 (SOP2 unsigned bit-field extract (D = (S0 >> off) & ((1<<width)-1))).
 *
 * S1 packs offset in bits [4:0] and width in bits [22:16]. Matches gem5 sop2.cc exactly; widths kept in 1..20 to avoid 1<<32 UB in the host ref.
 *
 * Pattern B (scalar): operands are kernel arguments (already in SGPRs); one
 * wavefront per block computes one result, broadcast via SDST; lane 0 writes
 * it.  One block is launched per test case.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o s_bfe_u32 s_bfe_u32.cpp
 * Run:     gem5.opt config.py -- ./s_bfe_u32
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
kernel_s_bfe_u32(uint32_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile(
        "s_bfe_u32 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { uint32_t s0, s1; } cases[] = {
        {0xdeadbeefu,(0u)|(8u<<16)},{0xdeadbeefu,(8u)|(8u<<16)},{0xdeadbeefu,(16u)|(8u<<16)},{0xdeadbeefu,(24u)|(8u<<16)},{0xffffffffu,(4u)|(12u<<16)},{0x80000000u,(31u)|(1u<<16)},{0x12345678u,(2u)|(20u<<16)},
    };
    const int nc = (int)(sizeof(cases) / sizeof(cases[0]));

    uint32_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, nc * sizeof(uint32_t)));

    for (int c = 0; c < nc; c++) {
        kernel_s_bfe_u32<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out + c);
        GPU_CHECK(hipGetLastError());
    }
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t *h_out = (uint32_t *)malloc(nc * sizeof(uint32_t));
    GPU_CHECK(hipMemcpy(h_out, d_out, nc * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint32_t s0 = cases[c].s0, s1 = cases[c].s1;
        uint32_t expected = ((uint32_t)((s0 >> (s1 & 0x1fu)) & ((1u << ((s1 >> 16) & 0x7fu)) - 1u)));
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
