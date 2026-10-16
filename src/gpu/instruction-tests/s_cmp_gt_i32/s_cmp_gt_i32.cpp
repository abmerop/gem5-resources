/*
 * HIP test for S_CMP_GT_I32 (ENC_SOPC, scalar signed greater-than compare).
 *
 * gem5 (src/arch/amdgpu/common/insts/sopc.cc) computes:
 *   SCC = (SSRC0.i32 > SSRC1.i32)        (signed)
 *
 * S_CMP_* writes only SCC, so the test captures SCC with a following
 * S_CSELECT_B32 (sdst = SCC ? 1 : 0).  Both operands are scalar kernel
 * arguments ("s" constraints) so the scalar SOPC encoding is emitted.  The
 * cases cover positive/negative, equal, and sign-boundary inputs.
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
kernel_s_cmp_gt_i32(int32_t s0, int32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile(
        "s_cmp_gt_i32 %1, %2\n\t"
        "s_cselect_b32 %0, 1, 0\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { int32_t s0, s1; } cases[] = {
        {5, 3}, {3, 5}, {7, 7}, {-1, -2}, {-2, -1}, {-1, 0},
        {0, -1}, {2147483647, -2147483648}, {-2147483648, 2147483647},
        {100, 100}, {-100, -50}, {123456, 123455},
    };
    const int NC = sizeof(cases) / sizeof(cases[0]);

    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NC * sizeof(uint32_t)));

    for (int c = 0; c < NC; c++)
        kernel_s_cmp_gt_i32<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1,
                                               d_out + c);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out, d_out, NC * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < NC; c++) {
        uint32_t expected = (cases[c].s0 > cases[c].s1) ? 1u : 0u;
        if (h_out[c] != expected) {
            fprintf(stderr, "FAIL[%d]: (%d>%d) got %u exp %u\n", c,
                    cases[c].s0, cases[c].s1, h_out[c], expected);
            pass = 0;
        }
    }

    if (pass)
        printf("out[0]=%u out[1]=%u\n", h_out[0], h_out[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
