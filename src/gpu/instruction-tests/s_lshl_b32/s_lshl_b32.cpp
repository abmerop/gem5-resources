/*
 * HIP test for S_LSHL_B32 (ENC_SOP2, scalar logical shift left).
 *
 * gem5 (src/arch/amdgpu/common/insts/sop2.cc) computes:
 *   SDST.u32 = SSRC0.u32 << (SSRC1.u32 & 0x1f)
 *   SCC      = (SDST != 0)
 *
 * Pattern B (scalar): the operands are kernel arguments, which are uniform
 * across the wavefront and therefore live in SGPRs, so the "=s"/"s"
 * constraints force the scalar S_LSHL_B32 encoding.  Only the low 5 bits of
 * the shift count are used.  All arithmetic is exact 32-bit, so the check is
 * bit-exact.
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
kernel_s_lshl_b32(uint32_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile(
        "s_lshl_b32 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { uint32_t s0, s1; } cases[] = {
        {0x00000001u, 0},  {0x00000001u, 4},  {0x000000ffu, 8},
        {0x12345678u, 1},  {0x12345678u, 16}, {0x80000000u, 1},
        {0xdeadbeefu, 12}, {0x00000003u, 31}, {0xf0f0f0f0u, 33 /* &31=1 */},
        {0xabcdef01u, 20},
    };
    const int NC = sizeof(cases) / sizeof(cases[0]);

    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NC * sizeof(uint32_t)));

    for (int c = 0; c < NC; c++)
        kernel_s_lshl_b32<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1,
                                             d_out + c);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out, d_out, NC * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < NC; c++) {
        uint32_t expected = cases[c].s0 << (cases[c].s1 & 0x1f);
        if (h_out[c] != expected) {
            fprintf(stderr, "FAIL[%d]: %08x<<%u got %08x exp %08x\n",
                    c, cases[c].s0, cases[c].s1, h_out[c], expected);
            pass = 0;
        }
    }

    if (pass)
        printf("out[3]=%08x\n", h_out[3]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
