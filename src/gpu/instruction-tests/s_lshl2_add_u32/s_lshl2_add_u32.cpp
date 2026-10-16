/*
 * HIP test for S_LSHL2_ADD_U32 (SOP2 scalar left-shift-by-2 and add).
 *
 * S_LSHL2_ADD_U32 is a SOP2 instruction that shifts S0 left by two bits
 * and adds S1, storing the result in SDST and updating SCC:
 *
 *   SDST = (S0.u << 2) + S1.u
 *   SCC  = (SDST != 0) ? 1 : 0
 *
 * This is equivalent to multiplying S0 by 4 and adding S1, which is the
 * natural pattern for 32-bit element array indexing:
 *
 *   byte_offset = element_index * sizeof(uint32_t) + base_offset
 *               = (element_index << 2) + base_offset
 *
 * Both operands and the destination are scalar registers (SGPRs).  Because
 * kernel arguments are passed in SGPRs, passing s0 and s1 directly as
 * kernel arguments guarantees the compiler can address them with scalar
 * registers, making s_lshl2_add_u32 the natural choice.
 *
 * The test uses inline assembly to force the exact instruction, then
 * verifies the result against the equivalent C expression.
 *
 * Build:
 *   hipcc -O2 -o s_lshl2_add_u32 s_lshl2_add_u32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_lshl2_add_u32
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

/*
 * s_lshl2_add_u32_kernel
 *
 * Each wavefront calls the instruction once with the same scalar operands
 * (s0 = element index, s1 = base offset), which are kernel arguments stored
 * in SGPRs.  The single result is broadcast to all threads in the wavefront
 * via the shared SDST register; lane 0 of each block writes it to out[].
 *
 * s0    - first source (scalar, left-shifted by 2)
 * s1    - second source (scalar, added after shift)
 * out   - per-block output array; out[blockIdx.x] receives SDST
 * n     - number of output slots
 */
__global__ void
s_lshl2_add_u32_kernel(uint32_t s0, uint32_t s1, uint32_t *out, int n)
{
    if ((int)blockIdx.x >= n)
        return;

    uint32_t sdst;

    /*
     * s_lshl2_add_u32 SDST, S0, S1
     *
     * SDST = (S0 << 2) + S1
     *
     * "=s" — output scalar register (SGPR)
     * "s"  — input scalar register (SGPR); kernel args s0, s1 already
     *         live in SGPRs so the compiler places them there directly.
     */
    __asm__ volatile(
        "s_lshl2_add_u32 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));

    /* All lanes in the wavefront share the same SDST value.  Only lane 0
     * of each block writes it so we get one result entry per block. */
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sdst;
    }
}

int
main(void)
{
    /* Test cases: (s0, s1) pairs with expected result (s0 << 2) + s1. */
    struct {
        uint32_t s0, s1, expected;
    } cases[] = {
        { 0,          0,          0           },  /* zero case            */
        { 1,          0,          4           },  /* pure shift           */
        { 0,          7,          7           },  /* pure add             */
        { 3,          5,          17          },  /* 3*4 + 5              */
        { 0x3fffffff, 3,          0xFFFFFFFF  },  /* near max, no wrap    */
        { 0x40000000, 0,          0x00000000  },  /* overflow wraps to 0  */
        { 100,        28,         428         },  /* typical array index  */
        { 0xdeadbeef, 0x12345678, 0x8ceb5234 },  /* arbitrary            */
    };
    const int num_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    /* ----- device buffer: one slot per test case ----- */
    uint32_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, num_cases * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_out, 0xFF, num_cases * sizeof(uint32_t)));

    /* Launch one block per test case; each block uses its own (s0, s1). */
    for (int c = 0; c < num_cases; c++) {
        s_lshl2_add_u32_kernel<<<1, BLOCK_SIZE>>>(
            cases[c].s0, cases[c].s1, d_out + c, 1);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    }

    /* ----- copy back and verify ----- */
    uint32_t h_out[num_cases];
    GPU_CHECK(hipMemcpy(h_out, d_out, num_cases * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    printf("s_lshl2_add_u32 test  (SDST = (S0 << 2) + S1)\n");
    printf("  %-12s %-12s %-12s %-12s %s\n",
           "S0", "S1", "expected", "got", "");
    for (int c = 0; c < num_cases; c++) {
        int ok = (h_out[c] == cases[c].expected);
        printf("  0x%08X   0x%08X   0x%08X   0x%08X   %s\n",
               cases[c].s0, cases[c].s1, cases[c].expected, h_out[c],
               ok ? "OK" : "FAIL");
        if (!ok)
            pass = 0;
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
