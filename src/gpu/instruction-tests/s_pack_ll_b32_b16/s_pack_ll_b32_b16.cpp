/*
 * HIP test for s_pack_ll_b32_b16 (SOP2 scalar pack low-low).
 *
 * s_pack_ll_b32_b16 packs the LOW 16-bit half of two scalar registers
 * into a single 32-bit destination:
 *
 *   SDST[15:0]  = SSRC0[15:0]   (low half of first source)
 *   SDST[31:16] = SSRC1[15:0]   (low half of second source)
 *
 * The "LL" suffix means: Low half of SSRC0 → Low half of SDST,
 *                        Low half of SSRC1 → High half of SDST.
 *
 * This is a SOP2 instruction (scalar, wavefront-uniform) so both source
 * operands and the destination are SGPRs.  Passing values as kernel
 * arguments ensures they live in SGPRs, making the instruction a natural
 * fit without needing to move data from VGPRs.
 *
 * The companion instructions s_pack_lh and s_pack_hh select different
 * 16-bit halves of the sources.
 *
 * Build:
 *   hipcc -O2 -o s_pack_ll_b32_b16 s_pack_ll_b32_b16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_pack_ll_b32_b16
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

static const int NUM_WAVEFRONTS    = 8;
static const int THREADS_PER_BLOCK = 64;   /* one wavefront on GFX9 */

/*
 * s_pack_ll_b32_b16_kernel
 *
 * Each block receives two 32-bit scalar values (s0, s1) as kernel arguments
 * (which already live in SGPRs) and emits one s_pack_ll_b32_b16 instruction.
 * The result is the same for every lane in the wavefront; only lane 0 writes
 * it to the output buffer.
 *
 * "=s" — output SGPR (destination)
 * "s"  — input SGPR (both sources are kernel args, already in SGPRs)
 */
__global__ void
s_pack_ll_b32_b16_kernel(uint32_t s0, uint32_t s1, uint32_t *out, int n)
{
    int wf_id = blockIdx.x;
    if (wf_id >= n)
        return;

    uint32_t sdst;

    __asm__ volatile(
        "s_pack_ll_b32_b16 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));

    /* All lanes share the same SGPR result; only lane 0 writes. */
    if (threadIdx.x == 0)
        out[wf_id] = sdst;
}

int
main(void)
{
    /*
     * Test cases: (s0, s1) → expected SDST where
     *   SDST[15:0]  = s0[15:0]
     *   SDST[31:16] = s1[15:0]
     */
    struct {
        uint32_t s0, s1, expected;
    } cases[] = {
        /* Both zero */
        { 0x00000000, 0x00000000, 0x00000000 },
        /* Low halves only */
        { 0x0000ABCD, 0x00001234, 0x1234ABCD },
        /* High halves of sources are IGNORED */
        { 0xFFFFABCD, 0xFFFF1234, 0x1234ABCD },
        /* Typical use: pack two 16-bit values */
        { 0x00003F80, 0x00004000, 0x40003F80 }, /* bf16(1.0)|bf16(2.0) */
        /* All-ones low halves */
        { 0x0000FFFF, 0x0000FFFF, 0xFFFFFFFF },
        /* s1 low=0, s0 low=anything */
        { 0x00005A5A, 0x00000000, 0x00005A5A },
        /* s0 low=0, s1 low=anything */
        { 0x00000000, 0x0000A5A5, 0xA5A50000 },
        /* Realistic: pack two 16-bit indices */
        { 0x00000010, 0x00000020, 0x00200010 },
    };
    const int num_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    /* ----- device output buffer (one slot per wavefront) ----- */
    uint32_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, num_cases * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_out, 0xFF, num_cases * sizeof(uint32_t)));

    /* Launch one block per test case; each block uses the same (s0, s1). */
    for (int c = 0; c < num_cases; c++) {
        s_pack_ll_b32_b16_kernel<<<1, THREADS_PER_BLOCK>>>(
            cases[c].s0, cases[c].s1, d_out + c, 1);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    }

    /* ----- copy back and verify ----- */
    uint32_t h_out[num_cases];
    GPU_CHECK(hipMemcpy(h_out, d_out, num_cases * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    printf("s_pack_ll_b32_b16 test  "
           "(SDST[15:0]=S0[15:0], SDST[31:16]=S1[15:0])\n");
    printf("  %-12s %-12s %-12s %-12s %s\n",
           "S0", "S1", "expected", "got", "");

    for (int c = 0; c < num_cases; c++) {
        int ok = (h_out[c] == cases[c].expected);
        printf("  0x%08X   0x%08X   0x%08X   0x%08X   %s\n",
               cases[c].s0, cases[c].s1,
               cases[c].expected, h_out[c],
               ok ? "OK" : "FAIL");
        if (!ok)
            pass = 0;
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
