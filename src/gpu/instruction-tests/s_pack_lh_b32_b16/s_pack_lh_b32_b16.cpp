/*
 * HIP test for s_pack_lh_b32_b16 (SOP2 scalar pack low-high).
 *
 * s_pack_lh_b32_b16 packs one 16-bit half from each of two scalar registers
 * into a single 32-bit destination:
 *
 *   SDST[15:0]  = SSRC0[15:0]   (low  half of SSRC0 → low  half of SDST)
 *   SDST[31:16] = SSRC1[31:16]  (HIGH half of SSRC1 → high half of SDST)
 *
 * The "LH" suffix means: Low half of SSRC0 → Low half of SDST,
 *                        High half of SSRC1 → High half of SDST.
 *
 * This distinguishes it from the companion instructions:
 *   s_pack_ll: SDST[15:0]=SSRC0[15:0], SDST[31:16]=SSRC1[15:0]  (both LOW)
 *   s_pack_lh: SDST[15:0]=SSRC0[15:0], SDST[31:16]=SSRC1[31:16] (low/HIGH)
 *   s_pack_hh: SDST[15:0]=SSRC0[31:16], SDST[31:16]=SSRC1[31:16] (both HIGH)
 *
 * All three are SOP2 instructions (scalar, wavefront-uniform).  Passing
 * values as kernel arguments ensures they live in SGPRs.
 *
 * Build:
 *   hipcc -O2 -o s_pack_lh_b32_b16 s_pack_lh_b32_b16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_pack_lh_b32_b16
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

static const int THREADS_PER_BLOCK = 64;   /* one wavefront on GFX9 */

/*
 * s_pack_lh_b32_b16_kernel
 *
 * Both s0 and s1 are kernel arguments (already in SGPRs).
 * SDST[15:0] = s0[15:0], SDST[31:16] = s1[31:16].
 * Only lane 0 per block writes the wavefront-wide result.
 *
 * "=s" — output SGPR
 * "s"  — input SGPR (kernel args)
 */
__global__ void
s_pack_lh_b32_b16_kernel(uint32_t s0, uint32_t s1, uint32_t *out, int n)
{
    int wf_id = blockIdx.x;
    if (wf_id >= n)
        return;

    uint32_t sdst;

    __asm__ volatile(
        "s_pack_lh_b32_b16 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));

    if (threadIdx.x == 0)
        out[wf_id] = sdst;
}

int
main(void)
{
    /*
     * Test cases: (s0, s1) → expected SDST where
     *   SDST[15:0]  = s0[15:0]   (low  half of s0)
     *   SDST[31:16] = s1[31:16]  (HIGH half of s1)
     */
    struct {
        uint32_t s0, s1, expected;
    } cases[] = {
        /* Both zero */
        { 0x00000000, 0x00000000, 0x00000000 },
        /* s0 low only, s1 high only */
        { 0x0000ABCD, 0x12340000, 0x1234ABCD },
        /* Verify: low half of s1 is IGNORED */
        { 0x0000ABCD, 0x1234FFFF, 0x1234ABCD },
        /* Verify: high half of s0 is IGNORED */
        { 0xFFFFABCD, 0x12340000, 0x1234ABCD },
        /* Typical: split a packed u32 and recombine from two sources */
        { 0x00003F80, 0x40000000, 0x40003F80 }, /* s0_lo=bf16(1.0), s1_hi=bf16(2.0) */
        /* All-ones in relevant halves */
        { 0x0000FFFF, 0xFFFF0000, 0xFFFFFFFF },
        /* s1 high=0, s0 low=anything */
        { 0x00005A5A, 0x00001234, 0x00005A5A },
        /* s0 low=0, s1 high=anything */
        { 0xFFFF0000, 0xA5A50000, 0xA5A50000 },
    };
    const int num_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    uint32_t *d_out = nullptr;
    GPU_CHECK(hipMalloc(&d_out, num_cases * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_out, 0xFF, num_cases * sizeof(uint32_t)));

    for (int c = 0; c < num_cases; c++) {
        s_pack_lh_b32_b16_kernel<<<1, THREADS_PER_BLOCK>>>(
            cases[c].s0, cases[c].s1, d_out + c, 1);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    }

    uint32_t h_out[num_cases];
    GPU_CHECK(hipMemcpy(h_out, d_out, num_cases * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    printf("s_pack_lh_b32_b16 test  "
           "(SDST[15:0]=S0[15:0], SDST[31:16]=S1[31:16])\n");
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
