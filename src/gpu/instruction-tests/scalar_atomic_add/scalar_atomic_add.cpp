/*
 * HIP test for s_atomic_add (scalar memory atomic add).
 *
 * s_atomic_add is a GFX9 (Vega) SMEM atomic that operates on a single
 * wavefront-uniform address held in SGPRs.  It atomically adds the value in
 * SDATA to MEM[addr] and, with GLC=1, writes the old value back to SDATA.
 *
 * The ISA semantics are:
 *   tmp = MEM[addr]
 *   MEM[addr] = tmp + SDATA
 *   if (GLC) SDATA = tmp
 *
 * Unlike s_atomic_dec, SDATA is both an input (the value to add) and an
 * output (the old value with glc), so the inline-asm constraint is "+s"
 * (read-write scalar register) rather than "=s".
 *
 * Build:
 *   hipcc -O2 -o scalar_atomic_add scalar_atomic_add.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./scalar_atomic_add
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/*
 * Each wavefront performs one s_atomic_add on a shared counter using inline
 * GCN assembly.  With GLC=1 (glc modifier) the old value is returned.
 *
 * One thread per wavefront (lane 0) writes the returned old value into
 * results[] so the host can verify correctness.
 *
 * counter   - device pointer to a single uint32 counter (wavefront-uniform)
 * results   - one slot per wavefront to hold the pre-add value
 * add_val   - value each wavefront adds to the counter
 */
__global__ void
scalar_atomic_add_kernel(uint32_t *counter, uint32_t *results,
                         uint32_t add_val)
{
    int wf_id = blockIdx.x;    /* one wavefront per block */

    /*
     * s_atomic_add requires a 64-bit SGPR base address and a 32-bit SGPR
     * data operand.  Both counter (kernel arg pointer) and add_val (kernel
     * arg scalar) already live in SGPRs, making s_atomic_add the natural
     * instruction here.
     *
     * "+s" marks old_val as a read-write SGPR:
     *   - on entry  it holds the value to add (add_val)
     *   - on exit   it holds the old counter value (returned via glc)
     *
     * "s" marks counter as a read-only SGPR pair (64-bit pointer).
     *
     * s_waitcnt lgkmcnt(0) stalls until the scalar cache op completes so
     * the returned value in the SGPR is valid before we read it.
     */
    uint32_t old_val = add_val;
    asm volatile(
        "s_atomic_add %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(old_val)     /* in: addend; out: old counter value */
        : "s"(counter)      /* 64-bit SGPR pair: target address   */
        : "memory");

    /* s_atomic_add is a scalar (wavefront-wide) op.  All lanes share the
     * same SGPR result; only lane 0 writes it to the per-wavefront slot. */
    if (threadIdx.x == 0) {
        results[wf_id] = old_val;
    }
}

int
main(int argc, char **argv)
{
    /*
     * Launch one wavefront per block (64 threads on GFX9).  Each wavefront
     * adds add_val (= 1) to the shared counter once.
     */
    const int num_wavefronts   = 8;
    const int threads_per_block = 64;   /* one wavefront on GFX9/Vega */
    const uint32_t start_val   = 0;
    const uint32_t add_val     = 2;
    const uint32_t total_added = num_wavefronts * add_val;

    /* Allocate counter as a two-element array so d_counter can also be
     * used to test offset value without much modification. */
    uint32_t *d_counter = nullptr;
    GPU_CHECK(hipMalloc(&d_counter, 2 * sizeof(uint32_t)));

    /* Write start_val into slot 1; slot 0 is padding. */
    GPU_CHECK(hipMemcpy(d_counter, &start_val, sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    printf("d_counter allocated at %p (using slot at %p)\n",
           d_counter, d_counter);

    /* Allocate result buffer (one slot per wavefront). */
    uint32_t *d_results = nullptr;
    GPU_CHECK(hipMalloc(&d_results, num_wavefronts * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_results, 0xFF, num_wavefronts * sizeof(uint32_t)));

    /* Launch kernel. */
    hipLaunchKernelGGL(scalar_atomic_add_kernel,
                       dim3(num_wavefronts), dim3(threads_per_block), 0, 0,
                       d_counter, d_results, add_val);
    GPU_CHECK(hipDeviceSynchronize());

    /* Copy results back. */
    uint32_t h_results[num_wavefronts];
    GPU_CHECK(hipMemcpy(h_results, d_results,
                        num_wavefronts * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    uint32_t h_counter_final = 0;
    GPU_CHECK(hipMemcpy(&h_counter_final, d_counter, sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /*
     * Verify:
     *  - Each wavefront should have received a unique old value in
     *    [start_val .. start_val + total_added - 1] (order unspecified).
     *  - The final counter should be start_val + total_added.
     */
    printf("s_atomic_add test\n");
    printf("  start_val      = %u\n", start_val);
    printf("  add_val        = %u\n", add_val);
    printf("  num_wavefronts = %d\n", num_wavefronts);

    /* Build a presence bitmap to verify all expected old values were seen. */
    uint32_t seen[total_added];
    for (int i = 0; i < total_added; i++)
        seen[i] = 0;

    int pass = 1;
    for (int i = 0; i < num_wavefronts; i++) {
        uint32_t v = h_results[i];
        printf("  wavefront[%d] old_val = %u\n", i, v);

        /* Expected range: [start_val .. start_val + total_added - 1] */
        if (v < start_val || v >= start_val + total_added) {
            printf("  FAIL: old_val %u out of expected range [%u, %u)\n",
                   v, start_val, start_val + total_added);
            pass = 0;
        } else {
            int idx = (int)(v - start_val);
            if (seen[idx]) {
                printf("  FAIL: duplicate old_val %u (seen by WF %d)\n", v, seen[idx]);
                pass = 0;
            }
            seen[idx] = i;
        }
    }

    uint32_t expected_final = start_val + total_added;
    printf("  final counter = %u (expected %u)\n",
           h_counter_final, expected_final);
    if (h_counter_final != expected_final) {
        printf("  FAIL: final counter mismatch\n");
        pass = 0;
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_counter));
    GPU_CHECK(hipFree(d_results));

    return pass ? 0 : 1;
}
