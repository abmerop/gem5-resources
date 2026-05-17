/*
 * HIP test for s_atomic_dec (scalar memory atomic decrement).
 *
 * s_atomic_dec is a GFX9 (Vega) SMEM atomic that operates on a single
 * wavefront-uniform address held in SGPRs. It atomically decrements the
 * value at the given address (wrapping to max_uint if the current value is
 * 0 or > compare), writing the old value back to an SGPR when GLC=1.
 *
 * The ISA semantics are:
 *   tmp = MEM[addr]
 *   MEM[addr] = (tmp == 0 || tmp > val) ? val : tmp - 1
 *   if (GLC) SDST = tmp
 *
 * Build:
 *   hipcc -O2 -o scalar_atomic_dec scalar_atomic_dec.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./scalar_atomic_dec
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
 * Each wavefront performs one s_atomic_dec on a shared counter using inline
 * GCN assembly. With GLC=1 (glc modifier) the old value is returned.
 *
 * One thread per wavefront (lane 0) writes the returned old value into
 * results[] so the host can verify correctness.
 *
 * counter  - device pointer to a single uint32 counter (wavefront-uniform)
 * results  - one slot per wavefront to hold the pre-decrement value
 * num_dec  - number of times to decrement per wave
 */
__global__ void
scalar_atomic_dec_kernel(uint32_t *counter, uint32_t *results, int num_dec)
{
    int wf_id = blockIdx.x;          // one wavefront per block
    uint32_t old_val = 0xFFFFFFFF;

    /*
     * s_atomic_dec_x2 / s_atomic_dec require a 64-bit SGPR base address and
     * a 32-bit SGPR data/return operand.  We load the pointer through inline
     * asm so the compiler places it in SGPRs.
     *
     * The clobber list marks s_data as modified (the returned old value) and
     * "memory" so the compiler does not reorder around the atomic.
     *
     * Constraints used:
     *   "s"  - scalar register (SGPR)
     *   "=s" - output SGPR
     */
    for (int i = 0; i < num_dec; ++i) {
        asm volatile(
            /* s_atomic_dec sdata, sbase, offset glc
             *
             * sbase  : 64-bit SGPR pair holding *counter (pointer)
             * sdata  : 32-bit SGPR; receives old value (GLC=1 implied by glc)
             * offset : immediate byte offset (0)
             * glc    : return old value to sdata
             *
             * s_waitcnt lgkmcnt(0) stalls until the scalar cache/memory op
             * completes before we read sdata.
             */
            "s_atomic_dec %0, %1, 0x4 glc\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(old_val)
            : "s"(counter)
            : "memory");
    }

    /* Only lane 0 of each wavefront writes the result. */
    if (threadIdx.x == 0) {
        results[wf_id] = old_val;
    }
}

int
main(int argc, char **argv)
{
    /*
     * Launch one wavefront per block (64 threads on GFX9).  Use enough
     * wavefronts to stress the atomic serialization path without making the
     * test excessively slow in gem5.
     */
    const int num_wavefronts = 8;
    const int threads_per_block = 64;   // one wavefront on GFX9/Vega
    const uint32_t start_val = 20;      // counter starts here
    const uint32_t num_dec = 1;
    const uint32_t total_dec = num_wavefronts * num_dec;

    assert(start_val > num_wavefronts * num_dec);

    /* Allocate and initialise device counter. */
    uint32_t *d_counter = nullptr;
    GPU_CHECK(hipMalloc(&d_counter, 2*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_counter+1, &start_val, sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    printf("d_counter allocated at %p\n", d_counter+1);

    /* Allocate result buffer (one slot per wavefront). */
    uint32_t *d_results = nullptr;
    GPU_CHECK(hipMalloc(&d_results, num_wavefronts * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_results, 0xFF, num_wavefronts * sizeof(uint32_t)));

    /* Launch kernel. */
    hipLaunchKernelGGL(scalar_atomic_dec_kernel,
                       dim3(num_wavefronts), dim3(threads_per_block), 0, 0,
                       d_counter, d_results, num_dec);
    GPU_CHECK(hipDeviceSynchronize());

    /* Copy results back. */
    uint32_t h_results[num_wavefronts];
    GPU_CHECK(hipMemcpy(h_results, d_results,
                        num_wavefronts * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    uint32_t h_counter_final = 0;
    GPU_CHECK(hipMemcpy(&h_counter_final, d_counter+1, sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /*
     * Verify:
     *  - Each wavefront should have received a unique old value in
     *    [start_val - total_dec + 1 .. start_val] (order unspecified).
     *  - The final counter should be start_val - total_dec (if
     *    num_wavefronts <= start_val, so no wrap-around occurs).
     *  - ISA wrap rule: if old == 0 or old > compare, result = compare.
     *    With compare == 0xFFFFFFFF (s_atomic_dec implicit wrap value) and
     *    start_val = 10 < 0xFFFFFFFF, no wrap occurs here.
     */
    printf("s_atomic_dec test\n");
    printf("  start_val     = %u\n", start_val);
    printf("  num_wavefronts = %d\n", num_wavefronts);

    /* Build a presence bitmap to verify all expected old values were seen. */
    uint32_t seen[num_wavefronts];
    for (int i = 0; i < num_wavefronts; i++)
        seen[i] = 0;

    int pass = 1;
    for (int i = 0; i < num_wavefronts; i++) {
        uint32_t v = h_results[i];
        printf("  wavefront[%d] old_val = %u\n", i, v);

        /* Expected range: [start_val - total_dec + 1 .. start_val] */
        if (v < (start_val - total_dec + 1) || v > start_val) {
            printf("  FAIL: old_val %u out of expected range [%u, %u]\n",
                   v, start_val - total_dec + 1, start_val);
            pass = 0;
        } else {
            int idx = (int)(v - (start_val - total_dec + 1));
            if (seen[idx]) {
                printf("  FAIL: duplicate old_val %u\n", v);
                pass = 0;
            }
            seen[idx] = 1;
        }
    }

    uint32_t expected_final = start_val - total_dec;
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
