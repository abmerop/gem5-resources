/*
 * HIP test for s_atomic_swap (scalar memory atomic exchange).
 *
 * s_atomic_swap is a GFX9 (Vega) SMEM atomic that operates on a single
 * wavefront-uniform address held in SGPRs.  It atomically writes SDATA to
 * MEM[addr] and, with GLC=1, returns the old value into SDATA.
 *
 * The ISA semantics are:
 *   tmp = MEM[addr]
 *   MEM[addr] = SDATA
 *   if (GLC) SDATA = tmp
 *
 * Like s_atomic_add, SDATA is both an input (the new value to write) and an
 * output (the old value with glc), so the inline-asm constraint is "+s"
 * (read-write scalar register).
 *
 * Unlike add/dec, swap has no dependency between successive operations:
 * each wavefront simply replaces the memory value with its own tag.  The
 * correct invariant is therefore:
 *   - Every old_val returned is either the initial value or the tag of
 *     exactly one other wavefront (no duplicates, every value is a
 *     legitimate predecessor).
 *   - The final memory value is the tag of whichever wavefront ran last.
 *
 * Build:
 *   hipcc -O2 -o scalar_atomic_swap scalar_atomic_swap.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./scalar_atomic_swap
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

/*
 * Each wavefront atomically swaps its unique tag (new_val) into the shared
 * memory location and captures the old value via GLC=1.
 *
 * mem     - device pointer to a single uint32 (wavefront-uniform address)
 * results - one slot per wavefront to hold the pre-swap value
 * new_val - the value this wavefront writes into mem (unique per wavefront)
 */
__global__ void
scalar_atomic_swap_kernel(uint32_t *mem, uint32_t *results, uint32_t new_val)
{
    int wf_id = blockIdx.x;    /* one wavefront per block */

    /*
     * new_val (kernel arg) lives in an SGPR.  mem (kernel arg pointer) lives
     * in a 64-bit SGPR pair.  These scalar operands make s_atomic_swap the
     * natural instruction to emit.
     *
     * "+s" marks swap_val as a read-write SGPR:
     *   - on entry  it holds the new value to write to mem
     *   - on exit   it holds the old value that was in mem (returned via glc)
     *
     * "s" marks mem as a read-only SGPR pair (64-bit pointer).
     *
     * s_waitcnt lgkmcnt(0) stalls until the scalar cache op completes so
     * the returned value in the SGPR is valid before we read it.
     */
    uint32_t swap_val = new_val;
    asm volatile(
        "s_atomic_swap %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(swap_val)    /* in: new value to write; out: old value in mem */
        : "s"(mem)          /* 64-bit SGPR pair: target address              */
        : "memory");

    /* s_atomic_swap is a scalar (wavefront-wide) op.  All lanes share the
     * same SGPR result; only lane 0 writes it to the per-wavefront slot. */
    if (threadIdx.x == 0) {
        results[wf_id] = swap_val;
    }
}

int
main(int argc, char **argv)
{
    /*
     * Launch one wavefront per block (64 threads on GFX9).  Each wavefront
     * swaps a unique tag (wf_id + 1) into the shared location.
     *
     * Tags are 1-based so that 0 is reserved for the initial value, making
     * the set of valid old_vals unambiguous: {0, 1, ..., num_wavefronts}.
     */
    const int num_wavefronts    = 8;
    const int threads_per_block = 64;   /* one wavefront on GFX9/Vega */
    const uint32_t init_val     = 0;    /* initial memory value        */

    uint32_t *d_mem = nullptr;
    GPU_CHECK(hipMalloc(&d_mem, sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_mem, &init_val, sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    printf("d_mem allocated at %p\n", d_mem);

    /* Allocate result buffer (one slot per wavefront). */
    uint32_t *d_results = nullptr;
    GPU_CHECK(hipMalloc(&d_results, num_wavefronts * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_results, 0xFF, num_wavefronts * sizeof(uint32_t)));

    /*
     * Launch num_wavefronts kernels sequentially so each wavefront gets a
     * unique tag (blockIdx.x + 1) passed as new_val.  Sequential launches
     * also give a deterministic final memory value for easy verification.
     */
    for (int wf = 0; wf < num_wavefronts; wf++) {
        uint32_t tag = (uint32_t)(wf + 1);
        scalar_atomic_swap_kernel<<<1, threads_per_block>>>(
            d_mem, d_results + wf, tag);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    }

    /* Copy results back. */
    uint32_t h_results[num_wavefronts];
    GPU_CHECK(hipMemcpy(h_results, d_results,
                        num_wavefronts * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    uint32_t h_mem_final = 0;
    GPU_CHECK(hipMemcpy(&h_mem_final, d_mem, sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /*
     * Verify:
     *  - With sequential launches each wavefront wf sees the tag written by
     *    wavefront (wf-1), i.e. old_val[wf] == wf  (0-based index == tag of
     *    predecessor; wf==0 sees init_val==0).
     *  - The final memory value must be the last wavefront's tag:
     *    num_wavefronts.
     */
    printf("s_atomic_swap test\n");
    printf("  init_val       = %u\n", init_val);
    printf("  num_wavefronts = %d\n", num_wavefronts);

    int pass = 1;
    for (int wf = 0; wf < num_wavefronts; wf++) {
        uint32_t v        = h_results[wf];
        uint32_t expected = (uint32_t)wf;   /* predecessor tag; wf 0 sees 0 */
        printf("  wavefront[%d] old_val = %u (expected %u)\n", wf, v, expected);
        if (v != expected) {
            printf("  FAIL: mismatch\n");
            pass = 0;
        }
    }

    uint32_t expected_final = (uint32_t)num_wavefronts;
    printf("  final mem = %u (expected %u)\n", h_mem_final, expected_final);
    if (h_mem_final != expected_final) {
        printf("  FAIL: final mem mismatch\n");
        pass = 0;
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem));
    GPU_CHECK(hipFree(d_results));

    return pass ? 0 : 1;
}
