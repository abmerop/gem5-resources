/*
 * HIP test for s_atomic_cmpswap (scalar memory compare-and-swap).
 *
 * s_atomic_cmpswap is a GFX9 (Vega) SMEM atomic that operates on a single
 * wavefront-uniform address held in SGPRs.  SDATA is a 2-SGPR pair:
 *
 *   SDATA[0]  (low SGPR)  : compare value on input; old memory value on output
 *   SDATA[1]  (high SGPR) : new value to write if compare matches (input only)
 *
 * The ISA semantics are:
 *   tmp = MEM[addr]
 *   if (tmp == SDATA[0]) MEM[addr] = SDATA[1]
 *   if (GLC) SDATA[0] = tmp
 *
 * In inline assembly the 2-SGPR pair is expressed as a single uint64_t with a
 * "+s" (read-write scalar) constraint:
 *   bits [31:0]  = SDATA[0] = compare value / return register
 *   bits [63:32] = SDATA[1] = new value to write
 *
 * Build:
 *   hipcc -O2 -o scalar_atomic_cmpswap scalar_atomic_cmpswap.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./scalar_atomic_cmpswap
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
 * Perform one s_atomic_cmpswap on *mem.
 *
 * mem     - device pointer to the target uint32 (wavefront-uniform address)
 * results - per-wavefront slot to store the pre-swap memory value
 * cmp_val - value to compare against MEM[mem]; swap occurs only if equal
 * new_val - value written to MEM[mem] when the compare succeeds
 *
 * The returned old_val equals cmp_val when the swap succeeds, and differs
 * from cmp_val when it fails (the memory is left unchanged in that case).
 */
__global__ void
scalar_atomic_cmpswap_kernel(uint32_t *mem, uint32_t *results,
                             uint32_t cmp_val, uint32_t new_val)
{
    int wf_id = blockIdx.x;    /* one wavefront per block */

    /*
     * Pack the two operands into a single 64-bit SGPR pair:
     *   bits [31: 0] = cmp_val  -> SDATA[0]: compare value, also return reg
     *   bits [63:32] = new_val  -> SDATA[1]: value written on success
     *
     * s_atomic_cmpswap s[N:N+1], s[M:M+1], 0x0 glc
     *
     * "+s" = read-write scalar (SGPR pair for uint64_t):
     *   on entry  s[N]   = cmp_val, s[N+1] = new_val
     *   on exit   s[N]   = old memory value (glc)
     *             s[N+1] = new_val unchanged (only low SGPR is written back)
     *
     * "s" = read-only scalar (SGPR pair holding the 64-bit target address).
     *
     * s_waitcnt lgkmcnt(0) stalls until the scalar cache op completes so
     * that the returned value in the low SGPR is visible before we read it.
     */
    uint64_t sdata = ((uint64_t)new_val << 32) | (uint64_t)cmp_val;
    asm volatile(
        "s_atomic_cmpswap %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(sdata)   /* in: [new_val:cmp_val]; out: old_val in low 32 bits */
        : "s"(mem)      /* 64-bit SGPR pair: target address                   */
        : "memory");

    uint32_t old_val = (uint32_t)(sdata & 0xFFFFFFFFULL);

    /* s_atomic_cmpswap is a scalar (wavefront-wide) op.  All lanes share the
     * same SGPR result; only lane 0 writes it to the per-wavefront slot. */
    if (threadIdx.x == 0) {
        results[wf_id] = old_val;
    }
}

int
main(int argc, char **argv)
{
    /*
     * Run two phases:
     *
     * Phase 1 — success chain (num_wavefronts sequential launches):
     *   Each wavefront presents the correct compare value so the swap always
     *   succeeds.  Memory advances 0 → 1 → 2 → … → num_wavefronts.
     *   Expected old_val for wavefront wf: wf  (the predecessor's value).
     *
     * Phase 2 — failure (one extra launch):
     *   Present a wrong compare value so the swap fails.
     *   Expected old_val: num_wavefronts (current memory, unchanged).
     *   Memory stays at num_wavefronts.
     */
    const int num_wavefronts    = 8;
    const int threads_per_block = 64;   /* one wavefront on GFX9/Vega */
    const uint32_t init_val     = 0;

    uint32_t *d_mem = nullptr;
    GPU_CHECK(hipMalloc(&d_mem, sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_mem, &init_val, sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    printf("d_mem allocated at %p\n", d_mem);

    /* One result slot per success wavefront plus one for the failure case. */
    const int total_launches = num_wavefronts + 1;
    uint32_t *d_results = nullptr;
    GPU_CHECK(hipMalloc(&d_results, total_launches * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_results, 0xFF, total_launches * sizeof(uint32_t)));

    /* --- Phase 1: success chain --- */
    for (int wf = 0; wf < num_wavefronts; wf++) {
        uint32_t cmp = (uint32_t)wf;        /* correct compare: predecessor  */
        uint32_t nw  = (uint32_t)(wf + 1); /* new value: current tag        */
        scalar_atomic_cmpswap_kernel<<<1, threads_per_block>>>(
            d_mem, d_results + wf, cmp, nw);
        GPU_CHECK(hipGetLastError());
        GPU_CHECK(hipDeviceSynchronize());
    }

    /* --- Phase 2: intentional failure --- */
    const uint32_t wrong_cmp = 0xDEADBEEFU;  /* deliberately wrong compare */
    const uint32_t fail_new  = 0xDEADBEEFU;  /* would be written on success */
    scalar_atomic_cmpswap_kernel<<<1, threads_per_block>>>(
        d_mem, d_results + num_wavefronts, wrong_cmp, fail_new);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* Copy results back. */
    uint32_t h_results[total_launches];
    GPU_CHECK(hipMemcpy(h_results, d_results,
                        total_launches * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    uint32_t h_mem_final = 0;
    GPU_CHECK(hipMemcpy(&h_mem_final, d_mem, sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /*
     * Verify phase 1: old_val[wf] must equal wf (the predecessor's tag).
     * Verify phase 2: old_val must equal num_wavefronts (swap failed, mem
     *                 unchanged); final mem must still be num_wavefronts.
     */
    printf("s_atomic_cmpswap test\n");
    printf("  init_val       = %u\n", init_val);
    printf("  num_wavefronts = %d\n", num_wavefronts);

    int pass = 1;

    printf("  --- phase 1: success chain ---\n");
    for (int wf = 0; wf < num_wavefronts; wf++) {
        uint32_t v        = h_results[wf];
        uint32_t expected = (uint32_t)wf;
        printf("  wavefront[%d] cmp=%u new=%u old_val=%u (expected %u) %s\n",
               wf, (uint32_t)wf, (uint32_t)(wf + 1), v, expected,
               v == expected ? "OK" : "FAIL");
        if (v != expected) {
            pass = 0;
        }
    }

    printf("  --- phase 2: intentional failure ---\n");
    uint32_t fail_old     = h_results[num_wavefronts];
    uint32_t expected_old = (uint32_t)num_wavefronts;
    printf("  cmp=0x%08X new=0x%08X old_val=%u (expected %u) %s\n",
           wrong_cmp, fail_new, fail_old, expected_old,
           fail_old == expected_old ? "OK" : "FAIL");
    if (fail_old != expected_old) {
        pass = 0;
    }

    uint32_t expected_final = (uint32_t)num_wavefronts;
    printf("  final mem = %u (expected %u) %s\n",
           h_mem_final, expected_final,
           h_mem_final == expected_final ? "OK" : "FAIL");
    if (h_mem_final != expected_final) {
        pass = 0;
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem));
    GPU_CHECK(hipFree(d_results));

    return pass ? 0 : 1;
}
