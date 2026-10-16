/*
 * HIP test for DS_APPEND (DS encoding, LDS append-counter allocation).
 *
 * DS_APPEND (from gem5 .arch description):
 *   "LDS & GDS. Add (count_bits(exec_mask)) to the value stored in DS memory
 *    at (M0.base + instr_offset). Return the pre-operation value to VGPRs."
 *
 * Semantics exercised here (LDS variant, no 'gds' modifier -- gfx950 rejects
 * 'gds'):
 *   - A single 32-bit append counter lives in LDS at a fixed byte offset.
 *   - M0[15:0] holds that byte offset (M0.base) for the LDS counter.
 *   - When a wavefront executes ds_append, the current counter value is read
 *     once, the counter is atomically incremented by the number of active
 *     lanes (popcount of EXEC), and the SAME pre-operation value is broadcast
 *     to VDST of every active lane.
 *   - This is how hardware hands out a contiguous block allocation: the first
 *     append returns base index B to all lanes, and the counter advances to
 *     B + (#active lanes). A second append therefore returns B + (#active).
 *
 * Self-check strategy (no host-side arithmetic ambiguity):
 *   Launch ONE wavefront (64 threads, all active). The LDS counter starts at 0.
 *     Pass 1: every lane must receive 0        (pre-op value broadcast).
 *     Pass 2: every lane must receive 64       (counter advanced by 64 lanes).
 *   Passing both proves (a) each append returns a single contiguous base to
 *   the whole wavefront and (b) the counter advanced by exactly the active-lane
 *   count -- i.e. a distinct, contiguous 64-wide allocation [0,64) followed by
 *   [64,128).  We store both returned values per lane and verify on the host.
 *
 * NOTE: ds_append cannot be produced by plain C++ / HIP intrinsics, so inline
 * assembly is required.  M0 must be set to the LDS byte offset of the counter
 * and the counter LDS word must be zero-initialized before the append.
 *
 * Build:
 *   make -C hip_tests/ds_append AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./ds_append
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t e = (cmd);                                             \
        if (e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                               \
            exit(1);                                                      \
        }                                                                  \
    } while (0)

static const int WAVE_SIZE = 64; /* one full wavefront, all lanes active */

/*
 * kernel_ds_append
 *
 * Uses a __shared__ 32-bit counter as the LDS append counter.  We take its
 * address, mask to the LDS byte offset, load it into M0, and issue ds_append
 * twice.  s_waitcnt lgkmcnt(0) is required after each append before the
 * returned value is valid in the destination VGPR.
 *
 * out1[lane] = pre-op counter for the first  append  (expected 0  for all)
 * out2[lane] = pre-op counter for the second append  (expected 64 for all)
 */
__global__ void
kernel_ds_append(uint32_t *out1, uint32_t *out2)
{
    int lane = threadIdx.x;

    /* LDS append counter, shared across the wavefront. */
    __shared__ uint32_t counter;

    /* Lane 0 zero-initializes the LDS counter. */
    if (lane == 0)
        counter = 0u;
    __syncthreads();

    /*
     * M0.base for the LDS append counter = byte offset of 'counter' within the
     * kernel's LDS allocation.  Casting the __shared__ pointer to an integer
     * yields that LDS byte offset.  ds_append reads M0 (a scalar reg), so the
     * offset must be wavefront-uniform: use v_readfirstlane_b32 to move it from
     * a VGPR into an SGPR.
     */
    uint32_t lds_addr;
    __asm__ volatile("v_readfirstlane_b32 %0, %1\n\t"
                     : "=s"(lds_addr)
                     : "v"((uint32_t)(uintptr_t)&counter));

    uint32_t r1, r2;
    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"       /* M0.base = LDS byte offset of counter */
        "ds_append %0\n\t"           /* pass 1: return pre-op, add #active */
        "s_waitcnt lgkmcnt(0)\n\t"
        "ds_append %1\n\t"           /* pass 2: return new pre-op value     */
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(r1), "=v"(r2)
        : "s"(lds_addr)
        : "m0", "memory");

    out1[lane] = r1;
    out2[lane] = r2;
}

int
main(void)
{
    uint32_t *h_out1 = (uint32_t *)malloc(WAVE_SIZE * sizeof(uint32_t));
    uint32_t *h_out2 = (uint32_t *)malloc(WAVE_SIZE * sizeof(uint32_t));

    uint32_t *d_out1, *d_out2;
    GPU_CHECK(hipMalloc(&d_out1, WAVE_SIZE * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out2, WAVE_SIZE * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_ds_append, dim3(1), dim3(WAVE_SIZE), 0, 0,
                       d_out1, d_out2);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out1, d_out1, WAVE_SIZE * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_out2, d_out2, WAVE_SIZE * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /*
     * Verify contiguous allocation semantics:
     *   - Every lane in the wavefront must see the same pre-op value from each
     *     append (broadcast base of the contiguous block).
     *   - First append base must be 0 (counter started at 0).
     *   - Second append base must be 64 (counter advanced by the 64 active
     *     lanes), proving the block [0,64) was allocated contiguously.
     */
    int pass = 1;
    for (int i = 0; i < WAVE_SIZE; i++) {
        if (h_out1[i] != 0u) {
            fprintf(stderr,
                    "FAIL pass1 lane %d: got %u expected 0\n", i, h_out1[i]);
            pass = 0;
            break;
        }
        if (h_out2[i] != (uint32_t)WAVE_SIZE) {
            fprintf(stderr,
                    "FAIL pass2 lane %d: got %u expected %d\n", i, h_out2[i],
                    WAVE_SIZE);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("append1 base=%u  append2 base=%u  (advanced by %u)\n",
               h_out1[0], h_out2[0], h_out2[0] - h_out1[0]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out1));
    GPU_CHECK(hipFree(d_out2));
    free(h_out1);
    free(h_out2);

    return pass ? 0 : 1;
}
