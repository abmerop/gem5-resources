/*
 * HIP test for S_ATOMIC_DEC (ENC_SMEM, scalar atomic decrement, return-old).
 *
 * In gem5 the decrement atomic (base/amo.hh AtomicOpDec) is a plain
 * subtraction:
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp - 1
 *   SDATA      = tmp                 (old value, returned via glc)
 *
 * Pattern B (scalar / wavefront-uniform): each wavefront targets its own
 * 32-bit counter slot (base = counter + blockIdx.x is uniform, so it lives in
 * an SGPR pair), so there is no contention and both the returned old value and
 * the post-op memory contents are deterministic.  s_waitcnt lgkmcnt(0) stalls
 * until the scalar atomic completes before the old value is read.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_atomic_dec s_atomic_dec.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_atomic_dec
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

static const int NUM_WF     = 64;
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_s_atomic_dec(uint32_t *counter, uint32_t *old_out)
{
    int wf = blockIdx.x;
    uint32_t *slot = counter + wf;       /* uniform -> SGPR pair */
    uint32_t old_val;

    __asm__ volatile(
        "s_atomic_dec %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=s"(old_val)
        : "s"(slot)
        : "memory");

    if (threadIdx.x == 0)
        old_out[wf] = old_val;
}

int
main(void)
{
    uint32_t *h_init = (uint32_t *)malloc(NUM_WF * sizeof(uint32_t));
    uint32_t *h_mem  = (uint32_t *)malloc(NUM_WF * sizeof(uint32_t));
    uint32_t *h_old  = (uint32_t *)malloc(NUM_WF * sizeof(uint32_t));

    for (int wf = 0; wf < NUM_WF; wf++)
        h_init[wf] = 1000u + (uint32_t)wf * 7u;

    uint32_t *d_counter, *d_old;
    GPU_CHECK(hipMalloc(&d_counter, NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_old,     NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_counter, h_init, NUM_WF * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_s_atomic_dec,
                       dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       d_counter, d_old);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_counter, NUM_WF * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, NUM_WF * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int wf = 0; wf < NUM_WF; wf++) {
        uint32_t exp_old = h_init[wf];
        uint32_t exp_mem = h_init[wf] - 1u;
        if (h_old[wf] != exp_old || h_mem[wf] != exp_mem) {
            fprintf(stderr, "FAIL wf%d: old=%u(exp %u) mem=%u(exp %u)\n",
                    wf, h_old[wf], exp_old, h_mem[wf], exp_mem);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("old[0]=%u mem[0]=%u old[%d]=%u mem[%d]=%u\n",
               h_old[0], h_mem[0], NUM_WF - 1, h_old[NUM_WF - 1],
               NUM_WF - 1, h_mem[NUM_WF - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_counter)); GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_mem); free(h_old);

    return pass ? 0 : 1;
}
