/*
 * HIP test for S_ATOMIC_ADD_X2 (ENC_SMEM, 64-bit scalar memory atomic add,
 * return-old).
 *
 * S_ATOMIC_ADD_X2 is a GFX9 SMEM atomic operating on a single
 * wavefront-uniform 64-bit address held in an SGPR pair.  It atomically adds
 * the 64-bit SDATA to MEM[addr] and, with GLC=1, returns the old value:
 *
 *   tmp = MEM[addr]                    (64-bit)
 *   MEM[addr] = tmp + SDATA            (64-bit integer add)
 *   if (GLC) SDATA = tmp
 *
 * SDATA is both input (the addend) and output (the old value with glc), so the
 * inline-asm constraint is "+s" on a 64-bit value (an SGPR pair).  The address
 * is a 64-bit SGPR pair ("s" on the kernel-arg pointer).
 *
 * Each wavefront (one per block) targets its own counter slot, so there is no
 * cross-wavefront contention and the old value / final value are
 * deterministic.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_atomic_add_x2 s_atomic_add_x2.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_atomic_add_x2
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

static const int NUM_WF      = 8;
static const int BLOCK_SIZE  = 64;      /* one wavefront per block */

/*
 * kernel_s_atomic_add_x2
 *
 * Each wavefront atomically adds add_val (64-bit) to its own counter slot via
 * S_ATOMIC_ADD_X2 with glc, capturing the returned old value.  Only lane 0
 * writes the per-wavefront old value back to global memory.
 */
__global__ void
kernel_s_atomic_add_x2(uint64_t *counters, uint64_t *old_out, uint64_t add_val)
{
    int wf = blockIdx.x;
    uint64_t *slot = &counters[wf];

    uint64_t old = add_val;
    __asm__ volatile(
        "s_atomic_add_x2 %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(old)             /* in: addend; out: old value */
        : "s"(slot)             /* 64-bit SGPR pair: address  */
        : "memory");

    if (threadIdx.x == 0)
        old_out[wf] = old;
}

int
main(void)
{
    uint64_t *h_init = (uint64_t *)malloc(NUM_WF * sizeof(uint64_t));
    uint64_t *h_mem  = (uint64_t *)malloc(NUM_WF * sizeof(uint64_t));
    uint64_t *h_old  = (uint64_t *)malloc(NUM_WF * sizeof(uint64_t));

    const uint64_t add_val = 0x0000000100000007ULL;
    for (int i = 0; i < NUM_WF; i++)
        h_init[i] = ((uint64_t)(i + 1) << 32) | (uint64_t)(i * 11 + 5);

    uint64_t *d_counters, *d_old;
    GPU_CHECK(hipMalloc(&d_counters, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old,      NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_counters, h_init, NUM_WF * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_s_atomic_add_x2,
                       dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       d_counters, d_old, add_val);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_counters, NUM_WF * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, NUM_WF * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < NUM_WF; i++) {
        uint64_t exp_old = h_init[i];
        uint64_t exp_mem = h_init[i] + add_val;
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr,
                    "FAIL [%d]: old=%llx(exp %llx) mem=%llx(exp %llx)\n",
                    i, (unsigned long long)h_old[i],
                    (unsigned long long)exp_old,
                    (unsigned long long)h_mem[i],
                    (unsigned long long)exp_mem);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("old[0]=%llx mem[0]=%llx\n",
               (unsigned long long)h_old[0], (unsigned long long)h_mem[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_counters)); GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_mem); free(h_old);

    return pass ? 0 : 1;
}
