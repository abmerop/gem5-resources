/* HIP test for S_ATOMIC_CMPSWAP_X2 (ENC_SMEM 64-bit scalar compare-and-swap,
 * gfx950).
 *
 * SDATA is a 4-SGPR quad:
 *   SDATA[0:1] (low  64b) : compare value on input; old memory value on output
 *                           (with glc)
 *   SDATA[2:3] (high 64b) : new value written if the compare matches
 *   tmp = MEM[addr];                       (64-bit)
 *   if (tmp == cmp) MEM[addr] = new;       (64-bit)
 *   if (glc) SDATA[0:1] = tmp
 *
 * We pack the 128-bit SDATA into a __uint128_t: low 64 = cmp, high 64 = new.
 * One wavefront per counter (wavefront-uniform, no contention).  Even
 * wavefronts use cmp == init so the swap succeeds (mem becomes new); odd
 * wavefronts use cmp != init so it fails (mem stays init).  In both cases the
 * returned old value == init.  base only.
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int NUM_WF     = 8;
static const int BLOCK_SIZE = 64;   /* one wavefront per block */

__global__ void
kernel(uint64_t *counters, uint64_t *old_out, const uint64_t *cmp_in,
       const uint64_t *new_in)
{
    int wf = blockIdx.x;
    uint64_t *slot = &counters[wf];
    /* low 64 = compare value, high 64 = new value */
    unsigned __int128 sdata =
        ((unsigned __int128)new_in[wf] << 64) | (unsigned __int128)cmp_in[wf];
    __asm__ volatile(
        "s_atomic_cmpswap_x2 %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(sdata)
        : "s"(slot)
        : "memory");
    if (threadIdx.x == 0)
        old_out[wf] = (uint64_t)(sdata & 0xFFFFFFFFFFFFFFFFULL);
}

int
main(void)
{
    uint64_t h_init[NUM_WF], h_mem[NUM_WF], h_old[NUM_WF];
    uint64_t h_cmp[NUM_WF], h_new[NUM_WF];
    for (int w = 0; w < NUM_WF; w++) {
        h_init[w] = ((uint64_t)(w * 2654435761u + 123u) << 32) |
                    (uint64_t)(w * 40503u + 7u);
        h_new[w]  = ((uint64_t)(w * 22695477u + 99u) << 32) |
                    (uint64_t)(w * 12345u + 1u);
        /* even wf: cmp matches init (swap succeeds); odd wf: cmp differs */
        h_cmp[w]  = (w & 1) ? (h_init[w] ^ 0xdeadbeefcafef00dULL) : h_init[w];
    }

    uint64_t *d_counters, *d_old, *d_cmp, *d_new;
    GPU_CHECK(hipMalloc(&d_counters, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old,      NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_cmp,      NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_new,      NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_counters, h_init, NUM_WF * sizeof(uint64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp, h_cmp, NUM_WF * sizeof(uint64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_new, h_new, NUM_WF * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel, dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       d_counters, d_old, d_cmp, d_new);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_counters, NUM_WF * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, NUM_WF * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        uint64_t exp_old = h_init[w];
        uint64_t exp_mem = (w & 1) ? h_init[w] : h_new[w];
        if (h_old[w] != exp_old || h_mem[w] != exp_mem) {
            fprintf(stderr,
                    "FAIL [%d]: old=%llx(exp %llx) mem=%llx(exp %llx)\n",
                    w, (unsigned long long)h_old[w],
                    (unsigned long long)exp_old,
                    (unsigned long long)h_mem[w],
                    (unsigned long long)exp_mem);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("old[0]=%llx mem[0]=%llx mem[1]=%llx\n",
               (unsigned long long)h_old[0], (unsigned long long)h_mem[0],
               (unsigned long long)h_mem[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_counters)); GPU_CHECK(hipFree(d_old));
    GPU_CHECK(hipFree(d_cmp)); GPU_CHECK(hipFree(d_new));
    return pass ? 0 : 1;
}
