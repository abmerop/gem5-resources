/*
 * HIP test for S_MEMTIME (ENC_SMEM, read 64-bit timestamp counter into SDST).
 *
 *   SDST.u64 = current 64-bit timestamp
 *
 * A timer value cannot be predicted exactly, so the self-check is on its
 * defined properties: two sequential reads must be monotonic non-decreasing
 * (t1 >= t0) and non-zero (the simulation has been running).  Pattern B
 * (scalar): only lane 0 of each wavefront performs the reads, with
 * s_waitcnt lgkmcnt(0) to retire each scalar-memory result before use.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_memtime s_memtime.cpp
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

static const int BLOCK_SIZE = 64;
static const int NUM_WF      = 4;

__global__ void
kernel_s_memtime(uint64_t *t0_out, uint64_t *t1_out)
{
    if (threadIdx.x != 0)
        return;

    uint64_t t0, t1;
    __asm__ volatile(
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_memtime %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=s"(t0), "=s"(t1)
        :
        : "memory");

    t0_out[blockIdx.x] = t0;
    t1_out[blockIdx.x] = t1;
}

int
main(void)
{
    uint64_t *d_t0, *d_t1;
    GPU_CHECK(hipMalloc(&d_t0, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_t1, NUM_WF * sizeof(uint64_t)));

    hipLaunchKernelGGL(kernel_s_memtime, dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       d_t0, d_t1);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint64_t h_t0[NUM_WF], h_t1[NUM_WF];
    GPU_CHECK(hipMemcpy(h_t0, d_t0, NUM_WF * sizeof(uint64_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_t1, d_t1, NUM_WF * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        if (h_t1[w] < h_t0[w] || h_t1[w] == 0) {
            fprintf(stderr, "FAIL wf=%d: t0=%llu t1=%llu (need t1>=t0>0)\n",
                    w, (unsigned long long)h_t0[w], (unsigned long long)h_t1[w]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("t0[0]=%llu t1[0]=%llu\n",
               (unsigned long long)h_t0[0], (unsigned long long)h_t1[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_t0)); GPU_CHECK(hipFree(d_t1));
    return pass ? 0 : 1;
}
