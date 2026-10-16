/*
 * HIP test for S_ATOMIC_ADD (SMEM scalar atomic add, GFX9).
 *
 * S_ATOMIC_ADD operates on a wavefront-uniform 64-bit address held in SGPRs:
 *
 *   tmp = MEM[addr]
 *   MEM[addr] = tmp + SDATA
 *   if (GLC) SDATA = tmp          (old value returned)
 *
 * gem5 implements this via Inst_SMEM__S_ATOMIC_ADD (Inst_SMEM_ATOMIC<1>).
 *
 * Each of num_wavefronts blocks (one wavefront each) atomically adds ADD_VAL to
 * a shared counter with glc, capturing the pre-add value into a per-wavefront
 * slot (lane 0 only).  Verification:
 *   - final counter == num_wavefronts * ADD_VAL
 *   - the returned old values are a permutation of {0, ADD_VAL, 2*ADD_VAL, ...}
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_atomic_add s_atomic_add.cpp
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

#define NUM_WF   8
#define ADD_VAL  3u

__global__ void
kernel_s_atomic_add(uint32_t *counter, uint32_t *results, uint32_t add_val)
{
    int wf_id = blockIdx.x;

    uint32_t old_val = add_val;
    asm volatile(
        "s_atomic_add %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(old_val)
        : "s"(counter)
        : "memory");

    if (threadIdx.x == 0)
        results[wf_id] = old_val;
}

int
main(void)
{
    uint32_t start_val = 0;
    uint32_t *d_counter = nullptr;
    GPU_CHECK(hipMalloc(&d_counter, sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_counter, &start_val, sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    uint32_t *d_results = nullptr;
    GPU_CHECK(hipMalloc(&d_results, NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_results, 0xFF, NUM_WF * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_s_atomic_add, dim3(NUM_WF), dim3(64), 0, 0,
                       d_counter, d_results, ADD_VAL);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t h_counter = 0;
    uint32_t h_results[NUM_WF];
    GPU_CHECK(hipMemcpy(&h_counter, d_counter, sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_results, d_results, NUM_WF * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    uint32_t expected_total = NUM_WF * ADD_VAL;
    if (h_counter != expected_total) {
        fprintf(stderr, "FAIL counter: got %u expected %u\n",
                h_counter, expected_total);
        pass = 0;
    }

    /* Old values must be a permutation of {0, ADD_VAL, ..., (NUM_WF-1)*ADD_VAL}. */
    int seen[NUM_WF] = {0};
    for (int i = 0; i < NUM_WF && pass; i++) {
        uint32_t v = h_results[i];
        if (v % ADD_VAL != 0 || (v / ADD_VAL) >= NUM_WF) {
            fprintf(stderr, "FAIL results[%d]=%u not a valid old value\n", i, v);
            pass = 0; break;
        }
        int slot = v / ADD_VAL;
        if (seen[slot]++) {
            fprintf(stderr, "FAIL duplicate old value %u\n", v);
            pass = 0; break;
        }
    }

    if (pass)
        printf("counter=%u results[0]=%u results[%d]=%u\n",
               h_counter, h_results[0], NUM_WF - 1, h_results[NUM_WF - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_counter)); GPU_CHECK(hipFree(d_results));
    return pass ? 0 : 1;
}
