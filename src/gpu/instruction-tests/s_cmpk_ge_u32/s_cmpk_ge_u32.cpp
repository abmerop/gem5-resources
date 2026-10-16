/*
 * HIP test for S_CMPK_GE_U32 (ENC_SOPK, scalar unsigned compare vs a 16-bit
 * immediate, sets SCC).
 *
 *   SCC = (S0.u32 >= zeroext(SIMM16))
 *
 * Pattern B (scalar): the value under test is passed as a kernel argument so it
 * stays in an SGPR.  We compare it against an immediate equal to it (>= true)
 * and against one larger (>= false), capturing SCC after each via
 * s_cselect_b32.  Only lane 0 of each wavefront writes the result.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_cmpk_ge_u32 s_cmpk_ge_u32.cpp
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

static const int BLOCK_SIZE = 64;       /* one wavefront per block */
static const int NUM_WF      = 4;

__global__ void
kernel_s_cmpk_ge_u32(unsigned val, int *out_t, int *out_f)
{
    if (threadIdx.x != 0)
        return;

    /* val = 0x1234.
     *   ge vs 0x1234 -> (0x1234 >= 0x1234) = true  -> 1
     *   ge vs 0x1235 -> (0x1234 >= 0x1235) = false -> 0 */
    int r_t, r_f;
    __asm__ volatile(
        "s_cmpk_ge_u32 %2, 0x1234\n\t"
        "s_cselect_b32 %0, 1, 0\n\t"
        "s_cmpk_ge_u32 %2, 0x1235\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(r_t), "=s"(r_f)
        : "s"(val)
        : "scc");

    out_t[blockIdx.x] = r_t;
    out_f[blockIdx.x] = r_f;
}

int
main(void)
{
    const unsigned val = 0x1234;

    int *d_t, *d_f;
    GPU_CHECK(hipMalloc(&d_t, NUM_WF * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_f, NUM_WF * sizeof(int)));

    hipLaunchKernelGGL(kernel_s_cmpk_ge_u32, dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       val, d_t, d_f);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    int h_t[NUM_WF], h_f[NUM_WF];
    GPU_CHECK(hipMemcpy(h_t, d_t, NUM_WF * sizeof(int), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_f, d_f, NUM_WF * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        if (h_t[w] != 1 || h_f[w] != 0) {
            fprintf(stderr, "FAIL wf=%d: ge(val,val)=%d (exp 1) ge(val,val+1)=%d (exp 0)\n",
                    w, h_t[w], h_f[w]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("ge(val,val)=%d ge(val,val+1)=%d\n", h_t[0], h_f[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_t)); GPU_CHECK(hipFree(d_f));
    return pass ? 0 : 1;
}
