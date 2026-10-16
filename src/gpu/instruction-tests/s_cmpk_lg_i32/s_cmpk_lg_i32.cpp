/*
 * HIP test for S_CMPK_LG_I32 (ENC_SOPK, scalar compare not-equal vs a signed
 * 16-bit immediate, sets SCC).
 *
 *   SCC = (S0.i32 != signext(SIMM16))
 *
 * Pattern B (scalar): the value under test is passed as a kernel argument so it
 * stays in an SGPR.  We compare it against two immediates -- one equal, one not
 * -- and capture SCC after each via s_cselect_b32, exercising both outcomes.
 * Only lane 0 of each wavefront writes the per-wavefront result.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_cmpk_lg_i32 s_cmpk_lg_i32.cpp
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
kernel_s_cmpk_lg_i32(int val, int *out_eq, int *out_ne)
{
    if (threadIdx.x != 0)
        return;

    /* Compare against an immediate equal to val (0x1234): SCC = (val != val)=0.
     * Compare against val+1 (0x1235): SCC = (val != val+1) = 1. */
    int r_eq, r_ne;
    __asm__ volatile(
        "s_cmpk_lg_i32 %2, 0x1234\n\t"
        "s_cselect_b32 %0, 1, 0\n\t"
        "s_cmpk_lg_i32 %2, 0x1235\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(r_eq), "=s"(r_ne)
        : "s"(val)
        : "scc");

    out_eq[blockIdx.x] = r_eq;
    out_ne[blockIdx.x] = r_ne;
}

int
main(void)
{
    const int val = 0x1234;

    int *d_eq, *d_ne;
    GPU_CHECK(hipMalloc(&d_eq, NUM_WF * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_ne, NUM_WF * sizeof(int)));

    hipLaunchKernelGGL(kernel_s_cmpk_lg_i32, dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       val, d_eq, d_ne);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    int h_eq[NUM_WF], h_ne[NUM_WF];
    GPU_CHECK(hipMemcpy(h_eq, d_eq, NUM_WF * sizeof(int), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_ne, d_ne, NUM_WF * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        /* val == 0x1234 -> lg false -> 0;  val != 0x1235 -> lg true -> 1 */
        if (h_eq[w] != 0 || h_ne[w] != 1) {
            fprintf(stderr, "FAIL wf=%d: lg-vs-equal=%d (exp 0) lg-vs-diff=%d (exp 1)\n",
                    w, h_eq[w], h_ne[w]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("lg(val,val)=%d lg(val,val+1)=%d\n", h_eq[0], h_ne[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_eq)); GPU_CHECK(hipFree(d_ne));
    return pass ? 0 : 1;
}
