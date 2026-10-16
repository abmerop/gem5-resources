/*
 * HIP test for S_SETVSKIP (ENC_SOPC, enable/disable VSKIP mode).
 *
 *   VSKIP = S0.u[S1.u[4:0]]
 *
 * When VSKIP is enabled, no vector (VOP, MUBUF, MIMG, DS, FLAT, EXP)
 * instructions are issued (they behave as NOPs).  This test enables VSKIP,
 * issues a vector add
 * that must be skipped, disables VSKIP, then issues a vector add that must
 * apply.  On correct hardware the first add is skipped, so the result is 10
 * (only the second add applies); if VSKIP is ignored the result would be 11.
 *
 * Pattern B (scalar control).  Only lane 0 of each wavefront writes the result.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_setvskip s_setvskip.cpp
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
static const int NUM_WF     = 4;

__global__ void
kernel_s_setvskip(int *out)
{
    int r;
    __asm__ volatile(
        "v_mov_b32 %0, 0\n\t"
        "s_waitcnt 0\n\t"
        "s_setvskip 1, 0\n\t"        /* VSKIP = bit0 of 1 = 1 (enable)  */
        "s_nop 1\n\t"
        "v_add_u32 %0, %0, 1\n\t"    /* skipped while VSKIP enabled      */
        "s_setvskip 0, 0\n\t"        /* VSKIP = bit0 of 0 = 0 (disable) */
        "s_nop 1\n\t"
        "v_add_u32 %0, %0, 10\n\t"   /* applies -> result 10             */
        : "=v"(r)
        :
        :);

    if (threadIdx.x == 0)
        out[blockIdx.x] = r;
}

int
main(void)
{
    int *d_out;
    GPU_CHECK(hipMalloc(&d_out, NUM_WF * sizeof(int)));

    hipLaunchKernelGGL(kernel_s_setvskip, dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    int h_out[NUM_WF];
    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_WF * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        if (h_out[w] != 10) {   /* first add skipped, second applied */
            fprintf(stderr, "FAIL wf=%d: got %d expected 10\n", w, h_out[w]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("result=%d (first add skipped)\n", h_out[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
