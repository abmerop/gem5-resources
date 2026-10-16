/*
 * HIP test for S_MOVRELS_B32 (ENC_SOP1, scalar move with M0-relative source).
 *
 * S_MOVRELS_B32 computes (per gem5 src/arch/amdgpu/common/insts/sop1.cc):
 *
 *   SGPR[D] = SGPR[SSRC0 + M0]
 *
 * i.e. an ordinary scalar move whose *source* SGPR index is offset by the M0
 * register.  A compiler cannot express a runtime-relative source while
 * honouring an "s" input binding, so the test sets M0 = 0, under which
 * S_MOVRELS_B32 sdst, ssrc behaves as a plain scalar copy of ssrc into the
 * allocated sdst.  This exercises the decode and execute path deterministically.
 *
 * Pattern B (scalar / wavefront-uniform): the source is built from a scalar
 * kernel argument plus the (uniform) block index, so it stays in an SGPR, and
 * only lane 0 of each wavefront writes the per-wavefront result.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_movrels_b32 s_movrels_b32.cpp
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
kernel_s_movrels_b32(uint32_t base, uint32_t *out)
{
    int wf = blockIdx.x;
    uint32_t src = base + (uint32_t)wf;
    uint32_t dst;

    __asm__ volatile(
        "s_mov_b32 m0, 0\n\t"
        "s_movrels_b32 %0, %1\n\t"
        : "=s"(dst)
        : "s"(src)
        : "m0");

    if (threadIdx.x == 0)
        out[wf] = dst;
}

int
main(void)
{
    const uint32_t base = 0x1000;

    uint32_t *h_out = (uint32_t *)malloc(NUM_WF * sizeof(uint32_t));

    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NUM_WF * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_s_movrels_b32,
                       dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       base, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_WF * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int wf = 0; wf < NUM_WF; wf++) {
        uint32_t expected = base + (uint32_t)wf;
        if (h_out[wf] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    wf, h_out[wf], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], NUM_WF - 1, h_out[NUM_WF - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);

    return pass ? 0 : 1;
}
