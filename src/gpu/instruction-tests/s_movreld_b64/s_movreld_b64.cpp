/*
 * HIP test for S_MOVRELD_B64 (ENC_SOP1, 64-bit scalar move, M0-relative dest).
 *
 * S_MOVRELD_B64 computes (per gem5 src/arch/amdgpu/common/insts/sop1.cc):
 *
 *   SGPR[D + M0 : D + M0 + 1] = SGPR[SSRC0 : SSRC0 + 1]
 *
 * i.e. a 64-bit scalar move whose *destination* SGPR-pair index is offset by
 * M0.  A compiler cannot express a runtime-relative destination under an "=s"
 * output binding, so the test sets M0 = 0, under which S_MOVRELD_B64 sdst, ssrc
 * behaves as a plain 64-bit scalar copy, exercising decode+execute
 * deterministically.
 *
 * Pattern B (scalar / wavefront-uniform).
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_movreld_b64 s_movreld_b64.cpp
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
kernel_s_movreld_b64(uint64_t base, uint64_t *out)
{
    int wf = blockIdx.x;
    uint64_t src = base + (uint64_t)wf;
    uint64_t dst;

    __asm__ volatile(
        "s_mov_b32 m0, 0\n\t"
        "s_movreld_b64 %0, %1\n\t"
        : "=s"(dst)
        : "s"(src)
        : "m0");

    if (threadIdx.x == 0)
        out[wf] = dst;
}

int
main(void)
{
    const uint64_t base = 0x76543210000ULL;

    uint64_t *h_out = (uint64_t *)malloc(NUM_WF * sizeof(uint64_t));

    uint64_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NUM_WF * sizeof(uint64_t)));

    hipLaunchKernelGGL(kernel_s_movreld_b64,
                       dim3(NUM_WF), dim3(BLOCK_SIZE), 0, 0,
                       base, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_WF * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int wf = 0; wf < NUM_WF; wf++) {
        uint64_t expected = base + (uint64_t)wf;
        if (h_out[wf] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%016llx expected 0x%016llx\n",
                    wf, (unsigned long long)h_out[wf],
                    (unsigned long long)expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%016llx out[%d]=0x%016llx\n",
               (unsigned long long)h_out[0], NUM_WF - 1,
               (unsigned long long)h_out[NUM_WF - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);

    return pass ? 0 : 1;
}
