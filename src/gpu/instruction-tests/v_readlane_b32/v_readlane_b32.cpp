/*
 * HIP test for V_READLANE_B32 (VOP3, read one lane of a VGPR into an SGPR).
 *
 * gem5 (Inst_VOP3__V_READLANE_B32::execute in
 * src/arch/amdgpu/common/insts/vop3.cc):
 *
 *   SDST = SRC0[ SRC1 & 0x3f ]
 *
 * i.e. the scalar destination receives the value held in lane (SRC1 & 63) of
 * the source VGPR; the result is wavefront-uniform.
 *
 * Test: one wavefront per block (64 lanes).  Each lane loads a distinct value
 * into a VGPR, then v_readlane_b32 selects a fixed lane (LANE_SEL) and writes
 * it to an SGPR, which every lane stores.  Expected: out[i] equals the input of
 * lane LANE_SEL within thread i's wavefront.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_readlane_b32 v_readlane_b32.cpp
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;
#define LANE_SEL 17

__global__ void
kernel_v_readlane_b32(const uint32_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t v = in[i];
    uint32_t s;
    __asm__ volatile(
        "v_readlane_b32 %0, %1, %2\n\t"
        : "=s"(s)
        : "v"(v), "I"(LANE_SEL));

    out[i] = s;
}

int
main(void)
{
    uint32_t *h_in  = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_in[i] = (uint32_t)(i * 2654435761u + 12345u);

    uint32_t *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_in, h_in, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_readlane_b32, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int wf_base = (i / BLOCK_SIZE) * BLOCK_SIZE;
        uint32_t expected = h_in[wf_base + LANE_SEL];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return pass ? 0 : 1;
}
