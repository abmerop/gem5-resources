/*
 * HIP test for V_CMPX_GT_U16 (VOPC unsigned 16-bit greater-than compare-and-
 * set-exec).
 *
 * V_CMPX_GT_U16 computes, per lane:
 *   result = (S0.u16 > S1.u16)
 * and writes that mask into EXEC (gem5 also mirrors it into VCC).  Unlike the
 * plain V_CMP form, the "X" variant overwrites the execute mask, disabling the
 * lanes that fail the comparison.
 *
 * gem5 implementation (src/arch/amdgpu/common/insts/vopc.cc):
 *   for each active lane: vcc.setBit(lane, src0[lane] > src1[lane]);
 *   wf->execMask() = vcc.rawData();
 * (SDWA/DPP are explicitly unsupported for this opcode -> base only.)
 *
 * Testing strategy: capture the resulting EXEC mask without letting it corrupt
 * the surrounding control flow.  The inline-asm block
 *   1. saves the current EXEC,
 *   2. runs v_cmpx_gt_u16 (which sets EXEC = compare result),
 *   3. copies the post-compare EXEC into an SGPR pair we can read,
 *   4. restores EXEC.
 * Each lane then extracts its own bit from the captured mask.  Scalar moves
 * ignore EXEC, so step 3 reliably reads the full 64-bit result.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cmpx_gt_u16 v_cmpx_gt_u16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmpx_gt_u16
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
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_cmpx(const uint16_t *a, const uint16_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i];     /* low 16 bits hold the u16 operand */
    uint32_t vb = b[i];

    uint64_t saved, mask;
    __asm__ volatile(
        "s_mov_b64 %0, exec\n\t"          /* saved = exec               */
        "v_cmpx_gt_u16 %2, %3\n\t"        /* exec = (va.u16 > vb.u16)   */
        "s_mov_b64 %1, exec\n\t"          /* mask  = post-compare exec  */
        "s_mov_b64 exec, %0\n\t"          /* restore exec               */
        : "=s"(saved), "=s"(mask)
        : "v"(va), "v"(vb)
        : "vcc");

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    uint16_t *h_a   = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_b   = (uint16_t *)malloc(N * sizeof(uint16_t));
    int      *h_out = (int      *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        switch (i % 4) {
        case 0:  h_a[i] = 100;    h_b[i] = 50;     break; /* gt  -> 1 */
        case 1:  h_a[i] = 10;     h_b[i] = 200;    break; /* lt  -> 0 */
        case 2:  h_a[i] = 77;     h_b[i] = 77;     break; /* eq  -> 0 */
        case 3:  h_a[i] = 0x8000; h_b[i] = 0x0001; break; /* unsigned -> 1 */
        }
    }

    uint16_t *d_a, *d_b;
    int      *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint16_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_cmpx, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int exp = (h_a[i] > h_b[i]) ? 1 : 0;
        if (h_out[i] != exp) {
            fprintf(stderr, "FAIL out[%d] (a=%u b=%u): got %d expected %d\n",
                    i, h_a[i], h_b[i], h_out[i], exp);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("cmpx: [gt]=%d [lt]=%d [eq]=%d [unsigned]=%d\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);

    return pass ? 0 : 1;
}
