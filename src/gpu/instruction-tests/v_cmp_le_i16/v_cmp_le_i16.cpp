/*
 * HIP test for V_CMP_LE_I16 (VOPC / VOP3 signed 16-bit "less-or-equal").
 *
 * V_CMP_LE_I16 computes, per lane:
 *
 *   VCC[lane] = (S0.i16 <= S1.i16)      (signed 16-bit comparison)
 *
 * The VOP3 (_e64) encoding writes the per-lane result mask to an arbitrary
 * SGPR pair, which is convenient to read back: each lane extracts its own bit
 * from the wavefront-wide mask using its lane id (threadIdx.x, since the block
 * is a single 64-lane wavefront).
 *
 * The 16-bit operands are placed in the low half of a VGPR; the instruction
 * interprets bits[15:0] as a signed 16-bit integer.  Test data mixes positive
 * and negative values around zero to exercise signed ordering.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cmp_le_i16 v_cmp_le_i16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmp_le_i16
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

/*
 * kernel_v_cmp_le_i16
 *
 * Each lane compares a[i] <= b[i] (signed i16) via the VOP3 v_cmp_le_i16_e64
 * form, which writes the wavefront mask into an SGPR pair.  The lane then
 * extracts its own bit.
 */
__global__ void
kernel_v_cmp_le_i16(const int16_t *a, const int16_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = (uint32_t)(uint16_t)a[i];
    uint32_t vb = (uint32_t)(uint16_t)b[i];

    uint64_t mask;
    __asm__ volatile(
        "v_cmp_le_i16_e64 %0, %1, %2\n\t"
        : "=s"(mask)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    int16_t *h_a = (int16_t *)malloc(N * sizeof(int16_t));
    int16_t *h_b = (int16_t *)malloc(N * sizeof(int16_t));
    int     *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = (int16_t)(i - 128);             /* -128 .. 127 */
        h_b[i] = (int16_t)((i % 2) ? -(i / 2) : (i / 2));
    }

    int16_t *d_a, *d_b;
    int     *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int16_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmp_le_i16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (h_a[i] <= h_b[i]) ? 1 : 0;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d (a=%d b=%d)\n",
                    i, h_out[i], expected, h_a[i], h_b[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_out));
    free(h_a);
    free(h_b);
    free(h_out);

    return pass ? 0 : 1;
}
