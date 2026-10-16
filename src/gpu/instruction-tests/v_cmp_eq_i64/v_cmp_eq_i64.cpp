/*
 * HIP test for V_CMP_EQ_I64 (VOP3 i64 equal compare).
 *
 * gem5 (vop3_cmp.cc): lane mask bit = (S0.i64 == S1.i64).  The _e64 form
 * writes the per-lane mask to an SGPR pair.  Variants: base.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_cmp_eq_i64 v_cmp_eq_i64.cpp
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
kernel_v_cmp_eq_i64(const int64_t *a, const int64_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int64_t va = a[i];
    int64_t vb = b[i];

    uint64_t mask;
    __asm__ volatile(
        "v_cmp_eq_i64_e64 %0, %1, %2\n\t"
        : "=s"(mask)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    int64_t *h_a   = (int64_t *)malloc(N * sizeof(int64_t));
    int64_t *h_b   = (int64_t *)malloc(N * sizeof(int64_t));
    int     *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = (int64_t)0x1122334400000000LL + (i % 7);
        /* match every other lane */
        h_b[i] = (int64_t)0x1122334400000000LL + ((i % 2) ? (i % 5) : (i % 7));
    }

    int64_t *d_a, *d_b;
    int     *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int64_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int64_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmp_eq_i64, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (h_a[i] == h_b[i]) ? 1 : 0;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
