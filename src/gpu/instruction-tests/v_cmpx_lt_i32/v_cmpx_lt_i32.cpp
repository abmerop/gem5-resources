/*
 * HIP test for V_CMPX_LT_I32 (VOP3 compare-set-EXEC, i32 signed less-than).
 *
 * mask = exec = (S0.i32 < S1.i32)   (gem5: vopc.cc, vcc then exec = vcc)
 *
 * The kernel saves exec, issues v_cmpx_lt_i32, reads back the resulting exec
 * mask, then restores exec so the store predicates normally.  Each lane reports
 * its own mask bit, which is compared to the CPU reference (a[i] < b[i]).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmpx_lt_i32 v_cmpx_lt_i32.cpp
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
kernel_v_cmpx_lt_i32(const int *a, const int *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int va = a[i];
    int vb = b[i];

    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_lt_i32_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    int *h_a   = (int *)malloc(N * sizeof(int));
    int *h_b   = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));

    /* Mix of negatives, zero, positives so signed less-than is exercised. */
    for (int i = 0; i < N; i++) {
        h_a[i] = (i % 11) - 5;
        h_b[i] = (i % 7)  - 3;
    }

    int *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmpx_lt_i32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (h_a[i] < h_b[i]) ? 1 : 0;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d (a=%d b=%d)\n",
                    i, h_out[i], expected, h_a[i], h_b[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
