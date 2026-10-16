/*
 * HIP test for V_CMPX_NEQ_F64 (VOP3 compare-set-EXEC, f64 not-equal, unordered).
 *
 * mask = exec = !(S0.f64 == S1.f64)  (true when a != b OR either is NaN).
 * 64-bit f64 operands span two VGPRs.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmpx_neq_f64 v_cmpx_neq_f64.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

__global__ void
kernel_v_cmpx_neq_f64(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double va = a[i];
    double vb = b[i];

    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_neq_f64_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    double *h_a   = (double *)malloc(N * sizeof(double));
    double *h_b   = (double *)malloc(N * sizeof(double));
    int  *h_out = (int  *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = (double)((i % 11) - 5) * 0.5;
        h_b[i] = (i % 3 == 0) ? h_a[i] : (double)((i % 7) - 3) * 0.5;
    }
    /* NaN cases -> unordered -> NEQ true */
    h_a[5] = nan(""); h_b[9] = nan(""); h_a[40] = nan(""); h_b[40] = nan("");

    double *d_a, *d_b;
    int  *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(double), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmpx_neq_f64, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1, nt = 0;
    for (int i = 0; i < N; i++) {
        int expected = !(h_a[i] == h_b[i]) ? 1 : 0;
        nt += expected;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("true=%d/%d\n", nt, N);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
