/*
 * HIP test for V_CMPX_EQ_F64 (VOP3 compare-set-EXEC, f64 ordered equal).
 *
 * mask = exec = (S0.f64 == S1.f64)   (gem5: vopc.cc; NaN compares unequal).
 *
 * The kernel saves exec, issues v_cmpx_eq_f64, reads back the resulting exec
 * mask, then restores exec.  Each lane reports its mask bit, compared to the
 * CPU reference (a[i] == b[i], with NaN handled by C's == semantics).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmpx_eq_f64 v_cmpx_eq_f64.cpp
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
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_v_cmpx_eq_f64(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double va = a[i];
    double vb = b[i];

    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_eq_f64_e64 %0, %2, %3\n\t"
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
    int    *h_out = (int    *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        if (i % 4 == 0) {                  /* equal */
            h_a[i] = h_b[i] = (double)i * 1.5 - 7.0;
        } else if (i % 4 == 1) {           /* unequal */
            h_a[i] = (double)i;
            h_b[i] = (double)i + 1.0;
        } else if (i % 4 == 2) {           /* NaN vs anything -> false */
            h_a[i] = nan("");
            h_b[i] = nan("");
        } else {                           /* +0.0 == -0.0 -> true */
            h_a[i] = 0.0;
            h_b[i] = -0.0;
        }
    }

    double *d_a, *d_b;
    int    *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(double), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmpx_eq_f64, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (h_a[i] == h_b[i]) ? 1 : 0;   /* C ==: NaN -> false */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
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
