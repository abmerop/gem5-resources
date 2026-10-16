/*
 * HIP test for V_CMP_LG_F64 (VOPC / VOP3, 64-bit "less-greater" compare).
 *
 * V_CMP_LG_F64 sets, per lane:
 *
 *   VCC[lane] = (S0.f64 < S1.f64) || (S0.f64 > S1.f64)
 *
 * i.e. ordered not-equal: true iff the operands are ordered (neither is NaN)
 * and unequal; false for equal operands and for any unordered (NaN) pair.
 *
 * The VOP3 (_e64) encoding writes the per-lane mask to an SGPR pair; each lane
 * (the block is a single 64-lane wavefront) extracts its own bit.  Operands
 * are f64 VGPR pairs.  Test data includes equal pairs, ordered-unequal pairs,
 * and NaN operands to exercise the ordered semantics.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cmp_lg_f64 v_cmp_lg_f64.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmp_lg_f64
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
kernel_v_cmp_lg_f64(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double va = a[i];
    double vb = b[i];
    uint64_t mask;

    __asm__ volatile(
        "v_cmp_lg_f64_e64 %0, %1, %2\n\t"
        : "=s"(mask)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    double *h_a   = (double *)malloc(N * sizeof(double));
    double *h_b   = (double *)malloc(N * sizeof(double));
    int    *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        double base = (double)(i - 128) * 1.5;
        h_a[i] = base;
        if (i % 4 == 0)
            h_b[i] = base;                   /* equal -> false */
        else if (i % 4 == 1)
            h_b[i] = base + 1.0;             /* unequal -> true */
        else if (i % 4 == 2)
            h_b[i] = nan("");                /* NaN -> false (unordered) */
        else
            h_a[i] = nan("");                /* NaN -> false (unordered) */
    }

    double *d_a, *d_b;
    int    *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(double), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmp_lg_f64,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int ordered = (!isnan(h_a[i]) && !isnan(h_b[i]));
        int expected = (ordered && (h_a[i] != h_b[i])) ? 1 : 0;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d "
                    "(a=%g b=%g)\n", i, h_out[i], expected, h_a[i], h_b[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%d out[1]=%d out[2]=%d out[3]=%d\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
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
