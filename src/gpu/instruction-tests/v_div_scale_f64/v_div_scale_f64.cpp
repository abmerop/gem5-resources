/*
 * HIP test for V_DIV_SCALE_F64 (VOP3 with SGPR carry-out, f64 division scale).
 *
 * V_DIV_SCALE_F64 is a helper in the f64 division macro sequence.  Per the gem5
 * model (Inst_VOP3__V_DIV_SCALE_F64::execute in
 * src/arch/amdgpu/common/insts/vop3.cc) it inspects the exponents of the
 * denominator (src1) and numerator (src2) and conditionally rescales src0,
 * setting a per-lane carry bit (the SGPR/VCC dest):
 *
 *   - if src1==0 or src2==0:                  VDST = NaN,            carry=0
 *   - if exp(src2)-exp(src1) >= 768 and
 *     src0==src1:                             VDST = ldexp(src0,128), carry=1
 *   - (well-scaled operands leave VDST in the common pass-through path)
 *
 * This test drives the two deterministic branches above (selected per thread)
 * and checks VDST.  src0/src1/src2 are f64 VGPR pairs; the carry-out is an
 * SGPR pair we let the assembler allocate (its per-lane value is not checked
 * here — VDST fully determines correctness for these branches).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_div_scale_f64 v_div_scale_f64.cpp
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
kernel_v_div_scale_f64(const double *s0, const double *s1, const double *s2,
                       double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    double a = s0[i], b = s1[i], c = s2[i], d;
    uint64_t carry;

    __asm__ volatile(
        "v_div_scale_f64 %0, %1, %2, %3, %4\n\t"
        : "=v"(d), "=s"(carry)
        : "v"(a), "v"(b), "v"(c));

    out[i] = d;
}

int
main(void)
{
    double *h_s0 = (double *)malloc(N * sizeof(double));
    double *h_s1 = (double *)malloc(N * sizeof(double));
    double *h_s2 = (double *)malloc(N * sizeof(double));
    double *h_out = (double *)malloc(N * sizeof(double));

    /* Two deterministic branches, selected per thread parity. */
    for (int i = 0; i < N; i++) {
        if ((i & 1) == 0) {
            /* zero-denominator -> NaN branch */
            h_s0[i] = 3.5;
            h_s1[i] = 0.0;             /* src1 == 0 */
            h_s2[i] = 7.0;
        } else {
            /* large exponent gap, src0==src1 -> ldexp(src0,128), carry=1 */
            h_s0[i] = 1.0;
            h_s1[i] = 1.0;             /* src0 == src1 */
            h_s2[i] = ldexp(1.0, 800); /* exp(src2)-exp(src1) ~ 800 >= 768 */
        }
    }

    double *d_s0, *d_s1, *d_s2, *d_out;
    GPU_CHECK(hipMalloc(&d_s0, N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_s1, N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_s2, N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(double)));

    GPU_CHECK(hipMemcpy(d_s0, h_s0, N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s1, h_s1, N * sizeof(double), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s2, h_s2, N * sizeof(double), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_div_scale_f64, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_s0, d_s1, d_s2, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(double), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if ((i & 1) == 0) {
            if (!isnan(h_out[i])) {
                fprintf(stderr, "FAIL out[%d]: got %g expected NaN\n",
                        i, h_out[i]);
                pass = 0; break;
            }
        } else {
            double expected = ldexp(1.0, 128);
            if (h_out[i] != expected) {
                fprintf(stderr, "FAIL out[%d]: got %g expected %g\n",
                        i, h_out[i], expected);
                pass = 0; break;
            }
        }
    }

    if (pass)
        printf("out[0]=%g out[1]=%g\n", h_out[0], h_out[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_s0)); GPU_CHECK(hipFree(d_s1));
    GPU_CHECK(hipFree(d_s2)); GPU_CHECK(hipFree(d_out));
    free(h_s0); free(h_s1); free(h_s2); free(h_out);
    return pass ? 0 : 1;
}
