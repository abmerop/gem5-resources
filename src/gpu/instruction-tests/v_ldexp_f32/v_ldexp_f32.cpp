/*
 * HIP test for V_LDEXP_F32 (VOP3 ldexp, f32).
 *
 * gem5 (vop3.cc): vdst = std::ldexp(S0.f32, S1.i32).  src0 is the f32
 * mantissa value, src1 a signed integer exponent.  Power-of-two scaling is
 * exact in f32, so compared bit-exactly against host ldexpf().  Variants: base.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_ldexp_f32 v_ldexp_f32.cpp
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel_v_ldexp_f32(const float *a, const int *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float va = a[i], vd;
    int vb = b[i];
    __asm__ volatile("v_ldexp_f32 %0, %1, %2\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = vd;
}

int
main(void)
{
    float *h_a = (float *)malloc(N * sizeof(float));
    int   *h_b = (int *)malloc(N * sizeof(int));
    float *h_out = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)((i % 7) + 1) * 0.5f;   /* 0.5 .. 3.5, exact */
        h_b[i] = (i % 13) - 6;                  /* -6 .. 6 */
    }

    float *d_a, *d_out;
    int   *d_b;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));

    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_ldexp_f32, dim3(g), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected = ldexpf(h_a[i], h_b[i]);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %g expected %g\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%g out[%d]=%g\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
