/*
 * HIP test for V_PK_ADD_F32 (VOP3P packed f32 add).
 *
 * V_PK_ADD_F32 operates on a pair of f32 lanes packed into a 64-bit operand:
 *   D[0] = S0[0] + S1[0]
 *   D[1] = S0[1] + S1[1]
 *
 * Inputs are exactly-representable floats (small integers / halves) so the
 * sums are bit-exact and verification can use ==.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_pk_add_f32 v_pk_add_f32.cpp
 * Run:     gem5.opt config.py -- ./v_pk_add_f32
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

typedef float float2v __attribute__((ext_vector_type(2)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_pk_add_f32(const float2v *a, const float2v *b, float2v *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float2v va = a[i], vb = b[i], vd;
    __asm__ volatile(
        "v_pk_add_f32 %0, %1, %2\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb));

    out[i] = vd;
}

int
main(void)
{
    float2v *h_a = (float2v *)malloc(N * sizeof(float2v));
    float2v *h_b = (float2v *)malloc(N * sizeof(float2v));
    float2v *h_out = (float2v *)malloc(N * sizeof(float2v));

    for (int i = 0; i < N; i++) {
        h_a[i].x = (float)(i - 128) * 0.5f;
        h_a[i].y = (float)(i % 17) * 2.0f;
        h_b[i].x = (float)(i % 9) * 0.25f;
        h_b[i].y = (float)(64 - i) * 0.5f;
    }

    float2v *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(float2v)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(float2v)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(float2v)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float2v), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(float2v), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_pk_add_f32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float2v),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float ex = h_a[i].x + h_b[i].x;
        float ey = h_a[i].y + h_b[i].y;
        if (h_out[i].x != ex || h_out[i].y != ey) {
            fprintf(stderr, "FAIL out[%d]: got (%f,%f) expected (%f,%f)\n",
                    i, h_out[i].x, h_out[i].y, ex, ey);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=(%f,%f)\n", h_out[0].x, h_out[0].y);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
