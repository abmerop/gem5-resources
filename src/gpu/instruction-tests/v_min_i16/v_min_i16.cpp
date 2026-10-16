/*
 * HIP test for V_MIN_I16 (ENC_VOP2, signed 16-bit minimum).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop2.cc) computes per lane:
 *   D.i16 = min(S0.i16, S1.i16)         (signed)
 *
 * NOTE on variants: the ISA lists DPP and SDWA forms for V_MIN_I16, but gem5's
 * execute() contains panic_if(isDPPInst()) / panic_if(isSDWAInst()) guards, so
 * those encodings abort the simulator and cannot be exercised.  This test
 * therefore covers BASE only (recorded variants=base), matching gem5 reality.
 *
 * Pattern A (vector): distinct per-lane operands via "v" constraints force the
 * VOP2 V_MIN_I16 encoding.  Integer min is exact.
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
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_min_i16(const int16_t *a, const int16_t *b, int16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int16_t s0 = a[i], s1 = b[i], d;
    __asm__ volatile(
        "v_min_i16 %0, %1, %2\n\t"
        : "=v"(d)
        : "v"(s0), "v"(s1));
    out[i] = d;
}

int
main(void)
{
    int16_t *h_a = (int16_t *)malloc(N * sizeof(int16_t));
    int16_t *h_b = (int16_t *)malloc(N * sizeof(int16_t));
    int16_t *h_out = (int16_t *)malloc(N * sizeof(int16_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = (int16_t)((i * 277) - 30000);
        h_b[i] = (int16_t)(20000 - (i * 191));
    }

    int16_t *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int16_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int16_t), hipMemcpyHostToDevice));

    kernel_v_min_i16<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int16_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int16_t expected = h_a[i] < h_b[i] ? h_a[i] : h_b[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL[%d]: min(%d,%d) got %d exp %d\n", i,
                    h_a[i], h_b[i], h_out[i], expected);
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
