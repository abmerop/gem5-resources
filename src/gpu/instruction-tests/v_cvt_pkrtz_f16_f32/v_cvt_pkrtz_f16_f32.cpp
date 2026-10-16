/*
 * HIP test for V_CVT_PKRTZ_F16_F32 (VOP3 pack two f32->f16, round toward zero).
 *
 * gem5 (vop3.cc) converts the source to f16 with roundTowardZero and packs into
 * the low/high 16 bits of the dest.  NOTE: the gem5 implementation derives BOTH
 * packed halves from src0 (the high-half source is read from src0), so this
 * test passes src0 == src1 (the same value), making the result
 * order-independent: out = (f16(a) << 16) | f16(a).  Inputs are exactly
 * representable in f16 so round-toward-zero == round-to-nearest.  Variants: base.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_cvt_pkrtz_f16_f32 \
 *              v_cvt_pkrtz_f16_f32.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
kernel_v_cvt_pkrtz_f16_f32(const float *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float va = a[i], vb = a[i];
    uint32_t vd;
    __asm__ volatile("v_cvt_pkrtz_f16_f32 %0, %1, %2\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = vd;
}

int
main(void)
{
    float    *h_a   = (float *)malloc(N * sizeof(float));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_a[i] = (float)((i % 21) - 10) * 0.5f;   /* -5 .. 5 step 0.5, exact */

    float    *d_a;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));

    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cvt_pkrtz_f16_f32, dim3(g), dim3(BLOCK_SIZE),
                       0, 0, d_a, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        _Float16 hf = (_Float16)h_a[i];
        uint16_t bits;
        memcpy(&bits, &hf, sizeof(bits));
        uint32_t expected = ((uint32_t)bits << 16) | (uint32_t)bits;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
