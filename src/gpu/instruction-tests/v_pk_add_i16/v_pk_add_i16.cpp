/*
 * HIP test for V_PK_ADD_I16 (VOP3P packed 2x i16 add).
 *
 * gem5 (vop3p.cc): for each 16-bit lane, vdst = clampI16(S0 + S1) with clamp
 * off, i.e. wrapping signed 16-bit add of the low and high halves
 * independently.  Inputs kept small so no wrap ambiguity.  Variants: base.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_pk_add_i16 v_pk_add_i16.cpp
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel_v_pk_add_i16(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i], vb = b[i], vd;
    __asm__ volatile("v_pk_add_i16 %0, %1, %2\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = vd;
}

static inline uint32_t
pack2(int16_t lo, int16_t hi)
{
    return ((uint32_t)(uint16_t)hi << 16) | (uint32_t)(uint16_t)lo;
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = pack2((int16_t)((i % 100) - 50), (int16_t)((i % 60) - 30));
        h_b[i] = pack2((int16_t)((i % 40) - 20), (int16_t)((i % 70) - 35));
    }

    uint32_t *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_pk_add_i16, dim3(g), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int16_t alo = (int16_t)(h_a[i] & 0xffff), ahi = (int16_t)(h_a[i] >> 16);
        int16_t blo = (int16_t)(h_b[i] & 0xffff), bhi = (int16_t)(h_b[i] >> 16);
        uint32_t expected = pack2((int16_t)(alo + blo), (int16_t)(ahi + bhi));
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
