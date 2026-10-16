/*
 * HIP test for V_PK_MAX_I16 (VOP3P packed signed 16-bit maximum).
 *
 * V_PK_MAX_I16 computes on the two i16 halves of each 32-bit operand:
 *   D[15:0]  = max(S0[15:0],  S1[15:0])   (signed)
 *   D[31:16] = max(S0[31:16], S1[31:16])  (signed)
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_pk_max_i16 v_pk_max_i16.cpp
 * Run:     gem5.opt config.py -- ./v_pk_max_i16
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
kernel_v_pk_max_i16(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i], vb = b[i], vd;
    __asm__ volatile(
        "v_pk_max_i16 %0, %1, %2\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb));

    out[i] = vd;
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        int16_t a_lo = (int16_t)(i * 131 - 12000);
        int16_t a_hi = (int16_t)(20000 - i * 217);
        int16_t b_lo = (int16_t)(i * 251 + 7 - 8000);
        int16_t b_hi = (int16_t)(i * 89 - 5000);
        h_a[i] = ((uint32_t)(uint16_t)a_hi << 16) | (uint16_t)a_lo;
        h_b[i] = ((uint32_t)(uint16_t)b_hi << 16) | (uint16_t)b_lo;
    }

    uint32_t *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_pk_max_i16, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int16_t a_lo = (int16_t)(h_a[i] & 0xffff), a_hi = (int16_t)(h_a[i] >> 16);
        int16_t b_lo = (int16_t)(h_b[i] & 0xffff), b_hi = (int16_t)(h_b[i] >> 16);
        int16_t e_lo = a_lo > b_lo ? a_lo : b_lo;
        int16_t e_hi = a_hi > b_hi ? a_hi : b_hi;
        uint32_t expected = ((uint32_t)(uint16_t)e_hi << 16) | (uint16_t)e_lo;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
