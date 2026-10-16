/*
 * HIP test for V_MED3_U16 (ENC_VOP3, unsigned 16-bit median-of-three).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop3.cc) returns the middle of the three
 * unsigned 16-bit operands:
 *   D.u16 = median(S0.u16, S1.u16, S2.u16)
 *
 * NOTE on variants: gem5's execute() has panic_if(isDPPInst())/
 * panic_if(isSDWAInst()) guards (and this is a VOP3-only op anyway), so only
 * BASE is exercised (recorded variants=base).
 *
 * Pattern A (vector): distinct per-lane operands via "v" constraints force the
 * VOP3 V_MED3_U16 encoding.  Integer median is exact; the host computes it by
 * sorting the three values.
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
kernel_v_med3_u16(const uint16_t *a, const uint16_t *b, const uint16_t *c,
                  uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint16_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile(
        "v_med3_u16 %0, %1, %2, %3\n\t"
        : "=v"(d)
        : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static uint16_t
median_u16(uint16_t a, uint16_t b, uint16_t c)
{
    uint16_t lo = a < b ? a : b;
    uint16_t hi = a > b ? a : b;
    if (c < lo) return lo;
    if (c > hi) return hi;
    return c;
}

int
main(void)
{
    uint16_t *h_a = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_b = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_c = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc(N * sizeof(uint16_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint16_t)(i * 313);
        h_b[i] = (uint16_t)(40000 - i * 137);
        h_c[i] = (uint16_t)(i * 1009 + 7);
    }

    uint16_t *d_a, *d_b, *d_c, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_c, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint16_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(uint16_t), hipMemcpyHostToDevice));

    kernel_v_med3_u16<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t expected = median_u16(h_a[i], h_b[i], h_c[i]);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL[%d]: med3(%u,%u,%u) got %u exp %u\n", i,
                    h_a[i], h_b[i], h_c[i], h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%u out[%d]=%u\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_c); free(h_out);
    return pass ? 0 : 1;
}
