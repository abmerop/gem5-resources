/*
 * HIP test for V_MAX_U16 (ENC_VOP2, unsigned 16-bit maximum).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop2.cc) computes per lane:
 *   D.u16 = max(S0.u16, S1.u16)         (unsigned)
 *
 * NOTE on variants: the ISA lists DPP and SDWA forms, but gem5's execute()
 * has panic_if(isDPPInst())/panic_if(isSDWAInst()) guards, so those abort the
 * simulator.  This test covers BASE only (recorded variants=base).
 *
 * Pattern A (vector): distinct per-lane operands via "v" constraints force the
 * VOP2 V_MAX_U16 encoding.  Unsigned max is exact.
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
kernel_v_max_u16(const uint16_t *a, const uint16_t *b, uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint16_t s0 = a[i], s1 = b[i], d;
    __asm__ volatile(
        "v_max_u16 %0, %1, %2\n\t"
        : "=v"(d)
        : "v"(s0), "v"(s1));
    out[i] = d;
}

int
main(void)
{
    uint16_t *h_a = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_b = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_out = (uint16_t *)malloc(N * sizeof(uint16_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint16_t)(i * 257);
        h_b[i] = (uint16_t)(65535 - i * 191);
    }

    uint16_t *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint16_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint16_t), hipMemcpyHostToDevice));

    kernel_v_max_u16<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t expected = h_a[i] > h_b[i] ? h_a[i] : h_b[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL[%d]: max(%u,%u) got %u exp %u\n", i,
                    h_a[i], h_b[i], h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%u out[%d]=%u\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
