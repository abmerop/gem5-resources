/*
 * HIP test for V_LSHL_ADD_U32 (ENC_VOP3, fused shift-left then add).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop3.cc) computes per lane:
 *   D.u32 = (S0.u32 << (S1.u32 & 0x1f)) + S2.u32
 *
 * Pattern A (vector): each lane gets distinct operands from global arrays via
 * "v" constraints, forcing the VOP3 V_LSHL_ADD_U32 encoding.  All 32-bit
 * unsigned arithmetic wraps mod 2^32, matching the host check exactly.
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
kernel_v_lshl_add_u32(const uint32_t *a, const uint32_t *b,
                      const uint32_t *c, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile(
        "v_lshl_add_u32 %0, %1, %2, %3\n\t"
        : "=v"(d)
        : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_c = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = 0x01020304u * (i + 1);
        h_b[i] = (uint32_t)(i % 33);      /* shift, low 5 bits used */
        h_c[i] = 0xdeadbeefu ^ (uint32_t)(i * 2654435761u);
    }

    uint32_t *d_a, *d_b, *d_c, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    kernel_v_lshl_add_u32<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c,
                                                          d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = (h_a[i] << (h_b[i] & 0x1f)) + h_c[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL[%d]: got %08x exp %08x\n", i, h_out[i],
                    expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[1]=%08x out[%d]=%08x\n", h_out[1], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_c); free(h_out);
    return pass ? 0 : 1;
}
