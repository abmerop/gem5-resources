/*
 * HIP test for V_MAD_U64_U32 (VOP3 64-bit multiply-add with carry SDST).
 *
 * V_MAD_U64_U32 computes per lane:
 *   D.u64 = S0.u32 * S1.u32 + S2.u64        (SDST = carry-out)
 *
 * gem5 (vop3.cc) computes muladd(vdst, src0.u32, src1.u32, src2.u64) and
 * writes the carry to the SDST (VCC).  This test verifies the 64-bit product
 * sum; S2 is kept small so no 64-bit carry occurs.  The SDST operand is
 * required by the encoding and is wired to a dummy SGPR pair.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o v_mad_u64_u32 v_mad_u64_u32.cpp
 * Run:     gem5.opt config.py -- ./v_mad_u64_u32
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
kernel_v_mad_u64_u32(const uint32_t *a, const uint32_t *b,
                     const uint64_t *c, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i];
    uint32_t vb = b[i];
    uint64_t vc = c[i];
    uint64_t vd;
    uint64_t carry;

    __asm__ volatile(
        "v_mad_u64_u32 %0, %1, %2, %3, %4\n\t"
        : "=v"(vd), "=s"(carry)
        : "v"(va), "v"(vb), "v"(vc));

    out[i] = vd;
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint64_t *h_c = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        h_a[i] = 0x10000u + (uint32_t)i * 2654435761u;
        h_b[i] = 0xABCDu + (uint32_t)i * 40503u;
        h_c[i] = (uint64_t)i * 1234567u + 7u;
    }

    uint32_t *d_a, *d_b;
    uint64_t *d_c, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c,   N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(uint64_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_mad_u64_u32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_c, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t expected = (uint64_t)h_a[i] * (uint64_t)h_b[i] + h_c[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%llx expected 0x%llx\n",
                    i, (unsigned long long)h_out[i],
                    (unsigned long long)expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%llx out[%d]=0x%llx\n",
               (unsigned long long)h_out[0], N - 1,
               (unsigned long long)h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_c); free(h_out);
    return pass ? 0 : 1;
}
