/*
 * HIP test for FLAT_LOAD_DWORDX2 (ENC_FLAT, 64-bit / 2-dword flat load).
 *
 * FLAT_LOAD_DWORDX2 loads two consecutive dwords (64 bits) from a flat address
 * into a VGPR pair:
 *
 *   VDST[63:0] = MEM64[addr]
 *
 * Per-thread test: thread i reads src[i] (a uint64_t) via the forced
 * flat_load_dwordx2 instruction and stores it to out[i].  s_waitcnt vmcnt(0)
 * stalls until the load data is available.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_load_dwordx2 flat_load_dwordx2.cpp
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
kernel_flat_load_dwordx2(const uint64_t *src, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(src + i);
    uint64_t dst;

    __asm__ volatile(
        "flat_load_dwordx2 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(dst)
        : "v"(addr)
        : "memory");

    out[i] = dst;
}

int
main(void)
{
    uint64_t *h_src = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint64_t)(0x0123456789abcdefull + (uint64_t)i * 0x1111u);

    uint64_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_dwordx2, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%016llx expected 0x%016llx\n",
                    i, (unsigned long long)h_out[i],
                    (unsigned long long)h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%016llx out[%d]=0x%016llx\n",
               (unsigned long long)h_out[0], N - 1,
               (unsigned long long)h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
