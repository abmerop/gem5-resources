/*
 * HIP test for FLAT_ATOMIC_XOR (ENC_FLAT, 32-bit atomic XOR, return-old).
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp ^ DATA          (32-bit)
 *   VDST       = tmp                  (old value, sc0 return)
 *
 * The return form is requested with the sc0 cache modifier on CDNA4 (gfx950).
 * Each thread targets a disjoint 32-bit slot, so there is no contention.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_xor flat_atomic_xor.cpp
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
kernel_flat_atomic_xor(uint32_t *mem, const uint32_t *data,
                       uint32_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(mem + i);
    uint32_t vdata = data[i];
    uint32_t old;

    __asm__ volatile(
        "flat_atomic_xor %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(vdata)
        : "memory");

    old_out[i] = old;
}

int
main(void)
{
    uint32_t *h_init = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_data = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_mem  = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_old  = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = 0x89abcdefu ^ (uint32_t)(i * 0x01010101u);
        h_data[i] = 0xdeadbeefu + (uint32_t)(i << 8);
    }

    uint32_t *d_mem, *d_data, *d_old;
    GPU_CHECK(hipMalloc(&d_mem,  N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_old,  N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_mem,  h_init, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_atomic_xor, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_mem, d_data, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t exp_old = h_init[i];
        uint32_t exp_mem = h_init[i] ^ h_data[i];
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr, "FAIL [%d]: old=0x%08x(exp 0x%08x) mem=0x%08x(exp 0x%08x)\n",
                    i, h_old[i], exp_old, h_mem[i], exp_mem);
            pass = 0; break;
        }
    }

    if (pass)
        printf("old[1]=0x%08x mem[1]=0x%08x\n", h_old[1], h_mem[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_data); free(h_mem); free(h_old);
    return pass ? 0 : 1;
}
