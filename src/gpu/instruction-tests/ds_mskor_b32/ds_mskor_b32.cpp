/*
 * HIP test for DS_MSKOR_B32 (ENC_DS, masked-OR to LDS, no return).
 *
 * ISA:  MEM[ADDR] = (MEM[ADDR] & ~DATA0) | DATA1;   DATA0=mask, DATA1=value.
 * Operand order (llvm-mc gfx950):  ds_mskor_b32 addr, data0(mask), data1(value)
 *
 * Sibling of DS_MSKOR_RTN_B32 (confirmed working).  Each lane owns a private
 * LDS dword; we seed it, apply the masked-OR, then read the resulting contents.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_mskor_b32 ds_mskor_b32.cpp
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel(const uint32_t *oldv, uint32_t mask, uint32_t value,
       uint32_t *newv, int n)
{
    __shared__ uint32_t lds[BLOCK_SIZE];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (i >= n)
        return;

    lds[t] = oldv[i];
    __syncthreads();

    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    __asm__ volatile("ds_mskor_b32 %0, %1, %2\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     :
                     : "v"(addr), "v"(mask), "v"(value)
                     : "memory");
    __syncthreads();

    newv[i] = lds[t];
}

int
main(void)
{
    const uint32_t mask = 0x0F0F0F0Fu, value = 0x12345678u;
    uint32_t *h_old = (uint32_t *)malloc(N * 4);
    uint32_t *h_new = (uint32_t *)malloc(N * 4);
    for (int i = 0; i < N; i++)
        h_old[i] = 0xA5A5A5A5u ^ (uint32_t)(i * 2654435761u);

    uint32_t *d_old, *d_new;
    GPU_CHECK(hipMalloc(&d_old, N * 4));
    GPU_CHECK(hipMalloc(&d_new, N * 4));
    GPU_CHECK(hipMemcpy(d_old, h_old, N * 4, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel, dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_old, mask, value, d_new, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_new, d_new, N * 4, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint32_t exp = (h_old[i] & ~mask) | value;
        if (h_new[i] != exp) {
            fprintf(stderr, "FAIL[%d] new got 0x%08x exp 0x%08x\n",
                    i, h_new[i], exp);
            pass = 0;
        }
    }
    if (pass)
        printf("new[0]=0x%08x new[1]=0x%08x\n", h_new[0], h_new[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
