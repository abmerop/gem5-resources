/*
 * HIP test for DS_MSKOR_RTN_B64 (ENC_DS, 64-bit masked-OR to LDS with return).
 *
 * ISA:  tmp=MEM[ADDR]; MEM[ADDR]=(MEM[ADDR] & ~DATA0)|DATA1; RETURN=tmp;
 *       DATA0=mask, DATA1=value, all 64-bit.
 * Operand order:  ds_mskor_rtn_b64 vdst, addr, data0(mask), data1(value)
 *
 * Sibling of DS_MSKOR_RTN_B32 (confirmed working).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_mskor_rtn_b64 ds_mskor_rtn_b64.cpp
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
kernel(const uint64_t *oldv, uint64_t mask, uint64_t value,
       uint64_t *ret, uint64_t *newv, int n)
{
    __shared__ uint64_t lds[BLOCK_SIZE];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (i >= n)
        return;

    lds[t] = oldv[i];
    __syncthreads();

    __attribute__((address_space(3))) uint64_t *lp =
        (__attribute__((address_space(3))) uint64_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    uint64_t r;
    __asm__ volatile("ds_mskor_rtn_b64 %0, %1, %2, %3\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(r)
                     : "v"(addr), "v"(mask), "v"(value)
                     : "memory");
    __syncthreads();

    ret[i]  = r;
    newv[i] = lds[t];
}

int
main(void)
{
    const uint64_t mask  = 0x0F0F0F0F0F0F0F0FULL;
    const uint64_t value = 0x123456789ABCDEF0ULL;
    uint64_t *h_old = (uint64_t *)malloc(N * 8);
    uint64_t *h_ret = (uint64_t *)malloc(N * 8);
    uint64_t *h_new = (uint64_t *)malloc(N * 8);
    for (int i = 0; i < N; i++)
        h_old[i] = 0xA5A5A5A5A5A5A5A5ULL ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1));

    uint64_t *d_old, *d_ret, *d_new;
    GPU_CHECK(hipMalloc(&d_old, N * 8));
    GPU_CHECK(hipMalloc(&d_ret, N * 8));
    GPU_CHECK(hipMalloc(&d_new, N * 8));
    GPU_CHECK(hipMemcpy(d_old, h_old, N * 8, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel, dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_old, mask, value, d_ret, d_new, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_ret, d_ret, N * 8, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_new, d_new, N * 8, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint64_t exp_ret = h_old[i];
        uint64_t exp_new = (h_old[i] & ~mask) | value;
        if (h_ret[i] != exp_ret || h_new[i] != exp_new) {
            fprintf(stderr,
                    "FAIL[%d] ret got 0x%016llx exp 0x%016llx ; new got 0x%016llx exp 0x%016llx\n",
                    i, (unsigned long long)h_ret[i], (unsigned long long)exp_ret,
                    (unsigned long long)h_new[i], (unsigned long long)exp_new);
            pass = 0;
        }
    }
    if (pass)
        printf("ret[0]=0x%016llx new[0]=0x%016llx\n",
               (unsigned long long)h_ret[0], (unsigned long long)h_new[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
