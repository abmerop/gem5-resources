/*
 * HIP test for DS_MSKOR_RTN_B32 (ENC_DS, masked-OR to LDS with return).
 *
 * ISA (src/arch/amdgpu/common/insts/ds.cc description):
 *   tmp        = MEM[ADDR];
 *   MEM[ADDR]  = (MEM[ADDR] & ~DATA0) | DATA1;     // DATA0 = mask, DATA1 = new
 *   RETURN     = tmp;
 *
 * Operand order (llvm-mc gfx950):
 *   ds_mskor_rtn_b32 vdst, addr, data0(mask), data1(value)
 *
 * Each lane owns a private LDS dword.  We seed it with a known "old" value,
 * apply the masked-OR, then check BOTH the returned old value AND the resulting
 * LDS contents.  (gem5 currently routes MSKOR through cmpstExecute, i.e.
 * compare-and-swap, so the stored value is wrong -- this test is expected to
 * FAIL until the execute() is corrected to masked-OR.)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_mskor_rtn_b32 ds_mskor_rtn_b32.cpp
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
       uint32_t *ret, uint32_t *newv, int n)
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

    uint32_t r;
    __asm__ volatile("ds_mskor_rtn_b32 %0, %1, %2, %3\n\t"
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
    const uint32_t mask  = 0x0F0F0F0Fu;
    const uint32_t value = 0x12345678u;

    uint32_t *h_old = (uint32_t *)malloc(N * 4);
    uint32_t *h_ret = (uint32_t *)malloc(N * 4);
    uint32_t *h_new = (uint32_t *)malloc(N * 4);
    for (int i = 0; i < N; i++)
        h_old[i] = 0xA5A5A5A5u ^ (uint32_t)(i * 2654435761u);

    uint32_t *d_old, *d_ret, *d_new;
    GPU_CHECK(hipMalloc(&d_old, N * 4));
    GPU_CHECK(hipMalloc(&d_ret, N * 4));
    GPU_CHECK(hipMalloc(&d_new, N * 4));
    GPU_CHECK(hipMemcpy(d_old, h_old, N * 4, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel, dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_old, mask, value, d_ret, d_new, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_ret, d_ret, N * 4, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_new, d_new, N * 4, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint32_t exp_ret = h_old[i];
        uint32_t exp_new = (h_old[i] & ~mask) | value;
        if (h_ret[i] != exp_ret || h_new[i] != exp_new) {
            fprintf(stderr,
                    "FAIL[%d] ret got 0x%08x exp 0x%08x ; new got 0x%08x exp 0x%08x\n",
                    i, h_ret[i], exp_ret, h_new[i], exp_new);
            pass = 0;
            break;
        }
    }
    if (pass)
        printf("ret[0]=0x%08x new[0]=0x%08x\n", h_ret[0], h_new[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    return pass ? 0 : 1;
}
