/*
 * HIP test for GLOBAL_ATOMIC_CMPSWAP (ENC_FLAT_GLBL, 32-bit compare-and-swap,
 * return-old).
 *   tmp = MEM[addr];
 *   MEM[addr] = (tmp == cmp) ? src : tmp;
 *   VDST = tmp                                (old value, sc0 return)
 *
 * The data operand is 2 consecutive VGPRs: {src (swap value, low dword),
 * cmp (compare value, high dword)} -- same ordering as FLAT_ATOMIC_CMPSWAP_X2.
 *
 * Each thread targets a disjoint 32-bit slot (no contention).  Even lanes use
 * cmp == init so the swap succeeds (mem becomes src); odd lanes use cmp != init
 * so the swap fails (mem stays init).  In both cases the returned old value is
 * init.  Pattern C, base only.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o global_atomic_cmpswap \
 *         global_atomic_cmpswap.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./global_atomic_cmpswap
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void
kernel(uint32_t *mem, const uint32_t *src, const uint32_t *cmp,
       uint32_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t addr = (uint64_t)(uintptr_t)(&mem[i]);
    uint2 data;
    data.x = src[i];   /* swap value  (low dword)  */
    data.y = cmp[i];   /* compare value (high dword) */
    uint32_t old;
    __asm__ volatile(
        "global_atomic_cmpswap %0, %1, %2, off sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(data)
        : "memory");
    old_out[i] = old;
}

int
main(void)
{
    uint32_t *h_init = (uint32_t *)malloc(N * 4);
    uint32_t *h_src  = (uint32_t *)malloc(N * 4);
    uint32_t *h_cmp  = (uint32_t *)malloc(N * 4);
    uint32_t *h_mem  = (uint32_t *)malloc(N * 4);
    uint32_t *h_old  = (uint32_t *)malloc(N * 4);

    for (int i = 0; i < N; i++) {
        h_init[i] = (uint32_t)(i * 2654435761u + 17u);
        h_src[i]  = (uint32_t)(i * 40503u + 3u);
        /* even lanes: cmp matches init (swap succeeds);
           odd lanes: cmp differs (swap fails) */
        h_cmp[i]  = (i & 1) ? (h_init[i] ^ 0xdeadbeefu) : h_init[i];
    }

    uint32_t *d_mem, *d_src, *d_cmp, *d_old;
    GPU_CHECK(hipMalloc(&d_mem, N * 4));
    GPU_CHECK(hipMalloc(&d_src, N * 4));
    GPU_CHECK(hipMalloc(&d_cmp, N * 4));
    GPU_CHECK(hipMalloc(&d_old, N * 4));
    GPU_CHECK(hipMemcpy(d_mem, h_init, N * 4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_src, h_src,  N * 4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp, h_cmp,  N * 4, hipMemcpyHostToDevice));

    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_mem, d_src, d_cmp, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * 4, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * 4, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t exp_mem = (i & 1) ? h_init[i] : h_src[i];
        uint32_t exp_old = h_init[i];
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr, "FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",
                    i, h_old[i], exp_old, h_mem[i], exp_mem);
            pass = 0;
            break;
        }
    }
    if (pass)
        printf("old[0]=%u mem[0]=%u mem[1]=%u\n", h_old[0], h_mem[0], h_mem[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
