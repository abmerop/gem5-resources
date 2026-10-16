/*
 * HIP test for FLAT_ATOMIC_CMPSWAP_X2 (ENC_FLAT, 64-bit compare-and-swap,
 * return-old).
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = (tmp == cmp) ? src : tmp        (64-bit)
 *   VDST       = tmp                              (old value, sc0 return)
 *
 * The data operand is 4 consecutive VGPRs: {src (low 2 dwords), cmp (high 2
 * dwords)} per AMD convention.  The return form is requested with sc0.
 *
 * Each thread targets a disjoint 64-bit slot.  Even lanes use cmp == init so
 * the swap succeeds (mem becomes src); odd lanes use cmp != init so the swap
 * fails (mem stays init).  In both cases the returned old value is init.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o flat_atomic_cmpswap_x2 \
 *         flat_atomic_cmpswap_x2.cpp
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

typedef uint64_t u64x2 __attribute__((ext_vector_type(2)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_flat_atomic_cmpswap_x2(uint64_t *mem, const uint64_t *src,
                              const uint64_t *cmp, uint64_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(mem + i);
    u64x2 data;
    data.x = src[i];   /* swap value (low 2 dwords)  */
    data.y = cmp[i];   /* compare value (high 2 dwords) */
    uint64_t old;

    __asm__ volatile(
        "flat_atomic_cmpswap_x2 %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(data)
        : "memory");

    old_out[i] = old;
}

int
main(void)
{
    uint64_t *h_init = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_src  = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_cmp  = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_mem  = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_old  = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = 0x1111222233334444ULL + ((uint64_t)i << 16);
        h_src[i]  = 0xaaaabbbbccccddddULL ^ ((uint64_t)i * 0x100000001ULL);
        h_cmp[i]  = (i & 1) ? (h_init[i] ^ 0x1ULL)   /* odd: mismatch */
                            : h_init[i];             /* even: match   */
    }

    uint64_t *d_mem, *d_src, *d_cmp, *d_old;
    GPU_CHECK(hipMalloc(&d_mem, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_cmp, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old, N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_mem, h_init, N * sizeof(uint64_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_src, h_src,  N * sizeof(uint64_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp, h_cmp,  N * sizeof(uint64_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_atomic_cmpswap_x2, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_mem, d_src, d_cmp, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * sizeof(uint64_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t exp_old = h_init[i];
        uint64_t exp_mem = (h_init[i] == h_cmp[i]) ? h_src[i] : h_init[i];
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr,
                    "FAIL [%d]: old=0x%016llx(exp 0x%016llx) mem=0x%016llx(exp 0x%016llx)\n",
                    i, (unsigned long long)h_old[i], (unsigned long long)exp_old,
                    (unsigned long long)h_mem[i], (unsigned long long)exp_mem);
            pass = 0; break;
        }
    }

    if (pass)
        printf("old[0]=0x%016llx mem[0]=0x%016llx mem[1]=0x%016llx\n",
               (unsigned long long)h_old[0], (unsigned long long)h_mem[0],
               (unsigned long long)h_mem[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_src));
    GPU_CHECK(hipFree(d_cmp)); GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_src); free(h_cmp); free(h_mem); free(h_old);
    return pass ? 0 : 1;
}
