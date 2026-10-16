/*
 * HIP test for FLAT_ATOMIC_INC (ENC_FLAT, 32-bit atomic increment).
 *
 * FLAT_ATOMIC_INC performs, atomically at the flat address:
 *   old = mem; mem = (old >= src) ? 0 : old + 1; return old (with sc0)
 *
 * Each thread targets its own slot (no contention).  To stay agnostic to the
 * exact wrap semantics, every initial value is kept well below the compare
 * operand, so the result is unambiguously old + 1 and the returned value is
 * the original.  Both the returned old value and the post-increment memory are
 * checked.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o flat_atomic_inc flat_atomic_inc.cpp
 * Run:     gem5.opt config.py -- ./flat_atomic_inc
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
static const uint32_t CMP   = 1000000u;   /* far above any init value */

__global__ void
kernel_flat_atomic_inc(uint32_t *mem, const uint32_t *cmp,
                       uint32_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(mem + i);
    uint32_t vcmp = cmp[i];
    uint32_t old;

    __asm__ volatile(
        "flat_atomic_inc %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(vcmp)
        : "memory");

    old_out[i] = old;
}

int
main(void)
{
    uint32_t *h_mem = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_cmp = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_old = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_mem[i] = (uint32_t)(i * 7 + 3);   /* << CMP */
        h_cmp[i] = CMP;
    }

    uint32_t *d_mem, *d_cmp, *d_old;
    GPU_CHECK(hipMalloc(&d_mem, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_cmp, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_old, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_mem, h_mem, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp, h_cmp, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_atomic_inc, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_mem, d_cmp, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t *h_mem_after = (uint32_t *)malloc(N * sizeof(uint32_t));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_mem_after, d_mem, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t exp_old = (uint32_t)(i * 7 + 3);
        uint32_t exp_new = exp_old + 1;
        if (h_old[i] != exp_old || h_mem_after[i] != exp_new) {
            fprintf(stderr, "FAIL i=%d: old got %u exp %u; mem got %u exp %u\n",
                    i, h_old[i], exp_old, h_mem_after[i], exp_new);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("old[0]=%u mem[0]=%u\n", h_old[0], h_mem_after[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_cmp)); GPU_CHECK(hipFree(d_old));
    free(h_mem); free(h_cmp); free(h_old); free(h_mem_after);
    return pass ? 0 : 1;
}
