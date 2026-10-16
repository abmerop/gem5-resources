/*
 * HIP test for FLAT_ATOMIC_SMIN_X2 (ENC_FLAT, 64-bit signed atomic minimum).
 *
 * FLAT_ATOMIC_SMIN_X2 performs, atomically at the flat address:
 *   old = mem; mem = min(old, src) (signed i64); return old (with sc0)
 *
 * Each thread targets its own slot (no contention).  Inputs mix cases where
 * src < old and src > old, including negative values, so the signed min is
 * meaningfully exercised.  Both the returned old value and the final memory
 * contents are checked.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o flat_atomic_smin_x2 flat_atomic_smin_x2.cpp
 * Run:     gem5.opt config.py -- ./flat_atomic_smin_x2
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
kernel_flat_atomic_smin_x2(int64_t *mem, const int64_t *src,
                           int64_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(mem + i);
    int64_t vsrc = src[i];
    int64_t old;

    __asm__ volatile(
        "flat_atomic_smin_x2 %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(vsrc)
        : "memory");

    old_out[i] = old;
}

int
main(void)
{
    int64_t *h_mem = (int64_t *)malloc(N * sizeof(int64_t));
    int64_t *h_src = (int64_t *)malloc(N * sizeof(int64_t));
    int64_t *h_old = (int64_t *)malloc(N * sizeof(int64_t));

    for (int i = 0; i < N; i++) {
        h_mem[i] = ((int64_t)(i - 128)) * 1000000007LL;
        h_src[i] = ((int64_t)((i % 11) - 5)) * 777777777LL;
    }

    int64_t *d_mem, *d_src, *d_old;
    GPU_CHECK(hipMalloc(&d_mem, N * sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_old, N * sizeof(int64_t)));

    GPU_CHECK(hipMemcpy(d_mem, h_mem, N * sizeof(int64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(int64_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_atomic_smin_x2, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_mem, d_src, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    int64_t *h_mem_after = (int64_t *)malloc(N * sizeof(int64_t));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(int64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_mem_after, d_mem, N * sizeof(int64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int64_t exp_old = ((int64_t)(i - 128)) * 1000000007LL;
        int64_t s       = ((int64_t)((i % 11) - 5)) * 777777777LL;
        int64_t exp_new = exp_old < s ? exp_old : s;
        if (h_old[i] != exp_old || h_mem_after[i] != exp_new) {
            fprintf(stderr, "FAIL i=%d: old got %lld exp %lld; "
                    "mem got %lld exp %lld\n", i,
                    (long long)h_old[i], (long long)exp_old,
                    (long long)h_mem_after[i], (long long)exp_new);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("old[0]=%lld mem[0]=%lld\n",
               (long long)h_old[0], (long long)h_mem_after[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_old));
    free(h_mem); free(h_src); free(h_old); free(h_mem_after);
    return pass ? 0 : 1;
}
