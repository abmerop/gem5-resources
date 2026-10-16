/*
 * HIP test for V_CMP_T_U64 (VOPC / VOP3, 64-bit "true" compare).
 *
 * V_CMP_T_U64 is the unconditional-true comparison: regardless of the operand
 * values, each active lane's result bit is 1:
 *
 *   VCC[lane] = 1
 *
 * The VOP3 (_e64) form writes the per-lane mask to an SGPR pair; each lane (a
 * single 64-lane wavefront per block) extracts its own bit, which must always
 * be 1.  Operands are arbitrary u64 values to confirm they do not affect the
 * result.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cmp_t_u64 v_cmp_t_u64.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmp_t_u64
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
kernel_v_cmp_t_u64(const uint64_t *a, const uint64_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t va = a[i];
    uint64_t vb = b[i];
    uint64_t mask;

    __asm__ volatile(
        "v_cmp_t_u64_e64 %0, %1, %2\n\t"
        : "=s"(mask)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    uint64_t *h_a = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_b = (uint64_t *)malloc(N * sizeof(uint64_t));
    int      *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint64_t)i * 0x100000001ULL;
        h_b[i] = ~((uint64_t)i);
    }

    uint64_t *d_a, *d_b;
    int      *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint64_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint64_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmp_t_u64,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != 1) {
            fprintf(stderr, "FAIL out[%d]: got %d expected 1\n", i, h_out[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%d out[%d]=%d (all true)\n",
               h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);

    return pass ? 0 : 1;
}
