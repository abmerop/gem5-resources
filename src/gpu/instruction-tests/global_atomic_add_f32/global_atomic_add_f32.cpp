/*
 * HIP test for GLOBAL_ATOMIC_ADD_F32 (ENC_FLAT_GLBL, 32-bit float atomic add,
 * return-old).
 *
 * GLOBAL_ATOMIC_ADD_F32 performs, atomically on a 32-bit float in global
 * memory:
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp + DATA            (f32 add)
 *   VDST       = tmp                    (old value, with sc0 return)
 *
 * The instruction is emitted by the compiler from unsafeAtomicAdd() on a
 * global float pointer (verified: hipcc -O2 --offload-arch=gfx950 emits
 * "global_atomic_add_f32 ... sc0").  Each thread owns a distinct slot so there
 * is no contention; the final value is init + addend and the returned old
 * value is init.  All addends are exact small multiples of 0.25 so the f32
 * addition is bit-exact.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o global_atomic_add_f32 global_atomic_add_f32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./global_atomic_add_f32
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
kernel_global_atomic_add_f32(float *mem, const float *addend, float *old_out,
                             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* unsafeAtomicAdd -> hardware global_atomic_add_f32 (return-old) */
    old_out[i] = unsafeAtomicAdd(&mem[i], addend[i]);
}

int
main(void)
{
    float *h_init   = (float *)malloc(N * sizeof(float));
    float *h_addend = (float *)malloc(N * sizeof(float));
    float *h_mem    = (float *)malloc(N * sizeof(float));
    float *h_old    = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_init[i]   = (float)(i % 17) * 0.25f;
        h_addend[i] = (float)(i % 9) * 0.5f - 2.0f;
    }

    float *d_mem, *d_addend, *d_old;
    GPU_CHECK(hipMalloc(&d_mem,    N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_addend, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_old,    N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_mem, h_init, N * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_addend, h_addend, N * sizeof(float),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_global_atomic_add_f32,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_mem, d_addend, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * sizeof(float),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(float),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float exp_old = h_init[i];
        float exp_mem = h_init[i] + h_addend[i];
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr, "FAIL [%d]: old=%g(exp %g) mem=%g(exp %g)\n",
                    i, h_old[i], exp_old, h_mem[i], exp_mem);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("old[1]=%g mem[1]=%g\n", h_old[1], h_mem[1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_addend));
    GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_addend); free(h_mem); free(h_old);

    return pass ? 0 : 1;
}
