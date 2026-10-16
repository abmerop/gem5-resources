/*
 * HIP test for DS_ADD_F32 (ENC_DS, LDS float atomic add, no return).
 *
 * DS_ADD_F32 atomically adds a 32-bit float into an LDS location:
 *
 *   LDS[addr] = LDS[addr] + DATA       (f32, no value returned)
 *
 * Rather than hand-roll the LDS address operand, the instruction is emitted by
 * the compiler from a shared-memory atomicAdd whose return value is unused
 * (verified: hipcc -O2 --offload-arch=gfx950 emits "ds_add_f32 v0, v1").
 *
 * Each thread owns a distinct LDS slot, so there is no contention; the final
 * slot value is init + addend.  All values are exact small multiples of 0.25
 * so the float addition is bit-exact.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o ds_add_f32 ds_add_f32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./ds_add_f32
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_ds_add_f32(const float *init, const float *addend, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ float lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = init[i];
    __syncthreads();

    /* return value intentionally unused -> compiler emits ds_add_f32 */
    (void)atomicAdd(&lds[t], addend[i]);

    __syncthreads();
    out[i] = lds[t];
}

int
main(void)
{
    float *h_init   = (float *)malloc(N * sizeof(float));
    float *h_addend = (float *)malloc(N * sizeof(float));
    float *h_out    = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_init[i]   = (float)(i % 17) * 0.25f;
        h_addend[i] = (float)(i % 9) * 0.5f - 2.0f;
    }

    float *d_init, *d_addend, *d_out;
    GPU_CHECK(hipMalloc(&d_init,   N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_addend, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out,    N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_init, h_init, N * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_addend, h_addend, N * sizeof(float),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_add_f32,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_init, d_addend, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_init[i] + h_addend[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %g expected %g\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%g out[%d]=%g\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_init)); GPU_CHECK(hipFree(d_addend));
    GPU_CHECK(hipFree(d_out));
    free(h_init); free(h_addend); free(h_out);

    return pass ? 0 : 1;
}
