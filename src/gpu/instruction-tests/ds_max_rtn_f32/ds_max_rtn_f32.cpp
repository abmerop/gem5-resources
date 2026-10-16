/*
 * HIP test for DS_MAX_RTN_F32 (ENC_DS, LDS float max atomic, return-old).
 *
 *   tmp        = LDS[addr]
 *   LDS[addr]  = max(tmp, DATA)        (f32)
 *   VDST       = tmp                   (old value returned)
 *
 * The RTN form is emitted by the compiler from a shared-memory float atomicMax
 * whose return value is used (verified: hipcc -O2 --offload-arch=gfx950 emits
 * "ds_max_rtn_f32").  Each thread owns a distinct LDS slot (no contention), so
 * the final slot value is max(init, data) and the returned old value is init.
 * Inputs are finite, non-NaN floats so max() is unambiguous.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_max_rtn_f32 ds_max_rtn_f32.cpp
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
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_ds_max_rtn_f32(const float *init, const float *data, float *out_new,
                      float *out_old, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ float lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = init[i];
    __syncthreads();

    /* return value used -> compiler emits ds_max_rtn_f32 */
    float old = atomicMax(&lds[t], data[i]);

    __syncthreads();
    out_new[i] = lds[t];
    out_old[i] = old;
}

int
main(void)
{
    float *h_init = (float *)malloc(N * sizeof(float));
    float *h_data = (float *)malloc(N * sizeof(float));
    float *h_new  = (float *)malloc(N * sizeof(float));
    float *h_old  = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_init[i] = (float)(i - 128) * 1.5f;
        h_data[i] = (float)(64 - i) * 2.25f;
    }

    float *d_init, *d_data, *d_new, *d_old;
    GPU_CHECK(hipMalloc(&d_init, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_new,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_old,  N * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_init, h_init, N * sizeof(float), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(float), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_max_rtn_f32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_init, d_data, d_new, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_new, d_new, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(float), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float exp_old = h_init[i];
        float exp_new = h_init[i] > h_data[i] ? h_init[i] : h_data[i];
        if (h_old[i] != exp_old || h_new[i] != exp_new) {
            fprintf(stderr, "FAIL [%d]: old=%g(exp %g) new=%g(exp %g)\n",
                    i, h_old[i], exp_old, h_new[i], exp_new);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("old[0]=%g new[0]=%g\n", h_old[0], h_new[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_init)); GPU_CHECK(hipFree(d_data));
    GPU_CHECK(hipFree(d_new));  GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_data); free(h_new); free(h_old);
    return pass ? 0 : 1;
}
