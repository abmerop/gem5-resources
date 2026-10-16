/*
 * HIP test for DS_RSUB_U32 (ENC_DS, LDS reverse subtract, no return).
 *
 * DS_RSUB_U32 computes, on a 32-bit LDS location:
 *
 *   LDS[addr] = DATA - LDS[addr]      (reverse subtract, 32-bit, no return)
 *
 * Each thread owns a distinct LDS slot, so there is no contention.  The slot is
 * initialised to init[i], the instruction performs data[i] - init[i], and the
 * post-op slot value is read back and verified (mod 2^32).
 *
 * The 32-bit LDS byte offset is obtained by casting the __shared__ pointer to
 * the local (address_space(3)) space (same approach as ds_read_u16_d16).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_rsub_u32 ds_rsub_u32.cpp
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
kernel_ds_rsub_u32(const uint32_t *init, const uint32_t *data, uint32_t *out,
                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint32_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = init[i];
    __syncthreads();

    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t]);
    uint32_t addr  = (uint32_t)(uintptr_t)lp;
    uint32_t vdata = data[i];

    __asm__ volatile(
        "ds_rsub_u32 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        : "v"(addr), "v"(vdata)
        : "memory");

    __syncthreads();
    out[i] = lds[t];
}

int
main(void)
{
    uint32_t *h_init = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_data = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out  = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = (uint32_t)(i * 1234567u + 1);
        h_data[i] = (uint32_t)(0xfeedfaceu - i * 13);
    }

    uint32_t *d_init, *d_data, *d_out;
    GPU_CHECK(hipMalloc(&d_init, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_init, h_init, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_rsub_u32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_init, d_data, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = h_data[i] - h_init[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_init)); GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_out));
    free(h_init); free(h_data); free(h_out);
    return pass ? 0 : 1;
}
