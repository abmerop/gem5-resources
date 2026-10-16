/*
 * HIP test for DS_WRITE_B128 (ENC_DS, 128-bit LDS store).
 *
 * DS_WRITE_B128 stores 4 consecutive dwords (128 bits) from the data VGPR
 * range to LDS at the given byte offset:  LDS[addr .. addr+15] = DATA[127:0].
 *
 * Each thread owns a distinct 16-byte LDS slot.  The slot is pre-filled with a
 * sentinel, the 128-bit value is stored with the forced ds_write_b128
 * instruction, then read back with normal loads and verified.  The 32-bit LDS
 * byte offset is obtained by casting the __shared__ pointer to the local
 * (address_space(3)) space (same approach as the ds_write_b8 test).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_write_b128 ds_write_b128.cpp
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

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_ds_write_b128(const uint32_t *data, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint32_t lds[BLOCK_SIZE * 4];
    int t = threadIdx.x;

    /* Pre-fill with a sentinel so we know the store actually wrote. */
    for (int k = 0; k < 4; k++)
        lds[t * 4 + k] = 0xEEEEEEEEu;
    __syncthreads();

    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t * 4]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    u32x4 vdata;
    vdata.x = data[i * 4 + 0];
    vdata.y = data[i * 4 + 1];
    vdata.z = data[i * 4 + 2];
    vdata.w = data[i * 4 + 3];

    __asm__ volatile(
        "ds_write_b128 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        : "v"(addr), "v"(vdata)
        : "memory");

    __syncthreads();
    for (int k = 0; k < 4; k++)
        out[i * 4 + k] = lds[t * 4 + k];
}

int
main(void)
{
    uint32_t *h_data = (uint32_t *)malloc(N * 4 * sizeof(uint32_t));
    uint32_t *h_out  = (uint32_t *)malloc(N * 4 * sizeof(uint32_t));

    for (int i = 0; i < N * 4; i++)
        h_data[i] = (uint32_t)(i * 2654435761u + 1013904223u);

    uint32_t *d_data, *d_out;
    GPU_CHECK(hipMalloc(&d_data, N * 4 * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * 4 * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * 4 * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_write_b128, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_data, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * 4 * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N * 4; i++) {
        if (h_out[i] != h_data[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_data[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N * 4 - 1, h_out[N * 4 - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_out));
    free(h_data); free(h_out);
    return pass ? 0 : 1;
}
