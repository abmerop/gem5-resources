/*
 * HIP test for DS_WRITE_B8 (ENC_DS, 8-bit LDS store).
 *
 * DS_WRITE_B8 stores the low 8 bits of the data VGPR to LDS at the given byte
 * offset:  LDS[addr] = DATA[7:0].
 *
 * Each thread owns a distinct LDS byte slot.  The byte is written with the
 * forced ds_write_b8 instruction, then read back with a normal load and
 * verified.  The 32-bit LDS byte offset is obtained by casting the __shared__
 * pointer to the local (address_space(3)) space (same approach as the
 * ds_read_u16_d16 test).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_write_b8 ds_write_b8.cpp
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
kernel_ds_write_b8(const uint32_t *data, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint8_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    /* Pre-fill with a sentinel so we know the store actually wrote. */
    lds[t] = 0xEE;
    __syncthreads();

    __attribute__((address_space(3))) uint8_t *lp =
        (__attribute__((address_space(3))) uint8_t *)(&lds[t]);
    uint32_t addr  = (uint32_t)(uintptr_t)lp;
    uint32_t vdata = data[i];

    __asm__ volatile(
        "ds_write_b8 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        : "v"(addr), "v"(vdata)
        : "memory");

    __syncthreads();
    out[i] = (uint32_t)lds[t];
}

int
main(void)
{
    uint32_t *h_data = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out  = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_data[i] = (uint32_t)(i * 277 + 13);   /* low byte is the payload */

    uint32_t *d_data, *d_out;
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_write_b8, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_data, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = h_data[i] & 0xffu;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%02x expected 0x%02x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%02x out[%d]=0x%02x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_out));
    free(h_data); free(h_out);
    return pass ? 0 : 1;
}
