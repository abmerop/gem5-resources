/*
 * HIP test for DS_READ2_B32 (ENC_DS, dual 32-bit LDS read).
 *
 * DS_READ2_B32 reads two dwords from LDS into a VGPR pair, at two independent
 * dword offsets (gem5: offset0*4, offset1*4 from the address VGPR):
 *
 *   VDST[0] = LDS[addr + offset0*4]
 *   VDST[1] = LDS[addr + offset1*4]
 *
 * Each thread owns a 2-dword LDS slot.  We fill both dwords, then read them
 * back with offset0:0 offset1:1 and verify.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read2_b32 ds_read2_b32.cpp
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

typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_ds_read2_b32(const uint32_t *data, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint32_t lds[BLOCK_SIZE * 2];
    int t = threadIdx.x;

    lds[t * 2 + 0] = data[i * 2 + 0];
    lds[t * 2 + 1] = data[i * 2 + 1];
    __syncthreads();

    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t * 2]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    u32x2 vd;
    __asm__ volatile(
        "ds_read2_b32 %0, %1 offset0:0 offset1:1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(vd)
        : "v"(addr)
        : "memory");

    out[i * 2 + 0] = vd.x;
    out[i * 2 + 1] = vd.y;
}

int
main(void)
{
    uint32_t *h_data = (uint32_t *)malloc(N * 2 * sizeof(uint32_t));
    uint32_t *h_out  = (uint32_t *)malloc(N * 2 * sizeof(uint32_t));

    for (int i = 0; i < N * 2; i++)
        h_data[i] = (uint32_t)(i * 2654435761u + 17u);

    uint32_t *d_data, *d_out;
    GPU_CHECK(hipMalloc(&d_data, N * 2 * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * 2 * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * 2 * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_read2_b32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_data, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * 2 * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N * 2; i++) {
        if (h_out[i] != h_data[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_data[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N * 2 - 1, h_out[N * 2 - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_out));
    free(h_data); free(h_out);
    return pass ? 0 : 1;
}
