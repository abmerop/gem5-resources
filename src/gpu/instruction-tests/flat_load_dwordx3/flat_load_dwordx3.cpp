/*
 * HIP test for FLAT_LOAD_DWORDX3 (ENC_FLAT, 96-bit load into 3 VGPRs).
 *
 *   {VDST0,VDST1,VDST2} = MEM[addr .. addr+11]
 *
 * Thread i loads three consecutive dwords from in[i*3 .. i*3+2] with a single
 * 96-bit flat load and the host verifies all three.  s_waitcnt vmcnt(0) stalls
 * until the load data is available.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_load_dwordx3 flat_load_dwordx3.cpp
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

typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_flat_load_dwordx3(const uint32_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(in + (size_t)i * 3);
    u32x3 data;

    __asm__ volatile(
        "flat_load_dwordx3 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(addr)
        : "memory");

    out[i * 3 + 0] = data.x;
    out[i * 3 + 1] = data.y;
    out[i * 3 + 2] = data.z;
}

int
main(void)
{
    const int total = N * 3;
    uint32_t *h_in  = (uint32_t *)malloc(total * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(total * sizeof(uint32_t));

    for (int i = 0; i < total; i++)
        h_in[i] = (uint32_t)(i * 2654435761u + 99u);

    uint32_t *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  total * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_in, h_in, total * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_dwordx3, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, total * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        if (h_out[i] != h_in[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_in[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], total - 1, h_out[total - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return pass ? 0 : 1;
}
