/*
 * HIP test for FLAT_STORE_SHORT_D16_HI (ENC_FLAT, store high 16 bits of a VGPR
 * to a 16-bit memory location).
 *
 *   MEM16[addr] = VDATA[31:16]
 *
 * Thread i builds a 32-bit value whose high half is a known payload, stores its
 * high half with flat_store_short_d16_hi, and the host verifies the 16-bit
 * memory word equals that payload.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_store_short_d16_hi flat_store_short_d16_hi.cpp
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
kernel_flat_store_short_d16_hi(uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* high half is the payload to store; low half is filler that must be
     * ignored by the _hi store. */
    uint32_t val = ((uint32_t)(0x3000 + i * 5) << 16) | (uint32_t)(0xBEEF);
    uint64_t addr = (uint64_t)(out + i);

    __asm__ volatile(
        "flat_store_short_d16_hi %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "v"(addr), "v"(val)
        : "memory");
}

int
main(void)
{
    uint16_t *h_out = (uint16_t *)malloc(N * sizeof(uint16_t));

    uint16_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint16_t)));
    GPU_CHECK(hipMemset(d_out, 0xff, N * sizeof(uint16_t)));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_store_short_d16_hi,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t expected = (uint16_t)(0x3000 + i * 5);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%04x expected 0x%04x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%04x out[%d]=0x%04x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);
    return pass ? 0 : 1;
}
