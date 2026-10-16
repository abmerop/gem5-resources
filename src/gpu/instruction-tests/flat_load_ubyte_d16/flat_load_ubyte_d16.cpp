/*
 * HIP test for FLAT_LOAD_UBYTE_D16 (ENC_FLAT, unsigned-byte load into LOW half
 * of destination VGPR, preserving the high 16 bits).
 *
 *   VDST[15:0]  = zeroext16(MEM8[addr])
 *   VDST[31:16] = VDST[31:16]   (unchanged)
 *
 * The destination is a read-write operand ("+v"): we seed its high half with a
 * known value, perform the load, and confirm both the zero-extended loaded low
 * byte and the preserved high half.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_load_ubyte_d16 flat_load_ubyte_d16.cpp
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
kernel_flat_load_ubyte_d16(const uint8_t *src, const uint16_t *high,
                           uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t dst = (uint32_t)high[i] << 16;   /* known high half; low = 0 */
    uint64_t addr = (uint64_t)(src + i);

    __asm__ volatile(
        "flat_load_ubyte_d16 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(dst)
        : "v"(addr)
        : "memory");

    out[i] = dst;
}

int
main(void)
{
    uint8_t  *h_src  = (uint8_t  *)malloc(N * sizeof(uint8_t));
    uint16_t *h_high = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint32_t *h_out  = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_src[i]  = (uint8_t)(i * 7 + 3);
        h_high[i] = (uint16_t)(0xC000 + i);
    }

    uint8_t  *d_src;
    uint16_t *d_high;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src,  N * sizeof(uint8_t)));
    GPU_CHECK(hipMalloc(&d_high, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_src,  h_src,  N * sizeof(uint8_t),  hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_high, h_high, N * sizeof(uint16_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_ubyte_d16, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_high, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = ((uint32_t)h_high[i] << 16) | (uint32_t)h_src[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_high)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_high); free(h_out);
    return pass ? 0 : 1;
}
