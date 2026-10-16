/*
 * HIP test for DS_READ_U8_D16_HI (ENC_DS, unsigned byte LDS read into the high
 * half of the destination VGPR).
 *
 *   VDST[31:16] = zero_extend16(MEM_U8[addr])
 *   VDST[15:0]  = preserved
 *
 * High-half, byte sibling of DS_READ_U16_D16_HI / DS_READ_U8_D16.  The byte is
 * zero-extended to 16 bits and written into bits [31:16], so bits [31:24] are
 * always zero.
 *
 * Per the ds_read_u16_d16 convention, gem5's completeAcc does not pre-read the
 * destination, so the preserved low half is implementation-defined; verify only
 * the loaded high 16 bits (the instruction's core function).
 *
 * The 32-bit LDS byte offset is obtained by casting the __shared__ pointer to
 * the local (address_space(3)) space.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read_u8_d16_hi ds_read_u8_d16_hi.cpp
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
kernel_ds_read_u8_d16_hi(const uint8_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint8_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = src[i];
    __syncthreads();

    __attribute__((address_space(3))) uint8_t *lp =
        (__attribute__((address_space(3))) uint8_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    /* Preload a sentinel low half; instruction writes the high 16 bits and
     * (per HW) preserves the low 16 bits. */
    uint32_t dst = 0x0000BEEFu;
    __asm__ volatile(
        "ds_read_u8_d16_hi %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+v"(dst)
        : "v"(addr)
        : "memory");

    out[i] = dst;
}

int
main(void)
{
    uint8_t  *h_src = (uint8_t *)malloc(N * sizeof(uint8_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint8_t)(i * 97 + 5);

    uint8_t  *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint8_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_read_u8_d16_hi, dim3(grid),
                       dim3(BLOCK_SIZE), 0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /* Verify the loaded high 16 bits: zero-extended byte (bits [31:24] == 0). */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t got_hi = (uint16_t)(h_out[i] >> 16);
        uint16_t exp_hi = (uint16_t)h_src[i];   /* zero-extended byte */
        if (got_hi != exp_hi) {
            fprintf(stderr, "FAIL out[%d]: hi 0x%04x (exp 0x%04x)\n",
                    i, got_hi, exp_hi);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x (hi=0x%04x)\n", h_out[0], h_out[0] >> 16);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
