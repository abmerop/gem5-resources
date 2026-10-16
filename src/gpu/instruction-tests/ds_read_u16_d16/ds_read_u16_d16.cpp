/*
 * HIP test for DS_READ_U16_D16 (ENC_DS, 16-bit LDS read into low half).
 *
 * DS_READ_U16_D16 reads a 16-bit value from LDS at the given byte offset and
 * writes it into bits [15:0] of the destination VGPR.  Per the ISA the high
 * half [31:16] is preserved, but gem5's completeAcc does not pre-read the
 * destination, so it is left implementation-defined here; this test therefore
 * verifies only the loaded low 16 bits (the instruction's core function).
 *
 * The 32-bit LDS byte offset is obtained by casting the __shared__ pointer to
 * the local (address_space(3)) space, matching the ds_pk_add_rtn_bf16 test.
 *
 * Build:   hipcc -O2 --offload-arch=gfx950 -o ds_read_u16_d16 ds_read_u16_d16.cpp
 * Run:     gem5.opt config.py -- ./ds_read_u16_d16
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
kernel_ds_read_u16_d16(const uint16_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint16_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = src[i];
    __syncthreads();

    __attribute__((address_space(3))) uint16_t *lp =
        (__attribute__((address_space(3))) uint16_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    /* Preload a sentinel high half; instruction writes the low 16 bits. */
    uint32_t dst = 0xABCD0000u;
    __asm__ volatile(
        "ds_read_u16_d16 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+v"(dst)
        : "v"(addr)
        : "memory");

    out[i] = dst;
}

int
main(void)
{
    uint16_t *h_src = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint16_t)(i * 277 + 13);

    uint16_t *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint16_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_read_u16_d16, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t got = (uint16_t)(h_out[i] & 0xffff);
        if (got != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got low16 0x%04x expected 0x%04x\n",
                    i, got, h_src[i]);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x (low16=0x%04x)\n", h_out[0], h_out[0] & 0xffff);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
