/*
 * HIP test for DS_READ_I8_D16 (ENC_DS, 8-bit LDS read, SIGN-extended into the
 * low 16 bits of the destination VGPR).
 *
 *   VDST[15:0]  = sext8( LDS_byte[addr] )
 *   VDST[31:16] = preserved
 *
 * Modeled on ds_read_u8_d16, but the loaded byte is sign-extended (gem5 uses
 * sext<8>).  Only the low 16 bits are verified.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read_i8_d16_hi ds_read_i8_d16_hi.cpp
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

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kernel(const uint8_t *src, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    __shared__ uint8_t lds[BLOCK_SIZE];
    int t = threadIdx.x;
    lds[t] = src[i];
    __syncthreads();

    __attribute__((address_space(3))) uint8_t *lp =
        (__attribute__((address_space(3))) uint8_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    uint32_t dst = 0xABCD0000u;
    __asm__ volatile(
        "ds_read_i8_d16_hi %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+v"(dst) : "v"(addr) : "memory");
    out[i] = dst;
}

int main(void) {
    uint8_t  *h_src = (uint8_t  *)malloc(N);
    uint32_t *h_out = (uint32_t *)malloc(N * 4);
    for (int i = 0; i < N; i++)
        h_src[i] = (uint8_t)(i * 277 + 13);   // spans negative bytes too

    uint8_t *d_src; uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N)); GPU_CHECK(hipMalloc(&d_out, N * 4));
    GPU_CHECK(hipMemcpy(d_src, h_src, N, hipMemcpyHostToDevice));
    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * 4, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t e = (uint16_t)(int16_t)(int8_t)h_src[i];   // sign extend
        uint16_t got = (uint16_t)(h_out[i] >> 16);
        if (got != e) {
            fprintf(stderr, "FAIL out[%d]: got 0x%04x expected 0x%04x\n",
                    i, got, e);
            pass = 0; break;
        }
    }
    if (pass) printf("out[0]=0x%08x out[200]=0x%08x\n", h_out[0], h_out[200]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
