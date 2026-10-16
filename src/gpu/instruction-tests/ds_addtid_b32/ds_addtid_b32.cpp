/*
 * HIP test for DS_WRITE_ADDTID_B32 / DS_READ_ADDTID_B32 (ENC_DS, 32-bit LDS
 * access where the per-lane address is computed from M0 and the lane ID rather
 * than from an address VGPR):
 *
 *   addr = offset:{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4
 *   DS_WRITE_ADDTID_B32:  MEM[addr] = DATA0
 *   DS_READ_ADDTID_B32:   VDST      = MEM[addr]
 *
 * The test sets M0 to the byte offset of an LDS array, has every lane store its
 * input through DS_WRITE_ADDTID_B32 (landing at lds[laneID]) and read it back
 * through DS_READ_ADDTID_B32, then verifies the round-trip.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_addtid_b32 ds_addtid_b32.cpp
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

static const int BLOCK_SIZE = 64;
static const int N          = BLOCK_SIZE;

__global__ void
kernel_ds_addtid_b32(const uint32_t *src, uint32_t *out, int n)
{
    int t = threadIdx.x;
    if (t >= n)
        return;

    __shared__ uint32_t lds[BLOCK_SIZE];

    // Byte offset of lds[0] within the LDS segment; this is what M0 must hold
    // so that lane t (addr = M0 + t*4) lands on lds[t].
    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[0]);
    uint32_t base = (uint32_t)(uintptr_t)lp;

    uint32_t val = src[t];

    // Store val to lds[t] via the add-thread-id write.
    __asm__ volatile(
        "s_mov_b32 m0, %0\n\t"
        "ds_write_addtid_b32 %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        : "s"(base), "v"(val)
        : "memory");

    __syncthreads();

    // Read lds[t] back via the add-thread-id read.
    uint32_t res = 0;
    __asm__ volatile(
        "s_mov_b32 m0, %1\n\t"
        "ds_read_addtid_b32 %0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(res)
        : "s"(base)
        : "memory");

    out[t] = res;
}

int
main(void)
{
    uint32_t *h_src = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint32_t)(i * 2654435761u + 0xdeadbeefu);

    uint32_t *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_ds_addtid_b32, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[63]=0x%08x\n", h_out[0], h_out[63]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
