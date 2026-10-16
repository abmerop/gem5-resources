/*
 * HIP test for FLAT_LOAD_SHORT_D16_HI (ENC_FLAT, 16-bit load into high half).
 *
 * FLAT_LOAD_SHORT_D16_HI loads a 16-bit value from a flat address into the
 * HIGH 16 bits of the destination VGPR, preserving its low 16 bits (per gem5
 * src/arch/amdgpu/common/insts/flat.cc, completeAcc reads vdst then
 * replaceBits(vdst, 31, 16, loaded16)):
 *
 *   VDST[31:16] = MEM16[addr]
 *   VDST[15:0]  = VDST[15:0]   (unchanged)
 *
 * The destination is therefore a read-write operand ("+v"): we seed its low
 * half with a known value, perform the load, and confirm both the loaded high
 * half and the preserved low half.  s_waitcnt vmcnt(0) stalls until the load
 * data is available.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o flat_load_short_d16_hi flat_load_short_d16_hi.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./flat_load_short_d16_hi
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

/*
 * kernel_flat_load_short_d16_hi
 *
 * dst is seeded with low[i] in its low 16 bits; the 16-bit load from src[i]
 * fills the high 16 bits while the low half is preserved.
 */
__global__ void
kernel_flat_load_short_d16_hi(const uint16_t *src, const uint16_t *low,
                              uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t dst = (uint32_t)low[i];      /* known low half; high = 0 */
    uint64_t addr = (uint64_t)(src + i);

    __asm__ volatile(
        "flat_load_short_d16_hi %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(dst)
        : "v"(addr)
        : "memory");

    out[i] = dst;
}

int
main(void)
{
    uint16_t *h_src = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint16_t *h_low = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_src[i] = (uint16_t)(0x1000 + i * 7);
        h_low[i] = (uint16_t)(0xA000 + i);
    }

    uint16_t *d_src, *d_low;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_low, N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint16_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_low, h_low, N * sizeof(uint16_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_short_d16_hi,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_src, d_low, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = ((uint32_t)h_src[i] << 16) | (uint32_t)h_low[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_low));
    GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_low); free(h_out);

    return pass ? 0 : 1;
}
