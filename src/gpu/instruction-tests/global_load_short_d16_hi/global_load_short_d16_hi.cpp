/*
 * HIP test for GLOBAL_LOAD_SHORT_D16_HI (Inst_FLAT, GLOBAL segment).
 *
 * GLOBAL_LOAD_SHORT_D16_HI loads a 16-bit value from memory and writes it
 * into the HIGH 16 bits of the destination VGPR while PRESERVING the low
 * 16 bits:
 *
 *   VDST[31:16] = MEM[addr][15:0]
 *   VDST[15:0]  = VDST[15:0]         (unchanged)
 *
 * The gem5 implementation reads memory as VecElemU16 and does
 * replaceBits(vdst[lane], 31, 16, tmp) after a vdst.read() that preserves
 * the lower half.  This test verifies the 16-bit value lands in the high
 * half and that the low half (sentinel 0xBEEF) is untouched.
 *
 * The instruction reads-modifies the destination, so inline asm uses "+v".
 *
 * Build:
 *   make
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./global_load_short_d16_hi
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int      N          = 256;
static const int      BLOCK_SIZE = 64;
static const uint32_t SENTINEL   = 0xDEADBEEFu;   /* low half 0xBEEF kept */

/*
 * kernel_load_short_d16_hi
 *
 * Each thread loads in[i] (a 16-bit value) into the HIGH half of a VGPR
 * seeded with the sentinel.  "+v" keeps the preserved low half.
 */
__global__ void
kernel_load_short_d16_hi(const uint16_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(in + i);
    uint32_t vd   = out[i];               /* seed = sentinel */

    __asm__ volatile(
        "global_load_short_d16_hi %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(vd)
        : "v"(addr)
        : "memory");

    out[i] = vd;
}

int
main(void)
{
    uint16_t *h_in  = (uint16_t *)malloc(N * sizeof(uint16_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_in[i]  = (uint16_t)(i * 257 + 0x8000);
        h_out[i] = SENTINEL;
    }

    uint16_t *d_in;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(uint16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_out, h_out, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_load_short_d16_hi,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t hi       = (uint32_t)h_in[i];
        uint32_t expected = ((hi & 0x0000FFFFu) << 16) | (SENTINEL & 0x0000FFFFu);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d] (short 0x%04x): got 0x%08x expected 0x%08x\n",
                    i, h_in[i], h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in));
    GPU_CHECK(hipFree(d_out));
    free(h_in);
    free(h_out);

    return pass ? 0 : 1;
}
