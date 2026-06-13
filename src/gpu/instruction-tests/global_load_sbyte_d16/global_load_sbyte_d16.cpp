/*
 * HIP test for GLOBAL_LOAD_SBYTE_D16 (Inst_FLAT, GLOBAL segment).
 *
 * GLOBAL_LOAD_SBYTE_D16 loads one byte from memory, SIGN-extends it to 16
 * bits, and writes it into the LOW 16 bits of the destination VGPR while
 * PRESERVING the high 16 bits:
 *
 *   VDST[15:0]  = sign_extend(MEM[addr][7:0])   (8-bit -> 16-bit)
 *   VDST[31:16] = VDST[31:16]                    (unchanged)
 *
 * The gem5 implementation reads memory as VecElemI8 and casts to VecElemI16
 * (sign extension), then replaceBits(vdst[lane], 15, 0, tmp) after a
 * vdst.read() that preserves the upper half.  This test exercises bytes
 * with the sign bit set (0x80..0xFF) to confirm sign extension, and the
 * high half (sentinel 0xDEAD) preservation.
 *
 * The instruction reads-modifies the destination, so inline asm uses "+v".
 *
 * Build:
 *   make
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./global_load_sbyte_d16
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
static const uint32_t SENTINEL   = 0xDEADBEEFu;   /* high half 0xDEAD kept */

/*
 * kernel_load_sbyte_d16
 *
 * Each thread sign-extends in[i] (a single byte) into the LOW half of a
 * VGPR seeded with the sentinel.  "+v" keeps the preserved high half.
 */
__global__ void
kernel_load_sbyte_d16(const int8_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(in + i);
    uint32_t vd   = out[i];               /* seed = sentinel */

    __asm__ volatile(
        "global_load_sbyte_d16 %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(vd)
        : "v"(addr)
        : "memory");

    out[i] = vd;
}

int
main(void)
{
    int8_t   *h_in  = (int8_t   *)malloc(N * sizeof(int8_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_in[i]  = (int8_t)(i & 0xFF);    /* covers 0x00..0xFF, incl. neg. */
        h_out[i] = SENTINEL;
    }

    int8_t   *d_in;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(int8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(int8_t),   hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_out, h_out, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_load_sbyte_d16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int16_t  sext     = (int16_t)h_in[i];           /* 8-bit -> 16-bit */
        uint32_t lo       = (uint32_t)(uint16_t)sext;
        uint32_t expected = (SENTINEL & 0xFFFF0000u) | (lo & 0x0000FFFFu);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d] (byte 0x%02x): got 0x%08x expected 0x%08x\n",
                    i, (uint8_t)h_in[i], h_out[i], expected);
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
