/*
 * HIP test for GLOBAL_STORE_BYTE_D16_HI (Inst_FLAT, GLOBAL segment).
 *
 * GLOBAL_STORE_BYTE_D16_HI stores ONE byte to memory taken from bits
 * [23:16] of the data VGPR (the low byte of the high 16-bit word):
 *
 *   MEM[addr][7:0] = DATA[23:16]
 *
 * This is the store counterpart of the LOAD_*_D16_HI family: it lets you
 * write back the byte that lives in the upper half of a packed register
 * without first shifting it down.
 *
 * The data value is kept in a full 32-bit VGPR ("v" constraint on a
 * uint32_t) so that bits [23:16] are well defined.  Each input word is
 * constructed so every byte position is distinct:
 *
 *   in[i] = 0xAA | (target_byte << 16) | 0x00CD..  -> 0xAA<bb>CDEF
 *
 *   bits [31:24] = 0xAA      (must NOT be stored)
 *   bits [23:16] = target    (THE byte that must be stored)
 *   bits [15:8]  = 0xCD      (must NOT be stored)
 *   bits [7:0]   = 0xEF      (must NOT be stored -- a plain store_byte bug
 *                             would store this instead)
 *
 * so a correct implementation stores `target`, a plain store_byte would
 * store 0xEF, and a zero-extraction bug would store 0x00.
 *
 * NOTE on the gem5 implementation: commit 22990c6b11 originally read the
 * data operand as ConstVecOperandU8, which only captures bits [7:0] of the
 * VGPR, so bits(data[lane], 23, 16) was always 0.  The operand has since
 * been widened to ConstVecOperandU32 so bits [23:16] are extracted
 * correctly; this test verifies that ISA-correct behaviour (DATA[23:16]).
 *
 * Build:
 *   make
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./global_store_byte_d16_hi
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

static const int     N          = 256;
static const int     BLOCK_SIZE = 64;
static const uint8_t SENTINEL   = 0xFF;   /* pre-fill of the output buffer */

/*
 * kernel_store_byte_d16_hi
 *
 * Each thread stores in[i] bits [23:16] to out[i] using
 * global_store_byte_d16_hi.  The address goes in a 64-bit VGPR pair and the
 * data value stays in a full 32-bit VGPR.  s_waitcnt vmcnt(0) ensures the
 * store has drained before the kernel returns.
 */
__global__ void
kernel_store_byte_d16_hi(uint8_t *out, const uint32_t *in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(out + i);
    uint32_t data = in[i];

    __asm__ volatile(
        "global_store_byte_d16_hi %0, %1, off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "v"(addr), "v"(data)
        : "memory");
}

int
main(void)
{
    uint32_t *h_in  = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint8_t  *h_out = (uint8_t  *)malloc(N * sizeof(uint8_t));

    for (int i = 0; i < N; i++) {
        uint32_t target = (uint32_t)(i & 0xFF);   /* byte to be stored */
        h_in[i]  = 0xAA000000u | (target << 16) | 0x0000CDEFu;
        h_out[i] = SENTINEL;
    }

    uint32_t *d_in;
    uint8_t  *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint8_t)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_out, h_out, N * sizeof(uint8_t),  hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_store_byte_d16_hi,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_out, d_in, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint8_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint8_t expected = (uint8_t)((h_in[i] >> 16) & 0xFFu);   /* DATA[23:16] */
        if (h_out[i] != expected) {
            fprintf(stderr,
                    "FAIL out[%d] (data 0x%08x): got 0x%02x expected 0x%02x\n",
                    i, h_in[i], h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%02x out[%d]=0x%02x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in));
    GPU_CHECK(hipFree(d_out));
    free(h_in);
    free(h_out);

    return pass ? 0 : 1;
}
