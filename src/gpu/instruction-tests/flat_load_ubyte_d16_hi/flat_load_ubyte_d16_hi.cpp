/*
 * HIP test for FLAT_LOAD_UBYTE_D16_HI (FLAT encoding, D16-high byte load).
 *
 * FLAT_LOAD_UBYTE_D16_HI loads a single unsigned byte from memory and writes
 * it, zero-extended to 16 bits, into the HIGH half of the destination VGPR.
 * The LOW half of the destination VGPR is PRESERVED (left unchanged).
 *
 * gem5 semantics (src/arch/amdgpu/common/insts/flat.cc,
 * Inst_FLAT__FLAT_LOAD_UBYTE_D16_HI::completeAcc):
 *   vdst.read();                              // preserve existing bits
 *   tmp = (u16) byte_loaded_from_memory;      // 0..255, zero-extended
 *   replaceBits(vdst[lane], 31, 16, tmp);     // dst[31:16] = tmp
 *   // dst[15:0] remain whatever they were before the load
 *
 * So:
 *   dst[31:16] == byte value (0..255)
 *   dst[15:0]  == the value the VGPR was pre-initialized to (preserved)
 *
 * The compiler will not naturally emit this narrow high-half sub-word load
 * for a plain byte read, so inline assembly is used (SKILL pattern C).
 * The destination is a read-write ("+v") 32-bit operand: we pre-load it with
 * a known sentinel so the preserved low half can be verified, and the address
 * is passed as a 64-bit value ("v") for the FLAT vaddr pair.  s_waitcnt
 * vmcnt(0) stalls until the vector memory op completes before we read back.
 *
 * Build:
 *   make -C hip_tests/flat_load_ubyte_d16_hi AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./flat_load_ubyte_d16_hi
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/* Known sentinel written to the low half of the destination VGPR before the
 * load.  After FLAT_LOAD_UBYTE_D16_HI this low half must be preserved. */
static const uint32_t PRESERVE_LOW = 0xBEEF;

/*
 * kernel_flat_load_ubyte_d16_hi
 *
 * Each thread reads one byte src[i] via FLAT_LOAD_UBYTE_D16_HI into the high
 * half of a VGPR whose low half has been pre-set to PRESERVE_LOW, then stores
 * the full 32-bit result so the host can check both halves.
 */
__global__ void
kernel_flat_load_ubyte_d16_hi(const uint8_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(src + i);

    /* Pre-initialize destination with sentinel in the low half (and a
     * non-zero high half, which the load must overwrite). */
    uint32_t vd = 0xAAAA0000u | PRESERVE_LOW;

    __asm__ volatile(
        "flat_load_ubyte_d16_hi %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(vd)
        : "v"(addr)
        : "memory");

    out[i] = vd;
}

int
main(void)
{
    uint8_t  *h_src = (uint8_t *)malloc(N * sizeof(uint8_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    /* Cover the full unsigned byte range 0..255 across the N threads. */
    for (int i = 0; i < N; i++)
        h_src[i] = (uint8_t)(i & 0xFF);

    uint8_t  *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint8_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_ubyte_d16_hi,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t hi = (h_out[i] >> 16) & 0xFFFF;   /* dst[31:16] */
        uint32_t lo = h_out[i] & 0xFFFF;           /* dst[15:0]  */
        uint32_t exp_hi = (uint32_t)h_src[i];      /* zero-extended byte */

        if (hi != exp_hi || lo != PRESERVE_LOW) {
            fprintf(stderr,
                    "FAIL out[%d]=0x%08x: hi got 0x%04x expected 0x%04x, "
                    "lo got 0x%04x expected 0x%04x\n",
                    i, h_out[i], hi, exp_hi, lo, PRESERVE_LOW);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src));
    GPU_CHECK(hipFree(d_out));
    free(h_src);
    free(h_out);

    return pass ? 0 : 1;
}
