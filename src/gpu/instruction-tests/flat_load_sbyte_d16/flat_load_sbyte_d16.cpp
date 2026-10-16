/*
 * HIP test for FLAT_LOAD_SBYTE_D16 (FLAT encoding, sign-extending byte load
 * into the low 16 bits of a VGPR with the high 16 bits preserved).
 *
 * FLAT_LOAD_SBYTE_D16 computes (per gem5,
 *   src/arch/amdgpu/common/insts/flat.cc,
 *   Inst_FLAT__FLAT_LOAD_SBYTE_D16::completeAcc):
 *
 *   tmp        = sign_extend_16( (int8_t) MEM[addr] )   // signed byte -> i16
 *   dst[15:0]  = tmp
 *   dst[31:16] = PRESERVED (unchanged)
 *
 * The D16 variant only writes the low 16 bits of the destination VGPR and
 * leaves the upper 16 bits untouched.  To exercise that, we pre-initialize
 * the destination VGPR to a known 32-bit "sentinel" value, perform the
 * flat_load_sbyte_d16 into it, and then verify:
 *   - low 16 bits  == (uint16_t)(int16_t)(int8_t) byte
 *   - high 16 bits == the sentinel's high 16 bits (preserved)
 *
 * The compiler will not naturally emit this D16 encoding for a plain byte
 * load (it lowers to flat_load_i8 into a full 32-bit reg), so inline asm
 * (pattern C) is used.  The byte address is passed as a uint64_t for the
 * "v" (VGPR) input constraint, and s_waitcnt vmcnt(0) drains the load
 * before the result VGPR is read.
 *
 * Build:
 *   make -C hip_tests/flat_load_sbyte_d16 AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./flat_load_sbyte_d16
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

/* Sentinel written to the destination VGPR before the D16 load, so the
 * preserved high 16 bits are verifiable.  High half = 0xABCD. */
static const uint32_t SENTINEL = 0xABCD1234u;

/*
 * kernel_flat_load_sbyte_d16
 *
 * Each thread loads one signed byte from d_bytes[i] using
 * flat_load_sbyte_d16 into a VGPR that has been pre-seeded with SENTINEL.
 * The full 32-bit result VGPR is stored to d_out[i] so the host can check
 * both the sign-extended low half and the preserved high half.
 */
__global__ void
kernel_flat_load_sbyte_d16(const int8_t *d_bytes, uint32_t *d_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Byte address as a 64-bit value for the "v" VGPR constraint. */
    uint64_t addr = reinterpret_cast<uint64_t>(&d_bytes[i]);

    /* Seed destination VGPR with the sentinel; the D16 load must preserve
     * its upper 16 bits. */
    uint32_t vd = SENTINEL;

    __asm__ volatile(
        "flat_load_sbyte_d16 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(vd)
        : "v"(addr)
        : "memory");

    d_out[i] = vd;
}

int
main(void)
{
    /* ----- host buffers ----- */
    int8_t   *h_bytes = (int8_t *)malloc(N * sizeof(int8_t));
    uint32_t *h_out   = (uint32_t *)malloc(N * sizeof(uint32_t));

    /* Test data covering positive, negative, zero, and boundary bytes. */
    for (int i = 0; i < N; i++) {
        /* i in [0,255] maps directly to all 256 byte patterns. */
        h_bytes[i] = (int8_t)(i & 0xFF);
    }
    /* A few explicit interesting cases. */
    h_bytes[0]   = (int8_t)0x00;   /*  0        */
    h_bytes[1]   = (int8_t)0x7F;   /* +127      */
    h_bytes[2]   = (int8_t)0x80;   /* -128      */
    h_bytes[3]   = (int8_t)0xFF;   /*  -1       */
    h_bytes[4]   = (int8_t)0x01;   /* +1        */

    /* ----- device buffers ----- */
    int8_t   *d_bytes;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_bytes, N * sizeof(int8_t)));
    GPU_CHECK(hipMalloc(&d_out,   N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_bytes, h_bytes, N * sizeof(int8_t),
                        hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_load_sbyte_d16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_bytes, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        /* Reference: low 16 = sign_extend_16(byte), high 16 = preserved. */
        uint16_t lo_exp = (uint16_t)(int16_t)(int8_t)h_bytes[i];
        uint16_t hi_exp = (uint16_t)(SENTINEL >> 16);
        uint32_t expected = ((uint32_t)hi_exp << 16) | (uint32_t)lo_exp;

        if (h_out[i] != expected) {
            fprintf(stderr,
                    "FAIL out[%d]: byte=0x%02x got 0x%08x expected 0x%08x "
                    "(lo=0x%04x hi=0x%04x)\n",
                    i, (uint8_t)h_bytes[i], h_out[i], expected,
                    (uint16_t)(h_out[i] & 0xFFFF),
                    (uint16_t)(h_out[i] >> 16));
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1,
               h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_bytes));
    GPU_CHECK(hipFree(d_out));
    free(h_bytes);
    free(h_out);

    return pass ? 0 : 1;
}
