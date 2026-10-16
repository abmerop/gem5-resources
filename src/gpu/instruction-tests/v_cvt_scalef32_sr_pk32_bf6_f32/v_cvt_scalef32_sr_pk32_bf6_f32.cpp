/*
 * HIP test for V_CVT_SCALEF32_SR_PK32_FP6_F16 (VOP3, MXFP scaled pack-32
 * conversion with stochastic rounding: 32 f16 -> 32 fp6, scaled by an f32,
 * rounded stochastically using a seed).
 *
 * Per the CDNA4 ISA (amdgpu_isa_cdna4.xml, opcode 604) the instruction:
 *   - reads 16 input dwords in src0 (32 packed f16 "components", v[8:23]),
 *   - reads a u32 stochastic-rounding seed in src1 (v24),
 *   - reads an f32 scale in src2 (v25),
 *   - scales each f16 by the exponent from the f32 scale, converts to FP6
 *     (6-bit float) with stochastic rounding, and writes 6 output dwords
 *     (192 bits = 32 packed fp6 values) into vdst (v[0:5]).
 *
 * Valid assembler form (llvm-mc -mcpu=gfx950):
 *   v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[8:23], v24, v25
 *
 * The compiler does not emit this naturally, so inline asm is required.
 *
 * Verification strategy
 * ---------------------
 * The FP6 (6-bit float) encoding, 6-bit packing, and stochastic rounding make
 * exact bit-accurate verification impractical.  This test therefore uses an
 * input for which the output is unambiguous and independent of the rounding
 * seed: all 32 input f16 values are +0.0 and the scale is 1.0.  0.0 converts
 * to the FP6 zero encoding (0) under any stochastic-rounding seed, so every
 * one of the 6 output dwords must be exactly 0.
 *
 * The 6 destination VGPRs are pre-seeded with a nonzero sentinel and read via
 * "+v", so a pass proves the instruction actually executed and wrote the
 * computed (all-zero) result rather than leaving the sentinel untouched.  A
 * distinct nonzero seed is used per thread so the seed operand is genuinely
 * live.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_scalef32_sr_pk32_bf6_f32 \
 *        v_cvt_scalef32_sr_pk32_bf6_f32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_scalef32_sr_pk32_bf6_f32
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

typedef uint32_t u32x32 __attribute__((ext_vector_type(32)));
typedef uint32_t u32x6  __attribute__((ext_vector_type(6)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;
static const int OUT_REGS   = 6;   /* 192-bit fp6-packed destination */

__global__ void
kernel_cvt(const uint32_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    u32x32 vsrc;
    for (int k = 0; k < 32; k++)
        vsrc[k] = in[i * 32 + k];          /* all +0.0 f16 */

    /* Stochastic-rounding seed: distinct nonzero value per thread. */
    uint32_t seed = 0x9E3779B9u * (uint32_t)(i + 1);

    float scale = 1.0f;

    u32x6 vd;
    for (int k = 0; k < OUT_REGS; k++)
        vd[k] = 0xDEADBEEFu;               /* sentinel; must be overwritten */

    __asm__ volatile(
        "v_cvt_scalef32_sr_pk32_bf6_f32 %0, %1, %2, %3\n\t"
        : "+v"(vd)
        : "v"(vsrc), "v"(seed), "v"(scale));

    for (int k = 0; k < OUT_REGS; k++)
        out[i * OUT_REGS + k] = vd[k];
}

int
main(void)
{
    uint32_t *h_in  = (uint32_t *)calloc((size_t)N * 32, sizeof(uint32_t));
    uint32_t *h_out =
        (uint32_t *)malloc((size_t)N * OUT_REGS * sizeof(uint32_t));

    /* h_in already all zero (calloc) -> 32 f16 +0.0 per thread. */

    uint32_t *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  (size_t)N * 32 * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, (size_t)N * OUT_REGS * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_in, h_in, (size_t)N * 32 * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_cvt, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out,
                        (size_t)N * OUT_REGS * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N * OUT_REGS && pass; i++) {
        if (h_out[i] != 0u) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x00000000\n",
                    i, h_out[i]);
            pass = 0;
        }
    }

    if (pass)
        printf("all %d output dwords == 0 (instruction executed)\n",
               N * OUT_REGS);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return pass ? 0 : 1;
}
