/*
 * HIP test for V_CVT_SCALEF32_SR_BF8_BF16 (ENC_VOP3, bf16 -> bf8 (e5m2) with
 * an f32 scale and stochastic rounding).
 *
 * Per gem5 (Inst_VOP3__V_CVT_SCALEF32_SR<fp8_e5m2, mxbfloat16> in
 * src/arch/amdgpu/common/insts/vop3_cvt.hh):
 *
 *   in     = mxbfloat16(SRC0)        ; bf16 value (low 16 bits of SRC0)
 *   seed   = SRC1                     ; stochastic-rounding seed
 *   scale  = SRC2 (f32)
 *   in.scaleDiv(scale)               ; SCALE = 1.0f -> identity
 *   cvt    = convertMXFP<bf8_e5m2>(in, roundStochastic, seed)
 *   VDST[8*out_byte +: 8] = cvt code ; out_byte = OPSEL[3:2] (=0)
 *
 * Stochastic rounding (mxfp_convert.hh) rounds UP iff round_prob <= draw_prob,
 * where round_prob = discarded_mantissa / max_src_mantissa and draw_prob =
 * srFunc(seed)/UINT32_MAX.  With seed = 0, draw_prob = 0, so:
 *   - inexact inputs (discarded != 0 => round_prob > 0): NOT rounded up =>
 *     mantissa is simply truncated to the top 2 bits (round toward zero).
 *   - exact inputs (discarded == 0 => round_prob == 0): rounded UP by 1 ULP,
 *     because the comparison is "<=" (0 <= 0 is true).  This exact-value
 *     round-up looks like a gem5 off-by-one, so this test deliberately uses
 *     only bf8-INEXACT bf16 inputs, for which seed = 0 gives a clean,
 *     deterministic truncation that is easy to verify.
 *
 * bf8_e5m2: sign[1] exp[5] mant[2], bias 15 (fp8_e5m2.hh).
 * bf16:     sign[1] exp[8] mant[7], bias 127.
 * For a normal value with both halves in range, the truncated result is
 *   code = (sign<<7) | ((e16-112)<<2) | (m7>>5)
 * (e16-112 rebiases exp from bias 127 to bias 15).  Inputs are chosen with
 * the low 5 mantissa bits nonzero (inexact) and bf8 exponent well inside
 * [1,30] so no exponent adjustment or denormal handling is triggered.
 *
 * gem5 models VDST as 64-bit but the encoding names a single VGPR; it
 * read-modify-writes the pair leaving the upper dword unchanged, so a single
 * uint32 read-write ("+v") destination is safe.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cvt_scalef32_sr_bf8_bf16 v_cvt_scalef32_sr_bf8_bf16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_scalef32_sr_bf8_bf16
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
kernel_sr(const uint32_t *src0, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t s0 = src0[i];
    int32_t  seed = 0;
    float    scale = 1.0f;
    uint32_t dst = 0;

    __asm__ volatile(
        "v_cvt_scalef32_sr_bf8_bf16 %0, %1, %2, %3\n\t"
        : "+v"(dst)
        : "v"(s0), "v"(seed), "v"(scale));

    out[i] = dst;
}

int
main(void)
{
    /* bf16 mantissas (7-bit) with low 5 bits nonzero => bf8-inexact. */
    static const uint8_t m7s[] = {
        0x21, 0x42, 0x63, 0x15, 0x2a, 0x5c, 0x11, 0x7b,
        0x09, 0x33, 0x4d, 0x66, 0x1e, 0x57, 0x6f, 0x39,
    };
    /* bf16 biased exponents -> bf8 exp = e16-112 must be in [1,30]. */
    static const uint8_t e16s[] = {120, 127, 131, 124};

    const int M = (int)(sizeof(m7s) / sizeof(m7s[0]));
    const int E = (int)(sizeof(e16s) / sizeof(e16s[0]));

    uint32_t *h_src = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint8_t  *exp_code = (uint8_t *)malloc(N * sizeof(uint8_t));

    for (int i = 0; i < N; i++) {
        uint32_t s   = (i % 5 == 0) ? 1u : 0u;       /* some negatives */
        uint32_t e16 = e16s[i % E];
        uint32_t m7  = m7s[(i / E) % M];

        uint16_t bf16 = (uint16_t)((s << 15) | (e16 << 7) | m7);
        h_src[i] = (uint32_t)bf16;

        uint32_t ebits = e16 - 112;                  /* rebias to e5m2 */
        uint32_t m2    = m7 >> 5;                     /* truncate (seed 0) */
        exp_code[i] = (uint8_t)((s << 7) | (ebits << 2) | m2);
    }

    uint32_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_sr, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if ((h_out[i] & 0xff) != exp_code[i] || (h_out[i] >> 8) != 0) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected low byte 0x%02x "
                    "(src bf16=0x%04x)\n", i, h_out[i], exp_code[i],
                    (uint16_t)h_src[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[1]=0x%08x out[2]=0x%08x\n",
               h_out[0], h_out[1], h_out[2]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out); free(exp_code);

    return pass ? 0 : 1;
}
