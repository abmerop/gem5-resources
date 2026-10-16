/*
 * HIP test for V_CVT_SCALEF32_PK_FP4_F16 (ENC_VOP3, packed f16 -> packed FP4
 * (e2m1) with an f32 scale).
 *
 * Per gem5 (Inst_VOP3__V_CVT_SCALE_PK<fp4_e2m1, mxfloat16> in
 * src/arch/amdgpu/common/insts/vop3_cvt.hh):
 *
 *   in.first, in.second = the two packed f16 lanes of SRC0
 *   each lane: value.scaleDiv(SCALE)      ; SCALE = SRC1 read as f32
 *   convert each scaled f16 -> fp4_e2m1   ; two 4-bit codes
 *   pack the two 4-bit codes into the low byte of VDST (OPSEL=0),
 *   preserving the rest of VDST.
 *
 * scaleDiv uses only the *exponent* of the scale float: scale_val =
 * scale.exp - 127, then value.exp -= scale_val.  With SCALE = 1.0f
 * (exp == 127) scale_val == 0, so scaleDiv is the identity and the f16 value
 * passes through unchanged before the fp4 conversion.
 *
 * fp4_e2m1 encoding (union in src/arch/amdgpu/common/dtype/fp4_e2m1.hh:
 * sign[1] exp[2] mant[1], bias 1, max = 6.0, no inf/nan):
 *
 *   +0.0->0  +0.5->1  +1.0->2  +1.5->3  +2.0->4  +3.0->5  +4.0->6  +6.0->7
 *   negatives set the sign bit (code | 8).
 *
 * To make the result independent of the lo/hi nibble packing order, BOTH f16
 * halves of SRC0 are set to the same value v; the produced low byte is then
 * (code(v) << 4) | code(v) regardless of ordering.  All test values are exact
 * in both f16 and fp4, so the conversion is bit-exact (no rounding ambiguity).
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cvt_scalef32_pk_fp4_f16 v_cvt_scalef32_pk_fp4_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_scalef32_pk_fp4_f16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
 * kernel: dst's low byte = packed pair of fp4 codes from the two (equal) f16
 * halves of src0, scaled by 1.0 (identity).  dst is preloaded to 0 and used
 * read-write because the instruction preserves the unwritten bits.
 */
__global__ void
kernel_v_cvt_scalef32_pk_fp4_f16(const uint32_t *src0, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t s0 = src0[i];
    float scale = 1.0f;
    uint32_t dst = 0;

    __asm__ volatile(
        "v_cvt_scalef32_pk_fp4_f16 %0, %1, %2\n\t"
        : "+v"(dst)
        : "v"(s0), "v"(scale));

    out[i] = dst;
}

struct VC { float v; uint8_t code; };

int
main(void)
{
    static const VC tbl[] = {
        {0.0f, 0}, {0.5f, 1}, {1.0f, 2}, {1.5f, 3}, {2.0f, 4}, {3.0f, 5},
        {4.0f, 6}, {6.0f, 7}, {-0.5f, 9}, {-1.0f, 10}, {-1.5f, 11},
        {-2.0f, 12}, {-3.0f, 13}, {-4.0f, 14}, {-6.0f, 15},
    };
    const int T = (int)(sizeof(tbl) / sizeof(tbl[0]));

    uint32_t *h_src = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint8_t  *exp_code = (uint8_t *)malloc(N * sizeof(uint8_t));

    for (int i = 0; i < N; i++) {
        VC e = tbl[i % T];
        __half h = __float2half(e.v);
        uint16_t hb;
        __builtin_memcpy(&hb, &h, sizeof(hb));
        h_src[i] = ((uint32_t)hb << 16) | (uint32_t)hb;   /* both halves = v */
        exp_code[i] = e.code;
    }

    uint32_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cvt_scalef32_pk_fp4_f16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t expected = ((uint32_t)exp_code[i] << 4) | (uint32_t)exp_code[i];
        if ((h_out[i] & 0xff) != expected || (h_out[i] >> 8) != 0) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected low byte 0x%02x "
                    "(v=%g)\n", i, h_out[i], expected, tbl[i % T].v);
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
