/*
 * HIP test for V_CVT_SCALEF32_2XPK16_FP6_F32 (VOP3, MXFP scaled conversion:
 * 32 f32 -> 32 fp6, scaled by an f32).
 *
 * Per the gem5 model (Inst_VOP3__V_CVT_SCALEF32_2XPK16_F32 in
 * src/arch/amdgpu/common/insts/vop3_cvt.hh), the instruction reads TWO 16-dword
 * source operands (src0 = 16 f32, src1 = 16 f32 -> 32 "components"), an f32
 * scale in src2, and writes 6 output dwords (32 packed fp6 values).  Because
 * dst is narrower than src, each value is divided by the scale before
 * conversion.
 *
 * Verification strategy
 * ---------------------
 * The fp6 (6-bit float) encoding and 6-bit packing are intricate, so this test
 * uses inputs where the output is unambiguous: all 32 input f32 values are
 * +0.0 and the scale is 1.0, so every converted fp6 value is 0 and all 6 output
 * dwords must be exactly 0.  The 6 destination VGPRs are pre-seeded with a
 * nonzero sentinel and read-modified ("+v"), so a pass proves the instruction
 * actually executed and wrote the computed (all-zero) result rather than
 * leaving the sentinel untouched.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

typedef uint32_t u32x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u32x6  __attribute__((ext_vector_type(6)));

static const int N = 256, BLOCK_SIZE = 64, OUT_REGS = 6;

__global__ void
kernel_cvt(uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u32x16 vsrc0, vsrc1;
    for (int k = 0; k < 16; k++) { vsrc0[k] = 0u; vsrc1[k] = 0u; }  /* +0.0 */
    float scale = 1.0f;

    u32x6 vd;
    for (int k = 0; k < OUT_REGS; k++) vd[k] = 0xDEADBEEFu;  /* sentinel */

    __asm__ volatile(
        "v_cvt_scalef32_2xpk16_fp6_f32 %0, %1, %2, %3\n\t"
        : "+v"(vd)
        : "v"(vsrc0), "v"(vsrc1), "v"(scale));

    for (int k = 0; k < OUT_REGS; k++)
        out[i * OUT_REGS + k] = vd[k];
}

int
main(void)
{
    uint32_t *h_out = (uint32_t *)malloc((size_t)N * OUT_REGS * sizeof(uint32_t));
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, (size_t)N * OUT_REGS * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_cvt, dim3(N/BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0,
                       d_out, N);
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

    GPU_CHECK(hipFree(d_out)); free(h_out);
    return pass ? 0 : 1;
}
