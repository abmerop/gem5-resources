/*
 * HIP test for V_CVT_SCALEF32_PK32_BF6_BF16 (VOP3, MXFP scaled pack-32
 * conversion: 32 bf16 -> 32 bf6, scaled by an f32).
 *
 * Per the gem5 model (Inst_VOP3__V_CVT_SCALE_PK32 in
 * src/arch/amdgpu/vega/insts/vop3_cvt.hh), the instruction reads 16 input
 * dwords (32 packed bf16 "components"), an f32 scale in src1, and writes 6
 * output dwords (32 packed bf6 values).  Because dst is narrower than src,
 * each value is divided by the scale before conversion.
 *
 * Verification strategy
 * ---------------------
 * The bf6 (6-bit float) encoding and 6-bit packing are intricate, so this test
 * uses an input where the output is unambiguous: all 32 input bf16 values are
 * +0.0 and the scale is 1.0, so every converted bf6 value is 0 and all 6
 * output dwords must be exactly 0.  The 6 destination VGPRs are pre-seeded with
 * a nonzero sentinel and read-modified ("+v"), so a pass proves the instruction
 * actually executed and wrote the computed (all-zero) result rather than
 * leaving the sentinel untouched.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_scalef32_pk32_bf6_bf16 \
 *        v_cvt_scalef32_pk32_bf6_bf16.cpp
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

typedef uint32_t u32x16 __attribute__((ext_vector_type(16)));
typedef uint32_t u32x6  __attribute__((ext_vector_type(6)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;
static const int OUT_REGS    = 6;

__global__ void
kernel_cvt(const uint32_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    u32x16 vsrc;
    for (int k = 0; k < 16; k++)
        vsrc[k] = in[i * 16 + k];          /* all +0.0 bf16 */

    float scale = 1.0f;

    u32x6 vd;
    for (int k = 0; k < OUT_REGS; k++)
        vd[k] = 0xDEADBEEFu;               /* sentinel; must be overwritten */

    __asm__ volatile(
        "v_cvt_scalef32_pk32_bf6_bf16 %0, %1, %2\n\t"
        : "+v"(vd)
        : "v"(vsrc), "v"(scale));

    for (int k = 0; k < OUT_REGS; k++)
        out[i * OUT_REGS + k] = vd[k];
}

int
main(void)
{
    uint32_t *h_in  = (uint32_t *)calloc((size_t)N * 16, sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc((size_t)N * OUT_REGS * sizeof(uint32_t));

    /* h_in already all zero (calloc) -> 32 bf16 +0.0 per thread. */

    uint32_t *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  (size_t)N * 16 * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, (size_t)N * OUT_REGS * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_in, h_in, (size_t)N * 16 * sizeof(uint32_t),
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
