/*
 * HIP test for V_MFMA_SCALE_F32_16X16X128_F8F6F4 (ENC_VOP3PX2, scaled MFMA
 * matrix multiply with f8/f6/f4 inputs).  Pattern D.
 *
 * This is a scaled matrix-fused-multiply-add: D[16x16 f32] = A[16x128] *
 * B[128x16] + C[16x16], with per-operand exponent scales (scaleA/scaleB).
 * The instruction assembles for gfx950 (validated with llvm-mc) as a 16-byte
 * VOP3PX2 dual encoding.  We initialise the A/B operand VGPRs and the f32
 * accumulator, issue the MFMA, and write the accumulator out; the host checks
 * that the accumulator changed from its sentinel (i.e. the instruction
 * executed and produced a result).
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[0:7], v[8:15], v[0:3], v20, v21
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mfma_scale_f32_16x16x128_f8f6f4 \
 *        v_mfma_scale_f32_16x16x128_f8f6f4.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));

static const int BLOCK_SIZE = 64; /* one wavefront */

__global__ void
kernel(const i32x8 *ain, const i32x8 *bin, int sa, int sb, f32x4 *out)
{
    f32x4 acc = {-1.0f, -1.0f, -1.0f, -1.0f};
    i32x8 a = ain[0];
    i32x8 b = bin[0];
    __asm__ volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %4\n\t"
        "s_nop 15\n\t"
        : "+v"(acc)
        : "v"(a), "v"(b), "v"(sa), "v"(sb));
    out[threadIdx.x] = acc;
}

int
main(void)
{
    i32x8 h_a, h_b;
    for (int i = 0; i < 8; i++) { h_a[i] = 0x3c3c3c3c; h_b[i] = 0x3c3c3c3c; }

    i32x8 *d_a, *d_b;
    f32x4 *d_out;
    GPU_CHECK(hipMalloc(&d_a, sizeof(i32x8)));
    GPU_CHECK(hipMalloc(&d_b, sizeof(i32x8)));
    GPU_CHECK(hipMalloc(&d_out, BLOCK_SIZE * sizeof(f32x4)));
    GPU_CHECK(hipMemcpy(d_a, &h_a, sizeof(i32x8), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, &h_b, sizeof(i32x8), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_out, 0, BLOCK_SIZE * sizeof(f32x4)));

    kernel<<<1, BLOCK_SIZE>>>(d_a, d_b, 0, 0, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    f32x4 h_out;
    GPU_CHECK(hipMemcpy(&h_out, d_out, sizeof(f32x4), hipMemcpyDeviceToHost));

    /* acc initialised to -1; a completed MFMA overwrites it. */
    int pass = (h_out[0] != -1.0f);
    if (!pass)
        fprintf(stderr, "FAIL acc unchanged: %f\n", (double)h_out[0]);
    else
        printf("acc0=%f\n", (double)h_out[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
