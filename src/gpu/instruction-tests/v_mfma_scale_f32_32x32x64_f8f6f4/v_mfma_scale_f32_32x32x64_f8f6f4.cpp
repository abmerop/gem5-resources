/*
 * HIP test for V_MFMA_SCALE_F32_32X32X64_F8F6F4 (ENC_VOP3PX2, scaled MFMA
 * matrix multiply with f8/f6/f4 inputs).  Pattern D.
 *
 * Scaled matrix-fused-multiply-add: D[32x32 f32] = A[32x64] * B[64x32] +
 * C[32x32], with per-operand exponent scales.  The instruction assembles for
 * gfx950 (validated with llvm-mc) as a 16-byte VOP3PX2 dual encoding.  We
 * initialise A/B and a 16-element f32 accumulator, issue the MFMA, and write
 * the accumulator out; the host checks that it changed from its sentinel.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[0:15], \
 *       v32, v33
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mfma_scale_f32_32x32x64_f8f6f4 \
 *        v_mfma_scale_f32_32x32x64_f8f6f4.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef int   i32x8  __attribute__((ext_vector_type(8)));

static const int BLOCK_SIZE = 64; /* one wavefront */

__global__ void
kernel(const i32x8 *ain, const i32x8 *bin, int sa, int sb, f32x16 *out)
{
    f32x16 acc;
    for (int i = 0; i < 16; i++) acc[i] = -1.0f;
    i32x8 a = ain[0];
    i32x8 b = bin[0];
    __asm__ volatile(
        "v_mfma_scale_f32_32x32x64_f8f6f4 %0, %1, %2, %0, %3, %4\n\t"
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
    f32x16 *d_out;
    GPU_CHECK(hipMalloc(&d_a, sizeof(i32x8)));
    GPU_CHECK(hipMalloc(&d_b, sizeof(i32x8)));
    GPU_CHECK(hipMalloc(&d_out, BLOCK_SIZE * sizeof(f32x16)));
    GPU_CHECK(hipMemcpy(d_a, &h_a, sizeof(i32x8), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, &h_b, sizeof(i32x8), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_out, 0, BLOCK_SIZE * sizeof(f32x16)));

    kernel<<<1, BLOCK_SIZE>>>(d_a, d_b, 0, 0, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    f32x16 h_out;
    GPU_CHECK(hipMemcpy(&h_out, d_out, sizeof(f32x16), hipMemcpyDeviceToHost));

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
