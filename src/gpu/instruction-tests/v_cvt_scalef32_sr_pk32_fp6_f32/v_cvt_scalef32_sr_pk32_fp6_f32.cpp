/*
 * HIP test for V_CVT_SCALEF32_SR_PK32_FP6_F32 (VOP3, MXFP stochastic-rounding
 * pack-32 conversion: 32 f32 -> 32 fp6, scaled by an f32, with an SR seed).
 *
 * Per the gem5 model (Inst_VOP3__V_CVT_SCALE_SR_PK32 in
 * src/arch/amdgpu/common/insts/vop3_cvt.hh), the instruction reads 32 input
 * dwords (32 f32 "components") in src0, an i32 SR seed in src1, an f32 scale in
 * src2, and writes 6 output dwords (32 packed fp6 values).  Each value is
 * divided by the scale before conversion.
 *
 * Verification strategy: all 32 input f32 values are +0.0 and the scale is 1.0,
 * so every converted fp6 value is 0 (0 rounds to 0 regardless of SR seed) and
 * all 6 output dwords must be exactly 0.  The 6 destination VGPRs are
 * pre-seeded with a nonzero sentinel and read-modified ("+v"), so a pass proves
 * the instruction actually executed and wrote the computed all-zero result.  A
 * distinct nonzero SR seed is used per thread so the seed operand is live.
 * Pattern A, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

typedef uint32_t u32x32 __attribute__((ext_vector_type(32)));
typedef uint32_t u32x6  __attribute__((ext_vector_type(6)));

static const int N=256, BLOCK_SIZE=64, OUT_REGS=6;

__global__ void
kernel_cvt(uint32_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    u32x32 vsrc;
    for (int k=0;k<32;k++) vsrc[k]=0u;          /* 32 f32 +0.0 */
    uint32_t seed = 0x9E3779B9u * (uint32_t)(i+1);
    float scale = 1.0f;
    u32x6 vd;
    for (int k=0;k<OUT_REGS;k++) vd[k]=0xDEADBEEFu;  /* sentinel */
    __asm__ volatile(
        "v_cvt_scalef32_sr_pk32_fp6_f32 %0, %1, %2, %3\n\t"
        : "+v"(vd)
        : "v"(vsrc), "v"(seed), "v"(scale));
    for (int k=0;k<OUT_REGS;k++) out[i*OUT_REGS+k]=vd[k];
}

int main(void)
{
    uint32_t *h_out=(uint32_t*)malloc((size_t)N*OUT_REGS*4);
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out,(size_t)N*OUT_REGS*4));
    kernel_cvt<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,(size_t)N*OUT_REGS*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*OUT_REGS && pass;i++){
        if(h_out[i]!=0u){fprintf(stderr,"FAIL out[%d]: got 0x%08x exp 0\n",i,h_out[i]);pass=0;}
    }
    if(pass)printf("all %d output dwords == 0 (instruction executed)\n",N*OUT_REGS);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
