/*
 * HIP test for V_DIV_FIXUP_F32 (ENC_VOP3, division fixup), pattern A.
 *
 * gem5 (vop3.cc): src0 = quotient estimate, src1 = denominator,
 * src2 = numerator.  For finite, non-zero, in-range operands (the case
 * exercised here) the result is:
 *
 *   sign_out = signbit(src1) ^ signbit(src2)
 *   D = sign_out ? -fabs(src0) : fabs(src0)
 *
 * All inputs are finite, non-zero, with modest exponents so the special-case
 * branches (NaN, 0, inf, huge exponent difference) are never taken; only the
 * sign-fixup branch runs.  Signs of src1/src2 are varied to exercise the
 * sign_out logic.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_div_fixup_f32 v_div_fixup_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_div_fixup_f32 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}
int main(void){
    float*h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++){
        h_a[i]=(float)(i+1)*0.5f;                 /* quotient, positive */
        h_b[i]=((i&1)?-1.0f:1.0f)*(float)(i+2);   /* denom, alternating sign */
        h_c[i]=((i&2)?-1.0f:1.0f)*(float)(i+3);   /* numer, alternating sign */
    }
    float*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int sign_out=std::signbit(h_b[i])^std::signbit(h_c[i]);
        float exp=sign_out?-fabsf(h_a[i]):fabsf(h_a[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[1]=%g out[%d]=%g\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
