/*
 * HIP test for V_DIV_FIXUP_F16 (ENC_VOP3, half-precision division fixup),
 * pattern A.
 *
 * gem5 (src/arch/amdgpu/common/insts/vop3.cc): src0 = quotient estimate,
 * src1 = denominator, src2 = numerator (each read from the low 16 bits with
 * opsel=0).  For finite, non-zero, in-range operands (the case exercised here)
 * the result is:
 *
 *   sign_out = signbit(src1) ^ signbit(src2)
 *   D = sign_out ? -fabs(src0) : fabs(src0)
 *
 * All inputs are finite, non-zero, exactly-representable half values with
 * modest exponents so the special-case branches (NaN, 0, inf) are never taken;
 * only the sign-fixup path runs.  Signs of src1/src2 are varied.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const _Float16*a,const _Float16*b,const _Float16*c,_Float16*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_div_fixup_f16 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}

int main(void){
    _Float16*h_a=(_Float16*)malloc(N*2),*h_b=(_Float16*)malloc(N*2),*h_c=(_Float16*)malloc(N*2),*h_out=(_Float16*)malloc(N*2);
    for(int i=0;i<N;i++){
        h_a[i]=(_Float16)(0.5f*(float)((i%13)+1));      /* quotient, positive, representable */
        h_b[i]=(_Float16)(((i&1)?-1.0f:1.0f)*(float)((i%7)+1));
        h_c[i]=(_Float16)(((i&2)?-1.0f:1.0f)*(float)((i%5)+1));
    }
    _Float16*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int sign_out=signbit((float)h_b[i])^signbit((float)h_c[i]);
        float mag=fabsf((float)h_a[i]);
        _Float16 exp=(_Float16)(sign_out?-mag:mag);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,(float)h_out[i],(float)exp);pass=0;break;}}
    if(pass)printf("out[1]=%g out[%d]=%g\n",(float)h_out[1],N-1,(float)h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
