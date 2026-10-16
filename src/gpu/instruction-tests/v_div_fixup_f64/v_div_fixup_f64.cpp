/*
 * HIP test for V_DIV_FIXUP_F64 (VOP3).
 * gem5 (vop3.cc): for finite non-zero, non-inf src1 (denom) and src2 (numer)
 * with a moderate exponent difference, vdst = copysign(|src0|, sign) where
 * sign = signbit(src1) ^ signbit(src2).  We use finite moderate operands so we
 * land in that branch; the result is exact (a sign fix-up of src0).  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const double*s0,const double*s1,const double*s2,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double a=s0[i],b=s1[i],c=s2[i],d;
    __asm__ volatile("v_div_fixup_f64 %0, %1, %2, %3\n\t":"=v"(d):"v"(a),"v"(b),"v"(c));
    out[i]=d;
}
int main(void){
    double*h0=(double*)malloc(N*8),*h1=(double*)malloc(N*8),*h2=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){
        h0[i]=(double)((i%17)+1)*1.25;          /* quotient magnitude, finite */
        h1[i]=(i&1)?-((double)((i%7)+2)):((double)((i%7)+2));   /* denom sign varies */
        h2[i]=(i&2)?-((double)((i%5)+3)):((double)((i%5)+3));   /* numer sign varies */
    }
    double*d0,*d1,*d2,*doo;
    GPU_CHECK(hipMalloc(&d0,N*8));GPU_CHECK(hipMalloc(&d1,N*8));GPU_CHECK(hipMalloc(&d2,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(d0,h0,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d1,h1,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d2,h2,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d0,d1,d2,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int sign=(signbit(h1[i])?1:0)^(signbit(h2[i])?1:0);
        double e=sign?-fabs(h0[i]):fabs(h0[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[1]=%g\n",ho[0],ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
