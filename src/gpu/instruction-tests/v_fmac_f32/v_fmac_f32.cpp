/*
 * HIP test for V_FMAC_F32 (VOP2, fused multiply-accumulate, f32).
 * gem5 (vop2.cc): vdst = fma(src0, src1, vdst) -- vdst is the accumulator.
 * gem5 panics on DPP/SDWA for this op, so base (e32) only.
 * Small integer operands -> product+accum exact -> bit-exact compare.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],d=c[i];
    __asm__ volatile("v_fmac_f32_e32 %0, %1, %2\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}

int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*hc=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++){
        ha[i]=(float)((i%13)-6);      // -6..6
        hb[i]=(float)((i%7)-3);       // -3..3
        hc[i]=(float)((i%11)-5);      // -5..5
    }
    float*da,*db,*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
    GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ float e=fmaf(ha[i],hb[i],hc[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] %g*%g+%g got %g exp %g\n",i,ha[i],hb[i],hc[i],ho[i],e);pass=0;break;} }
    if(pass)printf("out[0]=%g out[100]=%g\n",ho[0],ho[100]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
