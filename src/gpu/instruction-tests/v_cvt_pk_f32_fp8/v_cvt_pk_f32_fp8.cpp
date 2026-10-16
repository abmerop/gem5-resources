/*
 * HIP test for V_CVT_PK_F32_FP8 (VOP1, packed fp8 e4m3 pair -> 2x f32).
 * gem5 (vop1.cc): in1 = fp8(byte0), in2 = fp8(byte1); vdst.lo = float(in1),
 * vdst.hi = float(in2) (64-bit VGPR pair result). DPP panics; SDWA aborts
 * gem5, so base (e32) only.
 * On gfx950 this instruction decodes fp8 as E4M3FNUZ: bias 8, no
 * Inf, the only NaN is 0x80 (sign=1,exp=0,mant=0; no negative zero). Decode
 * to f32 is exact; all byte pairs are checked (NaN via isnan).
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,uint64_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i]; uint64_t d;
    __asm__ volatile("v_cvt_pk_f32_fp8 %0, %1\n\t":"=v"(d):"v"(s0));
    out[i]=d;
}

// e4m3fnuz decode; bias 8, no Inf, only NaN is 0x80.
static float e4m3_ref(uint8_t b,int*is_nan){
    int s=(b>>7)&1,e=(b>>3)&0xF,m=b&0x7; *is_nan=0; double v;
    if(b==0x80){ *is_nan=1; return 0.0f; }  // sole NaN (also no -0)
    if(e==0) v=ldexp((double)m,-10);        // +0 or subnormal, bias 8
    else v=ldexp(1.0+m/8.0,e-8);            // normal, bias 8
    return (float)(s?-v:v);
}

int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4); uint64_t*ho=(uint64_t*)malloc(N*8);
    uint8_t*b0=(uint8_t*)malloc(N),*b1=(uint8_t*)malloc(N);
    for(int i=0;i<N;i++){ b0[i]=(uint8_t)i; b1[i]=(uint8_t)((i*5+37)&0xff);
        ha[i]=((uint32_t)b1[i]<<8)|b0[i]; }
    uint32_t*da; uint64_t*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        float lo,hi; uint32_t l=(uint32_t)ho[i],h=(uint32_t)(ho[i]>>32);
        memcpy(&lo,&l,4); memcpy(&hi,&h,4);
        int n0,n1; float el=e4m3_ref(b0[i],&n0), eh=e4m3_ref(b1[i],&n1);
        if(n0){ if(!isnan(lo)){fprintf(stderr,"FAIL[%d] lo fp8=0x%02x got %g exp NaN\n",i,b0[i],lo);pass=0;} }
        else if(lo!=el){fprintf(stderr,"FAIL[%d] lo fp8=0x%02x got %g exp %g\n",i,b0[i],lo,el);pass=0;}
        if(pass){ if(n1){ if(!isnan(hi)){fprintf(stderr,"FAIL[%d] hi fp8=0x%02x got %g exp NaN\n",i,b1[i],hi);pass=0;} }
                  else if(hi!=eh){fprintf(stderr,"FAIL[%d] hi fp8=0x%02x got %g exp %g\n",i,b1[i],hi,eh);pass=0;} }
    }
    if(pass)printf("packed fp8->2xf32 ok\n");
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
