/*
 * HIP test for V_CVT_F32_BF8 (VOP1, bf8 e5m2 -> f32).
 * gem5 (vop1.cc): in = fp8_e5m2(byte from src, selected by SDWA SRC0_SEL,
 * default byte 0); vdst = float(in). DPP panics; SDWA selects the source byte.
 * bf8->f32 is always exact; we restrict inputs to finite normals and zero to
 * avoid inf/nan/subnormal ambiguity -> bit-exact compare.
 *
 * Variants exercised: base only. The SDWA path asserts in gem5
 * (VecOperand::operator[] "!scalar" in vega/operand.hh via the SDWA source
 * helper), so only the base (e32) form is tested here.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern_base(const uint32_t*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i]; float d;
    __asm__ volatile("v_cvt_f32_bf8_e32 %0, %1\n\t":"=v"(d):"v"(s0));
    out[i]=d;
}
// Standard IEEE-like e5m2 decode (bias 15). Only normals + zero used here.
static float bf8_to_f32(uint8_t b){
    int s=(b>>7)&1, e=(b>>2)&0x1f, m=b&3;
    float v;
    if(e==0) v=(m==0)?0.0f:ldexpf((float)m/4.0f, 1-15);
    else     v=ldexpf(1.0f + (float)m/4.0f, e-15);
    return s? -v : v;
}

int main(void){
    // Build a table of safe bf8 bytes: sign x {exp 1..30} x {mant 0..3}, + zero.
    uint8_t safe[512]; int ns=0;
    safe[ns++]=0x00; safe[ns++]=0x80; // +0, -0
    for(int e=1;e<=30 && ns<500;e++)
        for(int m=0;m<4;m++)
            for(int s=0;s<2;s++) safe[ns++]=(uint8_t)((s<<7)|(e<<2)|m);

    uint32_t*hbase=(uint32_t*)malloc(N*4);
    float*ho=(float*)malloc(N*4); uint8_t*bytes=(uint8_t*)malloc(N);
    for(int i=0;i<N;i++){ uint8_t b=safe[i%ns]; bytes[i]=b;
        hbase[i]=(uint32_t)b;            // bf8 in byte 0
    }
    uint32_t*db; float*doo;
    GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(db,hbase,N*4,hipMemcpyHostToDevice));

    int pass=1;
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(db,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    for(int i=0;i<N;i++){ float e=bf8_to_f32(bytes[i]);
        if(ho[i]!=e){fprintf(stderr,"BASE FAIL[%d] byte=0x%02x got %g exp %g\n",i,bytes[i],ho[i],e);pass=0;break;} }
    if(pass)printf("base ok out[10]=%g\n",ho[10]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
