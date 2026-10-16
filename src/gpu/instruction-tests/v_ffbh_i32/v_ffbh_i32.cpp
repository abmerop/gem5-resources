/*
 * HIP test for V_FFBH_I32 (VOP1, find-first-bit-high, signed).
 * gem5 (vop1.cc): vdst = firstOppositeSignBit(src) -- index (from MSB) of the
 * first bit differing from the sign bit; 0xFFFFFFFF if src is 0 or -1.
 * gem5 panics on DPP/SDWA for this op, so base (e32) only.
 * Integer semantics -> bit-exact compare.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const int32_t*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int32_t s0=a[i]; uint32_t d;
    __asm__ volatile("v_ffbh_i32_e32 %0, %1\n\t":"=v"(d):"v"(s0));
    out[i]=d;
}

// Host mirror of gem5 firstOppositeSignBit(ScalarRegI32).
static uint32_t ref_ffbh(int32_t val){
    bool sign_bit = (val & 0x80000000) != 0;
    uint32_t tmp_val=0; int count=0;
    if(!val || val==-1) return 0xFFFFFFFFu;
    for(int i=0;i<32;++i){
        tmp_val = (uint32_t)val & (0x80000000u >> i);
        if(!sign_bit){ if(tmp_val) return (uint32_t)count; }
        else { if(!tmp_val) return (uint32_t)count; }
        ++count;
    }
    return 0xFFFFFFFFu;
}

int main(void){
    int32_t*ha=(int32_t*)malloc(N*4); uint32_t*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        // spread of signs/magnitudes incl 0 and -1 edge cases
        int32_t v = (int32_t)((i*2654435761u) ^ (i<<7));
        if(i==0) v=0; else if(i==1) v=-1; else if(i==2) v=1;
        else if(i==3) v=-2; else if(i==4) v=0x40000000; else if(i==5) v=(int32_t)0x80000000;
        ha[i]=v;
    }
    int32_t*da; uint32_t*doo;
    GPU_CHECK(hipMalloc(&da,N*4)); GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,doo,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t e=ref_ffbh(ha[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] in=0x%08x got=0x%08x exp=0x%08x\n",i,(uint32_t)ha[i],ho[i],e);pass=0;break;} }
    if(pass)printf("out[2]=%u out[4]=%u\n",ho[2],ho[4]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
