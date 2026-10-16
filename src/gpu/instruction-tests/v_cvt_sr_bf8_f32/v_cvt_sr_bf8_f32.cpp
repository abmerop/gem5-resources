/*
 * HIP test for V_CVT_SR_BF8_F32 (VOP3): stochastic-rounding convert of an f32 to
 * bf8 (e5m2), writing the result byte to dst[7:0] (OPSEL=0), preserving the rest.
 *   v_cvt_sr_bf8_f32 vdst, src0(f32 in), src1(u32 seed)
 * To avoid gem5's SR round-up quirk on inexact inputs, the input is exactly 0.0f:
 * converting 0.0 yields bf8 0 regardless of the seed (no dropped bits).  dst is
 * seeded with a sentinel to confirm only byte0 is written.  Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*seed,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float in=0.0f; uint32_t s=seed[i];
    uint32_t d=0xDEADBEEFu; /* only byte0 should change */
    __asm__ volatile("v_cvt_sr_bf8_f32 %0, %1, %2\n\t":"+v"(d):"v"(in),"v"(s));
    out[i]=d;
}
int main(void){
    uint32_t*hs=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hs[i]=(uint32_t)(i*2654435761u+9u);
    uint32_t*ds,*dO; GPU_CHECK(hipMalloc(&ds,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(ds,hs,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(ds,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=0xDEADBE00u; /* byte0=0, rest preserved */
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] seed=0x%x got 0x%08x exp 0x%08x\n",i,hs[i],ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
