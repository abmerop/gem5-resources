/*
 * HIP test for V_PERMLANE32_SWAP_B32 (VOP1, cross-lane swap by 32).
 * gem5 (vop1.cc): swaps lane L with lane L+32 across the wave, i.e.
 *   vdst[L] = src[L ^ 32]   (bit 5 of the lane index flipped).
 * Ignores EXEC mask; panics on DPP/SDWA -> base only. Integer -> bit-exact.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i],d;
    __asm__ volatile("v_permlane32_swap_b32 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}

int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) ha[i]=(uint32_t)(0x5A000000u + i);
    uint32_t*da,*doo; GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int base=(i/BLOCK_SIZE)*BLOCK_SIZE, lane=i%BLOCK_SIZE;
        uint32_t e=ha[base + (lane ^ 32)];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] lane=%d got 0x%08x exp 0x%08x\n",i,lane,ho[i],e);pass=0;break;} }
    if(pass)printf("out[0]=0x%08x out[32]=0x%08x\n",ho[0],ho[32]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
