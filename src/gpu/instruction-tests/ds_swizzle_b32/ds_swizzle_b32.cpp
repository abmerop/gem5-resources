/*
 * HIP test for DS_SWIZZLE_B32 (ENC_DS, cross-lane swizzle, no LDS access).
 * QDMode (ds_pattern bit15 set): within each group of 4 lanes, lane (L+k) reads
 * data[L + bits(pattern, 2k+1:2k)].  We use pattern 0x801B (QUAD_PERM reverse):
 * k=0->+3, k=1->+2, k=2->+1, k=3->+0, i.e. out[L+k] = data[L + (3-k)].
 * Full wave, all lanes active.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t d=in[i],r;
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:0x801b\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(r):"v"(d));
    out[i]=r;
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hi[i]=(uint32_t)(i*2654435761u+123u);
    uint32_t*di,*dO; GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int wbase=(i/BLOCK_SIZE)*BLOCK_SIZE, lane=i%BLOCK_SIZE, L=lane&~3, k=lane&3;
        uint32_t exp=hi[wbase + L + (3-k)];
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;}
    }
    if(pass)printf("out[0..3]=%08x %08x %08x %08x\n",ho[0],ho[1],ho[2],ho[3]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
