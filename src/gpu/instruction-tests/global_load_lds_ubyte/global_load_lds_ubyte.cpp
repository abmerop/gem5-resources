/*
 * HIP test for GLOBAL_LOAD_LDS_UBYTE (ENC_FLAT_GLBL, direct global->LDS DMA of
 * one byte per lane, zero-extended).  Per the gem5 ldsComplete model each lane
 * writes a full zero-extended dword to lds_base + lane*4, so a dword LDS slot
 * per lane holds the zero-extended byte.  SADDR form (base in SGPR pair, per-lane
 * byte offset in VGPR).  Verified against the zero-extended source byte.  base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint8_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE];   /* one dword slot per lane */
    int t=threadIdx.x;
    lds[t]=0xDEAD; __syncthreads();
    const uint8_t*base=g+(size_t)blockIdx.x*BLOCK_SIZE;
    uint64_t saddr=(uint64_t)(uintptr_t)base;
    uint32_t voff=(uint32_t)t;
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;   /* array base, dword-strided */
    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"
        "global_load_lds_ubyte %0, %1\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        ::"v"(voff),"s"(saddr),"s"(lds_off):"memory","m0");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint8_t*hg=(uint8_t*)malloc(N); uint32_t*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hg[i]=(uint8_t)(i*131+7);
    uint8_t*dg; uint32_t*dO; GPU_CHECK(hipMalloc(&dg,N));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(dg,hg,N,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dg,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(uint32_t)hg[i];
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
