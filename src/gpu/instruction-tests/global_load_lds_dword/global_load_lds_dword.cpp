/*
 * HIP test for GLOBAL_LOAD_LDS_DWORD (ENC_FLAT_GLBL, direct global->LDS DMA of
 * one dword per lane).  Each active lane reads 4 bytes from its global address
 * and the hardware writes them to LDS at m0-relative offset lds[L].  The gfx950
 * assembler requires the SADDR form (base in an SGPR pair, per-lane byte offset
 * in a VGPR).  Verified against the source dword.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE];
    int t=threadIdx.x;
    lds[t]=0; __syncthreads();
    const uint32_t*base=g+(size_t)blockIdx.x*BLOCK_SIZE;
    uint64_t saddr=(uint64_t)(uintptr_t)base;
    uint32_t voff=(uint32_t)(t*4);
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;
    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"
        "global_load_lds_dword %0, %1\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        ::"v"(voff),"s"(saddr),"s"(lds_off):"memory","m0");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint32_t*hg=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hg[i]=(uint32_t)(i*0x9e3779b9u+11u);
    uint32_t*dg,*dO; GPU_CHECK(hipMalloc(&dg,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(dg,hg,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dg,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){if(ho[i]!=hg[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],hg[i]);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
