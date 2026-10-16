/*
 * HIP test for GLOBAL_LOAD_LDS_DWORDX3 (ENC_FLAT_GLBL, direct global->LDS DMA of
 * 3 dwords per lane).  Per the gem5 ldsComplete model lane L writes its 3 dwords
 * to lds_base + L*3*4 + i*4.  SADDR form (base in SGPR pair, per-lane byte offset
 * in VGPR).  Verified against the source dwords.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE*3]; int t=threadIdx.x;
    for(int j=0;j<3;j++) lds[t*3+j]=0; __syncthreads();
    const uint32_t*base=g+(size_t)blockIdx.x*BLOCK_SIZE*3;
    uint64_t saddr=(uint64_t)(uintptr_t)base;
    uint32_t voff=(uint32_t)(t*12);
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;
    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"
        "global_load_lds_dwordx3 %0, %1\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        ::"v"(voff),"s"(saddr),"s"(lds_off):"memory","m0");
    __syncthreads();
    for(int j=0;j<3;j++) out[i*3+j]=lds[t*3+j];
}
int main(void){
    uint32_t*hg=(uint32_t*)malloc((size_t)N*3*4),*ho=(uint32_t*)malloc((size_t)N*3*4);
    for(size_t k=0;k<(size_t)N*3;k++) hg[k]=(uint32_t)(k*0x9e3779b9u+11u);
    uint32_t*dg,*dO; GPU_CHECK(hipMalloc(&dg,(size_t)N*3*4));GPU_CHECK(hipMalloc(&dO,(size_t)N*3*4));
    GPU_CHECK(hipMemcpy(dg,hg,(size_t)N*3*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dg,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,(size_t)N*3*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(size_t k=0;k<(size_t)N*3;k++){if(ho[k]!=hg[k]){fprintf(stderr,"FAIL[%zu] got 0x%08x exp 0x%08x\n",k,ho[k],hg[k]);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
