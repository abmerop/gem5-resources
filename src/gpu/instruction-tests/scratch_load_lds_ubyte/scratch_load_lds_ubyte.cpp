/*
 * HIP test for SCRATCH_LOAD_LDS_UBYTE (ENC_FLAT_SCRATCH, direct scratch->LDS DMA
 * of one zero-extended byte per lane).  Each lane stores a byte to its private
 * scratch slot, then DMAs it into LDS (m0 = &lds); per the gem5 ldsComplete
 * model each lane writes a full zero-extended dword to lds_base + lane*4, so
 * reading lds[t] yields the zero-extended byte.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint8_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=0xDEAD; __syncthreads();
    volatile int pad[8]; pad[t&7]=i;
    int val=(int)g[i];
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;
    __asm__ volatile(
        "scratch_store_byte off, %1, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        "s_mov_b32 m0, %2\n\t"
        "scratch_load_lds_ubyte off, off offset:0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        ::"v"(lds),"v"(val),"s"(lds_off):"memory","m0");
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
