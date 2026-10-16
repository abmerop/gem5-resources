/*
 * HIP test for GLOBAL_ATOMIC_AND (ENC_FLAT_GLBL, 32-bit atomic AND, return-old).
 *   old = MEM[addr]; MEM[addr] = old & DATA; VDST = old
 * Emitted from atomicAnd() with its return value used. Disjoint per-thread
 * slots (no contention): mem = init & data.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,const uint32_t*data,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    old_out[i]=atomicAnd(&mem[i],data[i]);
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*hm=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i*2654435761u);hd[i]=(uint32_t)(i*40503u+0x0F0F0F0Fu);}
    uint32_t*dm,*dd,*dold; GPU_CHECK(hipMalloc(&dm,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dold,N*4));
    GPU_CHECK(hipMemcpy(dm,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dm,dd,dold,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(ho,dold,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t eo=hi[i],em=hi[i]&hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(exp %u) mem=%u(exp %u)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
