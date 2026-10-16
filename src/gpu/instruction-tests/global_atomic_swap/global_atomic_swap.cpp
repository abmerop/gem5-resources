/*
 * HIP test for GLOBAL_ATOMIC_SWAP (32-bit atomic, return-old).
 *   old = MEM[addr]; MEM[addr] = b; VDST = old.
 * Each lane owns a disjoint 4-byte slot (no contention).  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,const uint32_t*data,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]); uint32_t v=data[i],old;
    __asm__ volatile("global_atomic_swap %0, %1, %2, off sc0\n\ts_waitcnt vmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    old_out[i]=old;
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4),*hm=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i*2654435761u+17u);hd[i]=(uint32_t)(i*40503u+3u);}
    uint32_t*dm,*dd,*doo;
    GPU_CHECK(hipMalloc(&dm,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(dm,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dm,dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t a=hi[i],b=hd[i];uint32_t em=b;
        if(ho[i]!=a||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",i,ho[i],a,hm[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%u mem[0]=%u\n",ho[0],hm[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
