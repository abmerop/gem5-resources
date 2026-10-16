/*
 * HIP test for DS_ADD_RTN_U64 (ENC_DS 64-bit LDS add, return-old, gfx950).
 *   old = LDS[addr]; LDS[addr] = old + DATA; VDST = old
 * Private per-thread 64-bit LDS slot (no contention).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*init,const uint64_t*data,uint64_t*old_out,uint64_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint64_t*lp=(__attribute__((address_space(3))) uint64_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint64_t v=data[i],old;
    __asm__ volatile("ds_add_rtn_u64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint64_t*hi=(uint64_t*)malloc(N*8),*hd=(uint64_t*)malloc(N*8),*ho=(uint64_t*)malloc(N*8),*hm=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){hi[i]=(uint64_t)i*0x100000001ULL+7ULL;hd[i]=(uint64_t)(i*2654435761u)*3ULL+1ULL;}
    uint64_t*di,*dd,*dold,*dm; GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&dold,N*8));GPU_CHECK(hipMalloc(&dm,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dold,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dold,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm,dm,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t eo=hi[i],em=hi[i]+hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%llu(exp %llu) mem=%llu(exp %llu)\n",i,(unsigned long long)ho[i],(unsigned long long)eo,(unsigned long long)hm[i],(unsigned long long)em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
