/* HIP test for DS_AND_B64 (ENC_DS 64-bit LDS bitwise AND, gfx950).
 * LDS[a]=old & DATA. Private per-thread 64-bit slot (no contention). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*init,const uint64_t*data,uint64_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint64_t*lp=(__attribute__((address_space(3))) uint64_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint64_t v=data[i];
    __asm__ volatile("ds_and_b64 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads(); mem_out[i]=lds[t];
}
int main(void){
    uint64_t*hi=(uint64_t*)malloc(N*8),*hd=(uint64_t*)malloc(N*8),*hm=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){hi[i]=(uint64_t)(i*2654435761u)*0x100000001ULL;hd[i]=0x0F0F0F0F0F0F0F0FULL^((uint64_t)i<<3);}
    uint64_t*di,*dd,*dm; GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&dm,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t em=hi[i]&hd[i];
        if(hm[i]!=em){fprintf(stderr,"FAIL[%d] got %llu exp %llu\n",i,(unsigned long long)hm[i],(unsigned long long)em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
