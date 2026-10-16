/*
 * HIP test for DS_DEC_U32 (ENC_DS LDS atomic dec-with-wrap, gfx950).
 * gem5 functor AtomicOpDecWraparound: mem = (old==0 || old>data) ? data : old-1.
 * Private per-thread LDS slot (no contention): single functor application.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*init,const uint32_t*data,uint32_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v=data[i];
    __asm__ volatile("ds_dec_u32 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads(); mem_out[i]=lds[t];
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*hm=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i%50);hd[i]=25u;}
    uint32_t*di,*dd,*dm; GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dm,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t em=(hi[i]==0u||hi[i]>hd[i])?hd[i]:hi[i]-1u;
        if(hm[i]!=em){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,hm[i],em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
