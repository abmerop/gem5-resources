/*
 * HIP test for DS_OR_B64 (ENC_DS LDS atomic, gfx950/CDNA4).
 * Each thread owns a private 1-slot LDS location (no contention). Base only.
 * Host reference matches gem5 atomic functor semantics exactly.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_or_b64 ds_or_b64.cpp
 */
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
    __asm__ volatile("ds_or_b64 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads();
    mem_out[i]=lds[t];
}
int main(void){
    uint64_t*h_init=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_data=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_mem=(uint64_t*)malloc(N*sizeof(uint64_t));
    
    for(int i=0;i<N;i++){h_init[i]=((uint64_t)(i*2654435761u)<<20 | (uint32_t)(i*7u));h_data[i]=((uint64_t)(i*40503u+7u)<<15 | 0x55u);}
    uint64_t*di,*dd,*dmem; GPU_CHECK(hipMalloc(&di,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&dmem,N*sizeof(uint64_t)));
    
    GPU_CHECK(hipMemcpy(di,h_init,N*sizeof(uint64_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_data,N*sizeof(uint64_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dmem,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,dmem,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    
    int pass=1;
    for(int i=0;i<N;i++){ uint64_t em=h_init[i] | h_data[i];
        if(h_mem[i]!=em){fprintf(stderr,"FAIL[%d] mem=%lu(exp %lu)\n",i,h_mem[i],em);pass=0;break;} }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
