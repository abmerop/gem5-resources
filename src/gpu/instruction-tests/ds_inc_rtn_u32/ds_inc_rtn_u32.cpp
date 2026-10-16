/*
 * HIP test for DS_INC_RTN_U32 (ENC_DS LDS atomic, gfx950/CDNA4).
 * Each thread owns a private 1-slot LDS location (no contention). Base only.
 * Host reference matches gem5 atomic functor semantics exactly.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_inc_rtn_u32 ds_inc_rtn_u32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*init,const uint32_t*data,uint32_t*old_out,uint32_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v=data[i];
    uint32_t old;
    __asm__ volatile("ds_inc_rtn_u32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint32_t*h_init=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_data=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_mem=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t*h_old=(uint32_t*)malloc(N*sizeof(uint32_t));
    for(int i=0;i<N;i++){h_init[i]=(uint32_t)(i & 0x3f);h_data[i]=(uint32_t)(1000u + i);}
    uint32_t*di,*dd,*dmem; GPU_CHECK(hipMalloc(&di,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&dmem,N*sizeof(uint32_t)));
    uint32_t*dold;GPU_CHECK(hipMalloc(&dold,N*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(di,h_init,N*sizeof(uint32_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_data,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dold,dmem,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,dmem,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,dold,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t eo=h_init[i]; uint32_t em=h_init[i] + 1u;
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(exp %u) mem=%u(exp %u)\n",i,h_old[i],eo,h_mem[i],em);pass=0;break;} }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
