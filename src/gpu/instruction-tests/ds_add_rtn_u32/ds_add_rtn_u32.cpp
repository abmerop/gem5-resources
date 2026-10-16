/*
 * HIP test for DS_ADD_RTN_U32 (ENC_DS, 32-bit LDS add, return-old).
 *   old = LDS[addr]; LDS[addr] = old + DATA; VDST = old
 * Each thread owns a 1-dword LDS slot (no contention). Pattern C, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_add_rtn_u32 ds_add_rtn_u32.cpp
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
    uint32_t addr=(uint32_t)(uintptr_t)lp,v=data[i],old;
    __asm__ volatile("ds_add_rtn_u32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint32_t*h_init=(uint32_t*)malloc(N*4),*h_data=(uint32_t*)malloc(N*4),*h_old=(uint32_t*)malloc(N*4),*h_mem=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_init[i]=(uint32_t)(i*2654435761u);h_data[i]=(uint32_t)(i*40503u+7u);}
    uint32_t*di,*dd,*dold,*dmem;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dold,N*4));GPU_CHECK(hipMalloc(&dmem,N*4));
    GPU_CHECK(hipMemcpy(di,h_init,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dold,dmem,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_old,dold,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(h_mem,dmem,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t eo=h_init[i],em=h_init[i]+h_data[i];
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(exp %u) mem=%u(exp %u)\n",i,h_old[i],eo,h_mem[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%u mem[0]=%u\n",h_old[0],h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
