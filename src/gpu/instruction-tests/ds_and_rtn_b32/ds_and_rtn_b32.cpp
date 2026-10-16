/*
 * HIP test for DS_AND_RTN_B32 (ENC_DS, 32-bit LDS atomic, return-old).
 *   old = LDS[addr]; LDS[addr] = a & b; VDST = old.
 * Each lane owns a private LDS slot (no contention).  Pattern C, base only.
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
    __asm__ volatile("ds_and_rtn_b32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4),*hm=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i*2654435761u+123u);hd[i]=(uint32_t)(i*40503u+7u);}
    uint32_t*di,*dd,*doo,*dm;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&doo,N*4));GPU_CHECK(hipMalloc(&dm,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,doo,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t a=hi[i],b=hd[i];uint32_t em=a & b;
        if(ho[i]!=a||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",i,ho[i],a,hm[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%u mem[0]=%u\n",ho[0],hm[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
