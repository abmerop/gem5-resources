/*
 * HIP test for DS_MIN_U32 (ENC_DS, unsigned-min into LDS, no return).
 *   LDS[addr] = min(LDS[addr], DATA)  (unsigned 32-bit)
 * Pattern C, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_min_u32 ds_min_u32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*init,const uint32_t*data,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp,v=data[i];
    __asm__ volatile("ds_min_u32 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4),*h_o=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_i[i]=(uint32_t)(i*2654435761u);h_d[i]=(uint32_t)((N-i)*40503u+7u);}
    uint32_t*di,*dd,*doo;GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(di,h_i,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=h_i[i]<h_d[i]?h_i[i]:h_d[i];
        if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_o[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
