/*
 * HIP test for DS_MAX_U32 (ENC_DS, unsigned-max into LDS, no return).
 *
 *   LDS[addr] = max(LDS[addr], DATA)   (unsigned 32-bit)
 *
 * Each thread owns a 1-dword LDS slot: seed it with init, ds_max_u32 with data,
 * then read it back and verify max(init, data).  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_max_u32 ds_max_u32.cpp
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
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v=data[i];
    __asm__ volatile("ds_max_u32 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint32_t*h_init=(uint32_t*)malloc(N*4),*h_data=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_init[i]=(uint32_t)(i*2654435761u);h_data[i]=(uint32_t)((N-i)*40503u+7u);}
    uint32_t*d_init,*d_data,*d_out;
    GPU_CHECK(hipMalloc(&d_init,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_init,h_init,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_init,d_data,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=h_init[i]>h_data[i]?h_init[i]:h_data[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
