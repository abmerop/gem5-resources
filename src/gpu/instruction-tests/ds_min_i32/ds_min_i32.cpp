/*
 * HIP test for DS_MIN_I32 (ENC_DS, signed-min into LDS, no return).
 *   LDS[addr] = min(LDS[addr], DATA)  (signed 32-bit)
 * Each thread owns a 1-dword LDS slot: seed init, ds_min_i32 with data, read
 * back, verify. Pattern C, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_min_i32 ds_min_i32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int32_t*init,const int32_t*data,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ int32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) int32_t*lp=(__attribute__((address_space(3))) int32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; int32_t v=data[i];
    __asm__ volatile("ds_min_i32 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    int32_t*h_i=(int32_t*)malloc(N*4),*h_d=(int32_t*)malloc(N*4),*h_o=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_i[i]=(int32_t)(i*131-9000);h_d[i]=(int32_t)(-i*177+5000);}
    int32_t*di,*dd,*doo;GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(di,h_i,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t exp=h_i[i]<h_d[i]?h_i[i]:h_d[i];
        if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_o[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
