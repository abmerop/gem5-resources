/*
 * HIP test for DS_XOR_RTN_B64 (ENC_DS, 64-bit LDS bitwise-XOR, return-old).
 *   old = LDS[addr]; LDS[addr] = old ^ DATA; VDST = old
 * Each thread owns a disjoint 64-bit LDS slot. Pattern C, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_xor_rtn_b64 ds_xor_rtn_b64.cpp
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
    __asm__ volatile("ds_xor_rtn_b64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint64_t*h_i=(uint64_t*)malloc(N*8),*h_d=(uint64_t*)malloc(N*8),*h_o=(uint64_t*)malloc(N*8),*h_m=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){h_i[i]=(uint64_t)i*0x0123456789abcdefULL;h_d[i]=(uint64_t)(i+1)*0xfedcba9876543210ULL;}
    uint64_t*di,*dd,*doo,*dm;GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&doo,N*8));GPU_CHECK(hipMalloc(&dm,N*8));
    GPU_CHECK(hipMemcpy(di,h_i,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,doo,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,doo,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(h_m,dm,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t eo=h_i[i],em=h_i[i]^h_d[i];
        if(h_o[i]!=eo||h_m[i]!=em){fprintf(stderr,"FAIL[%d]\n",i);pass=0;break;}}
    if(pass)printf("old[0]=0x%llx mem[0]=0x%llx\n",(unsigned long long)h_o[0],(unsigned long long)h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
