/* HIP test for FLAT_LOAD_USHORT (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint16_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&src[i]);
    uint32_t d;
    __asm__ volatile("flat_load_ushort %0, %1\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"=v"(d):"v"(addr):"memory");
    out[i]=d;
}
int main(void){
    uint16_t*h_src=(uint16_t*)malloc(N*sizeof(uint16_t)); uint32_t*h_out=(uint32_t*)malloc(N*sizeof(uint32_t));
    for(int i=0;i<N;i++)h_src[i]=(uint16_t)(i*131+((i%2)?-7:200));
    uint16_t*d_src;uint32_t*d_out;GPU_CHECK(hipMalloc(&d_src,N*sizeof(uint16_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){ uint32_t e=(uint32_t)h_src[i]; if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] %u!=%u\n",i,h_out[i],e);pass=0;break;} }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
