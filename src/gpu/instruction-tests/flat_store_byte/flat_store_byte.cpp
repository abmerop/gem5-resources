/* HIP test for FLAT_STORE_BYTE (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint8_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t val=(uint32_t)(0xABCD1200u + i*7u);
    uint64_t addr=(uint64_t)(uintptr_t)(&out[i]);
    __asm__ volatile("flat_store_byte %0, %1\n\ts_waitcnt vmcnt(0)\n\t"::"v"(addr),"v"(val):"memory");
}
int main(void){
    uint8_t*h_out=(uint8_t*)malloc(N*sizeof(uint8_t));
    uint8_t*d_out;GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint8_t)));GPU_CHECK(hipMemset(d_out,0,N*sizeof(uint8_t)));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint8_t),hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){uint8_t e=(uint8_t)((uint32_t)(0xABCD1200u+i*7u)&0xffu);if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] %u!=%u\n",i,(unsigned)h_out[i],(unsigned)e);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
