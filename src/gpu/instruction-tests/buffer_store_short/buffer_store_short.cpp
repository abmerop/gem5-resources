/* HIP test for BUFFER_STORE_SHORT (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint16_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;u32x4 srd;srd.x=(uint32_t)(base&0xffffffffu);srd.y=(uint32_t)((base>>32)&0xffffu);srd.z=0xffffffffu;srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t val=(uint32_t)(0xABCD1200u+i*7u); uint32_t voff=(uint32_t)(i*2);
    __asm__ volatile("buffer_store_short %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t"::"v"(val),"v"(voff),"s"(srd):"memory");
}
int main(void){
    uint16_t*h_out=(uint16_t*)malloc(N*sizeof(uint16_t));
    uint16_t*d_out;GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint16_t)));GPU_CHECK(hipMemset(d_out,0,N*sizeof(uint16_t)));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint16_t),hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){uint16_t e=(uint16_t)((uint32_t)(0xABCD1200u+i*7u)&0xffffu);if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] %u!=%u\n",i,(unsigned)h_out[i],(unsigned)e);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
