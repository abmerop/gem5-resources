/* HIP test for BUFFER_STORE_DWORDX3 (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;u32x4 srd;srd.x=(uint32_t)(base&0xffffffffu);srd.y=(uint32_t)((base>>32)&0xffffu);srd.z=0xffffffffu;srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t base_v=(uint32_t)(i*2654435761u+7u); u32x3 d;
    d[0]=base_v+0;
    d[1]=base_v+1;
    d[2]=base_v+2;
    uint32_t voff=(uint32_t)(i*12);
    __asm__ volatile("buffer_store_dwordx3 %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t"::"v"(d),"v"(voff),"s"(srd):"memory");
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(N*3*4);
    uint32_t*d_out;GPU_CHECK(hipMalloc(&d_out,N*3*4));GPU_CHECK(hipMemset(d_out,0,N*3*4));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){uint32_t b=(uint32_t)(i*2654435761u+7u);for(int k=0;k<3;k++){uint32_t e=b+k;if(h_out[i*3+k]!=e){fprintf(stderr,"FAIL[%d,%d] %u!=%u\n",i,k,h_out[i*3+k],e);pass=0;break;}}if(!pass)break;}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
