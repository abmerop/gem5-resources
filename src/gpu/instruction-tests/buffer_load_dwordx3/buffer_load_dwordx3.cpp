/* HIP test for BUFFER_LOAD_DWORDX3 (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;u32x4 srd;srd.x=(uint32_t)(base&0xffffffffu);srd.y=(uint32_t)((base>>32)&0xffffu);srd.z=0xffffffffu;srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t voff=(uint32_t)(i*12); u32x3 d;
    __asm__ volatile("buffer_load_dwordx3 %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t":"=v"(d):"v"(voff),"s"(srd):"memory");
    out[i*3+0]=d[0];
    out[i*3+1]=d[1];
    out[i*3+2]=d[2];
}
int main(void){
    uint32_t*h_src=(uint32_t*)malloc(N*3*4),*h_out=(uint32_t*)malloc(N*3*4);
    for(int i=0;i<N*3;i++)h_src[i]=(uint32_t)(i*2654435761u+7u);
    uint32_t*d_src,*d_out;GPU_CHECK(hipMalloc(&d_src,N*3*4));GPU_CHECK(hipMalloc(&d_out,N*3*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*3*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N*3;i++)if(h_out[i]!=h_src[i]){fprintf(stderr,"FAIL[%d] %u!=%u\n",i,h_out[i],h_src[i]);pass=0;break;}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
