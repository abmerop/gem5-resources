/* HIP test for BUFFER_LOAD_SBYTE (gfx950/CDNA4 load/store roundtrip). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const int8_t*src,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;u32x4 srd;srd.x=(uint32_t)(base&0xffffffffu);srd.y=(uint32_t)((base>>32)&0xffffu);srd.z=0xffffffffu;srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t voff=(uint32_t)(i*1); int32_t d;
    __asm__ volatile("buffer_load_sbyte %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t":"=v"(d):"v"(voff),"s"(srd):"memory");
    out[i]=d;
}
int main(void){
    int8_t*h_src=(int8_t*)malloc(N*sizeof(int8_t)); int32_t*h_out=(int32_t*)malloc(N*sizeof(int32_t));
    for(int i=0;i<N;i++)h_src[i]=(int8_t)(i*131+((i%2)?-7:200));
    int8_t*d_src;int32_t*d_out;GPU_CHECK(hipMalloc(&d_src,N*sizeof(int8_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int32_t)));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*sizeof(int8_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){ int32_t e=(int32_t)h_src[i]; if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] %d!=%d\n",i,h_out[i],e);pass=0;break;} }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
