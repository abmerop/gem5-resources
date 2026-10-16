/* HIP test for FLAT_STORE_DWORDX4 (gfx950/CDNA4 load/store roundtrip). */
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
    uint32_t base=(uint32_t)(i*2654435761u+7u); u32x4 d;
    d[0]=base+0;
    d[1]=base+1;
    d[2]=base+2;
    d[3]=base+3;
    uint64_t addr=(uint64_t)(uintptr_t)(&out[i*4]);
    __asm__ volatile("flat_store_dwordx4 %0, %1\n\ts_waitcnt vmcnt(0)\n\t"::"v"(addr),"v"(d):"memory");
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(N*4*4);
    uint32_t*d_out;GPU_CHECK(hipMalloc(&d_out,N*4*4));GPU_CHECK(hipMemset(d_out,0,N*4*4));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4*4,hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){uint32_t base=(uint32_t)(i*2654435761u+7u);for(int k=0;k<4;k++){uint32_t e=base+k;if(h_out[i*4+k]!=e){fprintf(stderr,"FAIL[%d,%d] %u!=%u\n",i,k,h_out[i*4+k],e);pass=0;break;}}if(!pass)break;}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
