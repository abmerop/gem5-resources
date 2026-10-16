/*
 * HIP test for BUFFER_INV (ENC_MUBUF, cache invalidate).
 * gem5 (mubuf.cc) implements this as a cache-control op with no data effect.
 * We compute a value, store it, issue buffer_inv, then re-read our own store
 * through memory and confirm it is unchanged.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t v=in[i]*3u+1u;
    out[i]=v;
    __asm__ volatile("buffer_inv sc1\n\ts_waitcnt vmcnt(0)\n\t":::"memory");
    out[i]=out[i]+0u;
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hi[i]=(uint32_t)(i*7+3);
    uint32_t*di,*doo; GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t e=hi[i]*3u+1u; if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%u\n",ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
