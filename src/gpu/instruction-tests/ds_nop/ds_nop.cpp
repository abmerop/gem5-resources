/*
 * HIP test for DS_NOP (ENC_DS, LDS no-op), pattern C.
 *
 * gem5 (src/arch/amdgpu/common/insts/ds.cc): DS_NOP performs no memory or
 * register action (it only balances the wavefront's DS/LGKM issue counters).
 *
 * IMPORTANT: DS_NOP must not be placed while a real DS (LDS) memory op is still
 * in flight -- gem5 models DS_NOP by decrementing the wavefront's shared LGKM
 * counter, which would prematurely satisfy the s_waitcnt for a pending LDS
 * access.  So this test issues DS_NOP with no other DS op outstanding: it does a
 * pure global-memory computation with DS_NOP inserted in the middle, and
 * verifies the surrounding computation.  A pass proves DS_NOP decoded and
 * executed without disturbing register/memory state or crashing the simulator.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t v=in[i]*2u+1u;
    __asm__ volatile("ds_nop\n\t":::"memory");
    out[i]=v ^ 0x5a5a5a5au;
}
int main(void){
    uint32_t*h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++)h_in[i]=(uint32_t)(i*2654435761u+3u);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t e=(h_in[i]*2u+1u)^0x5a5a5a5au;
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%u\n",h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
