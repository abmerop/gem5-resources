/*
 * HIP test for BUFFER_WBL2 (ENC_MUBUF, write back the L2 cache).
 *
 * gem5 (src/arch/amdgpu/common/insts/mubuf.cc): this is a cache-maintenance
 * op with no register/memory data effect.  It is a NOP unless SC1 is set, in
 * which case it issues a system write-back; either way it does not change
 * program-visible memory contents.
 *
 * Verification strategy: each thread writes a known value to global memory,
 * then executes BUFFER_WBL2 (with sc1 to exercise the write-back path), then
 * the value is read back and verified.  A pass proves the instruction decoded
 * and executed without corrupting memory or crashing the simulator.  Pattern C.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*in, uint32_t*out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t v = in[i]*3u + 1u;
    out[i] = v;                        /* global store */
    __asm__ volatile("buffer_wbl2 sc1\n\t"
                     "s_waitcnt vmcnt(0)\n\t" ::: "memory");
    /* Re-read our own store through memory to confirm it survived the WB. */
    out[i] = out[i] + 0u;
}

int main(void){
    uint32_t*h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++)h_in[i]=(uint32_t)(i*2654435761u+7u);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t e=h_in[i]*3u+1u;
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%u\n",h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
